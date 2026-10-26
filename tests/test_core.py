"""不依赖 AstrBot 的单元测试：python -m pytest tests"""

import asyncio
import io
import json
import sys
import zipfile
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audit  # noqa: E402
import gallery  # noqa: E402
import prompting  # noqa: E402
import quota  # noqa: E402
import usage  # noqa: E402
from backends import novelai as nai_client  # noqa: E402
from backends.base import DrawError, DrawRequest  # noqa: E402
from backends.novelai import ImageRequest, NovelAIBackend, NovelAIClient, NovelAIError  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def make_zip(data: bytes = PNG) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("image_0.png", data)
    return buf.getvalue()


def make_request(**kw) -> ImageRequest:
    base = dict(
        model="nai-diffusion-4-5-full", prompt="1girl, smile", negative="lowres",
        width=832, height=1216, steps=28, scale=5.0, cfg_rescale=0.0,
        sampler="k_euler_ancestral", noise_schedule="karras", seed=123,
    )
    base.update(kw)
    return ImageRequest(**base)


# ---------------------------------------------------------------- novelai


@pytest.mark.parametrize(
    ("text", "expected"),
    [("832x1216", (832, 1216)), (" 1216 X 832 ", (1216, 832)), ("1024×1024", (1024, 1024)),
     ("bad", (1, 2)), ("0x100", (1, 2)), (None, (1, 2))],
)
def test_parse_size(text, expected):
    assert nai_client.parse_size(text, (1, 2)) == expected


def test_fit_size_free_guard():
    assert nai_client.fit_size(832, 1216) == (832, 1216)
    assert nai_client.fit_size(1024, 1024) == (1024, 1024)
    w, h = nai_client.fit_size(1536, 2048)
    assert w * h <= nai_client.OPUS_FREE_MAX_PIXELS and w % 64 == 0 and h % 64 == 0
    assert nai_client.fit_size(1536, 2048, free_guard=False) == (1536, 2048)
    assert nai_client.fit_size(830, 1220, free_guard=False) == (768, 1216)
    assert nai_client.fit_size(10, 10) == (64, 64)


def test_is_opus_free():
    assert nai_client.is_opus_free("nai-diffusion-4-5-full", 832, 1216, 28)
    assert not nai_client.is_opus_free("nai-diffusion-4-5-full", 832, 1216, 29)
    assert not nai_client.is_opus_free("nai-diffusion-4-5-full", 1088, 1088, 28)
    assert not nai_client.is_opus_free("nai-diffusion-5-full", 832, 1216, 28)


def test_build_payload():
    body = nai_client.build_payload(make_request())
    p = body["parameters"]
    assert body["input"] == "1girl, smile" and body["action"] == "generate"
    assert body["model"] == "nai-diffusion-4-5-full"
    assert p["v4_prompt"]["caption"]["base_caption"] == "1girl, smile"
    assert p["v4_negative_prompt"]["caption"]["base_caption"] == "lowres"
    assert p["negative_prompt"] == "lowres"
    assert p["n_samples"] == 1 and p["seed"] == 123 and p["steps"] == 28
    assert p["deliberate_euler_ancestral_bug"] is False and p["prefer_brownian"] is True
    json.dumps(body)  # 必须可序列化

    other = nai_client.build_payload(make_request(sampler="k_euler", seed=0))["parameters"]
    assert "prefer_brownian" not in other
    assert 1 <= other["seed"] <= nai_client.MAX_SEED


def test_extract_image():
    assert nai_client.extract_image(make_zip()) == PNG
    assert nai_client.extract_image(PNG) == PNG
    with pytest.raises(NovelAIError):
        nai_client.extract_image(b"not an image")


def test_error_from_response():
    e = nai_client.error_from_response(401, {"statusCode": 401, "message": "Invalid token"})
    assert "Token" in str(e) and "Invalid token" in str(e) and not e.retryable
    locked = nai_client.error_from_response(429, {"message": "Concurrent generation is locked"})
    assert locked.retryable and locked.concurrent
    assert nai_client.error_from_response(409, {"message": "concurrent generation"}).concurrent
    assert not nai_client.error_from_response(409, {"message": "conflict"}).retryable
    assert not nai_client.error_from_response(402, {}).retryable
    # 可能已在服务端开始生成的错误不重试
    for status in (500, 504, 524):
        assert not nai_client.error_from_response(status, {}).retryable
    page = "<!DOCTYPE html><html><head><title>502 Bad Gateway</title></head><body>x</body></html>"
    bad_gateway = nai_client.error_from_response(502, page)
    assert bad_gateway.retryable and "<" not in str(bad_gateway) and "502 Bad Gateway" in str(bad_gateway)
    assert "<" not in str(nai_client.error_from_response(503, "<html><body>no title</body></html>"))


def test_subscription_helpers():
    sub = {"tier": 3, "active": True, "trainingStepsLeft": {"fixedTrainingStepsLeft": 10000,
                                                            "purchasedTrainingSteps": 500}}
    assert nai_client.anlas_total(sub) == 10500
    assert nai_client.anlas_total({}) is None
    assert nai_client.tier_name(3) == "Opus" and nai_client.tier_name(None) == "未知"
    assert nai_client.format_expiry(1767225600) == nai_client.format_expiry(1767225600000) != ""
    assert nai_client.format_expiry(None) == ""
    assert nai_client.mask_token("pst-abcdefghijklmnop") == "pst-…mnop"


def run(coro):
    return asyncio.run(coro)


def test_client_generate_success():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=make_zip())

    async def go():
        client = NovelAIClient(transport=httpx.MockTransport(handler))
        try:
            return await client.generate(make_request(), token="pst-x", base_url="https://image.novelai.net/")
        finally:
            await client.close()

    assert run(go()) == PNG
    assert seen["url"] == "https://image.novelai.net/ai/generate-image"
    assert seen["auth"] == "Bearer pst-x"
    assert seen["body"]["parameters"]["width"] == 832


def test_client_generate_retries_on_429():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, json={"statusCode": 429, "message": "Concurrent generation is locked"})
        return httpx.Response(200, content=make_zip())

    async def go():
        client = NovelAIClient(transport=httpx.MockTransport(handler))
        try:
            return await client.generate(make_request(), token="t", retry_delay=0)
        finally:
            await client.close()

    assert run(go()) == PNG and len(calls) == 3


def run_generate(handler):
    async def go():
        client = NovelAIClient(transport=httpx.MockTransport(handler))
        try:
            return await client.generate(make_request(), token="t", retry_delay=0)
        finally:
            await client.close()

    return run(go())


def test_client_generate_retries_on_409_concurrent():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(409, json={"message": "Concurrent generation is locked"})
        return httpx.Response(200, content=make_zip())

    assert run_generate(handler) == PNG and len(calls) == 3


def test_client_generate_does_not_retry_timeout():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(NovelAIError) as info:
        run_generate(handler)
    assert len(calls) == 1 and "超时" in str(info.value)


def test_client_generate_retries_connect_error_once():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(NovelAIError):
        run_generate(handler)
    assert len(calls) == 2


def test_client_refuses_after_close():
    async def go():
        client = NovelAIClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=make_zip())))
        await client.close()
        await client.generate(make_request(), token="t", retry_delay=0)

    with pytest.raises(NovelAIError):
        run(go())


def test_normalize_urls():
    assert nai_client.normalize_base_url("") == nai_client.DEFAULT_BASE_URL
    assert nai_client.normalize_base_url(" image.novelai.net/ ") == "https://image.novelai.net"
    assert nai_client.normalize_base_url("http://relay.example.com/nai/") == "http://relay.example.com/nai"
    assert nai_client.normalize_proxy_url("  ") == ""
    assert nai_client.normalize_proxy_url("127.0.0.1:7890") == "http://127.0.0.1:7890"
    assert nai_client.normalize_proxy_url("socks5://127.0.0.1:7890") == "socks5://127.0.0.1:7890"
    assert nai_client.mask_url_credentials("http://user:pw@1.2.3.4:8080") == "http://***@1.2.3.4:8080"
    assert nai_client.mask_url_credentials("http://1.2.3.4:8080") == "http://1.2.3.4:8080"


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [("image.novelai.net", "https://image.novelai.net/ai/generate-image"),
     ("https://relay.example.com/nai/", "https://relay.example.com/nai/ai/generate-image")],
)
def test_client_uses_configured_base_url(base_url, expected):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=make_zip())

    async def go():
        client = NovelAIClient(transport=httpx.MockTransport(handler))
        try:
            return await client.generate(make_request(), token="t", base_url=base_url)
        finally:
            await client.close()

    assert run(go()) == PNG and seen == [expected]


def test_client_invalid_proxy_reports_error():
    async def go():
        client = NovelAIClient()
        try:
            await client.generate(make_request(), token="t", proxy="ftp://127.0.0.1:21")
        finally:
            await client.close()

    with pytest.raises(NovelAIError) as info:
        run(go())
    assert "代理地址无效" in str(info.value) and not info.value.retryable


def test_client_proxy_error_retried_once():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ProxyError("407 Proxy Authentication Required", request=request)

    with pytest.raises(NovelAIError) as info:
        run_generate(handler)
    assert len(calls) == 2 and "代理" in str(info.value)


def test_client_unsupported_protocol_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.UnsupportedProtocol("Request URL has an unsupported protocol 'ftp://'.", request=request)

    with pytest.raises(NovelAIError) as info:
        run_generate(handler)
    assert len(calls) == 1 and "协议" in str(info.value)


def test_client_generate_no_retry_on_401():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"statusCode": 401, "message": "Unauthorized"})

    async def go():
        client = NovelAIClient(transport=httpx.MockTransport(handler))
        try:
            await client.generate(make_request(), token="t", retry_delay=0)
        finally:
            await client.close()

    with pytest.raises(NovelAIError) as info:
        run(go())
    assert info.value.status == 401 and len(calls) == 1


def test_client_subscription():
    def handler(request):
        assert request.url.path == "/user/subscription"
        return httpx.Response(200, json={"tier": 3, "active": True})

    async def go():
        client = NovelAIClient(transport=httpx.MockTransport(handler))
        try:
            return await client.subscription(token="t")
        finally:
            await client.close()

    assert run(go())["tier"] == 3


def test_novelai_error_is_draw_error():
    assert issubclass(NovelAIError, DrawError)


def test_novelai_backend_settings():
    backend = NovelAIBackend({"token": "pst-x", "portrait_size": "1536x2048", "steps": 40})
    assert backend.missing() == "" and backend.prompt_mode == "tags" and backend.max_concurrency == 1
    w, h = backend.size("portrait")
    assert w * h <= nai_client.OPUS_FREE_MAX_PIXELS and backend.steps == 28  # 免费参数保护
    free = NovelAIBackend({"token": "t", "opus_free_guard": False, "steps": 40, "square_size": "bad"})
    assert free.steps == 40 and free.size("square") == (1024, 1024) and free.size("unknown") == (832, 1216)
    assert "NovelAI Token" in NovelAIBackend({}).missing()
    req = backend.build_request(DrawRequest(prompt="1girl", negative="lowres", orientation="landscape", seed=7))
    assert (req.width, req.height, req.seed, req.model) == (1216, 832, 7, "nai-diffusion-4-5-full")


def test_novelai_backend_generate():
    from backends.base import Clients, HttpPool

    def handler(request):
        body = json.loads(request.content)
        assert body["parameters"]["width"] == 1024 and request.headers["authorization"] == "Bearer t"
        return httpx.Response(200, content=make_zip())

    async def go():
        client = NovelAIClient(transport=httpx.MockTransport(handler))
        try:
            backend = NovelAIBackend({"token": "t"})
            return await backend.generate(Clients(http=HttpPool(), novelai=client),
                                          DrawRequest(prompt="cat", orientation="square"))
        finally:
            await client.close()

    result = run(go())
    assert result.image == PNG and result.mime == "image/png"
    assert result.params["size"] == "1024x1024" and result.params["opus_free"] is True


# ---------------------------------------------------------------- prompting


def test_normalize_orientation():
    assert prompting.normalize_orientation("Portrait") == "portrait"
    assert prompting.normalize_orientation("横图") == "landscape"
    assert prompting.normalize_orientation("square") == "square"
    assert prompting.normalize_orientation("diagonal") == ""
    assert prompting.normalize_orientation(None) == ""


def test_parse_tag_output_json():
    raw = '<think>hmm</think>```json\n{"prompt": "1girl, cat ears, 1girl", "negative": "", "orientation": "landscape"}\n```'
    out = prompting.parse_tag_output(raw)
    assert out == {"prompt": "1girl, cat ears", "negative": "", "orientation": "landscape"}


def test_parse_tag_output_plain_text():
    out = prompting.parse_tag_output("Prompt: 1boy，sword\nnight sky")
    assert out["prompt"] == "1boy, sword, night sky" and out["orientation"] == ""


def test_parse_tag_output_list_and_bad_orientation():
    out = prompting.parse_tag_output('{"prompt": ["a", "b"], "orientation": "weird"}')
    assert out["prompt"] == "a, b" and out["orientation"] == ""


def test_contains_non_english():
    assert prompting.contains_non_english("1girl, 猫耳")
    assert prompting.contains_non_english("ねこ") and prompting.contains_non_english("고양이")
    assert prompting.contains_non_english("ｶﾀｶﾅ")  # 半角片假名
    assert prompting.contains_non_english("кошка") and prompting.contains_non_english("แมว")
    assert prompting.contains_non_english("1girl。")  # 中文标点
    assert not prompting.contains_non_english("1girl, cat ears, pokémon, 1.2::smile::, {{blush}}, ♥, :3")
    assert not prompting.contains_non_english("ｃａｔ　ｅａｒｓ，１ｇｉｒｌ")  # 全角字母和标点会转成半角
    assert not prompting.contains_non_english("")


def test_drop_non_english_tags():
    kept, removed = prompting.drop_non_english_tags("1girl, 白色长发, cat ears，smile、穿着JK制服")
    assert kept == "1girl, cat ears, smile" and removed == ["白色长发", "穿着JK制服"]
    long_text = ", ".join(f"tag{i}" for i in range(500))
    assert prompting.drop_non_english_tags(long_text)[0] == long_text  # 不截断
    assert prompting.drop_non_english_tags("") == ("", [])


def test_drop_non_english_tags_config_cases():
    # 每行一个标签的配置：只去掉中文那一行
    kept, removed = prompting.drop_non_english_tags("nsfw\nlowres\n低质量\nbad hands")
    assert kept == "nsfw, lowres, bad hands" and removed == ["低质量"]
    # 不去重，跨逗号的权重分组保持原样
    weights = "1.2::cat ears, smile::, 0.8::dog ears, smile::, very aesthetic"
    assert prompting.drop_non_english_tags(weights)[0] == weights
    # 去掉分组中的中文标签时保留未闭合的权重 / 括号符号
    assert prompting.drop_non_english_tags("1.2::猫耳, smile::, blush")[0] == "1.2::, smile::, blush"
    assert prompting.drop_non_english_tags("{{白发, smile}}")[0] == "{{, smile}}"
    assert prompting.drop_non_english_tags("1.2::猫耳::, smile")[0] == "smile"
    # 、。 视为分隔符，「」【】 等括号去掉
    assert prompting.drop_non_english_tags("smile。blush、「cat ears」")[0] == "smile, blush, cat ears"


def test_choose_tag_result():
    first = {"prompt": "1girl, 白色长发, cat ears, smile", "negative": "blurry", "orientation": "portrait"}
    better = prompting.choose_tag_result(first, {"prompt": "1girl, white hair, cat ears", "negative": "", "orientation": ""})
    assert better == {"prompt": "1girl, white hair, cat ears", "negative": "blurry", "orientation": "portrait"}
    # 改写结果更差（英文标签更少或全是中文）时沿用第一次的结果
    assert prompting.choose_tag_result(first, {"prompt": "一个女孩, 猫耳", "negative": "", "orientation": ""}) is first
    assert prompting.choose_tag_result(first, {"prompt": "", "negative": "", "orientation": ""}) is first
    assert prompting.english_tag_count("1girl, 白发, smile") == 2 and prompting.english_tag_count("") == 0


def test_tag_system_prompt_is_configurable():
    custom = prompting.build_tag_system_prompt("自定义：只画风景。", sfw=False)
    assert custom.startswith("自定义：只画风景。")
    # 输出格式、英文要求和安全规则始终附加
    assert prompting.TAG_OUTPUT_RULES in custom and "ENGLISH ONLY" in custom and "minors" in custom
    default = prompting.build_tag_system_prompt("   ")
    assert default.startswith(prompting.DEFAULT_TAG_PROMPT)
    assert "english prompt that will be sent directly to novelai" in prompting.DEFAULT_TAG_PROMPT.lower()
    schema = json.loads((Path(__file__).resolve().parent.parent / "_conf_schema.json").read_text(encoding="utf-8"))
    assert schema["prompt"]["items"]["tag_system_prompt"]["default"] == prompting.DEFAULT_TAG_PROMPT


def test_parse_tag_output_normalizes_fullwidth():
    out = prompting.parse_tag_output('{"prompt": "１girl，ｃａｔ ears", "negative": "", "orientation": "portrait"}')
    assert out["prompt"] == "1girl, cat ears"


def test_tag_prompts_require_english():
    assert "ENGLISH ONLY" in prompting.build_tag_system_prompt()
    assert "minors" in prompting.build_tag_system_prompt(sfw=False)
    retry = prompting.build_tag_retry_prompt("画一只猫", "square", '{"prompt": "猫"}')
    assert "画一只猫" in retry and '{"prompt": "猫"}' in retry and "English" in retry


def test_clean_tags_limit():
    tags = prompting.clean_tags(", ".join(f"tag{i}" for i in range(1000)), limit=50)
    assert len(tags) <= 50 and not tags.endswith(",")


def test_join_tags():
    assert prompting.join_tags("a, b,", "", " c ") == "a, b, c"
    assert prompting.join_tags("", None) == ""


def test_strip_think():
    assert prompting.strip_think("<think>x</think>hello") == "hello"
    assert prompting.strip_think("reasoning...</think>\nhi") == "hi"
    assert prompting.strip_think("plain") == "plain"


def test_clean_line():
    assert prompting.clean_line('“我开始画啦！”\n', 40) == "我开始画啦！"
    assert prompting.clean_line("第一行\n第二行", 40) == "第一行 第二行"
    assert len(prompting.clean_line("长" * 500, 20)) == 41
    # 超长时尽量截在句末标点
    assert prompting.clean_line("好" * 15 + "！" + "多" * 60, 20) == "好" * 15 + "！"


def test_prompt_builders():
    sp = prompting.build_tag_system_prompt(sfw=True, extra="watercolor")
    assert "safe for work" in sp and "watercolor" in sp
    assert "safe for work" not in prompting.build_tag_system_prompt(sfw=False)
    assert "Requested orientation: square" in prompting.build_tag_user_prompt("猫", "square")
    assert "猫娘" in prompting.build_persona_system_prompt("你是猫娘", 30)
    for kind in ("start", "success", "fail", "fallback", "limit"):
        text = prompting.build_persona_user_prompt(kind, description="一只猫", prompt="cat", error="boom",
                                                   reason="这个群今天的 5 次画画次数已经用完了")
        assert "一只猫" in text and "{" not in text
        assert prompting.FALLBACK_LINES[kind]
    assert "正在画" in prompting.build_persona_user_prompt("start", description="一只猫")
    assert "正在画" in prompting.FALLBACK_LINES["start"]


def test_persona_prompt_templates():
    custom = prompting.build_persona_user_prompt(
        "success", "画了{description}，标签{prompt}，保留{未知}和{{花括号}}", description="猫", prompt="cat")
    assert custom == "画了猫，标签cat，保留{未知}和{{花括号}}"
    assert prompting.build_persona_user_prompt("fail", "  ", error="x") == \
        prompting.build_persona_user_prompt("fail", error="x")  # 空模板使用默认
    assert len(prompting.build_persona_user_prompt("fail", "{error}", error="e" * 1000)) == 200


def test_natural_prompts():
    sp = prompting.build_natural_system_prompt(sfw=True, extra="watercolor")
    assert "JSON" in sp and "minors" in sp and "safe for work" in sp and "watercolor" in sp
    assert "safe for work" not in prompting.build_natural_system_prompt(sfw=False)
    assert prompting.build_natural_system_prompt("custom base").startswith("custom base")
    out = prompting.parse_natural_output(
        '<think>x</think>```json\n{"prompt": "A cat\\n sleeping   under a tree.", "negative": "text", '
        '"orientation": "横图"}\n```')
    assert out == {"prompt": "A cat sleeping under a tree.", "negative": "text", "orientation": "landscape"}
    plain = prompting.parse_natural_output("Prompt: a red fox in snow")
    assert plain["prompt"] == "a red fox in snow" and plain["orientation"] == ""
    # 自然语言模式保留中文（例如需要画进图里的文字）
    assert "「新年快乐」" in prompting.parse_natural_output('{"prompt": "A banner that says 「新年快乐」"}')["prompt"]


def test_clean_description_limit():
    text = ("This is a sentence. " * 400).strip()
    out = prompting.clean_description(text, limit=100)
    assert len(out) <= 100 and out.endswith(".")
    assert prompting.clean_description(["a", "b"]) == "a b"


def test_persona_prompt_with_image():
    text = prompting.build_persona_user_prompt("success", description="猫", prompt="cat", with_image=True)
    assert text.endswith(prompting.WITH_IMAGE_NOTE)
    assert prompting.WITH_IMAGE_NOTE not in prompting.build_persona_user_prompt("success", description="猫")


def test_schema_persona_prompt_defaults_match_code():
    schema = json.loads((Path(__file__).resolve().parent.parent / "_conf_schema.json").read_text(encoding="utf-8"))
    for kind, template in prompting.DEFAULT_PERSONA_PROMPTS.items():
        # 达到每日次数限制时的两句台词在「使用范围与每日次数」中
        section = "quota" if kind in ("fallback", "limit") else "persona"
        assert schema[section]["items"][f"{kind}_prompt"]["default"] == template


# ---------------------------------------------------------------- audit


def test_audit_page_and_trim(tmp_path):
    log = audit.AuditLog(tmp_path / "audit.jsonl", max_records=10)

    async def go():
        for i in range(30):
            await log.append({"status": "success", "request": f"r{i}"})
        return await log.page(1, 5), await log.page(99, 5)

    (first, page, pages, total), (_, last_page, _, _) = run(go())
    assert total <= 10 + 10 and total >= 10
    assert page == 1 and pages == -(-total // 5) and last_page == pages
    assert first[0]["request"] == "r29"  # 从新到旧


def test_audit_append_survives_cancellation(tmp_path):
    log = audit.AuditLog(tmp_path / "audit.jsonl")

    async def go():
        tasks = [asyncio.create_task(log.append({"status": "success", "request": str(i)})) for i in range(50)]
        await asyncio.sleep(0)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return await log.page(1, 100)

    _, _, _, total = run(go())
    assert total == 50


def test_audit_skips_corrupt_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text('{"status":"success"}\nnot json\n[1]\n', encoding="utf-8")
    records, _, _, total = run(audit.AuditLog(path).page(1, 10))
    assert total == 1 and records[0]["_no"] == 1


def test_format_record():
    ok = audit.format_record({
        "_no": 3, "status": "success", "time": "2026-10-23T14:05:11+08:00", "user_name": "小明",
        "user_id": "123", "group_id": "456", "model": "nai-diffusion-4-5-full", "size": "832x1216",
        "steps": 28, "seed": 9, "duration": 12.3, "anlas_cost": 0, "request": "画猫", "prompt": "cat",
    })
    assert "#3 ✅ 2026-10-23 14:05:11" in ok and "群 456" in ok and "消耗 0 Anlas" in ok
    bad = audit.format_record({"status": "failed", "stage": "generate", "error": "Token 无效", "request": "x"})
    assert "❌" in bad and "失败（生成图片）：Token 无效" in bad and "私聊" in bad


# ---------------------------------------------------------------- gallery


def test_gallery_save_list_read_trim(tmp_path):
    g = gallery.Gallery(tmp_path / "images")

    async def go():
        paths = [await g.save(PNG, "image/png", "open-ai!", keep=3) for _ in range(5)]
        return paths, await g.recent(10)

    paths, recent = run(go())
    assert len(recent) == 3 and all(r["name"].count("/") == 1 for r in recent)
    assert paths[-1].name.endswith(".png") and "_openai_" in paths[-1].name
    assert recent[0]["name"].endswith(paths[-1].name)  # 从新到旧
    data, mime = run(g.read(recent[0]["name"]))
    assert data == PNG and mime == "image/png"
    for bad in ("../x.png", "20261025/../../x.png", "abc", "", "20261025\\..\\x.png","20261025/x.exe"):
        assert g.resolve(bad) is None
    assert run(g.read("20990101/missing.png")) is None


# ---------------------------------------------------------------- 审查后补充的回归测试


def test_parse_tag_output_tolerates_malformed_json():
    escaped = r'{"prompt": "1girl, furina \(genshin impact\), smile", "negative": "nsfw, extra fingers", ' \
              r'"orientation": "portrait"}'
    out = prompting.parse_tag_output(escaped)
    assert out == {"prompt": "1girl, furina (genshin impact), smile", "negative": "nsfw, extra fingers",
                   "orientation": "portrait"}
    trailing = prompting.parse_tag_output('{"prompt": "1girl", "negative": "", "orientation": "square"}\n'
                                          "Note: wrap tags in {braces}")
    assert trailing["prompt"] == "1girl" and trailing["orientation"] == "square"
    assert prompting.parse_tag_output('{"prompt": "1girl, smile", "negative": "lowres",}')["negative"] == "lowres"
    # 无法解析的 JSON 不能原样当作提示词
    broken = prompting.parse_tag_output('{"prompt": "1girl, never closed')
    assert broken["prompt"] == ""
    assert prompting.parse_tag_output("1girl, smile")["prompt"] == "1girl, smile"


def test_parse_natural_output_unescaped_quotes():
    out = prompting.parse_natural_output('{"prompt": "A shop with a neon sign that reads "OPEN" at night.", '
                                         '"negative": "people, watermark", "orientation": "landscape"}')
    assert out["prompt"] == 'A shop with a neon sign that reads "OPEN" at night.'
    assert out["negative"] == "people, watermark" and out["orientation"] == "landscape"


def test_parse_size_rejects_huge_numbers():
    from backends import base

    assert base.parse_size("9" * 400 + "x1", (1, 2)) == (1, 2)
    assert base.parse_size("200000x100", (1, 2)) == (1, 2)
    assert base.parse_ratio("1:" + "9" * 400) is None
    assert nai_client.parse_size("9" * 5000 + "x1", (1, 2)) == (1, 2)


def test_is_local_url():
    from backends.base import is_local_url

    for url in ("http://127.0.0.1:8188", "http://192.168.1.20:8188", "http://comfyui:8188", "http://100.101.1.2",
                "http://host.docker.internal:8188", "http://nas.local"):
        assert is_local_url(url), url
    for url in ("https://api.openai.com/v1", "http://8.8.8.8", "http://[bad", ""):
        assert not is_local_url(url), url


def test_novelai_client_skips_env_proxy_for_lan(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://10.255.255.1:7890")

    async def go():
        client = NovelAIClient()
        try:
            lan = await client._http("http://192.168.1.20:8000", "")
            lan_trust = lan._trust_env
            public = await client._http("https://image.novelai.net", "")
            return lan_trust, public._trust_env
        finally:
            await client.close()

    assert run(go()) == (False, True)


def test_limiter_fifo_and_live_limit():
    import limiter

    async def go():
        limit = {"n": 1}
        lim = limiter.Limiter(lambda: limit["n"])
        order, gates = [], {i: asyncio.Event() for i in range(4)}

        async def job(i):
            await lim.acquire()
            order.append(i)
            try:
                await gates[i].wait()
            finally:
                lim.release()

        tasks = [asyncio.create_task(job(i)) for i in range(3)]
        await asyncio.sleep(0)
        assert order == [0] and lim.waiting == 2 and lim.queued_ahead() == 3
        # 调低上限：正在进行的不受影响，新请求必须等到低于新上限
        limit["n"] = 1
        tasks.append(asyncio.create_task(job(3)))
        gates[0].set()
        await asyncio.sleep(0.01)
        assert order == [0, 1]  # 先来先服务，后来的 3 不会插队
        limit["n"] = 3
        lim.dispatch()
        await asyncio.sleep(0)
        assert order == [0, 1, 2, 3] and lim.active == 3
        for gate in gates.values():
            gate.set()
        await asyncio.gather(*tasks)
        assert lim.active == 0 and lim.waiting == 0

    run(go())


def test_limiter_cancelled_waiter_releases_nothing():
    import limiter

    async def go():
        lim = limiter.Limiter(lambda: 1)
        await lim.acquire()
        waiter = asyncio.create_task(lim.acquire())
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        assert lim.waiting == 0 and lim.active == 1
        lim.release()
        assert lim.active == 0

    run(go())


def test_parse_llm_output_more_shapes():
    assert prompting.parse_tag_output("{{1girl}}, solo, smile")["prompt"] == "{{1girl}}, solo, smile"
    single = prompting.parse_tag_output("{'prompt': '1girl, smile', 'negative': 'lowres', 'orientation': 'square'}")
    assert single == {"prompt": "1girl, smile", "negative": "lowres", "orientation": "square"}
    no_commas = prompting.parse_tag_output('{\n "prompt": "1girl, smile"\n "negative": "nsfw, lowres"\n'
                                           ' "orientation": "portrait"\n}')
    assert no_commas == {"prompt": "1girl, smile", "negative": "nsfw, lowres", "orientation": "portrait"}
    escaped = prompting.parse_tag_output(r'{"prompt": "furina \(genshin impact\), smile", "negative": "nsfw",}')
    assert escaped["prompt"] == "furina (genshin impact), smile" and escaped["negative"] == "nsfw"
    prose = prompting.parse_natural_output('Here you go {json}:\n{"prompt": "Line one.\nLine two, caf\u00e9.", '
                                           '"negative": "", "orientation": "square"}')
    assert prose["prompt"] == "Line one. Line two, café." and prose["orientation"] == "square"
    extra_key = prompting.parse_natural_output('{"prompt": "A sign that reads "OPEN" at night.", "style": "photo", '
                                               '"negative": "people", "orientation": "landscape"}')
    assert extra_key["prompt"] == 'A sign that reads "OPEN" at night.' and extra_key["negative"] == "people"
    nested = prompting.parse_natural_output('{"prompt": {"positive": "A cat."}, "orientation": "square"}')
    assert nested["prompt"] == "A cat."


def test_size_parsers_accept_trailing_period_and_reject_overflow():
    from backends.seedream import pick_size
    from backends.xai import RATIOS, parse_aspect

    assert pick_size("16:9.", "landscape", "5.0-pro") == pick_size("16:9", "landscape", "5.0-pro")
    assert pick_size("2048 x 2048。", "square", "5.0-pro") == "2048x2048"
    assert parse_aspect("1" * 160 + "x" + "1" * 160, RATIOS, "2:3") == ("2:3", 0)


# ---------------------------------------------------------------- 消耗统计


def test_estimate_cost_prefers_platform_cost_then_tokens_then_per_image():
    section = {"price_per_image": 0.05, "price_input_per_million": 2, "price_output_per_million": 30,
               "price_currency": "USD"}
    assert usage.estimate_cost({"images": 1, "cost": 0.07, "currency": "USD"}, section) == (0.07, "USD", False)
    assert usage.estimate_cost({"images": 1, "input_tokens": 500, "output_tokens": 1290}, section) == \
        (0.0397, "USD", True)
    assert usage.estimate_cost({"images": 2, "total_tokens": 5000}, section) == (0.1, "USD", True)
    # 只填了一个 token 价格时按张估算
    only_input = dict(section, price_output_per_million=0)
    assert usage.estimate_cost({"images": 1, "input_tokens": 500, "output_tokens": 1290}, only_input) == \
        (0.05, "USD", True)
    nan_cost = {"images": 1, "cost": float("nan"), "currency": "USD"}
    assert usage.estimate_cost(nan_cost, section)[2] is True
    assert usage.estimate_cost({"images": 1}, {}) == (None, "", False)
    assert usage.estimate_cost({"images": 1}, {"price_per_image": "abc"}) == (None, "", False)
    assert usage.estimate_cost({"images": 1}, {"price_per_image": -1}) == (None, "", False)


def test_apply_usage_respects_switch():
    record = {"usage": {"images": 1, "cost": 0.04, "currency": "USD"}}
    usage.apply_usage(record, {"track_usage": True})
    assert record["cost"] == 0.04 and "cost_estimated" not in record
    record = {"usage": {"images": 1}}
    usage.apply_usage(record, {"track_usage": False, "price_per_image": 1})
    assert record == {}
    record = {"usage": {"images": 1}}
    usage.apply_usage(record, {})
    assert record == {"usage": {"images": 1}}  # 没有配置价格时只记录用量


def test_format_cost():
    assert usage.format_cost(0.16690, "USD", True) == "约 0.1669 USD"
    assert usage.format_cost(2.0, "元") == "2 元"
    assert usage.format_cost(0.00001, "USD") == "<0.0001 USD"
    assert usage.format_cost(0, "USD") == "0 USD"


def _rec(time, backend="openai", status="success", user="1", **extra):
    return {"time": time, "backend": backend, "backend_label": backend.title(), "status": status, "user_id": user,
            **extra}


def test_summarize_filters_and_totals():
    from datetime import datetime, timezone

    now = datetime(2026, 10, 25, 15, 0, tzinfo=timezone.utc)
    records = [
        _rec("2026-10-20T10:00:00+00:00", usage={"images": 1, "total_tokens": 100}),
        _rec("2026-10-25T09:00:00+00:00", usage={"images": 1, "input_tokens": 10, "output_tokens": 90,
                                                   "total_tokens": 100}, cost=0.5, currency="USD",
             cost_estimated=True),
        _rec("2026-10-25T10:00:00+00:00", backend="novelai", anlas_cost=20, user="2"),
        _rec("2026-10-25T11:00:00+00:00", backend="novelai", anlas_cost=-500, user="2"),  # 期间充值
        _rec("2026-10-25T12:00:00+00:00", backend="xai", status="failed"),
        _rec("2026-10-25T13:00:00+00:00", backend="xai", usage={"images": 1, "cost": 0.02, "currency": "USD"},
             cost=0.02, currency="USD"),
        _rec("broken time"),
    ]
    today = usage.period_start(1, now)
    assert today == datetime(2026, 10, 25, tzinfo=timezone.utc)
    assert usage.period_start(7, now) == datetime(2026, 10, 19, tzinfo=timezone.utc)
    assert usage.period_start(0, now) is None

    s = usage.summarize(records, today, order=["novelai", "openai", "xai"])
    assert [r["backend"] for r in s["backends"]] == ["novelai", "openai", "xai"] and s["count"] == 5
    nai, oai, xai = s["backends"]
    assert nai["images"] == 2 and nai["anlas"] == 20 and nai["anlas_records"] == 2
    assert oai["images"] == 1 and oai["total_tokens"] == 100 and oai["costs"] == {"USD": 0.5}
    assert xai["success"] == 1 and xai["failed"] == 1 and xai["costs"] == {"USD": 0.02} and xai["estimated"] == []
    total = s["total"]
    assert total["costs"] == {"USD": 0.52} and total["estimated"] == ["USD"] and total["images"] == 4
    assert not s["truncated"]

    assert usage.summarize(records, None)["count"] == 7  # 不限时间时时间格式错误的记录也计入
    assert usage.summarize(records, today, user_id="2")["count"] == 2
    assert usage.summarize(records, today, backend="xai")["count"] == 2
    # 删除过的旧记录在统计范围内时说明统计不完整
    yesterday = datetime(2026, 10, 24, 23, 0, tzinfo=timezone.utc)
    assert usage.summarize(records, today, trimmed_until=yesterday)["truncated"] is False
    assert usage.summarize(records, None, trimmed_until=yesterday)["truncated"] is True
    assert usage.summarize(records, today, trimmed_until=today)["truncated"] is True
    assert usage.summarize(records, today)["truncated"] is False

    text = usage.format_summary(s, 1)
    assert text.startswith("绘图消耗（今天），共 5 次绘图：")
    assert "【Novelai】成功 2 张；消耗 20 Anlas" in text
    assert "【Openai】成功 1 张；100 tokens（输入 10 / 输出 90）；约 0.5 USD" in text
    assert "【Xai】成功 1 张，失败 1 次；0.02 USD" in text
    assert "合计：成功 4 张，失败 1 次；消耗 20 Anlas；100 tokens（输入 10 / 输出 90）；约 0.52 USD" in text
    assert "按配置的价格估算" in text
    empty = usage.summarize([], today)
    assert usage.format_summary(empty, 7, "你的") == "你的绘图消耗（最近 7 天）：没有绘图记录。"


def test_summarize_survives_malformed_records():
    bad = [
        {"time": "0001-01-01T00:00:00", "backend": "openai", "status": "success"},
        {"time": "2026-10-25T10:00:00+00:00", "backend": "openai", "status": "success",
         "usage": {"images": float("nan"), "total_tokens": float("inf")}, "cost": float("nan"), "currency": "USD"},
        {"time": "2026-10-25T10:00:00+00:00", "backend": "novelai", "status": "success", "anlas_cost": float("inf")},
    ]
    s = usage.summarize(bad, None)
    assert s["count"] == 3 and s["total"]["costs"] == {} and s["total"]["anlas_records"] == 0
    assert s["total"]["images"] == 3 and s["total"]["total_tokens"] == 0
    assert usage.format_record_usage(bad[1]) == ""
    assert usage.summarize(bad, usage.period_start(1))["count"] <= 2


def test_audit_remembers_trimmed_records(tmp_path):
    log = audit.AuditLog(tmp_path / "audit.jsonl", max_records=10)
    assert asyncio.run(log.trimmed_until()) is None

    async def fill():
        for i in range(25):
            await log.append({"time": f"2026-10-{i + 1:02d}T08:00:00+08:00", "status": "success"})

    asyncio.run(fill())
    records = asyncio.run(log.records())
    assert len(records) < 25 and records[-1]["time"].startswith("2026-10-25")
    until = asyncio.run(log.trimmed_until())
    first_kept = usage.record_time(records[0])
    assert until is not None and until < first_kept and (log.path.parent / "audit.meta.json").is_file()


def test_audit_records_and_format_usage(tmp_path):
    log = audit.AuditLog(tmp_path / "audit.jsonl")
    asyncio.run(log.append({"status": "success", "backend": "xai", "model": "grok-imagine-image",
                            "usage": {"images": 2, "cost": 0.14, "currency": "USD"}, "cost": 0.14, "currency": "USD"}))
    records = asyncio.run(log.records())
    assert len(records) == 1 and records[0]["_no"] == 1
    assert "2 张 · 0.14 USD" in audit.format_record(records[0])


# ---------------------------------------------------------------- 使用范围与每日次数


def test_session_blocked_matching():
    umo = "aiocqhttp:GroupMessage:123456"
    assert quota.session_blocked(umo, "123456", [], []) == ""
    assert quota.session_blocked(umo, "123456", ["123456"], ["123456"]) == ""  # 白名单优先
    assert quota.session_blocked(umo, "123456", [umo], []) == ""
    assert quota.session_blocked(umo, "123456", ["999"], []) == "whitelist"
    assert quota.session_blocked(umo, "123456", [], [" 123456 "]) == "blacklist"
    assert quota.session_blocked(umo, "123456", "", "111，123456") == "blacklist"  # 字符串也能识别
    private = "aiocqhttp:FriendMessage:10001"
    assert quota.session_blocked(private, "", ["10001"], []) == ""
    assert quota.session_blocked(private, "", None, None) == ""


def test_daily_quota_reserve_release_and_rollover(tmp_path, monkeypatch):
    path = tmp_path / "quota.json"
    q = quota.DailyQuota(path)
    monkeypatch.setattr(quota.DailyQuota, "today", staticmethod(lambda: "2026-10-26"))
    r1 = q.reserve("s", "u")
    r2 = q.reserve("s", "v")
    assert q.used("session", "s") == 2 and q.used("user", "u") == 1
    asyncio.run(q.release(r1, True))
    asyncio.run(q.release(r1, True))  # 重复释放不会重复计数
    asyncio.run(q.release(r2, False))
    assert q.used("session", "s") == 1 and q.used("user", "v") == 0
    assert json.loads(path.read_text(encoding="utf-8")) == {"date": "2026-10-26", "session": {"s": 1}, "user": {"u": 1}}
    assert quota.DailyQuota(path).used("session", "s") == 1
    # 第二天清零；文件中的旧日期也不会被读入
    monkeypatch.setattr(quota.DailyQuota, "today", staticmethod(lambda: "2026-10-27"))
    assert q.used("session", "s") == 0 and quota.DailyQuota(path).used("session", "s") == 0
    path.write_text("not json", encoding="utf-8")
    assert quota.DailyQuota(path).used("user", "u") == 0
