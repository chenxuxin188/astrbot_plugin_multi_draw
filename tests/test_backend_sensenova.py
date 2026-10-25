"""SenseNova 后端的单元测试（不访问网络）：python -m pytest tests/test_backend_sensenova.py"""

import asyncio
import base64
import json
import sys
from pathlib import Path

import httpx
import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

from backends import base  # noqa: E402
from backends import sensenova as sn  # noqa: E402
from backends.base import Clients, DrawError, DrawRequest, HttpPool, InputImage  # noqa: E402
from backends.sensenova import SenseNovaBackend  # noqa: E402


PNG =b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
GIF = b"GIF89a" + b"\x00" * 32
GEN_URL = "https://token.sensenova.cn/v1/images/generations"
EDIT_URL = "https://token.sensenova.cn/v1/images/edits"
KEY = "sk-test-1234567890"
FORBIDDEN_FIELDS = ("negative_prompt", "seed", "n", "quality", "style", "aspect_ratio", "steps")


def ok_body(data: bytes = PNG, **extra) -> dict:
    body = {"created": 1, "data": [{"b64_json": base64.b64encode(data).decode()}], "output_format": "png"}
    body.update(extra)
    return body


def image(width: int = 800, height: int = 1200, data: bytes = PNG, mime: str = "image/png") -> InputImage:
    return InputImage(data=data, mime=mime, width=width, height=height)


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def edit_req(*images: InputImage, prompt: str = "Replace the background of image 1 with a glacier",
             orientation: str = "portrait", **kwargs) -> DrawRequest:
    return DrawRequest(prompt=prompt, orientation=orientation, images=list(images or [image()]), **kwargs)


class Recorder:
    """MockTransport 处理函数：按顺序返回预设响应（或抛出异常），并记录收到的请求。"""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(request)
        return item

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.method == "POST"]


def run(handler, coro_factory):
    async def main():
        pool = HttpPool(transport=httpx.MockTransport(handler))
        try:
            return await coro_factory(Clients(http=pool))
        finally:
            await pool.close()

    return asyncio.run(main())


def generate(section: dict | None, handler, req: DrawRequest | None = None):
    backend = SenseNovaBackend(section)
    return run(handler, lambda clients: backend.generate(clients, req or DrawRequest(prompt="一只猫")))


def draw_error(section, handler, req=None) -> DrawError:
    with pytest.raises(DrawError) as info:
        generate(section, handler, req)
    return info.value


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """重试等待不真正 sleep，只记录时长。"""
    waits: list[float] = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(base.asyncio, "sleep", fake_sleep)
    return waits


def assert_valid_size(size: str):
    w, h = (int(v) for v in size.split("x"))
    assert w % 32 == 0 and h % 32 == 0, size
    assert 512 <= w <= 4096 and 512 <= h <= 4096, size
    assert max(w, h) <= 3 * min(w, h), size


# ---------------------------------------------------------------- 请求


@pytest.mark.parametrize(
    ("orientation", "size"),
    [("portrait", "1664x2496"), ("landscape", "2496x1664"), ("square", "2048x2048"), ("bogus", "1664x2496")],
)
def test_request_per_orientation(orientation, size):
    rec = Recorder(httpx.Response(200, json=ok_body(size=size, usage={"total_tokens": 5636})))
    result = generate({"api_key": KEY}, rec, DrawRequest(prompt="一只猫", orientation=orientation))
    req = rec.requests[0]
    assert req.method == "POST" and str(req.url) == GEN_URL
    assert req.headers["authorization"] == f"Bearer {KEY}"
    assert req.headers["content-type"] == "application/json"
    assert rec.bodies[0] == {
        "model": "sensenova-u1.5-lite",
        "prompt": "一只猫",
        "size": size,
        "response_format": "b64_json",
        "output_format": "png",
        "watermark": False,
        "prompt_extend": True,
    }
    assert result.image == PNG and result.mime == "image/png"
    assert result.params == {"model": "sensenova-u1.5-lite", "size": size, "output_format": "png",
                             "watermark": False, "prompt_extend": True, "tokens": 5636}
    assert len(rec.requests) == 1


def test_negative_folded_into_prompt_and_no_undocumented_fields():
    rec = Recorder(httpx.Response(200, json=ok_body()))
    generate({"api_key": KEY}, rec, DrawRequest(prompt="a cat", negative="text, watermark", seed=42))
    body = rec.bodies[0]
    assert body["prompt"] == "a cat\n\nAvoid: text, watermark"
    for field in FORBIDDEN_FIELDS:
        assert field not in body


@pytest.mark.parametrize(
    ("model", "size_cfg", "expected"),
    [
        ("sensenova-u1.5-lite", "4K", "2720x4096"),
        ("SenseNova-U1.5-Lite", "4k", "2720x4096"),
        ("sensenova-u1.5-fast", "4K", "2720x4096"),  # 文档中 U1.5 Fast 同样列出了 4K
        ("sensenova-u1.5-fast", "1K", "1088x1632"),
        ("sensenova-u1.5-fast", "", "1664x2496"),
        ("my-custom-model", "2K", "1664x2496"),
        ("my-custom-model", "4K", "1664x2496"),  # 文档之外的模型填 4K 时降为 2K
    ],
)
def test_model_families(model, size_cfg, expected):
    rec = Recorder(httpx.Response(200, json=ok_body()))
    result = generate({"api_key": KEY, "model": model, "portrait_size": size_cfg}, rec)
    assert rec.bodies[0]["model"] == model
    assert rec.bodies[0]["size"] == expected
    assert result.params["model"] == model and result.params["size"] == expected


def test_auto_size_reports_actual_size():
    rec = Recorder(httpx.Response(200, json=ok_body(size="2048x2048")))
    result = generate({"api_key": KEY, "square_size": "Auto"}, rec, DrawRequest(prompt="x", orientation="square"))
    assert rec.bodies[0]["size"] == "auto"
    assert result.params["size"] == "2048x2048"


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("", GEN_URL),
        (None, GEN_URL),
        ("token.sensenova.cn", GEN_URL),
        ("https://token.sensenova.cn/", GEN_URL),
        ("https://token.sensenova.cn/v1", GEN_URL),
        ("https://token.sensenova.cn/v1/", GEN_URL),
        ("https://token.sensenova.cn/v1/images/generations", GEN_URL),
        ("https://token.sensenova.cn/v1/images/edits", GEN_URL),
        ("https://token.sensenova.cn/v1/chat/completions", GEN_URL),
        ("https://token.sensenova.cn/v1/models?x=1", GEN_URL),
        # 大写的 /V1 也识别，不再拼成 /V1/images/generations/v1/...
        ("https://token.sensenova.cn/V1/images/generations", "https://token.sensenova.cn/V1/images/generations"),
        ("https://token.sensenova.ai/v1", "https://token.sensenova.ai/v1/images/generations"),
        ("http://relay.local:8080/proxy/v1/images", "http://relay.local:8080/proxy/v1/images/generations"),
        ("https://relay.example.com/openai", "https://relay.example.com/openai/v1/images/generations"),
        ("https://relay.example.com/v1beta", "https://relay.example.com/v1beta/v1/images/generations"),
    ],
)
def test_base_url_normalisation(base_url, expected):
    backend = SenseNovaBackend({"api_key": KEY, "base_url": base_url})
    assert backend.url("/images/generations") == expected
    assert backend.url("/models") == expected.replace("/images/generations", "/models")


def test_base_url_used_for_request():
    rec = Recorder(httpx.Response(200, json=ok_body()))
    generate({"api_key": KEY, "base_url": "https://token.sensenova.ai/v1/images/edits"}, rec)
    assert str(rec.requests[0].url) == "https://token.sensenova.ai/v1/images/generations"


# ---------------------------------------------------------------- 尺寸


@pytest.mark.parametrize(
    ("value", "orientation", "expected"),
    [
        ("2048 x 2048", "square", "2048x2048"),
        ("2048*2048", "square", "2048x2048"),
        ("1536×2720", "portrait", "1536x2720"),
        ("832x1216", "portrait", "832x1216"),
        ("1000x1500", "portrait", "992x1504"),
        ("5000x1000", "landscape", "4096x1376"),
        ("1000x5000", "portrait", "1376x4096"),
        ("300x300", "square", "512x512"),
        ("8192x8192", "square", "4096x4096"),
        ("100x1000", "portrait", "512x1536"),
        ("2k", "landscape", "2496x1664"),
        (" 1K ", "square", "1344x1344"),
        ("auto", "portrait", "auto"),
        ("3:4", "portrait", "1760x2368"),
        ("16:9", "landscape", "2720x1536"),
        ("2：3", "portrait", "1664x2496"),
        ("10:1", "landscape", "4096x1376"),
        ("garbage", "landscape", "2496x1664"),
        ("0x100", "square", "2048x2048"),
        ("", "portrait", "1664x2496"),
        (None, "square", "2048x2048"),
        (1024, "square", "2048x2048"),
    ],
)
def test_resolve_size(value, orientation, expected):
    assert sn.resolve_size(value, orientation, "sensenova-u1.5-lite") == expected


def test_snap_size_always_valid():
    for w in (1, 100, 511, 512, 513, 777, 1000, 1023, 1500, 2049, 3000, 4096, 4097, 6000, 10**6):
        for h in (1, 100, 333, 512, 900, 1216, 2000, 2731, 4096, 5000, 10**6):
            assert_valid_size(sn.snap_size(w, h))


def test_presets_valid():
    for table in sn.PRESETS.values():
        for size in table.values():
            assert_valid_size(size)


# ---------------------------------------------------------------- 响应


def test_url_response_downloaded_without_key():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"data": [{"url": "https://cdn.example.com/gen/a.jpeg"}],
                                             "output_format": "jpeg", "size": "1664x2496"})
        return httpx.Response(200, content=JPEG, headers={"content-type": "image/jpeg"})

    rec = Recorder(handler)
    result = generate({"api_key": KEY, "extra_body": '{"response_format": "url"}', "output_format": "jpg"}, rec)
    assert rec.bodies[0]["response_format"] == "url"
    assert rec.bodies[0]["output_format"] == "jpeg"
    download = rec.requests[1]
    assert download.method == "GET" and str(download.url) == "https://cdn.example.com/gen/a.jpeg"
    assert "authorization" not in download.headers
    assert result.image == JPEG and result.mime == "image/jpeg"


def test_b64_with_data_prefix():
    data_url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    result = generate({"api_key": KEY}, Recorder(httpx.Response(200, json={"data": [{"b64_json": data_url}]})))
    assert result.image == PNG


@pytest.mark.parametrize("body", [{"data": []}, {"data": [{}]}, {"data": None}, {"created": 1}])
def test_no_image_in_response(body):
    err = draw_error({"api_key": KEY}, Recorder(httpx.Response(200, json=body)))
    assert "没有返回图片" in str(err)


def test_error_object_in_200_response():
    body = {"error": {"type": "failed_precondition_error", "code": "9", "message": "安全检查未通过"}}
    err = draw_error({"api_key": KEY}, Recorder(httpx.Response(200, json=body)))
    assert "内容安全" in str(err) and not err.retryable


def test_error_object_with_image_keeps_image():
    # 同时带有图片和错误对象时以图片为准，避免丢掉已经扣费的结果
    body = ok_body(error={"code": 0, "message": "ok"})
    rec = Recorder(httpx.Response(200, json=body))
    assert generate({"api_key": KEY}, rec).image == PNG and len(rec.requests) == 1


def test_invalid_b64_and_non_json():
    err = draw_error({"api_key": KEY}, Recorder(httpx.Response(200, json={"data": [{"b64_json": "aGVsbG8="}]})))
    assert "不是可识别的图片" in str(err)
    err = draw_error({"api_key": KEY}, Recorder(httpx.Response(200, text="<html><title>oops</title></html>")))
    assert "不是 JSON" in str(err)


# ---------------------------------------------------------------- 错误与重试


def test_safety_refusal_400_not_retried():
    body = {"error": {"type": "failed_precondition_error", "code": "9", "message": "safety check not passed"}}
    rec = Recorder(httpx.Response(400, json=body))
    err = draw_error({"api_key": KEY}, rec)
    assert "未通过内容安全检查" in str(err) and "safety check not passed" in str(err)
    assert err.status == 400 and not err.retryable
    assert len(rec.requests) == 1


def test_failed_precondition_without_reason_and_engine_unavailable():
    rec = Recorder(httpx.Response(400, json={"error": {"type": "failed_precondition_error", "message": "x"}}))
    err = draw_error({"api_key": KEY}, rec)
    assert "拒绝生成" in str(err) and len(rec.requests) == 1
    body = {"error": {"type": "failed_precondition_error", "message": "engine unavailable"}}
    rec = Recorder(httpx.Response(400, json=body), httpx.Response(200, json=ok_body()))
    assert generate({"api_key": KEY}, rec).image == PNG
    assert len(rec.requests) == 2


@pytest.mark.parametrize("message", ["size violates constraint: width must be a multiple of 32",
                                     "model id is case-sensitive", "参数不符合安全范围"])
def test_invalid_request_not_reported_as_safety(message):
    body = {"error": {"type": "invalid_request_error", "code": "3", "message": message}}
    rec = Recorder(httpx.Response(400, json=body))
    err = draw_error({"api_key": KEY}, rec)
    assert "请求参数有误" in str(err) and "内容安全" not in str(err) and message in str(err)
    assert not err.retryable and len(rec.requests) == 1


def test_429_with_safety_words_is_rate_limit(no_sleep):
    body = {"error": {"type": "quota_exceeded_error", "code": "8", "message": "为保障服务安全，请求频率超限"}}
    rec = Recorder(httpx.Response(429, json=body), httpx.Response(200, json=ok_body()))
    assert generate({"api_key": KEY}, rec).image == PNG and len(rec.requests) == 2


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"error": {"code": 16, "message": "Forbidden"}}),
        httpx.Response(400, json={"error": {"type": "failed_precondition_error", "message": "安全检查未通过"}}),
        httpx.Response(429, json={"error": {"type": "quota_exceeded_error", "message": "quota exhausted"}}),
        httpx.Response(500, text="<html><title>boom</title></html>"),
        httpx.ReadTimeout("slow"),
        httpx.ConnectError("refused"),
    ],
)
def test_api_key_never_in_error(response):
    err = draw_error({"api_key": f"Bearer {KEY}"}, Recorder(response))
    assert KEY not in str(err) and KEY[-8:] not in str(err)


def test_403_policy_block():
    body = {"error": {"type": "permission_denied_error", "code": "7", "message": "policy blocks"}}
    rec = Recorder(httpx.Response(403, json=body))
    err = draw_error({"api_key": KEY}, rec)
    assert "HTTP 403" in str(err) and "内容策略" in str(err) and len(rec.requests) == 1


@pytest.mark.parametrize("message", ["Forbidden", "Authorization Not Found"])
def test_401_gateway_shape(message):
    rec = Recorder(httpx.Response(401, json={"error": {"code": 16, "message": message}}))
    err = draw_error({"api_key": KEY}, rec)
    assert "API Key" in str(err) and message in str(err) and "token.sensenova.ai" in str(err)
    assert err.status == 401 and not err.retryable and len(rec.requests) == 1


def test_404_wrong_host_or_model():
    rec = Recorder(httpx.Response(404, json={"message": "no Route matched with those values"}))
    err = draw_error({"api_key": KEY}, rec)
    assert "HTTP 404" in str(err) and "no Route matched" in str(err) and "token.sensenova.cn" in str(err)
    rec = Recorder(httpx.Response(404, json={"error": {"code": 5, "message": "NOT_FOUND", "details": []}}))
    assert "NOT_FOUND" in str(draw_error({"api_key": KEY}, rec))


@pytest.mark.parametrize(
    "message",
    ["5-hour points quota exhausted", "积分额度不足", "insufficient balance", "weekly quota exceeded"],
)
def test_429_quota_exhausted_not_retried(message):
    body = {"error": {"type": "quota_exceeded_error", "code": "8", "message": message}}
    rec = Recorder(httpx.Response(429, json=body), httpx.Response(200, json=ok_body()))
    err = draw_error({"api_key": KEY}, rec)
    assert "额度已用尽" in str(err) and not err.retryable and len(rec.requests) == 1


@pytest.mark.parametrize("message", ["rate limit exceeded", "Too many requests", "请求频率超限", ""])
def test_429_rate_limit_retried_once(no_sleep, message):
    body = {"error": {"type": "quota_exceeded_error", "code": "8", "message": message}}
    rec = Recorder(httpx.Response(429, json=body, headers={"Retry-After": "2"}), httpx.Response(200, json=ok_body()))
    assert generate({"api_key": KEY}, rec).image == PNG
    assert len(rec.requests) == 2 and no_sleep == [2.0]


def test_429_rate_limit_gives_up_after_two_attempts(no_sleep):
    body = {"error": {"type": "quota_exceeded_error", "message": "rate limit or quota exceeded"}}
    rec = Recorder(httpx.Response(429, json=body))
    err = draw_error({"api_key": KEY}, rec)
    assert "请求过于频繁" in str(err) and len(rec.requests) == 2 and no_sleep == [3.0]


def test_5xx_retry_rules():
    rec = Recorder(httpx.Response(500, json={"error": {"type": "internal_server_error", "message": "boom"}}))
    err = draw_error({"api_key": KEY}, rec)
    assert "服务器错误" in str(err) and len(rec.requests) == 1  # 可能已扣费，不重试
    rec = Recorder(httpx.Response(504, text="gateway timeout"))
    draw_error({"api_key": KEY}, rec)
    assert len(rec.requests) == 1
    rec = Recorder(httpx.Response(503, text="busy"), httpx.Response(200, json=ok_body()))
    assert generate({"api_key": KEY}, rec).image == PNG and len(rec.requests) == 2
    rec = Recorder(httpx.Response(502, text="bad gateway"))
    draw_error({"api_key": KEY}, rec)
    assert len(rec.requests) == 2  # 最多重试一次


def test_read_timeout_not_retried():
    rec = Recorder(httpx.ReadTimeout("slow"), httpx.Response(200, json=ok_body()))
    err = draw_error({"api_key": KEY}, rec)
    assert "超时" in str(err) and not err.retryable and len(rec.requests) == 1


def test_connect_error_retried_once():
    rec = Recorder(httpx.ConnectError("refused"), httpx.Response(200, json=ok_body()))
    assert generate({"api_key": KEY}, rec).image == PNG and len(rec.requests) == 2
    rec = Recorder(httpx.ConnectTimeout("refused"))
    err = draw_error({"api_key": KEY}, rec)
    assert "无法连接" in str(err) and len(rec.requests) == 2


# ---------------------------------------------------------------- 配置


def test_extra_body_merged_last():
    extra = '{"watermark": true, "size": "1024x1024", "prompt_extend": false, "model": "x-model", "foo": 1}'
    rec = Recorder(httpx.Response(200, json=ok_body()))
    result = generate({"api_key": KEY, "extra_body": extra}, rec)
    body = rec.bodies[0]
    assert body["watermark"] is True and body["size"] == "1024x1024" and body["prompt_extend"] is False
    assert body["model"] == "x-model" and body["foo"] == 1
    assert result.params["model"] == "x-model" and result.params["watermark"] is True


def test_extra_body_as_dict():
    rec = Recorder(httpx.Response(200, json=ok_body()))
    generate({"api_key": KEY, "extra_body": {"output_format": "webp"}}, rec)
    assert rec.bodies[0]["output_format"] == "webp"


@pytest.mark.parametrize("extra", ["{bad json", "[1, 2]", '"text"'])
def test_extra_body_invalid(extra):
    rec = Recorder(httpx.Response(200, json=ok_body()))
    err = draw_error({"api_key": KEY, "extra_body": extra}, rec)
    assert "额外请求参数" in str(err) and rec.requests == []


@pytest.mark.parametrize(
    ("value", "watermark", "prompt_extend"),
    [(True, True, True), (False, False, False), ("true", True, True), ("false", False, False),
     ("0", False, False), ("1", True, True), ("", False, True), (None, False, True), ("garbage", False, True),
     (1, True, True), (0, False, False)],
)
def test_bool_flags(value, watermark, prompt_extend):
    rec = Recorder(httpx.Response(200, json=ok_body()))
    generate({"api_key": KEY, "watermark": value, "prompt_extend": value}, rec)
    assert rec.bodies[0]["watermark"] is watermark and rec.bodies[0]["prompt_extend"] is prompt_extend


@pytest.mark.parametrize(("value", "expected"), [("webp", "webp"), ("JPEG", "jpeg"), ("jpg", "jpeg"), ("gif", "png"),
                                                 ("", "png"), (None, "png"), (3, "png")])
def test_output_format(value, expected):
    assert SenseNovaBackend({"output_format": value}).output_format == expected


def test_garbage_config_still_valid():
    section = {
        "api_key": f"  Bearer {KEY}  ", "base_url": None, "proxy_url": None, "model": None,
        "portrait_size": "abc", "landscape_size": None, "square_size": 12, "watermark": "",
        "prompt_extend": "maybe", "output_format": ["png"], "timeout": "soon", "extra_body": None,
    }
    backend = SenseNovaBackend(section)
    assert backend.timeout == 300.0 and backend.proxy == "" and backend.model == "sensenova-u1.5-lite"
    for orientation, size in sn.DEFAULT_SIZES.items():
        rec = Recorder(httpx.Response(200, json=ok_body()))
        generate(section, rec, DrawRequest(prompt="x", orientation=orientation))
        assert str(rec.requests[0].url) == GEN_URL
        assert rec.requests[0].headers["authorization"] == f"Bearer {KEY}"
        assert rec.bodies[0] == {"model": "sensenova-u1.5-lite", "prompt": "x", "size": size,
                                 "response_format": "b64_json", "output_format": "png", "watermark": False,
                                 "prompt_extend": True}
    assert SenseNovaBackend(None).section == {} and SenseNovaBackend("bad").missing()
    assert SenseNovaBackend({"timeout": 5}).timeout == 10.0 and SenseNovaBackend({"timeout": "60"}).timeout == 60.0


def test_missing_and_summary():
    assert SenseNovaBackend({}).missing() == "管理员还没有配置 SenseNova API Key"
    assert SenseNovaBackend({"api_key": "   "}).missing()
    assert SenseNovaBackend({"api_key": KEY}).missing() == ""
    assert SenseNovaBackend({}).summary() == "SenseNova / sensenova-u1.5-lite / 竖图 1664x2496"
    assert SenseNovaBackend({"model": "sensenova-u1.5-fast", "portrait_size": "4K"}).summary() == \
        "SenseNova / sensenova-u1.5-fast / 竖图 2720x4096"
    assert SenseNovaBackend({"model": "other-model", "portrait_size": "4K"}).summary() == \
        "SenseNova / other-model / 竖图 1664x2496"
    rec = Recorder(httpx.Response(200, json=ok_body()))
    err = draw_error({}, rec)
    assert "API Key" in str(err) and rec.requests == []


def test_class_attributes():
    assert SenseNovaBackend.key == "sensenova" and SenseNovaBackend.label == "SenseNova"
    assert SenseNovaBackend.default_prompt_mode == "natural" and SenseNovaBackend({}).prompt_mode == "natural"
    assert SenseNovaBackend.supports_negative is False and SenseNovaBackend.supports_models is True
    assert SenseNovaBackend.max_concurrency is None
    assert SenseNovaBackend.doc_url.startswith("https://platform.sensenova.cn/docs")
    assert SenseNovaBackend.max_edit_images == 5 and SenseNovaBackend.supports_strength is False
    assert SenseNovaBackend.supports_balance is False
    # 类属性与实例属性都是字符串（webapi 在实例上读取）
    assert SenseNovaBackend.billing_url == "https://platform.sensenova.cn/console"
    assert SenseNovaBackend({}).billing_url == "https://platform.sensenova.cn/console"


# ---------------------------------------------------------------- 模型列表


def test_list_models_filters_image_models():
    body = {"data": [
        {"id": "sensenova-6.8-flash-lite", "output_modalities": ["text"]},
        {"id": "sensenova-u1.5-lite", "output_modalities": ["image"]},
        {"id": "sensenova-u1.5-fast", "output_modalities": ["text", "image"]},
        {"id": "sensenova-u1-pro"},
        {"id": "sensenova-u1.5-lite", "output_modalities": ["image"]},
        {"id": "deepseek-v3", "output_modalities": None},
        "garbage", {"name": "no-id"},
    ]}
    rec = Recorder(httpx.Response(200, json=body))
    backend = SenseNovaBackend({"api_key": KEY, "base_url": "https://token.sensenova.cn/v1/images/generations"})
    models = run(rec, backend.list_models)
    assert models == ["sensenova-u1.5-lite", "sensenova-u1.5-fast", "sensenova-u1-pro"]
    req = rec.requests[0]
    assert req.method == "GET" and str(req.url) == "https://token.sensenova.cn/v1/models"
    assert req.headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("body", [{"data": [{"id": "sensenova-6.8-flash-lite", "output_modalities": ["text"]}]},
                                  {"data": "x"}, {}])
def test_list_models_fallback(body):
    backend = SenseNovaBackend({"api_key": KEY})
    assert run(Recorder(httpx.Response(200, json=body)), backend.list_models) == list(sn.KNOWN_MODELS)


def test_list_models_errors():
    backend = SenseNovaBackend({"api_key": KEY})
    with pytest.raises(DrawError, match="API Key"):
        run(Recorder(httpx.Response(401, json={"error": {"code": 16, "message": "Forbidden"}})), backend.list_models)
    with pytest.raises(DrawError, match="API Key"):
        run(Recorder(httpx.Response(200, json={})), SenseNovaBackend({}).list_models)


# ---------------------------------------------------------------- 图片修改：请求


def test_edit_request_shape():
    rec = Recorder(httpx.Response(200, json=ok_body(size="1664x2496", usage={"total_tokens": 12881})))
    first, second = image(800, 1200, PNG, "image/png"), image(600, 600, JPEG, "image/jpeg")
    result = generate({"api_key": KEY}, rec, edit_req(first, second))
    req = rec.requests[0]
    assert req.method == "POST" and str(req.url) == EDIT_URL and len(rec.requests) == 1
    assert req.headers["authorization"] == f"Bearer {KEY}"
    assert req.headers["content-type"] == "application/json"
    assert rec.bodies[0] == {
        "model": "sensenova-u1.5-lite",
        # 第 1 张为主编辑图，顺序与用户发送的一致；完整 data URL 前缀
        "images": [{"image_url": data_url(PNG, "image/png")}, {"image_url": data_url(JPEG, "image/jpeg")}],
        "prompt": "Replace the background of image 1 with a glacier",
        "size": "auto",
        "response_format": "b64_json",
        "output_format": "png",
        "watermark": False,
        "prompt_extend": True,
    }
    assert rec.bodies[0]["images"][0]["image_url"].startswith("data:image/png;base64,iVBORw0KGgo")
    assert result.image == PNG and result.mime == "image/png"
    assert result.params == {"mode": "edit", "model": "sensenova-u1.5-lite", "size": "1664x2496",
                             "output_format": "png", "watermark": False, "prompt_extend": True,
                             "input_images": 2, "tokens": 12881}


@pytest.mark.parametrize("strength", [0.0, 0.05, 0.4, 0.99, 1.0])
def test_edit_no_undocumented_fields_and_negative_in_prompt(strength):
    # 接口没有重绘幅度字段：任何 strength 都不影响请求体
    rec = Recorder(httpx.Response(200, json=ok_body()))
    generate({"api_key": KEY}, rec, edit_req(prompt="  make it night  ", negative="text", seed=7, strength=strength))
    body = rec.bodies[0]
    assert set(body) == {"model", "images", "prompt", "size", "response_format", "output_format", "watermark",
                         "prompt_extend"}
    assert body["prompt"] == "make it night  \n\nAvoid: text"
    for field in FORBIDDEN_FIELDS + ("mask", "image", "strength", "denoise", "input_fidelity"):
        assert field not in body


def test_edit_uses_configured_model_and_format():
    rec = Recorder(httpx.Response(200, json=ok_body(JPEG)))
    section = {"api_key": KEY, "model": "sensenova-u1.5-fast", "output_format": "jpg", "watermark": True,
               "prompt_extend": False, "base_url": "https://token.sensenova.ai/v1/images/generations"}
    result = generate(section, rec, edit_req())
    body = rec.bodies[0]
    assert str(rec.requests[0].url) == "https://token.sensenova.ai/v1/images/edits"
    assert body["model"] == "sensenova-u1.5-fast" and body["output_format"] == "jpeg"
    assert body["watermark"] is True and body["prompt_extend"] is False
    assert result.image == JPEG and result.params["model"] == "sensenova-u1.5-fast"


def test_edit_image_mime_from_file_header():
    # 以文件头为准：声明为 PNG 的 JPEG 数据按 JPEG 发送
    rec = Recorder(httpx.Response(200, json=ok_body()))
    generate({"api_key": KEY}, rec, edit_req(image(data=JPEG, mime="image/png")))
    assert rec.bodies[0]["images"] == [{"image_url": data_url(JPEG, "image/jpeg")}]
    assert rec.bodies[0]["images"][0]["image_url"].startswith("data:image/jpeg;base64,/9j/")


def test_edit_png_mime_from_header_even_if_declared_jpeg():
    rec = Recorder(httpx.Response(200, json=ok_body()))
    generate({"api_key": KEY}, rec, edit_req(image(data=PNG, mime="image/jpg")))
    assert rec.bodies[0]["images"] == [{"image_url": data_url(PNG, "image/png")}]


@pytest.mark.parametrize(("data", "mime", "text"), [(GIF, "image/gif", "PNG / JPEG"),
                                                     (b"not an image", "image/webp", "PNG / JPEG"),
                                                     # 声明为 PNG / JPEG 但文件头无法识别：不上传
                                                     (b"not an image", "image/png", "PNG / JPEG"),
                                                     (b"not an image", "image/jpeg", "PNG / JPEG"),
                                                     (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/png", "PNG / JPEG"),
                                                     (b"", "image/png", "空的")])
def test_edit_rejects_unsupported_input(data, mime, text):
    rec = Recorder(httpx.Response(200, json=ok_body()))
    err = draw_error({"api_key": KEY}, rec, edit_req(image(data=data, mime=mime)))
    assert text in str(err) and rec.requests == []


def test_edit_trims_to_five_images_in_order():
    images = [image(data=PNG + bytes([i])) for i in range(7)]
    rec = Recorder(httpx.Response(200, json=ok_body()))
    result = generate({"api_key": KEY}, rec, edit_req(*images))
    assert rec.bodies[0]["images"] == [{"image_url": data_url(PNG + bytes([i]), "image/png")} for i in range(5)]
    assert result.params["input_images"] == 5 and result.params["ignored_images"] == 2


def test_edit_five_images_sent_untrimmed():
    images = [image(data=PNG + bytes([i])) for i in range(5)]
    rec = Recorder(httpx.Response(200, json=ok_body()))
    result = generate({"api_key": KEY}, rec, edit_req(*images))
    assert len(rec.bodies[0]["images"]) == 5
    assert result.params["input_images"] == 5 and "ignored_images" not in result.params


def test_edit_blank_prompt_rejected_without_request():
    rec = Recorder(httpx.Response(200, json=ok_body()))
    err = draw_error({"api_key": KEY}, rec, edit_req(prompt="   ", negative="text"))
    assert "修改要求" in str(err) and rec.requests == []


def test_edit_missing_key_checked_first():
    rec = Recorder(httpx.Response(200, json=ok_body()))
    err = draw_error({}, rec, edit_req())
    assert "API Key" in str(err) and rec.requests == []


@pytest.mark.parametrize("model", ["sensenova-u1.5-lite", "sensenova-u1.5-fast", "SenseNova-U1.5-Lite", None, ""])
def test_edit_supported_models(model):
    assert SenseNovaBackend({"api_key": KEY, "model": model}).edit_missing() == ""


@pytest.mark.parametrize("model", ["sensenova-u1-fast", "sensenova-6.8-flash-lite", "my-custom-model"])
def test_edit_unsupported_model(model):
    reason = SenseNovaBackend({"api_key": KEY, "model": model}).edit_missing()
    assert f"当前模型 {model} 不支持修改图片" in reason and "sensenova-u1.5-lite" in reason
    rec = Recorder(httpx.Response(200, json=ok_body()))
    err = draw_error({"api_key": KEY, "model": model}, rec, edit_req())
    assert str(err) == reason and rec.requests == []
    # 文生图不受影响
    rec = Recorder(httpx.Response(200, json=ok_body()))
    generate({"api_key": KEY, "model": model}, rec)
    assert str(rec.requests[0].url) == GEN_URL


def test_edit_extra_body_cannot_override_fixed_fields():
    extra = {"images": [], "image": "x", "mask": "m", "prompt": "p", "size": "999x999", "n": 2,
             "watermark": True, "prompt_extend": False, "foo": 1}
    section = {"api_key": KEY, "extra_body": dict(extra), "edit_size": "2K"}
    rec = Recorder(httpx.Response(200, json=ok_body()))
    generate(section, rec, edit_req(image(800, 1200)))
    body = rec.bodies[0]
    assert body["images"] == [{"image_url": data_url(PNG, "image/png")}]
    assert body["prompt"] == "Replace the background of image 1 with a glacier" and body["size"] == "1664x2496"
    for field in ("image", "mask", "n"):
        assert field not in body
    assert body["watermark"] is True and body["prompt_extend"] is False and body["foo"] == 1
    assert section["extra_body"] == extra  # 配置本身没有被修改
    # 文生图仍然按原样合并
    rec = Recorder(httpx.Response(200, json=ok_body()))
    generate(section, rec)
    assert rec.bodies[0]["size"] == "999x999" and rec.bodies[0]["n"] == 2


def test_edit_extra_body_invalid():
    rec = Recorder(httpx.Response(200, json=ok_body()))
    err = draw_error({"api_key": KEY, "extra_body": "{bad"}, rec, edit_req())
    assert "额外请求参数" in str(err) and rec.requests == []


# ---------------------------------------------------------------- 图片修改：尺寸


@pytest.mark.parametrize(
    ("config", "model", "dims", "orientation", "expected"),
    [
        (None, "sensenova-u1.5-lite", (800, 1200), "portrait", "auto"),
        ("", "sensenova-u1.5-lite", (800, 1200), "portrait", "auto"),
        ("AUTO", "sensenova-u1.5-lite", (1920, 1080), "landscape", "auto"),
        ("garbage", "sensenova-u1.5-lite", (800, 1200), "portrait", "auto"),
        # 档位：按主图比例换算
        ("2K", "sensenova-u1.5-lite", (800, 1200), "portrait", "1664x2496"),
        ("1k", "sensenova-u1.5-lite", (800, 1200), "portrait", "1088x1632"),
        ("4K", "sensenova-u1.5-lite", (800, 1200), "portrait", "2720x4096"),
        ("4K", "sensenova-u1.5-fast", (800, 1200), "portrait", "2720x4096"),
        ("4K", "my-custom-model", (800, 1200), "portrait", "1664x2496"),
        ("2K", "sensenova-u1.5-lite", (1000, 1000), "square", "2048x2048"),
        ("2K", "sensenova-u1.5-lite", (1920, 1080), "landscape", "2720x1536"),
        # 固定尺寸
        ("1000x1500", "sensenova-u1.5-lite", (1920, 1080), "landscape", "992x1504"),
        # 用户要求改变方向：auto 会保持原图比例，改用该方向的尺寸
        ("", "sensenova-u1.5-lite", (800, 1200), "landscape", "2496x1664"),
        ("1K", "sensenova-u1.5-lite", (800, 1200), "landscape", "1632x1088"),
        ("4K", "my-custom-model", (800, 1200), "square", "2048x2048"),
        # 不知道主图尺寸时交给服务端
        ("2K", "sensenova-u1.5-lite", (0, 0), "landscape", "auto"),
        ("", "sensenova-u1.5-lite", (0, 0), "portrait", "auto"),
        # 方向无法识别时按主图方向
        ("2K", "sensenova-u1.5-lite", (1920, 1080), "bogus", "2720x1536"),
    ],
)
def test_edit_size(config, model, dims, orientation, expected):
    section = {"api_key": KEY, "model": model, "edit_size": config}
    req = edit_req(image(*dims), orientation=orientation)
    backend = SenseNovaBackend(section)
    assert backend.edit_size(req) == expected
    if expected != "auto":
        assert_valid_size(expected)
    if backend.edit_missing():  # 文档之外的模型不能修改图片，只检查尺寸换算
        return
    rec = Recorder(httpx.Response(200, json=ok_body()))
    result = generate(section, rec, req)
    assert rec.bodies[0]["size"] == expected and result.params["size"] == expected


def test_edit_orientation_change_uses_configured_size():
    backend = SenseNovaBackend({"api_key": KEY, "landscape_size": "16:9"})
    assert backend.edit_size(edit_req(image(800, 1200), orientation="landscape")) == "2720x1536"


def test_edit_size_extreme_aspect_is_valid():
    for w, h in ((4000, 1000), (1000, 4000), (2048, 300), (513, 2047)):
        img = image(w, h)
        req = edit_req(img, orientation=base.image_orientation(img))
        for tier in ("1K", "2K", "4K"):
            assert_valid_size(SenseNovaBackend({"edit_size": tier}).edit_size(req))


def test_edit_size_always_valid_or_auto():
    # 任意主图尺寸（含 1 像素、奇数边长、极端比例）、任意配置、任意方向，都不会产生文档不允许的尺寸
    dims = (1, 7, 31, 333, 511, 513, 1023, 1500, 2047, 2048)
    configs = ("", "auto", "1K", "2K", "4K", " 4k ", "1000x1500", "9999x1", "3:4", "garbage", None, 2048)
    section = {"portrait_size": "4K", "landscape_size": "auto", "square_size": "1:3"}
    for model in ("sensenova-u1.5-lite", "sensenova-u1.5-fast", "other"):
        for config in configs:
            backend = SenseNovaBackend({**section, "model": model, "edit_size": config})
            for w in dims:
                for h in dims:
                    for orientation in ("portrait", "landscape", "square", "bogus"):
                        size = backend.edit_size(edit_req(image(w, h), orientation=orientation))
                        if size != "auto":
                            assert_valid_size(size)


def test_edit_plan():
    backend = SenseNovaBackend({"api_key": KEY, "edit_size": "2K"})
    assert backend.plan(edit_req(image(800, 1200))) == {"mode": "edit", "model": "sensenova-u1.5-lite",
                                                        "input_images": 1, "size": "1664x2496"}
    assert backend.plan(edit_req(*[image()] * 7))["input_images"] == 5
    assert SenseNovaBackend({}).plan(edit_req())["size"] == "auto"
    # 文生图的预计参数不变
    assert backend.plan(DrawRequest(prompt="x", orientation="landscape")) == {"model": "sensenova-u1.5-lite",
                                                                             "size": "2496x1664"}


# ---------------------------------------------------------------- 图片修改：响应与错误


def test_edit_url_response_downloaded_without_key():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"data": [{"url": "https://cdn.example.com/edit/a.jpeg"}],
                                             "output_format": "jpeg", "size": "1536x2720"})
        return httpx.Response(200, content=JPEG, headers={"content-type": "image/jpeg"})

    rec = Recorder(handler)
    result = generate({"api_key": KEY, "extra_body": '{"response_format": "url"}'}, rec, edit_req())
    assert str(rec.requests[0].url) == EDIT_URL and rec.bodies[0]["response_format"] == "url"
    download = rec.requests[1]
    assert download.method == "GET" and str(download.url) == "https://cdn.example.com/edit/a.jpeg"
    assert "authorization" not in download.headers
    assert result.image == JPEG and result.params["size"] == "1536x2720" and result.params["mode"] == "edit"


@pytest.mark.parametrize("body", [{"data": []}, {"data": [{}]}, {"created": 1}])
def test_edit_no_image_in_response(body):
    err = draw_error({"api_key": KEY}, Recorder(httpx.Response(200, json=body)), edit_req())
    assert "没有返回图片" in str(err)


def test_edit_error_object_in_200_response():
    body = {"error": {"type": "failed_precondition_error", "code": "9", "message": "安全检查未通过"}}
    err = draw_error({"api_key": KEY}, Recorder(httpx.Response(200, json=body)), edit_req())
    assert "拒绝修改图片" in str(err) and "输入图片" in str(err) and not err.retryable


@pytest.mark.parametrize("message", ["image base64 decode failed", "failed to download image url",
                                     "images[0].image_url is not a valid image", "图片链接无法访问"])
def test_edit_unreadable_input(message):
    body = {"error": {"type": "invalid_request_error", "code": "3", "message": message}}
    rec = Recorder(httpx.Response(400, json=body), httpx.Response(200, json=ok_body()))
    err = draw_error({"api_key": KEY}, rec, edit_req())
    assert "无法读取输入图片" in str(err) and message in str(err)
    assert err.status == 400 and not err.retryable and len(rec.requests) == 1


@pytest.mark.parametrize("message", ["size must be a multiple of 32",
                                     # 文档的参数错误格式为 "invalid <字段>, should ..."
                                     "invalid images, should be at most 5 images",
                                     "invalid image size, should be multiples of 32",
                                     "invalid image_size", "invalid image count: 6", "invalid image-ratio",
                                     "size 不是有效值", "图片数量无效"])
def test_edit_invalid_param_not_reported_as_unreadable(message):
    body = {"error": {"type": "invalid_request_error", "code": "3", "message": message}}
    rec = Recorder(httpx.Response(400, json=body))
    err = draw_error({"api_key": KEY}, rec, edit_req())
    assert "请求参数有误" in str(err) and "无法读取" not in str(err) and "内容安全" not in str(err)
    assert message in str(err) and not err.retryable and len(rec.requests) == 1


@pytest.mark.parametrize("message", ["invalid image", "invalid image: cannot identify file", "invalid image data",
                                     "invalid image_url, should be a data URL or http(s) URL",
                                     "链接内容并非有效图片", "不是有效的图片", "无效的图片"])
def test_edit_unreadable_input_more_wordings(message):
    body = {"error": {"type": "invalid_request_error", "code": "3", "message": message}}
    err = draw_error({"api_key": KEY}, Recorder(httpx.Response(400, json=body)), edit_req())
    assert "无法读取输入图片" in str(err) and message in str(err) and not err.retryable


def test_edit_413_too_large():
    rec = Recorder(httpx.Response(413, text="<html><title>413 Request Entity Too Large</title></html>"),
                   httpx.Response(200, json=ok_body()))
    err = draw_error({"api_key": KEY}, rec, edit_req())
    assert "HTTP 413" in str(err) and "减少图片数量" in str(err)
    assert err.status == 413 and not err.retryable and len(rec.requests) == 1


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (400, {"error": {"type": "failed_precondition_error", "code": "9", "message": "security check failed"}}),
        (400, {"error": {"type": "failed_precondition_error", "code": "9", "message": "安全检查未通过"}}),
        (403, {"error": {"type": "permission_denied_error", "code": "7", "message": "sensitive content"}}),
        (400, {"error": {"code": 3, "message": "输入图片包含敏感内容"}}),
    ],
)
def test_edit_moderation_refusal(status, body):
    rec = Recorder(httpx.Response(status, json=body), httpx.Response(200, json=ok_body()))
    err = draw_error({"api_key": KEY}, rec, edit_req())
    assert "拒绝修改图片" in str(err) and "内容安全" in str(err) and f"HTTP {status}" in str(err)
    assert not err.retryable and len(rec.requests) == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.ReadTimeout("slow"),
        httpx.Response(500, json={"error": {"type": "internal_server_error", "message": "boom"}}),
        httpx.Response(504, text="gateway timeout"),
        httpx.Response(429, json={"error": {"type": "quota_exceeded_error", "message": "weekly quota exceeded"}}),
        httpx.Response(401, json={"error": {"code": 16, "message": "Forbidden"}}),
    ],
)
def test_edit_possibly_billed_or_fatal_errors_not_retried(response):
    rec = Recorder(response, httpx.Response(200, json=ok_body()))
    err = draw_error({"api_key": KEY}, rec, edit_req())
    assert not err.retryable and len(rec.requests) == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.ConnectError("refused"),
        httpx.Response(503, text="busy"),
        httpx.Response(429, json={"error": {"type": "quota_exceeded_error", "message": "rate limit exceeded"}}),
        httpx.Response(400, json={"error": {"type": "failed_precondition_error", "message": "engine unavailable"}}),
    ],
)
def test_edit_safe_errors_retried_once(no_sleep, response):
    rec = Recorder(response, httpx.Response(200, json=ok_body()))
    result = generate({"api_key": KEY}, rec, edit_req())
    assert result.params["mode"] == "edit" and len(rec.requests) == 2
    assert all(str(r.url) == EDIT_URL for r in rec.requests) and rec.bodies[0] == rec.bodies[1]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"error": {"code": 16, "message": "Forbidden"}}),
        httpx.Response(413, text="too large"),
        httpx.Response(400, json={"error": {"type": "invalid_request_error", "message": "base64 decode failed"}}),
        httpx.Response(429, json={"error": {"type": "quota_exceeded_error", "message": "quota exhausted"}}),
        httpx.Response(500, text="<html><title>boom</title></html>"),
        httpx.ReadTimeout("slow"),
        httpx.ConnectError("refused"),
    ],
)
def test_edit_api_key_never_in_error(response):
    err = draw_error({"api_key": f"Bearer {KEY}"}, Recorder(response), edit_req())
    assert KEY not in str(err) and KEY[-8:] not in str(err)


def test_edit_params_and_plan_have_no_secrets_or_image_data():
    backend = SenseNovaBackend({"api_key": KEY})
    result = run(Recorder(httpx.Response(200, json=ok_body())), lambda c: backend.generate(c, edit_req()))
    text = json.dumps(result.params) + json.dumps(backend.plan(edit_req())) + backend.summary()
    assert KEY not in text and KEY[-8:] not in text and "base64" not in text


# ---------------------------------------------------------------- 余额


@pytest.mark.parametrize(
    ("base_url", "console"),
    [
        ("", sn.CONSOLE_CN),
        (None, sn.CONSOLE_CN),
        ("https://token.sensenova.cn/v1", sn.CONSOLE_CN),
        ("token.sensenova.ai/v1/images/edits", sn.CONSOLE_INTL),
        ("https://TOKEN.SENSENOVA.AI/v1", sn.CONSOLE_INTL),
        ("https://relay.example.com/v1", sn.CONSOLE_CN),
        ("https://evil-sensenova.ai/v1", sn.CONSOLE_CN),
        ("http://[::1", sn.CONSOLE_CN),  # 地址格式错误时不影响创建后端
    ],
)
def test_billing_url_follows_site(base_url, console):
    backend = SenseNovaBackend({"api_key": KEY, "base_url": base_url})
    assert backend.billing_url == console
    assert console in backend.balance_missing()


def test_balance_not_supported_and_no_request():
    backend = SenseNovaBackend({"api_key": KEY})
    reason = backend.balance_missing()
    assert reason.startswith("SenseNova 没有可以用 API Key 查询积分的接口")
    assert sn.CONSOLE_CN in reason and "积分" in reason
    assert KEY not in reason and KEY[-4:] not in reason
    rec = Recorder(httpx.Response(200, json={}))
    with pytest.raises(DrawError) as info:
        run(rec, backend.balance)
    assert str(info.value) == reason and rec.requests == []
    # 没有配置 Key 时同样提示到控制台查看
    assert SenseNovaBackend({}).balance_missing() == reason


# ---------------------------------------------------------------- 配置模板


def load_schema_section() -> dict:
    return json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))["sensenova"]


def test_schema_defaults_produce_valid_request():
    section = load_schema_section()
    assert section["type"] == "object" and section["description"]
    items = section["items"]
    assert list(items) == ["api_key", "base_url", "proxy_url", "model", "portrait_size", "landscape_size",
                           "square_size", "edit_size", "watermark", "prompt_extend", "output_format", "timeout",
                           "extra_body"]
    for name, field in items.items():
        assert field["type"] in ("string", "text", "int", "float", "bool", "list"), name
        assert field.get("description") and field.get("hint"), name
        if "options" in field:
            assert field["default"] in field["options"], name
            assert len(field.get("labels", field["options"])) == len(field["options"]), name
    assert items["output_format"]["options"] == list(sn.OUTPUT_FORMATS)
    assert items["api_key"].get("secret") is True
    assert items["extra_body"]["editor_mode"] is True and items["extra_body"]["editor_language"] == "json"

    defaults = {name: field["default"] for name, field in items.items()}
    assert defaults["base_url"] == sn.DEFAULT_BASE_URL and defaults["model"] == sn.DEFAULT_MODEL
    assert defaults["timeout"] == sn.DEFAULT_TIMEOUT and defaults["output_format"] == "png"
    assert defaults["watermark"] is False and defaults["prompt_extend"] is True
    for orientation, size in sn.DEFAULT_SIZES.items():
        assert defaults[f"{orientation}_size"] == size
    assert defaults["edit_size"] == sn.DEFAULT_EDIT_SIZE == "auto"
    assert SenseNovaBackend(defaults).missing()  # 默认没有 Key
    # 默认模型可以修改图片；余额没有查询接口，只能到控制台查看
    assert SenseNovaBackend(defaults).edit_missing() == ""
    assert sn.CONSOLE_CN in SenseNovaBackend(defaults).balance_missing()

    defaults["api_key"] = KEY
    for orientation in ("portrait", "landscape", "square"):
        rec = Recorder(httpx.Response(200, json=ok_body()))
        result = generate(defaults, rec, DrawRequest(prompt="x", orientation=orientation))
        body = rec.bodies[0]
        assert str(rec.requests[0].url) == GEN_URL
        assert body["size"] == sn.DEFAULT_SIZES[orientation]
        assert_valid_size(body["size"])
        assert body["model"] == "sensenova-u1.5-lite" and body["watermark"] is False and body["prompt_extend"] is True
        assert body["output_format"] in sn.OUTPUT_FORMATS and body["response_format"] == "b64_json"
        assert set(body) == {"model", "prompt", "size", "response_format", "output_format", "watermark",
                             "prompt_extend"}
        assert result.image == PNG

    # 默认配置修改图片：保持主图比例（auto），字段与文档一致
    rec = Recorder(httpx.Response(200, json=ok_body()))
    result = generate(defaults, rec, DrawRequest(prompt="x", orientation="portrait", images=[image(800, 1200)]))
    body = rec.bodies[0]
    assert str(rec.requests[0].url) == EDIT_URL
    assert body["size"] == "auto" and body["model"] == "sensenova-u1.5-lite"
    assert set(body) == {"model", "images", "prompt", "size", "response_format", "output_format", "watermark",
                         "prompt_extend"}
    assert result.params["mode"] == "edit"
