"""xAI 后端的单元测试（不联网）：python -m pytest tests/test_backend_xai.py"""

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
from backends.base import Clients, DrawError, DrawRequest, HttpPool  # noqa: E402
from backends.xai import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_QUALITY,
    DEFAULT_RESOLUTION,
    DEFAULT_SIZES,
    RATIOS,
    RESOLUTIONS,
    V1_RATIOS,
    XAIBackend,
)


JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
GEN_URL = "https://api.x.ai/v1/images/generations"
ALLOWED_KEYS = {"model", "prompt", "n", "response_format", "aspect_ratio", "resolution", "quality"}


def ok_json(data: bytes = JPEG, **item) -> dict:
    return {"data": [{"b64_json": base64.b64encode(data).decode(), "mime_type": "image/jpeg", **item}],
            "usage": {"cost_in_usd_ticks": 400000000}}


class Recorder:
    """记录请求并按顺序返回预设响应（响应可以是 httpx.Response、异常或函数）。"""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        resp = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(resp, Exception):
            raise resp
        if callable(resp):
            return resp(request)
        return resp

    def body(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content)


def run(backend: XAIBackend, handler, req: DrawRequest | None = None, method: str = "generate"):
    async def main():
        pool = HttpPool(transport=httpx.MockTransport(handler))
        try:
            clients = Clients(http=pool)
            if method == "generate":
                return await backend.generate(clients, req or DrawRequest(prompt="a cat"))
            return await backend.list_models(clients)
        finally:
            await pool.close()

    return asyncio.run(main())


@pytest.fixture
def sleeps(monkeypatch):
    """跳过重试等待，并记录等待时间。"""
    delays = []

    async def fake_sleep(delay, *args, **kwargs):
        delays.append(delay)

    monkeypatch.setattr(base.asyncio, "sleep", fake_sleep)
    return delays


def backend(**section) -> XAIBackend:
    return XAIBackend({"api_key": "xai-test", **section})


# ---------------------------------------------------------------- 请求体


def test_class_attributes():
    assert XAIBackend.key == "xai" and XAIBackend.label == "xAI"
    assert XAIBackend.default_prompt_mode == "natural"
    assert XAIBackend.supports_negative is False and XAIBackend.supports_models is True
    assert XAIBackend.max_concurrency is None
    assert XAIBackend.doc_url.startswith("https://docs.x.ai/")


@pytest.mark.parametrize(("orientation", "ratio"), [("portrait", "2:3"), ("landscape", "3:2"), ("square", "1:1"),
                                                    ("diagonal", "2:3")])
def test_default_request_per_orientation(orientation, ratio):
    rec = Recorder(httpx.Response(200, json=ok_json()))
    result = run(backend(), rec, DrawRequest(prompt="a cat", orientation=orientation))
    request = rec.requests[0]
    assert request.method == "POST" and str(request.url) == GEN_URL
    assert request.headers["authorization"] == "Bearer xai-test"
    assert request.headers["content-type"] == "application/json"
    assert rec.body() == {"model": "grok-imagine-image-2.0", "prompt": "a cat", "n": 1, "response_format": "b64_json",
                          "aspect_ratio": ratio, "resolution": "1k", "quality": "medium"}
    assert result.image == JPEG and result.mime == "image/jpeg"
    assert result.params == {"model": "grok-imagine-image-2.0", "size": f"{ratio} 1k", "aspect_ratio": ratio,
                             "resolution": "1k", "quality": "medium", "cost_usd": 0.04}


def test_negative_prompt_folded_and_no_unsupported_fields():
    rec = Recorder(httpx.Response(200, json=ok_json()))
    run(backend(), rec, DrawRequest(prompt="a cat", negative="text", seed=42))
    body = rec.body()
    assert body["prompt"] == "a cat\n\nAvoid: text"
    assert set(body) <= ALLOWED_KEYS
    for key in ("size", "seed", "negative_prompt", "style"):
        assert key not in body


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # 1.0：不支持画质，21:9 换成最接近的 20:9
        ("grok-imagine-image", {"aspect_ratio": "20:9", "resolution": "1k"}),
        # 1.0 的日期别名以 grok-imagine-image-2 开头，不能当成 2.0
        ("grok-imagine-image-2026-03-02", {"aspect_ratio": "20:9", "resolution": "1k"}),
        ("grok-imagine-image-quality", {"aspect_ratio": "20:9", "resolution": "1k"}),
        ("grok-imagine-image-2.0", {"aspect_ratio": "21:9", "resolution": "1k", "quality": "medium"}),
        ("grok-imagine-image-2.0-latest", {"aspect_ratio": "21:9", "resolution": "1k", "quality": "medium"}),
        # 已下线的 grok-2-image 只支持 model / prompt / n / response_format
        ("grok-2-image-1212", {}),
        ("grok-2-image", {}),
    ],
)
def test_model_families(model, expected):
    rec = Recorder(httpx.Response(200, json=ok_json()))
    result = run(backend(model=model, landscape_size="21:9"), rec, DrawRequest(prompt="p", orientation="landscape"))
    assert rec.body() == {"model": model, "prompt": "p", "n": 1, "response_format": "b64_json", **expected}
    assert result.params["model"] == model
    if not expected:
        assert result.params["size"] == "auto"


@pytest.mark.parametrize(
    ("size", "ratio", "resolution"),
    [
        ("3:4", "3:4", "1k"),
        ("4:5", "3:4", "1k"),
        ("832x1216", "2:3", "1k"),
        ("832×1216", "2:3", "1k"),
        ("1664 X 2432", "2:3", "2k"),
        ("1536x1536", "1:1", "2k"),
        ("19.5:9", "19.5:9", "1k"),
        ("9：16", "9:16", "1k"),
        (" 2/3 ", "2:3", "1k"),
        ("5:2", "5:2", "1k"),
        ("13:19", "2:3", "1k"),
        ("auto", None, "1k"),
        ("AUTO", None, "1k"),
        ("garbage", "2:3", "1k"),
        ("0:5", "2:3", "1k"),
        ("", "2:3", "1k"),
        (None, "2:3", "1k"),
        (123, "2:3", "1k"),
    ],
)
def test_size_snapping(size, ratio, resolution):
    rec = Recorder(httpx.Response(200, json=ok_json()))
    result = run(backend(portrait_size=size), rec)
    body = rec.body()
    assert body.get("aspect_ratio") == ratio
    assert body["resolution"] == resolution
    assert result.params["aspect_ratio"] == (ratio or "auto")
    assert result.params["size"] == f"{ratio or 'auto'} {resolution}"


def test_huge_numbers_in_size_fall_back_to_default():
    # 400 位数字转成 float 是 inf，之前会在计算像素数时抛出 OverflowError
    for size in ("1" * 400 + "x3", "2:" + "9" * 400, "1" * 400 + "x" + "1" * 400):
        body = backend(portrait_size=size).build_body(DrawRequest(prompt="p"))
        assert body["aspect_ratio"] == "2:3" and body["resolution"] == "1k"


# 研究结论中提到的模型 → 预期的参数规则
MODEL_FAMILIES = {
    "grok-imagine-image-2.0": "v2",
    "grok-imagine-image-2.0-latest": "v2",
    "relay/grok-imagine-image-2.0": "v2",
    "GROK-IMAGINE-IMAGE-2.0": "v2",
    "grok-imagine-image": "v1",
    "grok-imagine-image-latest": "v1",
    "grok-imagine-image-2026-03-02": "v1",
    "grok-imagine-image-quality": "v1",
    "grok-imagine-image-quality-20260403": "v1",
    "grok-imagine-image-pro": "v1",
    "grok-2-image-1212": "legacy",
    "grok-2-image": "legacy",
    "grok-2-image-latest": "legacy",
    "": "v2",  # 留空使用默认的 2.0
}


def test_every_built_request_is_valid_for_its_model():
    """各模型 × 各方向 × 各种奇怪的配置值，生成的请求体都只含该模型接受的字段和取值。"""
    sizes = ("", None, "2K", "2048 x 2048", "abc", "auto", "21:9", "5:2", "4:5", "1:3", "0x0", "9:19.5", "13:19",
             "1920x1088", "-2:3", 3.5)
    for model, family in MODEL_FAMILIES.items():
        allowed = RATIOS if family == "v2" else V1_RATIOS
        for size in sizes:
            for resolution in ("", "2K", "1.5k", "auto", "1k", None):
                for quality in ("", "high", "LOW", "auto", None):
                    xai = backend(model=model, portrait_size=size, landscape_size=size, square_size=size,
                                  resolution=resolution, quality=quality)
                    for orientation in ("portrait", "landscape", "square"):
                        body = xai.build_body(DrawRequest(prompt="p", negative="n", orientation=orientation))
                        assert body["n"] == 1 and body["response_format"] == "b64_json"
                        assert body["model"] == (model or DEFAULT_MODEL)
                        if family == "legacy":
                            assert set(body) == {"model", "prompt", "n", "response_format"}
                            continue
                        assert set(body) <= ALLOWED_KEYS
                        assert body.get("aspect_ratio", "2:3") in allowed, (model, size)
                        assert body["resolution"] in RESOLUTIONS
                        assert "quality" not in body or (family == "v2" and body["quality"] in ("low", "medium"))


def test_v1_snaps_wide_ratios_and_all_snapped_values_are_allowed():
    xai = backend(model="grok-imagine-image")
    for text in ("5:2", "21:9", "3:1", "1:5", "7:3", "10:7", "1920x1088", "1000x700"):
        xai.section["portrait_size"] = text
        ratio, _ = xai.aspect("portrait", "v1")
        assert ratio in V1_RATIOS
        ratio, _ = xai.aspect("portrait", "v2")
        assert ratio in RATIOS


@pytest.mark.parametrize(
    ("value", "size", "expected"),
    [("2K", "2:3", "2k"), ("1k", "1664x2432", "1k"), ("2k", "832x1216", "2k"), ("1.5k", "2:3", "1k"),
     ("4k", "1664x2432", "2k"), (None, "2:3", "1k"), (7, "2:3", "1k")],
)
def test_resolution_setting(value, size, expected):
    rec = Recorder(httpx.Response(200, json=ok_json()))
    run(backend(resolution=value, portrait_size=size), rec)
    assert rec.body()["resolution"] == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("low", "low"), ("MEDIUM", "medium"), ("", "medium"), (None, "medium"), ("auto", None), ("high", None),
     (["x"], None)],
)
def test_quality_setting(value, expected):
    rec = Recorder(httpx.Response(200, json=ok_json()))
    result = run(backend(quality=value), rec)
    assert rec.body().get("quality") == expected
    assert result.params.get("quality") == expected


def test_quality_never_sent_to_v1():
    rec = Recorder(httpx.Response(200, json=ok_json()))
    run(backend(model="grok-imagine-image", quality="low"), rec)
    assert "quality" not in rec.body()


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("", GEN_URL),
        (None, GEN_URL),
        ("api.x.ai", GEN_URL),
        ("https://api.x.ai/", GEN_URL),
        ("https://api.x.ai/v1/", GEN_URL),
        ("https://api.x.ai/v1/images/generations", GEN_URL),
        ("https://us.api.x.ai/v1", GEN_URL),
        ("https://relay.example.com/openai/v1/", "https://relay.example.com/openai/v1/images/generations"),
        ("relay.example.com/v1/images/generations", "https://relay.example.com/v1/images/generations"),
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080/v1/images/generations"),
        # 误粘贴的其他接口地址
        ("https://api.x.ai/v1/chat/completions", GEN_URL),
        ("https://api.x.ai/v1/responses/", GEN_URL),
        ("https://api.x.ai/v1/images", GEN_URL),
        ("https://api.x.ai/v1/images/edits", GEN_URL),
        ("https://api.x.ai/v1/models", GEN_URL),
        ("https://api.x.ai/v1/image-generation-models", GEN_URL),
        ("https://api.x.ai/v1/messages", GEN_URL),
        ("https://eu-west-1.api.x.ai/v1/chat/completions", GEN_URL),
    ],
)
def test_base_url_normalisation(base_url, expected):
    rec = Recorder(httpx.Response(200, json=ok_json()))
    run(backend(base_url=base_url), rec)
    assert str(rec.requests[0].url) == expected


@pytest.mark.parametrize("base_url", ["http://[bad", "https://api.x.ai:abc/v1", "https://api.x.ai:99999/v1",
                                      "ftp://user:hunter2@api.x.ai/v1", "file:///etc/passwd"])
def test_bad_base_url_reports_error(base_url):
    xai = backend(base_url=base_url)
    assert "地址" in xai.missing() and "hunter2" not in xai.missing()
    rec = Recorder(httpx.Response(200, json=ok_json()))
    for method in ("generate", "list_models"):
        with pytest.raises(DrawError) as info:
            run(xai, rec, method=method)
        assert "地址" in str(info.value) and "hunter2" not in str(info.value)
    assert not rec.requests


def test_bearer_prefix_in_key_is_stripped():
    rec = Recorder(httpx.Response(200, json=ok_json()))
    run(XAIBackend({"api_key": " Bearer xai-abc "}), rec)
    assert rec.requests[0].headers["authorization"] == "Bearer xai-abc"


def test_whitespace_inside_key_removed():
    # Key 中间的换行会让 h11 报 Illegal header value，并把整个 Key 带进报错信息
    rec = Recorder(httpx.Response(200, json=ok_json()))
    xai = XAIBackend({"api_key": "xai-ab\ncd\t ef\u3000"})
    assert xai.missing() == ""
    run(xai, rec)
    assert rec.requests[0].headers["authorization"] == "Bearer xai-abcdef"


@pytest.mark.parametrize("key", ["xai-secret\u200b", "xai-\uff53ecret", "xai-secret\x00"])
def test_key_with_invisible_or_fullwidth_chars(key):
    # 非 ASCII 字符会在 httpx 编码请求头时抛出 UnicodeEncodeError（不是 DrawError）
    xai = XAIBackend({"api_key": key})
    assert "API Key" in xai.missing() and "secret" not in xai.missing()
    rec = Recorder(httpx.Response(200, json=ok_json()))
    for method in ("generate", "list_models"):
        with pytest.raises(DrawError) as info:
            run(xai, rec, method=method)
        assert "secret" not in str(info.value)
    assert not rec.requests


# ---------------------------------------------------------------- extra_body


def test_extra_body_merged_last():
    rec = Recorder(httpx.Response(200, json=ok_json()))
    extra = '{"quality": "low", "user": "bot", "aspect_ratio": "21:9", "resolution": "2k"}'
    result = run(backend(extra_body=extra), rec)
    body = rec.body()
    assert body["quality"] == "low" and body["user"] == "bot"
    assert body["aspect_ratio"] == "21:9" and body["resolution"] == "2k"
    assert result.params["size"] == "21:9 2k" and result.params["quality"] == "low"


def test_extra_body_dict_and_empty():
    rec = Recorder(httpx.Response(200, json=ok_json()))
    run(backend(extra_body={"user": "u1"}), rec)
    run(backend(extra_body="  "), rec)
    run(backend(extra_body=None), rec)
    assert rec.body(0)["user"] == "u1" and "user" not in rec.body(1) and "user" not in rec.body(2)


@pytest.mark.parametrize("extra", ["{bad json", "[1, 2]"])
def test_extra_body_invalid(extra):
    rec = Recorder(httpx.Response(200, json=ok_json()))
    with pytest.raises(DrawError) as info:
        run(backend(extra_body=extra), rec)
    assert "额外请求参数" in str(info.value) and not rec.requests


# ---------------------------------------------------------------- 响应解析


def test_b64_with_data_uri_prefix_and_png():
    data = {"data": [{"b64_json": "data:image/png;base64," + base64.b64encode(PNG).decode()}]}
    result = run(backend(), Recorder(httpx.Response(200, json=data)))
    assert result.image == PNG and result.mime == "image/png"
    assert "cost_usd" not in result.params and result.text == ""


def test_revised_prompt_kept_as_text():
    result = run(backend(), Recorder(httpx.Response(200, json=ok_json(revised_prompt="a fluffy cat"))))
    assert result.text == "a fluffy cat"


def test_url_response_downloaded_without_auth():
    image_url = "https://imgen.x.ai/xai-imgen/abc.jpg"

    def generate(request):
        return httpx.Response(200, json={"data": [{"url": image_url, "b64_json": None}]})

    rec = Recorder(generate, httpx.Response(200, content=JPEG))
    result = run(backend(extra_body='{"response_format": "url"}'), rec)
    assert rec.body()["response_format"] == "url"
    assert str(rec.requests[1].url) == image_url and rec.requests[1].method == "GET"
    assert "authorization" not in rec.requests[1].headers
    assert result.image == JPEG and result.mime == "image/jpeg"


@pytest.mark.parametrize(
    "item",
    [{"b64_json": base64.b64encode(JPEG).decode(), "respect_moderation": False}, {"b64_json": "", "url": None}],
)
def test_moderated_image(item):
    rec = Recorder(httpx.Response(200, json={"data": [item]}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec)
    assert "审核" in str(info.value) and not info.value.retryable and len(rec.requests) == 1


@pytest.mark.parametrize(
    ("data", "words"),
    [({"data": []}, "没有返回图片"), ({"error": "boom"}, "boom"), ({"data": [{"b64_json": "aGVsbG8="}]}, "不是可识别的图片")],
)
def test_bad_success_bodies(data, words):
    with pytest.raises(DrawError) as info:
        run(backend(), Recorder(httpx.Response(200, json=data)))
    assert words in str(info.value)


def test_non_json_success_body():
    with pytest.raises(DrawError) as info:
        run(backend(), Recorder(httpx.Response(200, text="<html><title>Bad gateway</title></html>")))
    assert "不是 JSON" in str(info.value)


# ---------------------------------------------------------------- 错误与重试


def test_moderation_refusal_status(sleeps):
    error = "Generated image rejected by content moderation."
    rec = Recorder(httpx.Response(400, json={"code": "400", "error": error}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec)
    assert "安全审核" in str(info.value) and "content moderation" in str(info.value)
    assert info.value.status == 400 and len(rec.requests) == 1 and not sleeps


def test_401_not_retried(sleeps):
    rec = Recorder(httpx.Response(401, json={"code": "unauthenticated:no-credentials",
                                             "error": "No credentials presented."}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec)
    message = str(info.value)
    assert "API Key" in message and "No credentials presented." in message and "HTTP 401" in message
    assert info.value.status == 401 and len(rec.requests) == 1 and not sleeps


def test_422_plain_text_shown(sleeps):
    text = "Failed to deserialize the JSON body into the target type: aspect_ratio: unknown variant `13:19`"
    rec = Recorder(httpx.Response(422, text=text, headers={"content-type": "text/plain"}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec)
    assert "HTTP 422" in str(info.value) and "unknown variant `13:19`" in str(info.value)
    assert len(rec.requests) == 1


def test_404_nested_integer_code():
    error = {"code": 404, "message": "The requested resource was not found."}
    rec = Recorder(httpx.Response(404, json={"error": error}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec)
    assert "HTTP 404" in str(info.value) and "not found" in str(info.value)


def test_prompt_too_long_hint():
    rec = Recorder(httpx.Response(400, json={"code": "400", "error": "Prompt length exceeds max_prompt_length"}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec)
    assert "提示词过长" in str(info.value)


@pytest.mark.parametrize("status", [402, 403, 429])
def test_billing_exhausted_not_retried(status, sleeps):
    error = ("Your team abc has either used all available credits or reached its monthly spending limit. "
             "To continue making API requests, please purchase more credits or raise your spending limit.")
    rec = Recorder(httpx.Response(status, json={"code": "Some resource has been exhausted", "error": error}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec)
    assert "额度" in str(info.value) and not info.value.retryable
    assert len(rec.requests) == 1 and not sleeps


def test_rate_limit_429_retried_once(sleeps):
    rec = Recorder(httpx.Response(429, json={"code": "429", "error": "Too many requests"},
                                  headers={"retry-after": "2"}),
                   httpx.Response(200, json=ok_json()))
    result = run(backend(), rec)
    assert result.image == JPEG and len(rec.requests) == 2 and sleeps == [2.0]


def test_rate_limit_429_gives_up_after_second_attempt(sleeps):
    rec = Recorder(httpx.Response(429, json={"error": "rate limited"}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec)
    assert info.value.status == 429 and len(rec.requests) == 2 and sleeps == [3.0]


@pytest.mark.parametrize("status", [500, 502, 504])
def test_5xx_not_retried(status, sleeps):
    # 500 / 502 / 504 时服务端可能已经生成并扣费
    rec = Recorder(httpx.Response(status, json={"error": "upstream"}, headers={"retry-after": "1"}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec)
    assert info.value.status == status and not info.value.retryable and info.value.retry_after is None
    assert len(rec.requests) == 1 and not sleeps


def test_503_retried_once(sleeps):
    rec = Recorder(httpx.Response(503, text="Service Unavailable"), httpx.Response(200, json=ok_json()))
    result = run(backend(), rec)
    assert result.image == JPEG and len(rec.requests) == 2 and sleeps == [3.0]


def test_error_messages_are_clean():
    """提示语放在 HTTP 状态码之前，不出现「.。」这样的标点。"""
    rec = Recorder(httpx.Response(401, json={"code": "unauthenticated", "error": "Incorrect API key provided."}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec)
    assert str(info.value) == ("xAI API Key 无效或已过期，请到 https://console.x.ai 检查（HTTP 401）："
                               "Incorrect API key provided.")
    rec = Recorder(httpx.Response(400, json={"code": "400", "error": "Prompt length exceeds the limit."}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec)
    assert str(info.value) == "xAI 提示词过长，请缩短描述（HTTP 400）：Prompt length exceeds the limit."
    assert not info.value.retryable


def test_read_timeout_not_retried(sleeps):
    rec = Recorder(httpx.ReadTimeout("slow"))
    with pytest.raises(DrawError) as info:
        run(backend(), rec)
    assert "超时" in str(info.value) and len(rec.requests) == 1 and not sleeps


def test_connect_error_retried_once(sleeps):
    rec = Recorder(httpx.ConnectError("refused"), httpx.Response(200, json=ok_json()))
    result = run(backend(), rec)
    assert result.image == JPEG and len(rec.requests) == 2 and len(sleeps) == 1


# ---------------------------------------------------------------- missing / summary / 配置健壮性


@pytest.mark.parametrize("section", [None, {}, {"api_key": ""}, {"api_key": "   "}, {"api_key": None}])
def test_missing_key(section):
    xai = XAIBackend(section)
    assert xai.missing() == "管理员还没有配置 xAI API Key"
    rec = Recorder(httpx.Response(200, json=ok_json()))
    with pytest.raises(DrawError) as info:
        run(xai, rec)
    assert "API Key" in str(info.value) and not rec.requests


def test_missing_ok():
    assert backend().missing() == ""


def test_summary():
    assert backend().summary() == "xAI / grok-imagine-image-2.0 / 竖图 2:3 1k / 画质 medium"
    assert backend(model="grok-imagine-image", portrait_size="1664x2432").summary() == \
        "xAI / grok-imagine-image / 竖图 2:3 2k"
    assert backend(portrait_size="auto", quality="auto").summary() == "xAI / grok-imagine-image-2.0 / 竖图 auto 1k"
    assert backend(model="grok-2-image-1212").summary() == "xAI / grok-2-image-1212 / 竖图 auto"
    assert XAIBackend(None).summary().startswith("xAI / grok-imagine-image-2.0")


def test_garbage_config_still_builds_valid_request():
    section = {"api_key": "xai-test", "base_url": None, "proxy_url": None, "model": None, "portrait_size": None,
               "landscape_size": 42, "square_size": "x", "resolution": [], "quality": {}, "timeout": "abc",
               "extra_body": None, "prompt_mode": 3}
    xai = XAIBackend(section)
    assert xai.timeout == 180.0 and xai.prompt_mode == "natural" and xai.proxy == ""
    for orientation in ("portrait", "landscape", "square", None):
        rec = Recorder(httpx.Response(200, json=ok_json()))
        run(xai, rec, DrawRequest(prompt="p", orientation=orientation))
        body = rec.body()
        assert str(rec.requests[0].url) == GEN_URL
        assert body["model"] == DEFAULT_MODEL and body["aspect_ratio"] in RATIOS and body["resolution"] == "1k"
        assert set(body) <= ALLOWED_KEYS and "quality" not in body


# ---------------------------------------------------------------- 模型列表


def test_list_models():
    payload = {"models": [{"id": "grok-imagine-image-2.0", "aliases": ["grok-imagine-image-2.0-latest"]},
                          {"id": "grok-imagine-image", "aliases": ["grok-imagine-image-2026-03-02"]},
                          {"id": "grok-imagine-image"}, {"no_id": True}, "junk"]}
    rec = Recorder(httpx.Response(200, json=payload))
    models = run(backend(base_url="https://api.x.ai"), rec, method="list_models")
    assert models == ["grok-imagine-image-2.0", "grok-imagine-image"]
    assert rec.requests[0].method == "GET"
    assert str(rec.requests[0].url) == "https://api.x.ai/v1/image-generation-models"
    assert rec.requests[0].headers["authorization"] == "Bearer xai-test"


def test_list_models_falls_back_to_models_endpoint():
    payload = {"object": "list", "data": [{"id": "grok-4.7"}, {"id": "grok-imagine-image-2.0"},
                                          {"id": "grok-imagine-video"}, {"id": "grok-imagine-image"}]}
    rec = Recorder(httpx.Response(404, json={"error": {"code": 404, "message": "not found"}}),
                   httpx.Response(200, json=payload))
    models = run(backend(base_url="https://relay.example.com/v1"), rec, method="list_models")
    assert models == ["grok-imagine-image-2.0", "grok-imagine-image"]
    assert [str(r.url) for r in rec.requests] == ["https://relay.example.com/v1/image-generation-models",
                                                   "https://relay.example.com/v1/models"]


def test_list_models_401_no_fallback():
    rec = Recorder(httpx.Response(401, json={"code": "unauthenticated", "error": "Incorrect API key provided"}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, method="list_models")
    assert info.value.status == 401 and len(rec.requests) == 1


def test_list_models_requires_key():
    rec = Recorder(httpx.Response(200, json={"models": []}))
    with pytest.raises(DrawError):
        run(XAIBackend({}), rec, method="list_models")
    assert not rec.requests


# ---------------------------------------------------------------- 配置段落


def load_schema_section() -> dict:
    return json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))["xai"]


def test_schema_section_shape_and_defaults_match_code():
    section = load_schema_section()
    items = section["items"]
    assert section["type"] == "object" and section["description"].endswith("设置")
    assert list(items) == ["api_key", "base_url", "proxy_url", "model", "portrait_size", "landscape_size",
                           "square_size", "resolution", "quality", "timeout", "extra_body"]
    for name, field in items.items():
        assert field["type"] in ("string", "text", "int", "float", "bool", "list"), name
        assert field.get("description") and field.get("hint"), name
        if "options" in field:
            assert field["default"] in field["options"] and len(field.get("labels", [])) == len(field["options"])
    assert items["api_key"]["secret"] is True and items["api_key"]["default"] == ""
    assert "options" not in items["model"]
    assert items["base_url"]["default"] == DEFAULT_BASE_URL
    assert items["model"]["default"] == DEFAULT_MODEL
    assert {o: items[f"{o}_size"]["default"] for o in DEFAULT_SIZES} == DEFAULT_SIZES
    assert items["resolution"]["default"] == DEFAULT_RESOLUTION
    assert set(items["resolution"]["options"]) == {"auto", *RESOLUTIONS}
    assert items["quality"]["default"] == DEFAULT_QUALITY
    assert items["timeout"]["default"] == int(XAIBackend({}).timeout)
    extra = items["extra_body"]
    assert extra["type"] == "text" and extra["default"] == "" and extra["editor_mode"] is True
    assert extra["editor_language"] == "json"


def test_schema_defaults_produce_valid_requests():
    items = load_schema_section()["items"]
    section = {name: field["default"] for name, field in items.items()}
    section["api_key"] = "xai-test"
    xai = XAIBackend(section)
    assert xai.missing() == ""
    expected = {"portrait": "2:3", "landscape": "3:2", "square": "1:1"}
    for orientation, ratio in expected.items():
        rec = Recorder(httpx.Response(200, json=ok_json()))
        result = run(xai, rec, DrawRequest(prompt="p", orientation=orientation))
        assert str(rec.requests[0].url) == GEN_URL
        assert rec.body() == {"model": DEFAULT_MODEL, "prompt": "p", "n": 1, "response_format": "b64_json",
                              "aspect_ratio": ratio, "resolution": "1k", "quality": "medium"}
        assert result.params["size"] == f"{ratio} 1k"
    # 每个可选的分辨率 / 画质都能生成合法请求
    for resolution in items["resolution"]["options"]:
        for quality in items["quality"]["options"]:
            body = XAIBackend({**section, "resolution": resolution, "quality": quality}).build_body(
                DrawRequest(prompt="p"))
            assert body["resolution"] in RESOLUTIONS and body.get("quality", "low") in ("low", "medium")
