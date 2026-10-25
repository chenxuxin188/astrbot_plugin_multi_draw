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
from backends import xai as xai_module  # noqa: E402
from backends.base import BalanceInfo, Clients, DrawError, DrawRequest, HttpPool, InputImage  # noqa: E402
from backends.xai import (  # noqa: E402
    BILLING_URL,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_QUALITY,
    DEFAULT_RESOLUTION,
    DEFAULT_SIZES,
    MANAGEMENT_BASE_URL,
    RATIOS,
    RESOLUTIONS,
    V1_RATIOS,
    XAIBackend,
    tag_image_refs,
)


JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 32
GIF = b"GIF89a" + b"\x00" * 32
GEN_URL = "https://api.x.ai/v1/images/generations"
EDIT_URL = "https://api.x.ai/v1/images/edits"
ALLOWED_KEYS = {"model", "prompt", "n", "response_format", "aspect_ratio", "resolution", "quality"}
EDIT_KEYS = {"model", "prompt", "n", "response_format", "aspect_ratio", "resolution", "quality", "image", "images"}


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
            if method == "balance":
                return await backend.balance(clients)
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
    # 类上读取时是所有模型中的最大值（2.0 的 5 张），按约定是 int
    assert XAIBackend.max_edit_images == 5 and isinstance(XAIBackend.max_edit_images, int)
    assert XAIBackend.supports_strength is False and XAIBackend.supports_balance is True
    assert XAIBackend.billing_url == BILLING_URL == "https://console.x.ai/team/default/billing"


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
                           "square_size", "resolution", "quality", "timeout", "extra_body", "management_key", "team_id"]
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
    # 余额查询用的凭据：可选，密钥类字段标记为 secret，提示只用于查询余额
    management = items["management_key"]
    assert management["type"] == "string" and management["default"] == "" and management["secret"] is True
    assert "余额" in management["description"] and "仅" in management["description"]
    assert "Management Keys" in management["hint"] and "只" in management["hint"]
    assert items["team_id"]["type"] == "string" and items["team_id"]["default"] == ""
    assert "余额" in items["team_id"]["description"]
    empty = XAIBackend({})
    assert empty.management_key == management["default"] and empty.team_id == items["team_id"]["default"]
    # 模型说明写明哪些模型能修改图片
    hint = items["model"]["hint"]
    assert "修改图片" in hint and "5 张" in hint and "3 张" in hint and "grok-2-image 不支持" in hint


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
            edit = XAIBackend({**section, "resolution": resolution, "quality": quality}).build_edit_body(
                DrawRequest(prompt="p", images=[image()]))
            assert set(edit) <= EDIT_KEYS and edit["resolution"] in RESOLUTIONS and "aspect_ratio" not in edit
    # 默认配置可以修改图片；没有填写管理密钥时不能查询余额
    assert xai.edit_missing() == "" and xai.max_edit_images == 5
    assert "管理密钥" in xai.balance_missing()


# ---------------------------------------------------------------- 修改图片


def image(data: bytes = JPEG, width: int = 1024, height: int = 1536) -> InputImage:
    mime = base.detect_mime(data) or "application/octet-stream"
    return InputImage(data=data, mime=mime, width=width, height=height)


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def edit_req(*images: InputImage, prompt: str = "make it night", **kwargs) -> DrawRequest:
    images = images or (image(),)
    kwargs.setdefault("orientation", base.image_orientation(images[0]) or "portrait")
    return DrawRequest(prompt=prompt, images=list(images), **kwargs)


def test_edit_single_image_request_shape():
    rec = Recorder(httpx.Response(200, json=ok_json()))
    result = run(backend(), rec, edit_req())
    request = rec.requests[0]
    assert request.method == "POST" and str(request.url) == EDIT_URL
    assert request.headers["authorization"] == "Bearer xai-test"
    assert request.headers["content-type"] == "application/json"
    # 一张图片放在 image（对象，url 为 data URL），不发送 images、size，也不发送宽高比（沿用原图比例）
    assert rec.body() == {"model": "grok-imagine-image-2.0", "prompt": "make it night", "n": 1,
                          "response_format": "b64_json", "resolution": "1k", "quality": "medium",
                          "image": {"url": data_url(JPEG, "image/jpeg")}}
    assert result.image == JPEG and result.mime == "image/jpeg"
    assert result.params == {"model": "grok-imagine-image-2.0", "size": "原图比例 1k", "aspect_ratio": "input",
                             "resolution": "1k", "quality": "medium", "cost_usd": 0.04, "mode": "edit",
                             "input_images": 1}


def test_edit_multiple_images_use_images_array_and_image_tokens():
    rec = Recorder(httpx.Response(200, json=ok_json()))
    images = (image(JPEG), image(PNG, 800, 800), image(WEBP, 640, 480))
    prompt = "Put the cat from image 2 and the hat from Image #3 into image 1; keep image 1's lighting. image 9"
    result = run(backend(), rec, edit_req(*images, prompt=prompt))
    body = rec.body()
    assert "image" not in body and body["images"] == [{"url": data_url(JPEG, "image/jpeg")},
                                                      {"url": data_url(PNG, "image/png")},
                                                      {"url": data_url(WEBP, "image/webp")}]
    assert body["prompt"] == ("Put the cat from <IMAGE_1> and the hat from <IMAGE_2> into <IMAGE_0>; "
                              "keep <IMAGE_0>'s lighting. image 9")
    assert set(body) <= EDIT_KEYS and "aspect_ratio" not in body
    assert result.params["input_images"] == 3 and result.params["mode"] == "edit"


def test_tag_image_refs():
    assert tag_image_refs("change image 1", 1) == "change image 1"  # 单图不改写
    assert tag_image_refs("IMAGE 2 onto image1", 2) == "<IMAGE_1> onto <IMAGE_0>"
    assert tag_image_refs("image 0, image 3, images 1", 2) == "image 0, image 3, images 1"


def test_edit_negative_folded_into_prompt():
    rec = Recorder(httpx.Response(200, json=ok_json()))
    run(backend(), rec, edit_req(prompt="make it night", negative="text", seed=7))
    body = rec.body()
    assert body["prompt"] == "make it night\n\nAvoid: text"
    for key in ("negative_prompt", "seed", "size", "strength", "mask", "input_fidelity"):
        assert key not in body


@pytest.mark.parametrize("strength", [0.0, 0.05, 0.6, 0.99, 1.0, -1.0, 7.5])
def test_edit_strength_ignored(strength):
    # xAI 没有重绘幅度参数：无论 req.strength 是多少，请求体都一样，也不写入记录
    reference = backend().build_edit_body(edit_req())
    rec = Recorder(httpx.Response(200, json=ok_json()))
    result = run(backend(), rec, edit_req(strength=strength))
    assert rec.body() == reference
    assert "strength" not in result.params


@pytest.mark.parametrize(
    ("model", "limit", "quality"),
    [("grok-imagine-image-2.0", 5, True), ("grok-imagine-image-2.0-latest", 5, True),
     ("grok-imagine-image", 3, False), ("grok-imagine-image-2026-03-02", 3, False),
     ("grok-imagine-image-quality", 3, False), ("grok-imagine-image-pro", 3, False)],
)
def test_edit_per_model_limits(model, limit, quality):
    xai = backend(model=model)
    assert xai.max_edit_images == limit and xai.edit_missing() == ""
    rec = Recorder(httpx.Response(200, json=ok_json()))
    result = run(xai, rec, edit_req(*[image() for _ in range(limit)]))
    body = rec.body()
    assert len(body["images"]) == limit and body["model"] == model
    assert ("quality" in body) is quality
    assert result.params["input_images"] == limit
    # 超过上限时给出明确的中文说明，不发送请求
    rec = Recorder(httpx.Response(200, json=ok_json()))
    with pytest.raises(DrawError) as info:
        run(xai, rec, edit_req(*[image() for _ in range(limit + 1)]))
    assert f"最多使用 {limit} 张图片" in str(info.value) and not rec.requests


@pytest.mark.parametrize("model", ["grok-2-image", "grok-2-image-1212", "grok-2-image-latest"])
def test_legacy_model_cannot_edit(model):
    xai = backend(model=model)
    assert xai.max_edit_images == 0
    assert xai.edit_missing() == f"当前模型 {model} 不支持修改图片，请改用 grok-imagine-image-2.0 或 grok-imagine-image"
    rec = Recorder(httpx.Response(200, json=ok_json()))
    with pytest.raises(DrawError) as info:
        run(xai, rec, edit_req())
    assert "不支持修改图片" in str(info.value) and not rec.requests


@pytest.mark.parametrize(
    ("size", "orientation", "section", "ratio"),
    [
        # 方向与原图相同：不发送宽高比，输出沿用原图比例
        ((1024, 1536), "portrait", {}, None),
        ((1536, 1024), "landscape", {}, None),
        ((1024, 1024), "square", {}, None),
        # 要求改变方向：发送配置中该方向的比例（换成接口接受的值，不发送 auto）
        ((1024, 1536), "landscape", {}, "3:2"),
        ((1536, 1024), "portrait", {"portrait_size": "832x1216"}, "2:3"),
        ((1024, 1536), "square", {}, "1:1"),
        ((1024, 1536), "landscape", {"landscape_size": "21:9"}, "21:9"),
        ((1024, 1536), "landscape", {"landscape_size": "21:9", "model": "grok-imagine-image"}, "20:9"),
        ((1024, 1536), "landscape", {"landscape_size": "auto"}, None),
        # 方向无效或不知道原图尺寸时保持原图比例
        ((1024, 1536), "diagonal", {}, None),
        ((0, 0), "landscape", {}, None),
    ],
)
def test_edit_aspect_ratio(size, orientation, section, ratio):
    rec = Recorder(httpx.Response(200, json=ok_json()))
    result = run(backend(**section), rec, edit_req(image(JPEG, *size), orientation=orientation))
    body = rec.body()
    assert body.get("aspect_ratio") == ratio
    assert body["aspect_ratio"] != "auto" if "aspect_ratio" in body else True
    if ratio:
        assert result.params["size"] == f"{ratio} 1k" and result.params["aspect_ratio"] == ratio
    else:
        assert result.params["size"] == "原图比例 1k" and result.params["aspect_ratio"] == "input"


@pytest.mark.parametrize(
    ("setting", "size", "expected"),
    [("auto", (1024, 1536), "1k"), ("auto", (1448, 1448), "2k"), ("auto", (2048, 1536), "2k"),
     ("auto", (0, 0), "1k"), ("1k", (2048, 1536), "1k"), ("2K", (512, 512), "2k"), ("1.5k", (2048, 2048), "2k")],
)
def test_edit_resolution_follows_input(setting, size, expected):
    xai = backend(resolution=setting, portrait_size="1664x2432")
    body = xai.build_edit_body(edit_req(image(JPEG, *size)))
    assert body["resolution"] == expected
    assert xai.plan(edit_req(image(JPEG, *size))) == {"model": DEFAULT_MODEL, "size": f"原图比例 {expected}"}


def test_edit_plan_with_orientation_change_and_text_plan_unchanged():
    xai = backend()
    assert xai.plan(edit_req(image(JPEG, 1024, 1536), orientation="landscape")) == {
        "model": DEFAULT_MODEL, "size": "3:2 1k"}
    assert xai.plan(DrawRequest(prompt="p", orientation="landscape")) == {"model": DEFAULT_MODEL, "size": "3:2 1k"}


@pytest.mark.parametrize(("quality", "expected"), [("low", "low"), ("medium", "medium"), ("auto", None),
                                                    ("bogus", None)])
def test_edit_quality(quality, expected):
    body = backend(quality=quality).build_edit_body(edit_req())
    assert body.get("quality") == expected
    assert "quality" not in backend(model="grok-imagine-image", quality="low").build_edit_body(edit_req())


def test_edit_extra_body_cannot_replace_images_or_send_size():
    extra = json.dumps({"user": "bot", "size": "1024x1024", "image": {"url": "https://evil.example/x.png"},
                        "images": [], "quality": "low"})
    rec = Recorder(httpx.Response(200, json=ok_json()))
    result = run(backend(extra_body=extra), rec, edit_req())
    body = rec.body()
    assert body["user"] == "bot" and body["quality"] == "low" and "size" not in body and "images" not in body
    assert body["image"] == {"url": data_url(JPEG, "image/jpeg")}
    assert result.params["quality"] == "low"


@pytest.mark.parametrize(("data", "kind"), [(GIF, "GIF 格式"), (b"BM" + b"\x00" * 32, "BMP 格式"),
                                            (b"not an image", "无法识别的格式")])
def test_edit_rejects_unsupported_formats(data, kind):
    rec = Recorder(httpx.Response(200, json=ok_json()))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, edit_req(image(JPEG), image(data)))
    assert str(info.value) == f"xAI 修改图片只支持 JPEG、PNG、WebP 格式，第 2 张图片是 {kind}"
    assert not rec.requests


def test_edit_mime_sniffed_not_trusted():
    # 调用方给错 MIME 时按文件头填写 data URL
    body = backend().build_edit_body(edit_req(InputImage(data=PNG, mime="image/jpeg", width=10, height=10)))
    assert body["image"]["url"].startswith("data:image/png;base64,")


def test_edit_total_size_guard(monkeypatch):
    monkeypatch.setattr(xai_module, "MAX_EDIT_BODY", 100)
    rec = Recorder(httpx.Response(200, json=ok_json()))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, edit_req(image(JPEG + b"\x00" * 200)))
    assert "总大小过大" in str(info.value) and not rec.requests


def test_edit_requires_key_and_valid_extra_body():
    rec = Recorder(httpx.Response(200, json=ok_json()))
    with pytest.raises(DrawError) as info:
        run(XAIBackend({}), rec, edit_req())
    assert "API Key" in str(info.value)
    with pytest.raises(DrawError) as info:
        run(backend(extra_body="{bad"), rec, edit_req())
    assert "额外请求参数" in str(info.value) and not rec.requests


def test_edit_url_response_downloaded_and_relay_base_url():
    image_url = "https://imgen.x.ai/xai-imgen/edit.jpg"
    rec = Recorder(httpx.Response(200, json={"data": [{"url": image_url}]}), httpx.Response(200, content=JPEG))
    result = run(backend(base_url="https://relay.example.com/v1/images/generations"), rec, edit_req())
    assert str(rec.requests[0].url) == "https://relay.example.com/v1/images/edits"
    assert str(rec.requests[1].url) == image_url and "authorization" not in rec.requests[1].headers
    assert result.image == JPEG and "cost_usd" not in result.params


@pytest.mark.parametrize(
    "item",
    [{"b64_json": base64.b64encode(JPEG).decode(), "respect_moderation": False}, {"b64_json": "", "url": None}],
)
def test_edit_moderated_output(item, sleeps):
    rec = Recorder(httpx.Response(200, json={"data": [item]}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, edit_req())
    assert str(info.value) == "xAI 拒绝修改：修改后的图片未通过内容审核，请修改描述或换一张图片后重试"
    assert not info.value.retryable and len(rec.requests) == 1 and not sleeps


@pytest.mark.parametrize("status", [400, 403, 422])
def test_edit_moderation_refusal_status(status, sleeps):
    rec = Recorder(httpx.Response(status, json={"code": str(status),
                                                "error": "Input image rejected by content moderation."}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, edit_req())
    message = str(info.value)
    assert message.startswith("xAI 拒绝修改：图片或描述未通过安全审核") and "换一张图片" in message
    assert info.value.status == status and len(rec.requests) == 1 and not sleeps


def test_wrong_key_400_mapped_to_key_message(sleeps):
    error = "Incorrect API key provided. You can obtain an API key from https://console.x.ai."
    for req in (None, edit_req()):
        rec = Recorder(httpx.Response(400, json={"code": "invalid-argument", "error": error}))
        with pytest.raises(DrawError) as info:
            run(backend(), rec, req)
        assert str(info.value).startswith("xAI API Key 无效或已过期") and "HTTP 400" in str(info.value)
        assert len(rec.requests) == 1
    assert not sleeps


@pytest.mark.parametrize("status", [402, 403, 429])
def test_edit_billing_exhausted_not_retried(status, sleeps):
    error = "Your team has either used all available credits or reached its monthly spending limit."
    rec = Recorder(httpx.Response(status, json={"code": "exhausted", "error": error}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, edit_req())
    assert "额度" in str(info.value) and not info.value.retryable and len(rec.requests) == 1 and not sleeps


@pytest.mark.parametrize("status", [500, 502, 504])
def test_edit_5xx_not_retried(status, sleeps):
    rec = Recorder(httpx.Response(status, json={"error": "upstream"}))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, edit_req())
    assert info.value.status == status and not info.value.retryable and len(rec.requests) == 1 and not sleeps


def test_edit_read_timeout_not_retried(sleeps):
    rec = Recorder(httpx.ReadTimeout("slow"))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, edit_req())
    assert "超时" in str(info.value) and len(rec.requests) == 1 and not sleeps


def test_edit_429_and_connect_error_retried_once(sleeps):
    rec = Recorder(httpx.Response(429, json={"error": "Too many requests"}, headers={"retry-after": "1"}),
                   httpx.Response(200, json=ok_json()))
    assert run(backend(), rec, edit_req()).image == JPEG and len(rec.requests) == 2 and sleeps == [1.0]
    rec = Recorder(httpx.ConnectError("refused"), httpx.Response(200, json=ok_json()))
    assert run(backend(), rec, edit_req()).image == JPEG and len(rec.requests) == 2
    assert all(str(r.url) == EDIT_URL for r in rec.requests)


def test_413_payload_too_large():
    rec = Recorder(httpx.Response(413, text="Payload Too Large"))
    with pytest.raises(DrawError) as info:
        run(backend(), rec, edit_req())
    assert "HTTP 413" in str(info.value) and "过大" in str(info.value)


# ---------------------------------------------------------------- 余额

TEAM = "8f0c2a4e-1b2c-4d5e-8f90-123456789abc"
OTHER_TEAM = "11111111-2222-4333-8444-555555555555"
MGMT_KEY = "xai-token-management-SECRET-1234"
API_KEY_URL = "https://api.x.ai/v1/api-key"
VALIDATION_URL = f"{MANAGEMENT_BASE_URL}/auth/management-keys/validation"


def prepaid_url(team: str = TEAM) -> str:
    return f"{MANAGEMENT_BASE_URL}/v1/billing/teams/{team}/prepaid/balance"


def preview_url(team: str = TEAM) -> str:
    return f"{MANAGEMENT_BASE_URL}/v1/billing/teams/{team}/postpaid/invoice/preview"


class Router:
    """按 URL 返回预设响应（每个 URL 可以是一个响应、异常或按顺序使用的列表），并记录请求。"""

    def __init__(self, routes: dict):
        self.routes = {url: list(v) if isinstance(v, list) else [v] for url, v in routes.items()}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        assert url in self.routes, f"unexpected request {url}"
        queue = self.routes[url]
        resp = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(resp, Exception):
            raise resp
        return resp

    @property
    def urls(self) -> list[str]:
        return [str(r.url) for r in self.requests]


def balance_backend(**section) -> XAIBackend:
    return XAIBackend({"api_key": "xai-test", "management_key": MGMT_KEY, **section})


PREPAID = {"changes": [{"changeOrigin": "PURCHASE", "amount": {"val": "-2000"}},
                       {"changeOrigin": "SPEND", "amount": {"val": "766"}}],
           "total": {"val": "-1234"}}
PREVIEW = {"coreInvoice": {"amountAfterVat": "550", "prepaidCredits": {"val": "-4500"}},
           "effectiveSpendingLimit": {"val": "0"}, "billingCycle": {"year": 2026, "month": 10}}


def assert_no_secrets(info: BalanceInfo | str):
    text = info if isinstance(info, str) else json.dumps([info.items, info.note], ensure_ascii=False)
    assert MGMT_KEY not in text and "xai-test" not in text


def test_balance_team_from_api_key():
    router = Router({API_KEY_URL: httpx.Response(200, json={"team_id": TEAM, "redacted_api_key": "xai-****",
                                                            "team_blocked": False}),
                     prepaid_url(): httpx.Response(200, json=PREPAID),
                     preview_url(): httpx.Response(200, json=PREVIEW)})
    info = run(balance_backend(), router, method="balance")
    assert router.urls == [API_KEY_URL, prepaid_url(), preview_url()]
    assert all(r.method == "GET" for r in router.requests)
    # /v1/api-key 用 API Key，管理 API 用管理密钥
    assert router.requests[0].headers["authorization"] == "Bearer xai-test"
    assert router.requests[1].headers["authorization"] == f"Bearer {MGMT_KEY}"
    assert router.requests[2].headers["authorization"] == f"Bearer {MGMT_KEY}"
    assert info.items == [("预付余额", "12.34 美元"), ("本月后付费", "5.50 美元"),
                          ("后付费上限", "0（只用预付额度，用完即停）")]
    assert info.note == "账单数据可能有延迟，以控制台为准"
    assert_no_secrets(info)


def test_balance_team_from_config_skips_lookup():
    router = Router({prepaid_url(): httpx.Response(200, json={"total": {"val": "-50000"}}),
                     preview_url(): httpx.Response(200, json={"coreInvoice": {"amountAfterVat": "0"},
                                                              "effectiveSpendingLimit": "250000"})})
    # 团队 ID 不带横线或大写时换成规范写法
    info = run(balance_backend(team_id=TEAM.replace("-", "").upper()), router, method="balance")
    assert router.urls == [prepaid_url(), preview_url()]
    assert info.items == [("预付余额", "500.00 美元"), ("后付费上限", "2,500.00 美元")]


def test_balance_relay_base_url_uses_management_key_team():
    router = Router({VALIDATION_URL: httpx.Response(200, json={"scope": "SCOPE_TEAM", "scopeId": TEAM,
                                                               "teamId": OTHER_TEAM}),
                     prepaid_url(): httpx.Response(200, json=PREPAID),
                     preview_url(): httpx.Response(403, json={"code": 7, "message": "permission denied"})})
    info = run(balance_backend(base_url="https://relay.example.com/v1"), router, method="balance")
    # 中转地址不请求 /v1/api-key；没有后付费账单权限时跳过
    assert router.urls == [VALIDATION_URL, prepaid_url(), preview_url()]
    assert info.items == [("预付余额", "12.34 美元")]
    assert "不是中转服务的余额" in info.note
    assert_no_secrets(info)


@pytest.mark.parametrize("status", [400, 401, 403])
def test_balance_wrong_api_key_falls_back_to_management_team(status, sleeps):
    error = {"code": "invalid-argument", "error": "Incorrect API key provided."}
    router = Router({API_KEY_URL: httpx.Response(status, json=error),
                     VALIDATION_URL: httpx.Response(200, json={"scope": "SCOPE_UNSPECIFIED", "teamId": TEAM}),
                     prepaid_url(): httpx.Response(200, json=PREPAID),
                     preview_url(): httpx.Response(404, json={"code": 5, "message": "not found"})})
    info = run(balance_backend(), router, method="balance")
    assert router.urls == [API_KEY_URL, VALIDATION_URL, prepaid_url(), preview_url()]
    assert info.items == [("预付余额", "12.34 美元")] and not sleeps


@pytest.mark.parametrize(
    "api_key_response",
    [httpx.Response(500, json={"code": "13", "error": "internal"}),
     httpx.Response(503, text="Service Unavailable"),
     httpx.Response(200, text="<html><title>maintenance</title></html>"),
     httpx.ConnectError("refused")],
)
def test_balance_api_key_lookup_failure_falls_back_to_management_team(api_key_response, sleeps):
    # /v1/api-key 只用来获取团队 ID，api.x.ai 暂时出错时不应影响用管理密钥查询余额
    router = Router({API_KEY_URL: api_key_response,
                     VALIDATION_URL: httpx.Response(200, json={"scope": "SCOPE_TEAM", "scopeId": TEAM}),
                     prepaid_url(): httpx.Response(200, json=PREPAID),
                     preview_url(): httpx.Response(200, json={})})
    info = run(balance_backend(), router, method="balance")
    assert router.urls[-3:] == [VALIDATION_URL, prepaid_url(), preview_url()]
    assert set(router.urls[:-3]) == {API_KEY_URL}
    assert info.items == [("预付余额", "12.34 美元")]
    assert_no_secrets(info)


def test_balance_unknown_team_404_message(sleeps):
    router = Router({prepaid_url(): httpx.Response(404, json={"code": 5, "message": "team not found"})})
    with pytest.raises(DrawError) as info:
        run(balance_backend(team_id=TEAM), router, method="balance")
    message = str(info.value)
    # 不能说成「检查 API 地址和模型名」：管理 API 的地址是固定的
    assert "团队" in message and "模型" not in message and "HTTP 404" in message and "team not found" in message
    assert info.value.status == 404 and not sleeps and router.urls == [prepaid_url()]


def test_balance_organization_scope_needs_team_id():
    router = Router({VALIDATION_URL: httpx.Response(200, json={"scope": "SCOPE_ORGANIZATION", "scopeId": "org-1"})})
    with pytest.raises(DrawError) as info:
        run(balance_backend(base_url="https://relay.example.com/v1"), router, method="balance")
    assert "团队 ID" in str(info.value) and router.urls == [VALIDATION_URL]


def test_balance_team_mismatch_retries_with_management_team():
    router = Router({API_KEY_URL: httpx.Response(200, json={"team_id": OTHER_TEAM}),
                     prepaid_url(OTHER_TEAM): httpx.Response(403, json={"code": 7, "message": "denied"}),
                     VALIDATION_URL: httpx.Response(200, json={"scope": "SCOPE_TEAM", "scopeId": TEAM}),
                     prepaid_url(): httpx.Response(200, json=PREPAID),
                     preview_url(): httpx.Response(200, json=PREVIEW)})
    info = run(balance_backend(), router, method="balance")
    assert router.urls == [API_KEY_URL, prepaid_url(OTHER_TEAM), VALIDATION_URL, prepaid_url(), preview_url()]
    assert info.items[0] == ("预付余额", "12.34 美元")
    assert "不属于同一团队" in info.note


def test_balance_403_on_own_team_reports_permission():
    router = Router({VALIDATION_URL: httpx.Response(200, json={"scope": "SCOPE_TEAM", "scopeId": TEAM}),
                     prepaid_url(): httpx.Response(403, json={"code": 7, "message": "denied"})})
    with pytest.raises(DrawError) as info:
        run(balance_backend(team_id=TEAM), router, method="balance")
    assert "没有读取该团队账单的权限" in str(info.value) and info.value.status == 403
    assert router.urls == [prepaid_url()]


def test_balance_invalid_management_key_not_retried(sleeps):
    body = {"code": 16, "message": "Invalid bearer token. Please ensure you use a valid management key.",
            "details": []}
    router = Router({API_KEY_URL: httpx.Response(200, json={"team_id": TEAM}),
                     prepaid_url(): httpx.Response(401, json=body)})
    with pytest.raises(DrawError) as info:
        run(balance_backend(), router, method="balance")
    message = str(info.value)
    assert message.startswith("xAI 管理密钥无效") and "不是 API Key" in message and "HTTP 401" in message
    assert info.value.status == 401 and not sleeps and len(router.requests) == 2
    assert_no_secrets(message)


def test_balance_error_echoing_secret_is_masked():
    body = {"code": 3, "message": f"bad token {MGMT_KEY}"}
    router = Router({prepaid_url(): httpx.Response(400, json=body)})
    with pytest.raises(DrawError) as info:
        run(balance_backend(team_id=TEAM), router, method="balance")
    assert_no_secrets(str(info.value))
    assert base.mask_secret(MGMT_KEY) in str(info.value) and info.value.status == 400


@pytest.mark.parametrize(
    ("total", "expected", "note"),
    [({"val": "0"}, "0.00 美元", ""), ({"val": "-1"}, "0.01 美元", ""), ("-123456789", "1,234,567.89 美元", ""),
     ({"val": "1500"}, "0.00 美元", "预付账本合计为 +15.00 美元")],
)
def test_balance_amounts(total, expected, note):
    router = Router({prepaid_url(): httpx.Response(200, json={"total": total}),
                     preview_url(): httpx.Response(200, json={})})
    info = run(balance_backend(team_id=TEAM), router, method="balance")
    assert info.items == [("预付余额", expected)]
    assert note in info.note and "-0.00" not in info.items[0][1]


@pytest.mark.parametrize("data", [{}, {"total": {"val": "abc"}}, {"total": None}, {"total": {"val": True}}])
def test_balance_bad_total(data):
    router = Router({prepaid_url(): httpx.Response(200, json=data)})
    with pytest.raises(DrawError) as info:
        run(balance_backend(team_id=TEAM), router, method="balance")
    assert "格式异常" in str(info.value)


def test_balance_blocked_key_and_preview_server_error_noted():
    router = Router({API_KEY_URL: httpx.Response(200, json={"team_id": TEAM, "api_key_disabled": True}),
                     prepaid_url(): httpx.Response(200, json=PREPAID),
                     preview_url(): httpx.Response(500, json={"code": 13, "message": "internal"})})
    info = run(balance_backend(), router, method="balance")
    assert info.items == [("预付余额", "12.34 美元")]
    assert "停用" in info.note and "未能读取后付费账单（HTTP 500）" in info.note


def test_balance_connect_error_retried(sleeps):
    router = Router({prepaid_url(): [httpx.ConnectError("refused"), httpx.Response(200, json=PREPAID)],
                     preview_url(): httpx.Response(200, json={})})
    info = run(balance_backend(team_id=TEAM), router, method="balance")
    assert info.items == [("预付余额", "12.34 美元")] and len(sleeps) == 1
    assert router.urls == [prepaid_url(), prepaid_url(), preview_url()]


@pytest.mark.parametrize(
    ("section", "words"),
    [({}, "管理密钥（Management Key，不是 API Key）"),
     ({"management_key": "   "}, "Management Keys"),
     ({"management_key": "xai-token-密钥"}, "中文、全角或不可见字符"),
     ({"management_key": MGMT_KEY, "team_id": "not-a-uuid"}, "UUID")],
)
def test_balance_missing(section, words):
    xai = XAIBackend({"api_key": "xai-test", **section})
    reason = xai.balance_missing()
    assert words in reason
    assert_no_secrets(reason)
    router = Router({})
    with pytest.raises(DrawError) as info:
        run(xai, router, method="balance")
    assert str(info.value) == reason and not router.requests


def test_balance_missing_ok_without_api_key():
    # 查询余额只需要管理密钥；Bearer 前缀和空白会被去掉
    xai = XAIBackend({"management_key": f" Bearer {MGMT_KEY}\n"})
    assert xai.balance_missing() == "" and xai.management_key == MGMT_KEY
    assert balance_backend(team_id=TEAM).balance_missing() == ""
    router = Router({VALIDATION_URL: httpx.Response(200, json={"scope": "SCOPE_TEAM", "scopeId": TEAM}),
                     prepaid_url(): httpx.Response(200, json=PREPAID),
                     preview_url(): httpx.Response(200, json={})})
    info = run(xai, router, method="balance")
    # 没有 API Key 时不请求 /v1/api-key
    assert router.urls == [VALIDATION_URL, prepaid_url(), preview_url()]
    assert info.items == [("预付余额", "12.34 美元")]
