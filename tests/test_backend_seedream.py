"""Seedream 后端测试（不联网）：python -m pytest tests/test_backend_seedream.py"""

import asyncio
import base64
import itertools
import json
import sys
from pathlib import Path

import httpx
import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

from backends import base  # noqa: E402
from backends.base import Clients, DrawError, DrawRequest, HttpPool  # noqa: E402
from backends.seedream import (  # noqa: E402
    _TABLE_4,
    _TABLE_5,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_SIZES,
    DEFAULT_TIMEOUT,
    FAMILIES,
    SeedreamBackend,
    detect_family,
    fit_pixels,
    pick_size,
    platform_model,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff" + b"\x00" * 16
B64 = base64.b64encode(PNG).decode()
KEY = "ark-test-key-0000"
ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
FORBIDDEN = ("seed", "guidance_scale", "n", "quality", "negative_prompt", "stream", "tools", "background",
             "sequential_image_generation")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """重试等待不真的睡眠，并记录等待时间。"""
    waits = []

    async def fake_sleep(seconds, *args, **kwargs):
        waits.append(seconds)

    monkeypatch.setattr(base.asyncio, "sleep", fake_sleep)
    return waits


def ok(item: dict | None = None, headers: dict | None = None) -> httpx.Response:
    item = item if item is not None else {"b64_json": B64, "size": "1664x2496", "output_format": "jpeg"}
    body = {"model": DEFAULT_MODEL, "created": 1784696685, "data": [item],
            "usage": {"generated_images": 1, "output_tokens": 16224, "total_tokens": 16224}}
    return httpx.Response(200, json=body, headers=headers or {})


def err(status: int, code: str, message: str = "boom", flat: bool = False, headers: dict | None = None):
    payload = {"code": code, "message": message} if flat else \
        {"error": {"code": code, "message": message, "param": "", "type": "BadRequest"}}
    return httpx.Response(status, json=payload, headers=headers or {})


def run(section, handler, req: DrawRequest | None = None):
    """用 MockTransport 调用 generate，返回 (结果或 DrawError, 收到的请求列表)。"""
    calls: list[httpx.Request] = []
    responses = list(handler) if isinstance(handler, list) else None

    def transport(request: httpx.Request):
        calls.append(request)
        result = responses.pop(0) if responses is not None else handler(request)
        if isinstance(result, Exception):
            raise result
        return result

    async def main():
        pool = HttpPool(transport=httpx.MockTransport(transport))
        try:
            return await SeedreamBackend(section).generate(Clients(http=pool), req or DrawRequest(prompt="a cat"))
        except DrawError as e:
            return e
        finally:
            await pool.close()

    return asyncio.run(main()), calls


def body_of(request: httpx.Request) -> dict:
    return json.loads(request.content)


def assert_valid(body: dict, family: str):
    """请求体只包含该模型系列接受的字段，尺寸在允许范围内。"""
    rules = FAMILIES[family]
    allowed = {"model", "prompt", "size", "response_format", "watermark"}
    allowed |= {"output_format"} if rules["output_format"] else set()
    allowed |= {"optimize_prompt_options"} if rules["fast"] else set()
    assert set(body) <= allowed, set(body) - allowed
    assert not any(k in body for k in FORBIDDEN)
    assert body["response_format"] == "b64_json"
    assert isinstance(body["watermark"], bool)
    assert body.get("output_format", "jpeg") in ("jpeg", "png")
    assert body.get("optimize_prompt_options", {"mode": "fast"}) == {"mode": "fast"}
    width, height = map(int, body["size"].split("x"))
    lo, hi = rules["pixels"]
    assert lo <= width * height <= hi and 1 / 16 <= width / height <= 16


# ---------------------------------------------------------------- 请求


def test_class_attributes():
    assert SeedreamBackend.key == "seedream" and SeedreamBackend.label == "Seedream"
    assert SeedreamBackend.default_prompt_mode == "natural"
    assert SeedreamBackend.supports_negative is False and SeedreamBackend.supports_models is False
    assert SeedreamBackend.max_concurrency is None
    assert SeedreamBackend.doc_url.startswith("https://www.volcengine.com/docs/82379/")


@pytest.mark.parametrize(("orientation", "size"), list(DEFAULT_SIZES.items()))
def test_default_request_per_orientation(orientation, size):
    result, calls = run({"api_key": KEY},
                        lambda r: ok(headers={"X-Request-Id": "req-1"}), DrawRequest("a cat", orientation=orientation))
    assert len(calls) == 1
    request = calls[0]
    assert request.method == "POST" and str(request.url) == ENDPOINT
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["content-type"] == "application/json"
    assert body_of(request) == {
        "model": DEFAULT_MODEL, "prompt": "a cat", "size": size, "response_format": "b64_json",
        "watermark": False, "output_format": "jpeg",
    }
    assert result.image == PNG and result.mime == "image/png" and result.text == ""
    assert result.params == {"model": DEFAULT_MODEL, "size": "1664x2496", "watermark": False,
                             "output_format": "jpeg", "request_id": "req-1"}


FAST = {"optimize_prompt_options": {"mode": "fast"}}
PNG_FORMAT = {"output_format": "png"}


@pytest.mark.parametrize(
    ("model", "family", "expected"),
    [
        ("doubao-seedream-5-0-pro-260628", "5.0-pro", {**PNG_FORMAT, **FAST}),
        ("doubao-seedream-5-0-flash-260915", "5.0-flash", PNG_FORMAT),
        ("doubao-seedream-5-0-260128", "5.0-lite", PNG_FORMAT),
        ("doubao-seedream-5-0-lite-260128", "5.0-lite", PNG_FORMAT),
        ("dola-seedream-5-0-flash-260915", "5.0-flash", PNG_FORMAT),
        ("doubao-seedream-4-5-251128", "4.5", {}),
        ("doubao-seedream-4-0-250828", "4.0", FAST),
        ("Doubao-Seedream-4.0-20260415", "4.0", FAST),
        # 中转服务常用不带日期的模型名
        ("seedream-5.0-lite", "5.0-lite", PNG_FORMAT),
        ("doubao-seedream-4.5", "4.5", {}),
        ("Seedream 4.0", "4.0", FAST),
        ("ep-20261025-abcde", "generic", {}),
        ("doubao-seedream-6-0-pro-270101", "generic", {}),
        ("doubao-seedream-5-0-ultra-270101", "generic", {}),
        ("doubao-seedream-4-05-1", "generic", {}),
        ("some-relay-model", "generic", {}),
    ],
)
def test_model_families(model, family, expected):
    assert detect_family(model) == family
    section = {"api_key": KEY, "model": model, "output_format": "png", "optimize_mode": "fast",
               "portrait_size": "1024x1024"}
    result, calls = run(section, lambda r: ok())
    body = body_of(calls[0])
    assert body["model"] == platform_model(model, DEFAULT_BASE_URL)
    assert_valid(body, family)
    optional = {k: body[k] for k in ("output_format", "optimize_prompt_options") if k in body}
    assert optional == expected
    lo = FAMILIES[family]["pixels"][0]
    assert body["size"] == ("1024x1024" if lo <= 1024 * 1024 else "1920x1920")


def test_model_family_override_for_endpoint_ids():
    section = {"api_key": KEY, "model": "ep-20261025-abcde", "model_family": "4.5", "square_size": "4K"}
    _, calls = run(section, lambda r: ok(), DrawRequest("x", orientation="square"))
    body = body_of(calls[0])
    assert body["size"] == "4096x4096" and "sequential_image_generation" not in body
    assert_valid(body, "4.5")
    # 未知的系列名按自动识别处理
    backend = SeedreamBackend({"model": "ep-1", "model_family": "nonsense"})
    assert backend.family == "generic"
    assert SeedreamBackend({"model_family": "AUTO"}).family == "5.0-pro"


# ---------------------------------------------------------------- 尺寸


@pytest.mark.parametrize(
    ("family", "text", "orientation", "expected"),
    [
        ("5.0-pro", "2048 x 2048", "square", "2048x2048"),
        ("5.0-pro", "2048X2048", "square", "2048x2048"),
        ("5.0-pro", "1536×1536", "square", "1536x1536"),
        ("5.0-pro", "1080*1920", "portrait", "1080x1920"),  # 合法尺寸原样保留
        ("5.0-pro", "2k", "portrait", "1664x2496"),
        ("5.0-pro", "2K", "landscape", "2496x1664"),
        ("5.0-pro", "1.5K", "square", "1536x1536"),
        ("5.0-pro", "1k", "portrait", "832x1248"),
        ("5.0-pro", "4K", "portrait", "1664x2496"),  # pro 没有 4K，换成最接近的 2K
        ("5.0-pro", "9:16", "portrait", "1584x2816"),
        ("5.0-pro", "16:9 1K", "landscape", "1424x800"),
        ("5.0-pro", "4：3", "landscape", "2368x1776"),
        ("5.0-pro", "9:21", "portrait", "1344x3136"),
        ("5.0-flash", "1.5k 3:4", "portrait", "1344x1792"),
        ("5.0-lite", "1K", "square", "2048x2048"),  # lite 没有 1K
        ("5.0-lite", "1024x1024", "square", "1920x1920"),  # 低于 368 万像素时放大
        ("5.0-lite", "3K", "portrait", "2496x3744"),
        ("5.0-lite", "4K", "landscape", "4992x3328"),
        ("5.0-lite", "9:16", "portrait", "1600x2848"),
        ("4.5", "3K", "square", "2048x2048"),  # 4.5 没有 3K，与 2K、4K 等距时取较低档
        ("4.5", "4k", "square", "4096x4096"),
        ("4.0", "1K", "landscape", "1248x832"),
        ("4.0", "16:9 1k", "landscape", "1312x736"),
        ("generic", "4K", "portrait", "1664x2496"),
        ("generic", "1024x1024", "square", "1920x1920"),
        ("generic", "", "landscape", "2496x1664"),
        ("5.0-pro", "abc", "portrait", "1664x2496"),
        ("5.0-pro", None, "square", "2048x2048"),
        ("5.0-pro", "0x100", "landscape", "2496x1664"),
        ("5.0-pro", "0:0", "landscape", "2496x1664"),
        ("5.0-pro", 12345, "portrait", "1664x2496"),
        ("5.0-pro", "2048x2048", "diagonal", "2048x2048"),
    ],
)
def test_pick_size(family, text, orientation, expected):
    assert pick_size(text, orientation, family) == expected


@pytest.mark.parametrize(
    ("family", "text"),
    [("5.0-pro", "1920x3413"), ("5.0-pro", "4096x4096"), ("5.0-flash", "512x512"), ("5.0-pro", "1x100"),
     ("4.5", "100000x10"), ("5.0-lite", "2.35:1"), ("5.0-pro", "5:4"), ("generic", "1:3 4K"), ("4.0", "1:100")],
)
def test_pick_size_snaps_into_range(family, text):
    width, height = map(int, pick_size(text, "portrait", family).split("x"))
    lo, hi = FAMILIES[family]["pixels"]
    assert lo <= width * height <= hi and 1 / 16 <= width / height <= 16


def test_fit_pixels_always_valid_and_keeps_orientation():
    values = [1, 7, 16, 333, 512, 720, 1024, 1080, 1920, 2048, 2560, 4096, 5000, 8192, 20000, 99999]
    for (lo, hi), (w, h) in itertools.product({f["pixels"] for f in FAMILIES.values()},
                                              itertools.product(values, values)):
        fw, fh = fit_pixels(w, h, lo, hi)
        assert lo <= fw * fh <= hi and 1 / 16 <= fw / fh <= 16, (w, h, lo, hi, fw, fh)
        assert (fw > fh) == (w > h) or w == h or fw == fh
    # 宽高比尽量保持
    w, h = fit_pixels(1920, 3413, 921_600, 4_624_220)
    assert abs(w / h - 1920 / 3413) < 0.01


def test_official_tables_fit_their_families():
    for family, rules in FAMILIES.items():
        tables = _TABLE_5 if family in ("5.0-pro", "5.0-flash") else _TABLE_4
        lo, hi = rules["pixels"]
        for tier in rules["tiers"]:
            for ratio, (w, h) in tables[tier].items():
                a, b = map(int, ratio.split(":"))
                assert lo <= w * h <= hi, (family, tier, ratio)
                # 宽高没有写反，比例与键一致
                assert abs(w / h - a / b) < 0.02, (family, tier, ratio)


# ---------------------------------------------------------------- 地址与模型 ID


@pytest.mark.parametrize(
    ("base_url", "url"),
    [
        ("", ENDPOINT),
        (None, ENDPOINT),
        ("ark.cn-beijing.volces.com", ENDPOINT),
        ("https://ark.cn-beijing.volces.com/", ENDPOINT),
        ("https://ark.cn-beijing.volces.com/api/v3/", ENDPOINT),
        (" https://ark.cn-beijing.volces.com/api/v3/images/generations ", ENDPOINT),
        ("https://ark.ap-southeast.bytepluses.com/api/v3",
         "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"),
        ("http://127.0.0.1:3000/v1", "http://127.0.0.1:3000/v1/images/generations"),
        # 误粘贴的其他接口地址
        ("https://ark.cn-beijing.volces.com/api/v3/chat/completions", ENDPOINT),
        ("https://ark.cn-beijing.volces.com/api/v3/images/", ENDPOINT),
        ("https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks", ENDPOINT),
        ("http://127.0.0.1:3000/v1/images/generations", "http://127.0.0.1:3000/v1/images/generations"),
    ],
)
def test_base_url_normalisation(base_url, url):
    _, calls = run({"api_key": KEY, "base_url": base_url}, lambda r: ok())
    assert str(calls[0].url) == url


@pytest.mark.parametrize("base_url", ["ftp://ark.cn-beijing.volces.com/api/v3", "https://ark.cn-beijing.volces.com:abc",
                                      "://ark.cn-beijing.volces.com", "https://user:pass@:99999/api/v3"])
def test_invalid_base_url(base_url):
    reason = SeedreamBackend({"api_key": KEY, "base_url": base_url}).missing()
    assert "Seedream API 地址" in reason and "pass" not in reason
    error, calls = run({"api_key": KEY, "base_url": base_url}, lambda r: ok())
    assert isinstance(error, DrawError) and str(error) == reason and calls == []


@pytest.mark.parametrize(
    ("model", "base_url", "expected"),
    [
        ("doubao-seedream-5-0-pro-260628", "https://ark.ap-southeast.bytepluses.com/api/v3",
         "dola-seedream-5-0-pro-260628"),
        ("doubao-seedream-5-0-flash-260915", "https://ark.ap-southeast.bytepluses.com/api/v3",
         "dola-seedream-5-0-flash-260915"),
        ("doubao-seedream-4-5-251128", "https://ark.ap-southeast.bytepluses.com/api/v3", "seedream-4-5-251128"),
        ("doubao-seedream-5-0-260128", "https://ark.eu-west.bytepluses.com/api/v3", "seedream-5-0-260128"),
        ("dola-seedream-5-0-pro-260628", DEFAULT_BASE_URL, "doubao-seedream-5-0-pro-260628"),
        ("seedream-4-0-250828", DEFAULT_BASE_URL, "doubao-seedream-4-0-250828"),
        ("doubao-seedream-5-0-pro-260628", DEFAULT_BASE_URL, "doubao-seedream-5-0-pro-260628"),
        ("ep-20261025-abcde", "https://ark.ap-southeast.bytepluses.com/api/v3", "ep-20261025-abcde"),
        ("doubao-seedream-5-0-pro-260628", "https://relay.example.com/v1", "doubao-seedream-5-0-pro-260628"),
    ],
)
def test_platform_model(model, base_url, expected):
    assert platform_model(model, base_url) == expected


def test_byteplus_default_model_is_translated():
    section = {"api_key": KEY, "base_url": "https://ark.ap-southeast.bytepluses.com/api/v3"}
    _, calls = run(section, lambda r: ok())
    assert body_of(calls[0])["model"] == "dola-seedream-5-0-pro-260628"


# ---------------------------------------------------------------- 响应


def test_url_response_is_downloaded_without_auth():
    image_url = "https://ark-content.tos-cn-beijing.volces.com/img.jpeg?X-Tos-Signature=abc"

    def handler(request: httpx.Request):
        if request.method == "POST":
            return ok({"url": image_url, "size": "2048x2048"})
        assert str(request.url) == image_url
        return httpx.Response(200, content=JPEG)

    result, calls = run({"api_key": KEY, "model": "doubao-seedream-4-5-251128"}, handler)
    assert [c.method for c in calls] == ["POST", "GET"]
    assert "authorization" not in calls[1].headers
    assert result.image == JPEG and result.mime == "image/jpeg"
    assert result.params["size"] == "2048x2048" and "output_format" not in result.params


def test_b64_with_data_prefix_and_fast_params():
    section = {"api_key": KEY, "optimize_mode": "fast", "output_format": "png"}
    result, _ = run(section, lambda r: ok({"b64_json": "data:image/png;base64," + B64, "output_format": "png"}))
    assert result.image == PNG
    assert result.params["optimize_mode"] == "fast" and result.params["output_format"] == "png"


@pytest.mark.parametrize(
    ("payload", "text"),
    [
        ({"data": [], "error": {"code": "OutputImageSensitiveContentDetected", "message": "x"}}, "生成的图片未通过内容安全审核"),
        ({"error": {"code": "InternalServiceError", "message": "oops"}}, "没有生成图片"),
        ({"data": [{"error": {"code": "InputTextSensitiveContentDetected.SevereViolation", "message": "x"}}]},
         "提示词未通过内容安全审核"),
        ({"data": []}, "没有返回图片"),
        ({"data": [{"b64_json": base64.b64encode(b"not an image").decode()}]}, "不是可识别的图片"),
    ],
)
def test_200_without_image(payload, text):
    error, calls = run({"api_key": KEY}, lambda r: httpx.Response(200, json=payload))
    assert isinstance(error, DrawError) and text in str(error)
    assert not error.retryable and len(calls) == 1


def test_non_json_success_body():
    error, _ = run({"api_key": KEY}, lambda r: httpx.Response(200, text="<html>gateway</html>"))
    assert isinstance(error, DrawError) and "不是 JSON" in str(error)


# ---------------------------------------------------------------- 错误与重试


@pytest.mark.parametrize(
    ("code", "text"),
    [
        ("InputTextSensitiveContentDetected", "提示词未通过内容安全审核"),
        ("OutputImageSensitiveContentDetected.DeepFake", "生成的图片未通过内容安全审核"),
        ("SensitiveContentDetected.Violence", "请求内容未通过内容安全审核"),
        ("InputTextRiskDetection", "提示词未通过内容安全审核"),
        ("InputImageSensitiveContentDetected.PrivacyInformation", "请求内容未通过内容安全审核"),
    ],
)
def test_moderation_refusal(code, text):
    error, calls = run({"api_key": KEY}, lambda r: err(400, code, "The request failed because ..."))
    assert isinstance(error, DrawError) and text in str(error) and code in str(error)
    assert "拒绝生成" in str(error) and error.status == 400 and not error.retryable
    assert len(calls) == 1


@pytest.mark.parametrize("flat", [False, True])
def test_401(flat):
    error, calls = run({"api_key": KEY}, lambda r: err(401, "AuthenticationError", "the API key is invalid", flat))
    assert isinstance(error, DrawError) and error.status == 401 and not error.retryable
    assert "API Key 无效" in str(error) and "AuthenticationError" in str(error)
    assert "the API key is invalid" in str(error)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("code", "flat", "text"),
    [
        ("SetLimitExceeded", False, "消费上限"),
        ("QuotaExceeded", True, "额度已用尽"),
        ("AccountQuotaExceeded", False, "额度已用尽"),
        ("ModelAccountTpdRateLimitExceeded", True, "每日用量上限"),
    ],
)
def test_billing_429_not_retried(code, flat, text, no_sleep):
    error, calls = run({"api_key": KEY}, lambda r: err(429, code, flat=flat, headers={"Retry-After": "0"}))
    assert isinstance(error, DrawError) and error.status == 429 and not error.retryable
    assert text in str(error) and code in str(error)
    assert len(calls) == 1 and no_sleep == []


@pytest.mark.parametrize("code", ["ModelAccountIpmRateLimitExceeded", "ServerOverloaded", "RequestBurstTooFast", ""])
def test_rate_limit_429_retried_once(code, no_sleep):
    result, calls = run({"api_key": KEY}, [err(429, code, headers={"Retry-After": "2"}), ok()])
    assert result.image == PNG and len(calls) == 2 and no_sleep == [2.0]
    # 两次都被限流：只重试一次，最后报错
    error, calls = run({"api_key": KEY}, lambda r: err(429, code))
    assert isinstance(error, DrawError) and error.retryable and len(calls) == 2
    assert "请求过于频繁" in str(error)


@pytest.mark.parametrize(
    ("status", "code", "attempts"),
    [(500, "InternalServiceError", 1), (504, "", 1), (500, "ModelLoading", 2), (503, "ServiceConnectionRefused", 2),
     (502, "", 2), (400, "InvalidParameter", 1), (403, "AccountOverdueError", 1),
     (404, "InvalidEndpointOrModel.NotFound", 1), (404, "InvalidEndpointOrModel.ModelIDAccessDisabled", 1)],
)
def test_server_errors(status, code, attempts):
    error, calls = run({"api_key": KEY}, lambda r: err(status, code))
    assert isinstance(error, DrawError) and error.status == status and len(calls) == attempts
    assert f"HTTP {status}" in str(error)
    if code:
        assert code in str(error)
    if code == "InvalidEndpointOrModel.ModelIDAccessDisabled":
        assert "接入点 ID" in str(error)


def test_html_error_page():
    error, calls = run({"api_key": KEY}, lambda r: httpx.Response(
        502, text="<html><head><title>502 Bad Gateway</title></head></html>"))
    assert isinstance(error, DrawError) and "502 Bad Gateway" in str(error) and len(calls) == 2


def test_timeout_not_retried():
    error, calls = run({"api_key": KEY}, lambda r: httpx.ReadTimeout("timed out", request=r))
    assert isinstance(error, DrawError) and "超时" in str(error) and not error.retryable
    assert len(calls) == 1
    # 请求发出后连接中断也可能已经计费，不重试
    error, calls = run({"api_key": KEY}, lambda r: httpx.RemoteProtocolError("disconnected", request=r))
    assert isinstance(error, DrawError) and not error.retryable and len(calls) == 1


def test_connect_error_retried_once(no_sleep):
    result, calls = run({"api_key": KEY}, [httpx.ConnectError("refused"), ok()])
    assert result.image == PNG and len(calls) == 2 and len(no_sleep) == 1


def test_timeout_setting():
    assert SeedreamBackend({}).timeout == DEFAULT_TIMEOUT
    assert SeedreamBackend({"timeout": "600"}).timeout == 600
    assert SeedreamBackend({"timeout": 1}).timeout == 10
    assert SeedreamBackend({"timeout": "abc"}).timeout == DEFAULT_TIMEOUT


# ---------------------------------------------------------------- 配置


def test_extra_body_merged_last():
    extra = '{"size": "4K", "watermark": true, "output_format": "png", "custom": {"a": 1}}'
    _, calls = run({"api_key": KEY, "extra_body": extra}, lambda r: ok())
    body = body_of(calls[0])
    assert body["size"] == "4K" and body["watermark"] is True and body["output_format"] == "png"
    assert body["custom"] == {"a": 1} and body["model"] == DEFAULT_MODEL
    _, calls = run({"api_key": KEY, "extra_body": {"seed": 7}}, lambda r: ok())
    assert body_of(calls[0])["seed"] == 7


@pytest.mark.parametrize("extra", ["{bad json", "[1, 2]", "\"text\""])
def test_invalid_extra_body(extra):
    error, calls = run({"api_key": KEY, "extra_body": extra}, lambda r: ok())
    assert isinstance(error, DrawError) and "额外请求参数" in str(error) and calls == []


def test_missing():
    assert SeedreamBackend({}).missing() == "管理员还没有配置 Seedream API Key"
    assert SeedreamBackend(None).missing() == "管理员还没有配置 Seedream API Key"
    assert SeedreamBackend({"api_key": "   "}).missing() != ""
    assert SeedreamBackend({"api_key": None}).missing() != ""
    assert SeedreamBackend({"api_key": KEY}).missing() == ""


@pytest.mark.parametrize("key", [f"Bearer {KEY}", f"bearer  {KEY}\n", "ark-test-\nkey-0000", " ark-test-key\r\n-0000 "])
def test_key_prefix_and_whitespace_are_stripped(key):
    """Key 中间的换行会让请求头非法，httpx 的报错会原样带出 Key，必须先去掉。"""
    assert SeedreamBackend({"api_key": key}).missing() == ""
    _, calls = run({"api_key": key}, lambda r: ok())
    assert calls[0].headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("key", ["ark-测试-key-0000", "ark-test\u200bkey-0000", "ark-test\x7fkey-0000"])
def test_key_with_invalid_characters_is_rejected(key):
    reason = SeedreamBackend({"api_key": key}).missing()
    assert "不可见字符" in reason and key not in reason
    error, calls = run({"api_key": key}, lambda r: ok())
    assert isinstance(error, DrawError) and str(error) == reason and calls == []


def test_key_echoed_by_relay_is_masked():
    secret = "sk-relay-secret-123456789"
    for response in (httpx.Response(401, json={"error": {"message": f"invalid token {secret}"}}),
                     httpx.Response(200, json={"error": {"code": "Oops", "message": f"bad key {secret}"}})):
        error, _ = run({"api_key": secret, "base_url": "https://relay.example.com/v1"}, lambda r, resp=response: resp)
        assert isinstance(error, DrawError) and secret not in str(error) and "6789" in str(error)


def test_negative_prompt_and_seed():
    _, calls = run({"api_key": KEY}, lambda r: ok(), DrawRequest("a cat", negative="dogs, text", seed=42))
    body = body_of(calls[0])
    assert body["prompt"].startswith("a cat") and "Avoid: dogs, text" in body["prompt"]
    assert not any(k in body for k in FORBIDDEN)


@pytest.mark.parametrize(("value", "expected"), [("false", False), ("False", False), ("true", True), (True, True),
                                                 (False, False), ("", False), (None, False), (1, True), (0, False)])
def test_watermark_parsing(value, expected):
    _, calls = run({"api_key": KEY, "watermark": value}, lambda r: ok())
    assert body_of(calls[0])["watermark"] is expected


def test_garbage_config_values():
    section = {
        "api_key": KEY, "base_url": None, "proxy_url": "", "model": None, "portrait_size": "abc",
        "landscape_size": 123, "square_size": None, "model_family": 5, "output_format": "gif",
        "optimize_mode": ["fast"], "watermark": "yes please", "timeout": "", "extra_body": None,
    }
    for orientation in DEFAULT_SIZES:
        _, calls = run(section, lambda r: ok(), DrawRequest("a cat", orientation=orientation))
        assert str(calls[0].url) == ENDPOINT
        body = body_of(calls[0])
        assert body == {"model": DEFAULT_MODEL, "prompt": "a cat", "size": DEFAULT_SIZES[orientation],
                        "response_format": "b64_json", "watermark": False, "output_format": "jpeg"}
    backend = SeedreamBackend({"model": "", "portrait_size": "", "base_url": "  "})
    assert backend.summary() == f"Seedream / {DEFAULT_MODEL} / 竖图 1664x2496"
    assert backend.timeout == DEFAULT_TIMEOUT


def test_summary():
    assert SeedreamBackend({}).summary() == "Seedream / doubao-seedream-5-0-pro-260628 / 竖图 1664x2496"
    section = {"model": "doubao-seedream-5-0-260128", "portrait_size": "3K"}
    assert SeedreamBackend(section).summary() == "Seedream / doubao-seedream-5-0-260128 / 竖图 2496x3744"


def test_list_models_not_supported():
    async def main():
        pool = HttpPool(transport=httpx.MockTransport(lambda r: ok()))
        try:
            return await SeedreamBackend({"api_key": KEY}).list_models(Clients(http=pool))
        finally:
            await pool.close()

    with pytest.raises(DrawError):
        asyncio.run(main())


# ---------------------------------------------------------------- 配置 schema


def load_schema_section() -> dict:
    return json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))["seedream"]


def test_schema_defaults_match_code_and_produce_valid_requests():
    section = load_schema_section()
    items = section["items"]
    assert list(items) == ["api_key", "base_url", "proxy_url", "model", "portrait_size", "landscape_size",
                           "square_size", "model_family", "output_format", "optimize_mode", "watermark",
                           "timeout", "extra_body"]
    for name, field in items.items():
        assert field["type"] in ("string", "text", "int", "float", "bool", "list"), name
        assert field.get("description") and field.get("hint"), name
        if "options" in field:
            assert field["default"] in field["options"] and len(field.get("labels", field["options"])) == len(
                field["options"])
    assert items["api_key"]["secret"] is True
    assert items["base_url"]["default"] == DEFAULT_BASE_URL
    assert items["model"]["default"] == DEFAULT_MODEL and "options" not in items["model"]
    assert {o: items[f"{o}_size"]["default"] for o in DEFAULT_SIZES} == DEFAULT_SIZES
    assert set(items["model_family"]["options"]) == {"auto"} | (set(FAMILIES) - {"generic"})
    assert items["output_format"]["default"] == DEFAULT_OUTPUT_FORMAT
    assert set(items["output_format"]["options"]) == {"png", "jpeg"}
    assert set(items["optimize_mode"]["options"]) == {"standard", "fast"}
    assert items["watermark"]["default"] is False
    assert items["timeout"]["default"] == DEFAULT_TIMEOUT
    assert items["extra_body"]["default"] == "" and items["extra_body"]["editor_mode"] is True

    defaults = {name: field["default"] for name, field in items.items()}
    defaults["api_key"] = KEY
    assert SeedreamBackend(defaults).missing() == ""
    family = detect_family(defaults["model"])
    for orientation in DEFAULT_SIZES:
        _, calls = run(defaults, lambda r: ok(), DrawRequest("a cat", orientation=orientation))
        assert str(calls[0].url) == ENDPOINT
        body = body_of(calls[0])
        assert_valid(body, family)
        assert body["size"] == DEFAULT_SIZES[orientation] and body["watermark"] is False
    # 默认尺寸对每个模型系列都不需要调整
    for fam, size in itertools.product(FAMILIES, DEFAULT_SIZES.values()):
        assert pick_size(size, "square", fam) == size
