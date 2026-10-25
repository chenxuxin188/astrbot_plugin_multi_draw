"""Seedream 后端测试（不联网）：python -m pytest tests/test_backend_seedream.py"""

import asyncio
import base64
import datetime
import hashlib
import itertools
import json
import struct
import sys
import zlib
from pathlib import Path

import httpx
import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

from backends import base  # noqa: E402
from backends.base import BalanceInfo, Clients, DrawError, DrawRequest, HttpPool, InputImage  # noqa: E402
from backends.seedream import (  # noqa: E402
    _TABLE_4,
    _TABLE_5,
    BILLING,
    BILLING_PLATFORMS,
    DEFAULT_BASE_URL,
    DEFAULT_EDIT_SIZE_MODE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_SIZES,
    DEFAULT_TIMEOUT,
    EDIT_MAX_IMAGES,
    EDIT_SIZE_MODES,
    FAMILIES,
    MAX_EDIT_IMAGES,
    SeedreamBackend,
    _hmac_sha256,
    canonical_query,
    canonical_request,
    detect_family,
    fit_pixels,
    pick_size,
    platform_model,
    png_has_alpha,
    sign_v4,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff" + b"\x00" * 16
B64 = base64.b64encode(PNG).decode()
KEY = "ark-test-key-0000"
ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
FORBIDDEN = ("seed", "guidance_scale", "n", "quality", "negative_prompt", "stream", "tools", "background",
             "sequential_image_generation")
# 修改图片时任何模型都不接受的字段
EDIT_FORBIDDEN = ("seed", "guidance_scale", "n", "quality", "negative_prompt", "stream", "tools", "mask", "strength",
                  "sequential_image_generation", "input_fidelity")


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
    # 类属性是所有模型系列的最大值
    assert SeedreamBackend.max_edit_images == MAX_EDIT_IMAGES == 14
    assert SeedreamBackend.supports_strength is False and SeedreamBackend.supports_balance is True
    assert SeedreamBackend.billing_url == "https://console.volcengine.com/finance/fund/recharge"


@pytest.mark.parametrize(
    ("section", "limit"),
    [
        ({}, 10),
        ({"model": "doubao-seedream-5-0-flash-260915"}, 10),
        ({"model": "doubao-seedream-5-0-260128"}, 14),
        ({"model": "doubao-seedream-4-5-251128"}, 14),
        ({"model": "seedream-4-0-250828"}, 14),
        ({"model": "ep-20261025-abcde"}, 10),
        ({"model": "ep-20261025-abcde", "model_family": "4.5"}, 14),
        (None, 10),
        ({"model": 123, "base_url": "https://[bad"}, 10),
    ],
)
def test_instance_edit_limit_follows_model_family(section, limit):
    """插件按实例上的 max_edit_images 截取图片，因此实例上给出当前模型系列的上限。"""
    backend = SeedreamBackend(section)
    assert backend.max_edit_images == limit == EDIT_MAX_IMAGES[backend.family]
    assert backend.edit_missing() == ""


def test_instance_billing_url_follows_platform():
    assert SeedreamBackend({}).billing_url == BILLING["volcengine"]["console"]
    byteplus = SeedreamBackend({"base_url": "https://ark.ap-southeast.bytepluses.com/api/v3"})
    assert byteplus.billing_url == "https://console.byteplus.com/finance/overview"
    relay = SeedreamBackend({"base_url": "https://relay.example.com/v1", "billing_platform": "byteplus"})
    assert relay.billing_url == BILLING["byteplus"]["console"]


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
                           "edit_size_mode", "keep_transparency", "timeout", "extra_body",
                           "access_key_id", "secret_access_key", "billing_platform", "billing_region"]
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

    # 修改图片与余额查询的字段
    assert items["edit_size_mode"]["default"] == DEFAULT_EDIT_SIZE_MODE
    assert tuple(items["edit_size_mode"]["options"]) == EDIT_SIZE_MODES
    assert items["keep_transparency"]["default"] is False and items["keep_transparency"]["type"] == "bool"
    for name in ("access_key_id", "secret_access_key"):
        assert items[name]["secret"] is True and items[name]["default"] == "" and items[name]["type"] == "string"
        assert "仅用于查询余额" in items[name]["description"] and "余额" in items[name]["hint"]
    assert "BillingCenterReadOnlyAccess" in items["access_key_id"]["hint"]
    assert tuple(items["billing_platform"]["options"]) == BILLING_PLATFORMS
    assert items["billing_platform"]["default"] == "auto" and items["billing_region"]["default"] == ""
    assert "修改图片" in items["model"]["hint"] and "10 张" in items["model"]["hint"]
    assert "14 张" in items["model"]["hint"]
    # 默认配置：可以修改图片，修改请求合法；没有 Access Key 时余额查询给出原因
    backend = SeedreamBackend(defaults)
    assert backend.edit_missing() == "" and backend.max_edit_images == EDIT_MAX_IMAGES[family]
    assert "Access Key" in backend.balance_missing()
    req = DrawRequest("make it night", orientation="portrait", images=[InputImage(JPEG, "image/jpeg", 1200, 1800)])
    _, calls = run(defaults, lambda r: ok(), req)
    body = body_of(calls[0])
    assert_valid_edit(body, family)
    assert "background" not in body and size_ratio(body["size"]) == pytest.approx(1200 / 1800, abs=0.01)


# ---------------------------------------------------------------- 图片修改


def png(color_type: int, trns: bool = False) -> bytes:
    """最小的合法 PNG：指定 IHDR 颜色类型，可选 tRNS 块。"""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, color_type, 0, 0, 0)
    extra = chunk(b"tRNS", b"\x00\x00") if trns else b""
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + extra + chunk(b"IDAT", zlib.compress(b"\x00" * 5))
            + chunk(b"IEND", b""))


RGBA = png(6)
RGB = png(2)


def image(data: bytes = JPEG, width: int = 1024, height: int = 1024) -> InputImage:
    return InputImage(data, "image/png" if data.startswith(b"\x89PNG") else "image/jpeg", width, height)


def edit_req(*images: InputImage, prompt: str = "Replace the background of image 1 with a beach", **kwargs):
    return DrawRequest(prompt, images=list(images) or [image()], **kwargs)


def data_url(data: bytes, fmt: str) -> str:
    return f"data:image/{fmt};base64,{base64.b64encode(data).decode()}"


def size_ratio(size: str) -> float:
    width, height = map(int, size.split("x"))
    return width / height


def assert_valid_edit(body: dict, family: str):
    """修改请求只包含该模型系列接受的字段，图片是小写格式的 data URL，数量和尺寸在允许范围内。"""
    rules = FAMILIES[family]
    allowed = {"model", "prompt", "image", "size", "response_format", "watermark"}
    allowed |= {"output_format"} if rules["output_format"] else set()
    allowed |= {"optimize_prompt_options"} if rules["fast"] else set()
    allowed |= {"background"} if family in ("5.0-pro", "5.0-flash") else set()
    assert set(body) <= allowed, set(body) - allowed
    assert not any(k in body for k in EDIT_FORBIDDEN)
    images = body["image"] if isinstance(body["image"], list) else [body["image"]]
    assert 1 <= len(images) <= EDIT_MAX_IMAGES[family]
    assert isinstance(body["image"], str) or len(images) >= 2  # 单张图片发送字符串
    for url in images:
        prefix, encoded = url.split(",", 1)
        assert prefix in ("data:image/jpeg;base64", "data:image/png;base64")
        assert base64.b64decode(encoded, validate=True)
    assert body["response_format"] == "b64_json" and isinstance(body["watermark"], bool)
    if "x" in body["size"]:
        width, height = map(int, body["size"].split("x"))
        lo, hi = rules["pixels"]
        assert lo <= width * height <= hi and 1 / 16 <= width / height <= 16
    else:
        assert body["size"] in rules["tiers"]


def test_edit_single_image_request_and_result():
    usage = {"generated_images": 1, "input_images": 1, "output_tokens": 16224, "total_tokens": 16224}

    def handler(request: httpx.Request):
        payload = {"model": DEFAULT_MODEL, "created": 1, "usage": usage,
                   "data": [{"b64_json": B64, "size": "1664x2496", "output_format": "jpeg"}]}
        return httpx.Response(200, json=payload, headers={"X-Request-Id": "req-9"})

    req = edit_req(image(JPEG, 1200, 1800), orientation="portrait")
    result, calls = run({"api_key": KEY}, handler, req)
    assert len(calls) == 1
    request = calls[0]
    assert request.method == "POST" and str(request.url) == ENDPOINT
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["content-type"] == "application/json"
    body = body_of(request)
    assert body == {
        "model": DEFAULT_MODEL, "prompt": req.prompt, "image": data_url(JPEG, "jpeg"), "size": "1664x2496",
        "response_format": "b64_json", "watermark": False, "output_format": "jpeg",
    }
    assert_valid_edit(body, "5.0-pro")
    assert result.image == PNG and result.mime == "image/png"
    assert result.params == {"model": DEFAULT_MODEL, "size": "1664x2496", "watermark": False, "output_format": "jpeg",
                             "mode": "edit", "input_images": 1, "request_id": "req-9"}


def test_edit_multi_image_order_and_formats():
    req = edit_req(image(RGB, 1000, 1000), image(JPEG, 800, 1200), orientation="square",
                   prompt="Replace the clothing in image 1 with the outfit from image 2")
    section = {"api_key": KEY, "model": "doubao-seedream-4-5-251128", "output_format": "png"}
    result, calls = run(section, lambda r: ok({"b64_json": B64, "size": "2048x2048"}), req)
    body = body_of(calls[0])
    assert body["image"] == [data_url(RGB, "png"), data_url(JPEG, "jpeg")]  # 按顺序，即 image 1、image 2
    assert body["prompt"] == req.prompt
    assert body["size"] == "2048x2048"  # 比例取第一张（方图）
    assert "output_format" not in body  # 4.5 不接受
    assert_valid_edit(body, "4.5")
    assert result.params == {"model": "doubao-seedream-4-5-251128", "size": "2048x2048", "watermark": False,
                             "mode": "edit", "input_images": 2}


@pytest.mark.parametrize(
    ("model", "width", "height"),
    [
        (DEFAULT_MODEL, 1920, 1080),
        (DEFAULT_MODEL, 600, 1800),
        (DEFAULT_MODEL, 2048, 2048),
        ("doubao-seedream-5-0-flash-260915", 1080, 1350),
        ("doubao-seedream-5-0-260128", 900, 1600),
        ("doubao-seedream-4-5-251128", 2048, 1024),
        ("doubao-seedream-4-0-250828", 1000, 700),
        ("ep-20261025-abcde", 3000, 1000),
    ],
)
def test_edit_size_keeps_input_aspect_ratio(model, width, height):
    """接口没有「跟随原图」的尺寸，按原图比例和配置尺寸的面积计算宽x高。"""
    req = edit_req(image(JPEG, width, height), orientation=base.image_orientation(image(JPEG, width, height)))
    _, calls = run({"api_key": KEY, "model": model}, lambda r: ok(), req)
    body = body_of(calls[0])
    family = detect_family(model)
    assert_valid_edit(body, family)
    assert size_ratio(body["size"]) == pytest.approx(width / height, rel=0.02)
    # 面积与该方向的配置尺寸接近（费用与画图相同）
    configured = SeedreamBackend({"model": model}).size(req.orientation)
    w, h = map(int, body["size"].split("x"))
    cw, ch = map(int, configured.split("x"))
    assert w * h == pytest.approx(cw * ch, rel=0.05)


def test_edit_size_follows_explicit_orientation_and_unknown_dimensions():
    backend = SeedreamBackend({"api_key": KEY})
    # 用户明确要求换方向：使用该方向的配置尺寸
    assert backend.edit_size(edit_req(image(JPEG, 1200, 1800), orientation="landscape")) == "2496x1664"
    # 不知道原图尺寸：按画面方向
    assert backend.edit_size(edit_req(image(JPEG, 0, 0), orientation="square")) == "2048x2048"
    # 无效的方向按原图比例处理
    size = backend.edit_size(edit_req(image(JPEG, 1920, 1080), orientation="diagonal"))
    assert size_ratio(size) == pytest.approx(16 / 9, rel=0.02)
    # 配置的面积低于模型下限时放大到允许的范围，比例不变
    lite = SeedreamBackend({"model": "doubao-seedream-4-5-251128", "portrait_size": "1024x1536"})
    w, h = map(int, lite.edit_size(edit_req(image(JPEG, 1000, 1500), orientation="portrait")).split("x"))
    assert w * h >= FAMILIES["4.5"]["pixels"][0] and w / h == pytest.approx(2 / 3, rel=0.02)


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ({}, "2K"),
        ({"portrait_size": "1.5K"}, "1.5K"),
        ({"portrait_size": "832x1248"}, "1K"),
        ({"model": "doubao-seedream-5-0-260128", "portrait_size": "4K"}, "4K"),
        ({"model": "doubao-seedream-5-0-260128", "portrait_size": "1K"}, "2K"),  # lite 没有 1K
        ({"model": "doubao-seedream-4-0-250828", "portrait_size": "1K"}, "1K"),
        ({"model": "ep-20261025-abcde"}, "2K"),
    ],
)
def test_edit_tier_mode(section, expected):
    section = {"api_key": KEY, "edit_size_mode": "TIER", **section}
    _, calls = run(section, lambda r: ok(), edit_req(image(JPEG, 1200, 1800), orientation="portrait"))
    body = body_of(calls[0])
    assert body["size"] == expected
    assert_valid_edit(body, SeedreamBackend(section).family)


def test_unknown_edit_size_mode_uses_match():
    backend = SeedreamBackend({"edit_size_mode": "nonsense"})
    assert backend.edit_size_mode == DEFAULT_EDIT_SIZE_MODE
    assert backend.edit_size(edit_req(image(JPEG, 1200, 1800), orientation="portrait")) == "1664x2496"


@pytest.mark.parametrize(("model", "limit"), [(DEFAULT_MODEL, 10), ("doubao-seedream-5-0-flash-260915", 10),
                                              ("doubao-seedream-5-0-260128", 14), ("doubao-seedream-4-5-251128", 14),
                                              ("doubao-seedream-4-0-250828", 14), ("ep-20261025-abcde", 10)])
def test_edit_image_limit_per_family(model, limit):
    section = {"api_key": KEY, "model": model}
    _, calls = run(section, lambda r: ok(), edit_req(*[image()] * limit))
    body = body_of(calls[0])
    assert len(body["image"]) == limit
    assert_valid_edit(body, detect_family(model))
    error, calls = run(section, lambda r: ok(), edit_req(*[image()] * (limit + 1)))
    assert isinstance(error, DrawError) and not error.retryable and calls == []
    assert f"最多使用 {limit} 张图片" in str(error) and f"本次有 {limit + 1} 张" in str(error)


@pytest.mark.parametrize(
    ("make", "text"),
    [
        (lambda: image(JPEG, 14, 100), "大于 14 像素"),
        (lambda: image(JPEG, 3400, 200), "1:16"),
        (lambda: image(JPEG, 7000, 6000), "6000x6000"),
        (lambda: InputImage(b"not an image", "application/octet-stream", 100, 100), "格式"),
        (lambda: InputImage(b"\xff\xd8\xff" + bytes(30_000_000), "image/jpeg", 100, 100), "30 MB"),
    ],
)
def test_edit_input_validation(make, text):
    error, calls = run({"api_key": KEY}, lambda r: ok(), edit_req(image(), make()))
    assert isinstance(error, DrawError) and text in str(error) and "第 2 张" in str(error) and calls == []


def test_edit_data_url_format_is_lowercase_and_falls_back_to_mime():
    # 无法按文件头识别时使用插件给出的 MIME（转成小写）
    heic = InputImage(b"\x00\x00\x00\x18ftypheic", "IMAGE/HEIC", 100, 100)
    _, calls = run({"api_key": KEY}, lambda r: ok(), edit_req(heic))
    assert body_of(calls[0])["image"] == data_url(heic.data, "heic")
    # 文件头优先于 MIME
    _, calls = run({"api_key": KEY}, lambda r: ok(), edit_req(InputImage(RGB, "image/jpeg", 100, 100)))
    assert body_of(calls[0])["image"] == data_url(RGB, "png")


def test_png_has_alpha():
    assert png_has_alpha(png(6)) and png_has_alpha(png(4))
    assert png_has_alpha(png(2, trns=True)) and png_has_alpha(png(3, trns=True))
    assert not png_has_alpha(png(2)) and not png_has_alpha(png(0)) and not png_has_alpha(png(3))
    assert not png_has_alpha(JPEG) and not png_has_alpha(PNG) and not png_has_alpha(png(6)[:20])


def test_keep_transparency_is_opt_in_and_checked():
    section = {"api_key": KEY, "keep_transparency": True, "output_format": "jpeg"}
    for model in (DEFAULT_MODEL, "doubao-seedream-5-0-flash-260915"):
        result, calls = run({**section, "model": model},
                            lambda r: ok({"b64_json": B64, "size": "2048x2048", "output_format": "png"}),
                            edit_req(image(RGBA), orientation="square"))
        body = body_of(calls[0])
        assert body["background"] == "transparent" and body["output_format"] == "png"
        assert_valid_edit(body, detect_family(model))
        assert result.params["background"] == "transparent" and result.params["output_format"] == "png"
    cases = [
        ({"keep_transparency": False}, [image(RGBA)]),
        ({"keep_transparency": "false"}, [image(RGBA)]),
        ({}, [image(RGB)]),  # 没有透明通道
        ({}, [image(JPEG)]),
        ({}, [image(RGBA), image(RGBA)]),  # 多张图片
        ({"model": "doubao-seedream-5-0-260128"}, [image(RGBA)]),  # 只有 5.0 pro / flash 支持
        ({"model": "doubao-seedream-4-0-250828"}, [image(RGBA)]),
    ]
    for extra, images in cases:
        result, calls = run({**section, **extra}, lambda r: ok(), edit_req(*images))
        body = body_of(calls[0])
        assert "background" not in body and body.get("output_format", "jpeg") == "jpeg"
        assert "background" not in result.params
    # 默认关闭
    _, calls = run({"api_key": KEY}, lambda r: ok(), edit_req(image(RGBA)))
    assert "background" not in body_of(calls[0])


@pytest.mark.parametrize(
    ("status", "code", "texts"),
    [
        (400, "InputImageSensitiveContentDetected.PrivacyInformation", ["拒绝修改", "真实人物", "换一张图片"]),
        (400, "InputImageSensitiveContentDetected.PolicyViolation", ["拒绝修改", "版权"]),
        (400, "InputImageSensitiveContentDetected", ["拒绝修改", "输入的图片未通过内容安全审核"]),
        (400, "InputImageRiskDetection", ["拒绝修改", "输入的图片未通过内容安全审核"]),
        (400, "InvalidImageURL.InvalidFormat", ["无法解析输入的图片"]),
        (400, "InvalidImageURL.EmptyURL", ["无法解析输入的图片"]),
        (400, "OutputImageSensitiveContentDetected.DeepFake", ["拒绝生成", "生成的图片未通过内容安全审核"]),
        (400, "InputTextSensitiveContentDetected", ["拒绝生成", "提示词未通过内容安全审核"]),
        (400, "InvalidParameter", ["请求参数有误", "msg"]),
        (413, "", ["图片数据过大"]),
        (401, "AuthenticationError", ["API Key 无效"]),
    ],
)
def test_edit_errors(status, code, texts):
    error, calls = run({"api_key": KEY}, lambda r: err(status, code, "msg"), edit_req())
    assert isinstance(error, DrawError) and error.status == status and not error.retryable
    assert all(text in str(error) for text in texts), str(error)
    assert code in str(error)
    if "拒绝" not in str(error):  # 审核拦截与文生图一致，只带错误码
        assert f"HTTP {status}" in str(error)
    assert len(calls) == 1


def test_edit_413_html_page():
    error, calls = run({"api_key": KEY}, lambda r: httpx.Response(
        413, text="<html><head><title>413 Request Entity Too Large</title></head></html>"), edit_req())
    assert isinstance(error, DrawError) and "图片数据过大" in str(error) and not error.retryable and len(calls) == 1


def test_edit_retry_rules(no_sleep):
    # 限流：请求没有开始生成，重试一次，请求体不变
    result, calls = run({"api_key": KEY}, [err(429, "ModelAccountIpmRateLimitExceeded", headers={"Retry-After": "1"}),
                                           ok()], edit_req())
    assert result.params["mode"] == "edit" and len(calls) == 2 and no_sleep == [1.0]
    assert body_of(calls[0]) == body_of(calls[1])
    # 额度用尽、500、超时、连接中断：可能已经计费或重试无用，不重试
    for handler in (lambda r: err(429, "SetLimitExceeded"), lambda r: err(500, "InternalServiceError"),
                    lambda r: httpx.ReadTimeout("timed out", request=r),
                    lambda r: httpx.RemoteProtocolError("disconnected", request=r)):
        error, calls = run({"api_key": KEY}, handler, edit_req())
        assert isinstance(error, DrawError) and not error.retryable and len(calls) == 1
    # 连接失败：请求没有发出，重试
    result, calls = run({"api_key": KEY}, [httpx.ConnectError("refused"), ok()], edit_req())
    assert result.image == PNG and len(calls) == 2


def test_edit_200_without_image_and_url_download():
    error, calls = run({"api_key": KEY}, lambda r: httpx.Response(
        200, json={"error": {"code": "InputImageSensitiveContentDetected", "message": "x"}}), edit_req())
    assert isinstance(error, DrawError) and "拒绝修改" in str(error)
    assert "InputImageSensitiveContentDetected" in str(error) and not error.retryable and len(calls) == 1
    image_url = "https://ark-content.tos-cn-beijing.volces.com/edit.jpeg?X-Tos-Signature=abc"

    def handler(request: httpx.Request):
        if request.method == "POST":
            return ok({"url": image_url, "size": "2048x2048"})
        assert str(request.url) == image_url
        return httpx.Response(200, content=JPEG)

    result, calls = run({"api_key": KEY}, handler, edit_req())
    assert [c.method for c in calls] == ["POST", "GET"] and "authorization" not in calls[1].headers
    assert result.image == JPEG and result.params["size"] == "2048x2048" and result.params["mode"] == "edit"


def test_edit_ignores_strength_and_seed():
    req = edit_req(strength=0.8, seed=42, negative="blurry")
    result, calls = run({"api_key": KEY, "optimize_mode": "fast"}, lambda r: ok(), req)
    body = body_of(calls[0])
    assert_valid_edit(body, "5.0-pro")
    assert body["prompt"].startswith(req.prompt) and "Avoid: blurry" in body["prompt"]
    assert body["optimize_prompt_options"] == {"mode": "fast"}
    assert "strength" not in result.params and result.params["optimize_mode"] == "fast"


def test_edit_byteplus_and_extra_body():
    section = {"api_key": KEY, "base_url": "https://ark.ap-southeast.bytepluses.com/api/v3",
               "extra_body": '{"watermark": true}'}
    _, calls = run(section, lambda r: ok(), edit_req())
    assert str(calls[0].url) == "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
    body = body_of(calls[0])
    assert body["model"] == "dola-seedream-5-0-pro-260628" and body["watermark"] is True
    assert body["image"] == data_url(JPEG, "jpeg")


def test_edit_key_echoed_by_relay_is_masked():
    secret = "sk-relay-secret-123456789"
    error, _ = run({"api_key": secret, "base_url": "https://relay.example.com/v1"}, lambda r: httpx.Response(
        400, json={"error": {"code": "InvalidParameter", "message": f"bad key {secret}"}}), edit_req())
    assert isinstance(error, DrawError) and secret not in str(error) and "6789" in str(error)


@pytest.mark.parametrize("model", ["doubao-seededit-3-0-i2i-250628", "doubao-seedream-3-0-t2i-250415", "Seedream 3.0"])
def test_edit_missing_for_models_that_cannot_edit(model):
    backend = SeedreamBackend({"api_key": KEY, "model": model})
    reason = backend.edit_missing()
    assert "不支持修改图片" in reason and model in reason
    error, calls = run({"api_key": KEY, "model": model}, lambda r: ok(), edit_req())
    assert isinstance(error, DrawError) and str(error) == reason and calls == []
    # 文生图不受影响；选择了模型系列时按所选系列处理
    _, calls = run({"api_key": KEY, "model": model}, lambda r: ok())
    assert len(calls) == 1
    assert SeedreamBackend({"model": model, "model_family": "4.0"}).edit_missing() == ""


def test_plan_for_edit():
    backend = SeedreamBackend({"api_key": KEY})
    req = edit_req(image(JPEG, 1920, 1080), orientation="landscape")
    plan = backend.plan(req)
    assert plan == {"model": DEFAULT_MODEL, "mode": "edit", "input_images": 1, "size": backend.edit_size(req)}
    assert size_ratio(plan["size"]) == pytest.approx(16 / 9, rel=0.02)
    assert backend.plan(DrawRequest("a cat")) == {"model": DEFAULT_MODEL, "size": "1664x2496"}


# ---------------------------------------------------------------- 余额查询：签名

# Access Key ID 不参与签名计算（只出现在 Credential 中），用占位值即可核对官方示例的签名
OFFICIAL_AK = "AK_DOC_EXAMPLE"
OFFICIAL_SK = "WkRZeE1EQmxPVGhsWWpWak5HVmtNbUUxTXpZeU9UVXlOMlE1TmpZeVlqTQ=="
OFFICIAL_NOW = datetime.datetime(2025, 3, 29, 18, 9, 37, tzinfo=datetime.timezone.utc)
BILLING_HOST = "billing.volcengineapi.com"


def test_signature_official_get_vector():
    """https://www.volcengine.com/docs/6369/67269 的 GET QueryBalanceAcct 示例，逐步核对。"""
    query = {"Action": "QueryBalanceAcct", "Version": "2022-01-01"}
    canonical = canonical_request("GET", BILLING_HOST, query, b"", "20250329T180937Z")
    assert canonical == ("GET\n/\nAction=QueryBalanceAcct&Version=2022-01-01\n"
                         "host:billing.volcengineapi.com\nx-date:20250329T180937Z\n\nhost;x-date\n"
                         "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    assert hashlib.sha256(canonical.encode()).hexdigest() == \
        "43171c1658c64b5db55c58d54988a4598d2d09a5613136beaa5eef40eae6e2c1"
    key = _hmac_sha256(OFFICIAL_SK.encode(), "20250329")
    assert key.hex() == "069b1da2ba9c0ecbd8e8aaf2a5742696ebc22f3fe95a649983d31b433ba94ff3"
    for part in ("cn-beijing", "billing", "request"):
        key = _hmac_sha256(key, part)
    assert key.hex() == "b491ed164936de3bb06c1eb23326aa9587b5aaa6a4e02144b9d523bbebb7ca9f"
    headers = sign_v4("GET", BILLING_HOST, query, b"", OFFICIAL_AK, OFFICIAL_SK, "cn-beijing", "billing",
                      now=OFFICIAL_NOW)
    assert headers == {
        "X-Date": "20250329T180937Z",
        "Authorization": "HMAC-SHA256 Credential=AK_DOC_EXAMPLE/20250329/"
                         "cn-beijing/billing/request, SignedHeaders=host;x-date, "
                         "Signature=1eda9e7e6b1728151a8e8791fdaf67cfbd28bd5c80d0fce2eb208746cf483105",
    }


def test_signature_official_post_vector():
    body = b'{"Limit":10,"BillPeriod":"2023-08"}'
    assert hashlib.sha256(body).hexdigest() == "e8cc56e129d9759d56c936e679a345d001a4235b58bee8e935ccad97f23ed663"
    query = {"Action": "ListBill", "Version": "2022-01-01"}
    canonical = canonical_request("POST", BILLING_HOST, query, body, "20250329T180937Z")
    assert hashlib.sha256(canonical.encode()).hexdigest() == \
        "27383e3b56d03850f5634483527fbddcbf06cf98de1bc8a6679ef2300bff3b15"
    headers = sign_v4("POST", BILLING_HOST, query, body, OFFICIAL_AK, OFFICIAL_SK, "cn-beijing", "billing",
                      now=OFFICIAL_NOW)
    assert headers["Authorization"].endswith(
        "Signature=5e8480ceea12d0000a23c054151c50dd02c1a7dec835004057d19f13d53a7658")


def test_canonical_query_encoding():
    assert canonical_query({"Version": "2022-01-01", "Action": "QueryBalanceAcct"}) == \
        "Action=QueryBalanceAcct&Version=2022-01-01"
    # RFC 3986：空格为 %20，只保留 -_.~，按名称的 ASCII 顺序（大写在前）
    assert canonical_query({"b": "a b", "a": "x/y*z~-_.+", "C": "中"}) == \
        "C=%E4%B8%AD&a=x%2Fy%2Az~-_.%2B&b=a%20b"


# ---------------------------------------------------------------- 余额查询：请求与结果

AK = "AKLTtestaccesskey000001"
SK = "c2VjcmV0LXNlY3JldC1rZXktMDAwMQ=="
BALANCE_OK = {
    "ResponseMetadata": {"RequestId": "r", "Action": "QueryBalanceAcct", "Version": "2022-01-01", "Service": "billing"},
    "Result": {"AccountID": 2100000001, "ArrearsBalance": "0", "AvailableBalance": "77.01",
               "CashBalance": "1083.01", "CreditLimit": "0.00", "FreezeAmount": "5.01", "Currency": "CNY"},
}
CREDS = {"access_key_id": AK, "secret_access_key": SK}


def run_balance(section, handler):
    """用 MockTransport 调用 balance，返回 (结果或 DrawError, 收到的请求列表)。"""
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
            return await SeedreamBackend(section).balance(Clients(http=pool))
        except DrawError as e:
            return e
        finally:
            await pool.close()

    return asyncio.run(main()), calls


def billing_err(status: int, code: str, message: str = "m") -> httpx.Response:
    meta = {"RequestId": "r", "Action": "QueryBalanceAcct", "Version": "2022-01-01", "Service": "billing",
            "Error": {"CodeN": 100009, "Code": code, "Message": message}}
    return httpx.Response(status, json={"ResponseMetadata": meta})


def assert_signed(request: httpx.Request, host: str, region: str):
    """请求的签名与按请求本身（方法、Host、查询参数、X-Date、Body）重新计算的结果一致。"""
    assert request.url.host == host and request.headers["host"] == host
    x_date = request.headers["x-date"]
    signed_at = datetime.datetime.strptime(x_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=datetime.timezone.utc)
    now = datetime.datetime.now(datetime.timezone.utc)
    assert abs((now - signed_at).total_seconds()) < 300  # 使用 UTC 时间
    expected = sign_v4(request.method, host, dict(request.url.params), request.content, AK, SK, region, "billing",
                       now=signed_at)
    assert request.headers["authorization"] == expected["Authorization"]
    assert request.headers["authorization"].startswith(
        f"HMAC-SHA256 Credential={AK}/{x_date[:8]}/{region}/billing/request, SignedHeaders=host;x-date, Signature=")
    assert SK not in str(request.headers) and SK not in str(request.url)


def test_balance_volcengine_request_and_result():
    result, calls = run_balance({"api_key": KEY, **CREDS}, lambda r: httpx.Response(200, json=BALANCE_OK))
    assert isinstance(result, BalanceInfo) and len(calls) == 1
    request = calls[0]
    assert request.method == "GET" and request.content == b""
    assert str(request.url) == "https://billing.volcengineapi.com/?Action=QueryBalanceAcct&Version=2022-01-01"
    assert_signed(request, BILLING_HOST, "cn-beijing")
    assert KEY not in str(request.headers)  # 方舟 API Key 不发送给费用中心
    assert result.items == [("可用余额", "77.01 元"), ("现金余额", "1,083.01 元"), ("冻结金额", "5.01 元")]
    assert result.note.startswith("火山引擎主账号的资金余额") and "不是单个 API Key 的额度" in result.note
    text = json.dumps([result.items, result.note], ensure_ascii=False)
    assert AK not in text and SK not in text


def test_balance_shows_credit_and_arrears_when_nonzero():
    payload = json.loads(json.dumps(BALANCE_OK))
    payload["Result"].update(CreditLimit="100", ArrearsBalance="3.5", FreezeAmount="0.00")
    del payload["Result"]["Currency"]  # 火山引擎没有返回币种时按人民币
    result, _ = run_balance(CREDS, lambda r: httpx.Response(200, json=payload))
    assert result.items == [("可用余额", "77.01 元"), ("现金余额", "1,083.01 元"), ("信控额度", "100.00 元"),
                            ("欠费金额", "3.50 元")]


def test_balance_byteplus_host_region_and_currency():
    section = {"base_url": "https://ark.ap-southeast.bytepluses.com/api/v3", **CREDS}
    payload = {"ResponseMetadata": {}, "Result": {"AvailableBalance": "12.5", "CashBalance": "12.5"}}
    result, calls = run_balance(section, lambda r: httpx.Response(200, json=payload))
    assert str(calls[0].url) == "https://billing.byteplusapi.com/?Action=QueryBalanceAcct&Version=2022-01-01"
    assert_signed(calls[0], "billing.byteplusapi.com", "ap-singapore-1")
    # BytePlus 不返回币种，不假定为美元
    assert result.items == [("可用余额", "12.50"), ("现金余额", "12.50")]
    assert result.note.startswith("BytePlus 主账号") and "结算币种" in result.note
    payload["Result"]["Currency"] = "USD"
    result, calls = run_balance({**section, "billing_region": " ap-southeast-1 "},
                                lambda r: httpx.Response(200, json=payload))
    assert_signed(calls[0], "billing.byteplusapi.com", "ap-southeast-1")
    assert result.items[0] == ("可用余额", "12.50 美元") and "结算币种" not in result.note


def test_balance_platform_override_for_relay_address():
    section = {"api_key": KEY, "base_url": "https://relay.example.com/v1", "billing_platform": "volcengine", **CREDS}
    result, calls = run_balance(section, lambda r: httpx.Response(200, json=BALANCE_OK))
    assert calls[0].url.host == BILLING_HOST and result.items[0] == ("可用余额", "77.01 元")
    result, calls = run_balance({**section, "billing_platform": "byteplus"},
                                lambda r: httpx.Response(200, json=BALANCE_OK))
    assert_signed(calls[0], "billing.byteplusapi.com", "ap-singapore-1")
    assert result.items[0] == ("可用余额", "77.01 元")  # 返回了币种时按币种显示


@pytest.mark.parametrize(
    ("status", "code", "text", "retried"),
    [
        (401, "InvalidAccessKey", "Access Key ID 不存在或已停用", False),
        (401, "SignatureDoesNotMatch", "Secret Access Key", False),
        (403, "AccessDenied", "BillingCenterReadOnlyAccess", False),
        (400, "InvalidTimestamp", "服务器时间", False),
        (404, "InvalidActionOrVersion", "查询失败", False),
        (500, "InternalError", "服务器错误", False),
        (429, "FlowLimitExceeded", "查询过于频繁", True),
        (503, "ServiceUnavailableTemp", "暂时不可用", True),
        (504, "InternalServiceTimeout", "响应超时", True),
    ],
)
def test_balance_errors_never_show_message(status, code, text, retried, no_sleep):
    """Error.Message 会原样带出 Access Key，只展示错误码对应的说明。"""
    message = f"The accesskey [{AK}] included in the request is invalid. {SK}"
    error, calls = run_balance(CREDS, lambda r: billing_err(status, code, message))
    assert isinstance(error, DrawError) and error.status == status
    assert text in str(error) and code in str(error) and f"HTTP {status}" in str(error)
    assert AK not in str(error) and SK not in str(error) and "accesskey" not in str(error)
    assert error.retryable is retried and len(calls) == (2 if retried else 1)


def test_balance_retries_resign_each_attempt(no_sleep, monkeypatch):
    from backends import seedream

    signed = []
    real_sign = seedream.sign_v4

    def counting_sign(*args, **kwargs):
        signed.append(args)
        return real_sign(*args, **kwargs)

    monkeypatch.setattr(seedream, "sign_v4", counting_sign)
    result, calls = run_balance(CREDS, [billing_err(429, "FlowLimitExceeded"), httpx.Response(200, json=BALANCE_OK)])
    assert isinstance(result, BalanceInfo) and len(calls) == 2 and len(signed) == 2 and no_sleep == [3.0]
    for request in calls:
        assert_signed(request, BILLING_HOST, "cn-beijing")
    # 连接失败重试；超时不重试
    result, calls = run_balance(CREDS, [httpx.ConnectError("refused"), httpx.Response(200, json=BALANCE_OK)])
    assert isinstance(result, BalanceInfo) and len(calls) == 2
    error, calls = run_balance(CREDS, lambda r: httpx.ReadTimeout("timed out", request=r))
    assert isinstance(error, DrawError) and "超时" in str(error) and len(calls) == 1


@pytest.mark.parametrize(
    ("make", "text"),
    [
        (lambda: httpx.Response(200, json={"ResponseMetadata": {"Error": {"Code": "AccessDenied", "Message": AK}}}),
         "BillingCenterReadOnlyAccess"),
        (lambda: httpx.Response(200, json={"ResponseMetadata": {}, "Result": {}}), "格式异常"),
        # 可用余额为 null 或空串时不能显示成 0
        (lambda: httpx.Response(200, json={"Result": {"AvailableBalance": None, "CashBalance": "1"}}), "格式异常"),
        (lambda: httpx.Response(200, json={"Result": {"AvailableBalance": " ", "CashBalance": "1"}}), "格式异常"),
        (lambda: httpx.Response(200, json={"ResponseMetadata": {}}), "格式异常"),
        (lambda: httpx.Response(200, json=[1, 2]), "格式异常"),
        (lambda: httpx.Response(200, text=f"<p>invalid key {AK}</p>"), "不是 JSON"),
    ],
)
def test_balance_bad_success_responses(make, text):
    error, calls = run_balance(CREDS, lambda r: make())
    assert isinstance(error, DrawError) and text in str(error) and len(calls) == 1
    assert AK not in str(error) and SK not in str(error)


def test_balance_missing():
    reason = SeedreamBackend({"api_key": KEY}).balance_missing()
    assert "Access Key ID" in reason and "Secret Access Key" in reason and "BillingCenterReadOnlyAccess" in reason
    assert BILLING["volcengine"]["console"] in reason
    byteplus = SeedreamBackend({"base_url": "https://ark.ap-southeast.bytepluses.com/api/v3", "access_key_id": AK})
    assert BILLING["byteplus"]["console"] in byteplus.balance_missing()
    relay = {"api_key": KEY, "base_url": "https://relay.example.com/v1", **CREDS}
    assert "官方地址" in SeedreamBackend(relay).balance_missing()
    assert "官方地址" in SeedreamBackend({**relay, "billing_platform": "auto"}).balance_missing()
    assert SeedreamBackend({**relay, "billing_platform": "volcengine"}).balance_missing() == ""
    assert SeedreamBackend({**relay, "billing_platform": "BytePlus"}).balance_missing() == ""
    # 余额查询不需要方舟 API Key
    assert SeedreamBackend(CREDS).balance_missing() == ""
    # 空白被去掉；中文、全角或不可见字符报错，且不回显密钥
    assert SeedreamBackend({"access_key_id": f" {AK}\n", "secret_access_key": f"{SK} "}).access_key == (AK, SK)
    for bad in ({"access_key_id": "AKLT密钥0001"}, {"secret_access_key": "c2Vj\u200bcmV0"}):
        reason = SeedreamBackend({**CREDS, **bad}).balance_missing()
        assert "不可见字符" in reason and SK not in reason and "c2Vj" not in reason


def test_balance_without_credentials_makes_no_request():
    section = {"api_key": KEY}
    error, calls = run_balance(section, lambda r: httpx.Response(200, json=BALANCE_OK))
    assert isinstance(error, DrawError) and str(error) == SeedreamBackend(section).balance_missing() and calls == []


def test_balance_region_is_normalised_and_validated():
    # 大写和空白：转成小写后参与签名（地域 ID 都是小写）
    result, calls = run_balance({**CREDS, "billing_region": " CN-Shanghai "},
                                lambda r: httpx.Response(200, json=BALANCE_OK))
    assert isinstance(result, BalanceInfo)
    assert_signed(calls[0], BILLING_HOST, "cn-shanghai")
    assert SeedreamBackend(CREDS).billing_region == "cn-beijing"
    assert SeedreamBackend({**CREDS, "billing_platform": "byteplus"}).billing_region == "ap-singapore-1"
    # 中文、全角或其他字符会让请求头非法（报错还会误导为 API Key 有误）：不发送请求，给出明确原因
    for bad in ("华北2", "cn_beijing", "cn-beijing/x", "ｃｎ-beijing", "a" * 65):
        section = {**CREDS, "billing_region": bad}
        reason = SeedreamBackend(section).balance_missing()
        assert "签名地域" in reason and SK not in reason
        error, calls = run_balance(section, lambda r: httpx.Response(200, json=BALANCE_OK))
        assert isinstance(error, DrawError) and str(error) == reason and calls == []


def test_balance_missing_cash_balance_is_not_shown_as_zero():
    payload = {"ResponseMetadata": {}, "Result": {"AvailableBalance": "8", "Currency": "CNY"}}
    result, _ = run_balance(CREDS, lambda r: httpx.Response(200, json=payload))
    assert result.items == [("可用余额", "8.00 元")]
    payload["Result"]["AvailableBalance"] = 0  # 数字 0 是合法的余额
    result, _ = run_balance(CREDS, lambda r: httpx.Response(200, json=payload))
    assert result.items == [("可用余额", "0.00 元")]


@pytest.mark.parametrize("data", [b"", None])
def test_edit_empty_image_data_is_rejected_locally(data):
    error, calls = run({"api_key": KEY}, lambda r: ok(), edit_req(image(), InputImage(data, "image/png", 100, 100)))
    assert isinstance(error, DrawError) and "第 2 张图片没有数据" in str(error) and calls == []
