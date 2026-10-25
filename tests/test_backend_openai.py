"""OpenAI 后端的单元测试（不访问网络）：python -m pytest tests/test_backend_openai.py"""

import asyncio
import base64
import email.parser
import email.policy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

from backends import base  # noqa: E402
from backends import openai as oai  # noqa: E402
from backends.base import BalanceInfo, Clients, DrawError, DrawRequest, HttpPool, InputImage  # noqa: E402
from backends.openai import OpenAIBackend  # noqa: E402


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
B64 = base64.b64encode(PNG).decode()
KEY = "sk-test-key-123456"
GEN_URL = "https://api.openai.com/v1/images/generations"


def ok_json(**extra):
    return httpx.Response(200, json={"created": 1, "data": [{"b64_json": B64}], **extra})


class Recorder:
    """MockTransport 的处理函数：按顺序返回预设响应（或抛出异常），并记录收到的请求。"""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request):
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if callable(item):
            item = item(request)
        if isinstance(item, Exception):
            raise item
        return item

    def body(self, index: int = 0) -> dict:
        return json.loads(self.requests[index].content)


def run(section, rec, req=None, *, action="generate"):
    async def main():
        pool = HttpPool(transport=httpx.MockTransport(rec))
        try:
            backend = OpenAIBackend(section)
            if action == "models":
                return await backend.list_models(Clients(http=pool))
            if action == "balance":
                return await backend.balance(Clients(http=pool))
            return await backend.generate(Clients(http=pool), req or DrawRequest("a cat"))
        finally:
            await pool.close()

    return asyncio.run(main())


@pytest.fixture
def sleeps(monkeypatch):
    waits = []

    async def fake_sleep(delay, *args, **kwargs):
        waits.append(delay)

    monkeypatch.setattr(base.asyncio, "sleep", fake_sleep)
    return waits


# ---------------------------------------------------------------- 请求


def test_default_request():
    rec = Recorder(ok_json(size="1024x1536", quality="medium", output_format="png",
                           usage={"total_tokens": 1234}))
    result = run({"api_key": KEY}, rec, DrawRequest("a cat", negative="text, watermark", seed=42))
    request = rec.requests[0]
    assert request.method == "POST" and str(request.url) == GEN_URL
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["content-type"] == "application/json"
    assert rec.body() == {
        "model": "gpt-image-2.5-flare",
        "prompt": "a cat\n\nAvoid: text, watermark",
        "n": 1,
        "size": "1024x1536",
        "quality": "medium",
    }
    assert result.image == PNG and result.mime == "image/png" and result.text == ""
    assert result.params == {"model": "gpt-image-2.5-flare", "size": "1024x1536", "quality": "medium",
                             "output_format": "png", "tokens": 1234}


@pytest.mark.parametrize(("orientation", "size"), [
    ("portrait", "1024x1536"), ("landscape", "1536x1024"), ("square", "1024x1024"), ("bogus", "1024x1536"),
])
def test_orientation_default_sizes(orientation, size):
    rec = Recorder(ok_json())
    result = run({"api_key": KEY}, rec, DrawRequest("x", orientation=orientation))
    assert rec.body()["size"] == size and result.params["size"] == size


@pytest.mark.parametrize(("model", "family"), [
    ("gpt-image-2.5-flare", "gpt25"), ("gpt-image-2.5-sunburst-2026-09-08", "gpt25"), ("GPT-Image-2", "gpt2"),
    ("gpt-image-2-2026-04-21", "gpt2"), ("openai/gpt-image-2", "gpt2"), ("gpt-image-1.5", "gpt"),
    ("gpt-image-1-mini", "gpt"), ("chatgpt-image-latest", "gpt"), ("dall-e-3", "dalle3"), ("dall-e-2", "dalle2"),
    ("flux-1.1-pro", "other"), ("", "other"),
])
def test_model_family(model, family):
    assert oai.model_family(model) == family


@pytest.mark.parametrize(("model", "orientation", "text", "expected"), [
    # gpt-image-2 / 2.5：任意尺寸，吸附到合法值
    ("gpt-image-2.5-flare", "landscape", "1920x1080", "1936x1088"),
    ("gpt-image-2", "square", "512x512", "816x816"),
    ("openai/gpt-image-2", "portrait", "1000 X 1500", "992x1488"),  # 保持 2:3
    ("gpt-image-2.5-flare", "portrait", "2K", "1152x2048"),
    ("gpt-image-2.5-flare", "landscape", "4k", "3840x2160"),
    ("gpt-image-2.5-flare", "square", "4K", "2880x2880"),
    ("gpt-image-2.5-flare", "portrait", "3:4", "1152x1536"),
    ("gpt-image-2.5-flare", "landscape", "16：9", "1536x864"),
    ("gpt-image-2.5-flare", "portrait", "auto", "auto"),
    ("gpt-image-2.5-flare", "portrait", "garbage", "1024x1536"),
    ("gpt-image-2.5-flare", "portrait", "", "1024x1536"),
    ("gpt-image-2.5-flare", "portrait", "99999999999999999999x1", "1024x1536"),
    # 旧 GPT Image 模型：只有三种标准尺寸
    ("gpt-image-1.5", "landscape", "1920x1080", "1536x1024"),
    ("chatgpt-image-latest", "portrait", "2K", "1024x1536"),
    ("gpt-image-1-mini", "square", "2048x2048", "1024x1024"),
    ("gpt-image-1.5", "portrait", "auto", "auto"),
    # 中转服务上的其他模型
    ("dall-e-3", "portrait", "1024x1536", "1024x1792"),
    ("dall-e-3", "landscape", "2K", "1792x1024"),
    ("dall-e-2", "square", "512x512", "512x512"),
    ("dall-e-2", "portrait", "1024x1536", "1024x1024"),
    ("dall-e-3", "portrait", "auto", ""),
    ("flux-pro", "square", "1000x1000", "1000x1000"),
    ("flux-pro", "portrait", "2K", "2K"),
    ("flux-pro", "portrait", "9:16", "864x1536"),
    ("flux-pro", "portrait", "auto", ""),
])
def test_resolve_size(model, orientation, text, expected):
    assert oai.resolve_size(model, orientation, text) == expected


@pytest.mark.parametrize(("width", "height"), [
    (1, 1), (100, 5000), (5000, 100), (3840, 3840), (16, 16), (7680, 4320), (1000, 1000), (1920, 1080),
    (4000, 1000), (1408, 464),
])
def test_snap_free_size_always_valid(width, height):
    w, h = map(int, oai.snap_free_size(width, height).split("x"))
    assert w % 16 == 0 and h % 16 == 0 and max(w, h) <= 3840 and max(w, h) <= 3 * min(w, h)
    assert 655_360 <= w * h <= 8_294_400


@pytest.mark.parametrize(("model", "section", "expected", "absent"), [
    ("gpt-image-2.5-sunburst", {"quality": "max", "output_format": "jpg", "moderation": "low",
                                "background": "opaque", "response_format": "url"},
     {"quality": "max", "output_format": "jpeg", "moderation": "low", "background": "opaque"},
     ("response_format", "style")),
    ("gpt-image-2.5-flare", {"quality": "xhigh"}, {"quality": "xhigh"}, ("response_format",)),
    # xhigh / max 只有 2.5 支持，其他 GPT Image 模型降为 high
    ("gpt-image-2", {"quality": "xhigh", "output_format": "webp"}, {"quality": "high", "output_format": "webp"},
     ("response_format",)),
    ("gpt-image-1.5", {"quality": "max", "output_format": "png", "moderation": "auto", "background": "auto"},
     {"quality": "high"}, ("output_format", "moderation", "background", "response_format")),
    ("chatgpt-image-latest", {"quality": "hd", "response_format": "b64_json"}, {},
     ("quality", "response_format")),
    ("gpt-image-2.5-flare", {"quality": "auto", "output_format": "gif"}, {},
     ("quality", "output_format")),
    ("dall-e-3", {"quality": "high", "moderation": "low"}, {"quality": "hd", "response_format": "b64_json",
                                                             "size": "1024x1792"},
     ("moderation", "output_format")),
    ("dall-e-2", {"quality": "high"}, {"response_format": "b64_json", "size": "1024x1024"}, ("quality",)),
    # 其他模型默认不发送 response_format（可能是改名的 GPT Image 模型，带上会 400）
    ("openai/my-image-alias", {"quality": "high", "output_format": "jpeg"}, {},
     ("quality", "output_format", "moderation", "background", "response_format")),
    ("flux-pro", {"response_format": "b64_json"}, {"response_format": "b64_json"}, ()),
    ("flux-pro", {"response_format": "url"}, {"response_format": "url"}, ()),
    ("flux-pro", {"response_format": "none"}, {}, ("response_format",)),
])
def test_body_per_model_family(model, section, expected, absent):
    rec = Recorder(ok_json())
    run({"api_key": KEY, "model": model, **section}, rec)
    body = rec.body()
    assert body["model"] == model and body["n"] == 1
    for key, value in expected.items():
        assert body[key] == value, key
    for key in absent:
        assert key not in body, key
    assert not {"seed", "negative_prompt", "style"} & set(body)


# ---------------------------------------------------------------- API 地址


@pytest.mark.parametrize(("url", "expected"), [
    ("", "https://api.openai.com/v1"),
    (None, "https://api.openai.com/v1"),
    ("https://api.openai.com", "https://api.openai.com/v1"),
    ("api.openai.com/v1/", "https://api.openai.com/v1"),
    ("https://relay.example.com/", "https://relay.example.com/v1"),
    ("https://relay.example.com/v1/chat/completions", "https://relay.example.com/v1"),
    ("https://relay.example.com/v1/images/generations", "https://relay.example.com/v1"),
    ("https://relay.example.com/v1/images", "https://relay.example.com/v1"),
    ("https://relay.example.com/v1/models/", "https://relay.example.com/v1"),
    ("https://relay.example.com/chat/completions", "https://relay.example.com/v1"),
    ("https://relay.example.com/api/v1", "https://relay.example.com/api/v1"),
    ("http://127.0.0.1:3000", "http://127.0.0.1:3000/v1"),
])
def test_api_base(url, expected):
    assert oai.api_base(url) == expected


@pytest.mark.parametrize("url", ["ftp://relay.example.com/v1", "ws://relay.example.com"])
def test_bad_base_url_rejected(url):
    backend = OpenAIBackend({"api_key": KEY, "base_url": url})
    assert "http://" in backend.missing() and "https://" in backend.missing()
    rec = Recorder(ok_json())
    with pytest.raises(DrawError, match="API 地址无效"):
        run({"api_key": KEY, "base_url": url}, rec)
    assert not rec.requests


@pytest.mark.parametrize("url", ["https://[relay.example.com/v1", "https://relay.example.com:abc/v1",
                                 "https://relay.example.com:99999"])
def test_malformed_base_url_reported_not_raised(url):
    # urlsplit / httpx 对这些地址抛出 ValueError / InvalidURL，必须转换成 DrawError
    assert "API 地址无效" in OpenAIBackend({"api_key": KEY, "base_url": url}).missing()
    rec = Recorder(ok_json())
    with pytest.raises(DrawError, match="API 地址无效"):
        run({"api_key": KEY, "base_url": url}, rec)
    assert not rec.requests


def test_redirect_not_followed(sleeps):
    # 跟随 301 会把 POST 变成 GET；应直接提示检查地址，且只发出一次请求
    location = "https://user:pw@relay.example.com/v1/images/generations"
    rec = Recorder(httpx.Response(301, headers={"location": location}), ok_json())
    match = r"跳转（HTTP 301 → https://\*\*\*@relay\.example\.com/v1/images/generations）.*https://"
    with pytest.raises(DrawError, match=match) as info:
        run({"api_key": KEY, "base_url": "http://relay.example.com/v1"}, rec)
    assert not info.value.retryable and len(rec.requests) == 1 and rec.requests[0].method == "POST"
    assert "pw" not in str(info.value) and not sleeps


def test_relay_url_used_for_request():
    rec = Recorder(ok_json())
    run({"api_key": KEY, "base_url": "relay.example.com/v1/chat/completions"}, rec)
    assert str(rec.requests[0].url) == "https://relay.example.com/v1/images/generations"


# ---------------------------------------------------------------- 响应


def test_url_response_downloaded_without_auth():
    def handler(request):
        if request.url.host == "cdn.example.com":
            return httpx.Response(200, content=JPEG, headers={"content-type": "image/jpeg"})
        return httpx.Response(200, json={"data": [{"url": "https://cdn.example.com/a.jpg",
                                                   "revised_prompt": "a fluffy cat"}]})

    rec = Recorder(handler)
    result = run({"api_key": KEY, "model": "dall-e-3"}, rec)
    assert result.image == JPEG and result.mime == "image/jpeg" and result.text == "a fluffy cat"
    assert len(rec.requests) == 2 and rec.requests[1].method == "GET"
    assert "authorization" not in rec.requests[1].headers
    assert result.params["response_format"] == "b64_json"


@pytest.mark.parametrize("item", [
    {"url": f"data:image/png;base64,{B64}"},
    {"b64_json": f"data:image/png;base64,{B64}"},
    {"b64Json": B64},
    {"b64_json": "", "url": f"data:image/png;base64,{B64}"},
])
def test_relay_response_variants(item):
    rec = Recorder(httpx.Response(200, json={"data": [item]}))
    assert run({"api_key": KEY, "model": "flux-pro"}, rec).image == PNG


def test_raw_image_body():
    rec = Recorder(httpx.Response(200, content=PNG, headers={"content-type": "image/png"}))
    assert run({"api_key": KEY}, rec).image == PNG


def test_raw_body_not_image():
    rec = Recorder(httpx.Response(200, content=b"nope", headers={"content-type": "image/png"}))
    with pytest.raises(DrawError, match="不是可识别的图片"):
        run({"api_key": KEY}, rec)


def test_error_with_http_200(sleeps):
    rec = Recorder(httpx.Response(200, json={"error": {"message": "upstream fail", "type": "new_api_error"}}))
    with pytest.raises(DrawError, match="upstream fail") as info:
        run({"api_key": KEY}, rec)
    assert not info.value.retryable and len(rec.requests) == 1 and not sleeps


@pytest.mark.parametrize(("body", "match"), [
    ({"data": []}, "没有图片"),
    ({"created": 1}, "没有图片"),
    ({"data": [{"error": {"message": "item failed"}}]}, "item failed"),
    ({"data": [{"b64_json": base64.b64encode(b"not an image").decode()}]}, "不是可识别的图片"),
])
def test_no_image_in_response(body, match):
    with pytest.raises(DrawError, match=match):
        run({"api_key": KEY}, Recorder(httpx.Response(200, json=body)))


def test_html_body_suggests_checking_url():
    rec = Recorder(httpx.Response(200, text="<html><title>Relay Panel</title></html>",
                                  headers={"content-type": "text/html"}))
    with pytest.raises(DrawError, match="不是 JSON.*Relay Panel.*请检查 API 地址"):
        run({"api_key": KEY}, rec)


# ---------------------------------------------------------------- 错误与重试


@pytest.mark.parametrize(("details", "match"), [
    ({"moderation_stage": "input", "categories": ["violence"]}, "提示词被 OpenAI 的安全系统拦截了（类别：violence）"),
    ({"moderation_stage": "output", "categories": []}, "生成的图片没有通过 OpenAI 的安全审核"),
    (None, "这次绘图被 OpenAI 的安全系统拦截了"),
])
def test_moderation_blocked(details, match, sleeps):
    err = {"type": "image_generation_user_error", "code": "moderation_blocked"}  # 官方示例没有 message
    if details is not None:
        err["moderation_details"] = details
    rec = Recorder(httpx.Response(400, json={"error": err}))
    with pytest.raises(DrawError, match=match) as info:
        run({"api_key": KEY}, rec)
    assert "换一种描述" in str(info.value) and info.value.status == 400 and not info.value.retryable
    assert len(rec.requests) == 1 and not sleeps


def test_legacy_content_policy_violation():
    rec = Recorder(httpx.Response(400, json={"error": {"code": "content_policy_violation",
                                                       "message": "Your request was rejected by our safety system"}}))
    with pytest.raises(DrawError, match="安全系统拦截"):
        run({"api_key": KEY, "model": "dall-e-3"}, rec)


def test_401_not_retried(sleeps):
    rec = Recorder(httpx.Response(401, json={"error": {"message": "Incorrect API key provided",
                                                       "type": "invalid_request_error", "code": "invalid_api_key"}}))
    with pytest.raises(DrawError, match="API Key 无效") as info:
        run({"api_key": KEY}, rec)
    assert info.value.status == 401 and "Incorrect API key" in str(info.value)
    assert len(rec.requests) == 1 and not sleeps


def test_api_key_echoed_by_relay_is_masked():
    secret = "sk-relay-secret-ABCDEFGH"
    rec = Recorder(httpx.Response(401, json={"error": {"message": f"invalid token {secret}"}}))
    with pytest.raises(DrawError, match="API Key 无效") as info:
        run({"api_key": secret}, rec)
    assert secret not in str(info.value) and "EFGH" in str(info.value)
    with pytest.raises(DrawError) as info:
        run({"api_key": secret}, Recorder(httpx.Response(401, json={"error": {"message": secret}})),
            action="models")
    assert secret not in str(info.value)
    # 中转服务以 HTTP 200 返回的错误、非 JSON 响应同样打码
    for resp in (httpx.Response(200, json={"error": {"message": f"bad token {secret}"}}),
                 httpx.Response(200, text=f"bad token {secret}")):
        with pytest.raises(DrawError) as info:
            run({"api_key": secret}, Recorder(resp))
        assert secret not in str(info.value) and "bad token" in str(info.value)


def test_api_key_bearer_prefix_stripped():
    rec = Recorder(ok_json())
    run({"api_key": f"Bearer {KEY}"}, rec)
    assert rec.requests[0].headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("key", ["sk-abc\ndef-123456", "sk-abc​def", "sk-全角密钥"])
def test_api_key_with_bad_chars_rejected(key):
    # 换行会让 httpx 报错时带出完整的 Authorization 头，全角 / 不可见字符会抛出 UnicodeEncodeError
    assert "重新复制" in OpenAIBackend({"api_key": key}).missing()
    rec = Recorder(ok_json())
    with pytest.raises(DrawError, match="重新复制") as info:
        run({"api_key": key}, rec)
    assert key not in str(info.value) and not rec.requests


def test_403_unverified_org_hint():
    rec = Recorder(httpx.Response(403, json={"error": {"message": "Your organization must be verified to use "
                                                                  "the model `gpt-image-2.5-flare`."}}))
    with pytest.raises(DrawError, match="组织验证"):
        run({"api_key": KEY}, rec)


@pytest.mark.parametrize("err", [
    {"code": "insufficient_quota", "type": "insufficient_quota", "message": "You exceeded your current quota"},
    {"code": "credit_balance_exhausted", "type": "insufficient_quota"},
    {"code": "some_new_billing_code", "type": "insufficient_quota"},
    {"code": "project_spend_limit_exceeded", "type": "requests"},
    {"message": "You exceeded your current quota, please check your plan and billing details."},
    # 中转服务的写法
    {"message": "用户额度不足, 剩余额度: $0.01", "type": "new_api_error", "code": "insufficient_user_quota"},
    "该令牌余额不足",
])
def test_billing_429_not_retried(err, sleeps):
    rec = Recorder(httpx.Response(429, json={"error": err}, headers={"retry-after": "1"}))
    with pytest.raises(DrawError, match="余额不足") as info:
        run({"api_key": KEY}, rec)
    assert info.value.status == 429 and not info.value.retryable
    assert len(rec.requests) == 1 and not sleeps


def test_rate_limit_429_retried_once(sleeps):
    # OpenAI 普通限流的说明里也带有 account/billing 链接，不能当成账单问题
    message = ("Rate limit reached for gpt-image-2.5-flare in organization org-x on images per minute (IPM): "
               "Limit 20, Used 20, Requested 1. Please try again in 2s. You can increase your rate limit by "
               "adding a payment method to your account at https://platform.openai.com/account/billing.")
    limited = httpx.Response(429, json={"error": {"message": message, "type": "requests",
                                                  "code": "rate_limit_exceeded"}},
                             headers={"retry-after": "2"})
    rec = Recorder(limited, ok_json())
    result = run({"api_key": KEY}, rec)
    assert result.image == PNG and len(rec.requests) == 2 and sleeps == [2.0]


def test_rate_limit_429_gives_up_after_second_attempt(sleeps):
    rec = Recorder(httpx.Response(429, json={"error": {"type": "requests", "code": "rate_limit_exceeded"}}))
    with pytest.raises(DrawError, match="429") as info:
        run({"api_key": KEY}, rec)
    assert info.value.retryable and len(rec.requests) == 2 and sleeps == [3.0]


def test_503_retried_once(sleeps):
    rec = Recorder(httpx.Response(503, json={"error": {"type": "service_unavailable_error",
                                                       "code": "server_is_overloaded"}}), ok_json())
    assert run({"api_key": KEY}, rec).image == PNG and len(rec.requests) == 2 and len(sleeps) == 1


@pytest.mark.parametrize("status", [500, 502, 504, 524])
def test_other_5xx_not_retried(status, sleeps):
    rec = Recorder(httpx.Response(status, text="<html><title>Bad Gateway</title></html>"))
    with pytest.raises(DrawError, match="服务器错误") as info:
        run({"api_key": KEY}, rec)
    assert info.value.status == status and not info.value.retryable
    assert len(rec.requests) == 1 and not sleeps


def test_read_timeout_not_retried(sleeps):
    rec = Recorder(lambda request: httpx.ReadTimeout("slow", request=request))
    with pytest.raises(DrawError, match="超时") as info:
        run({"api_key": KEY}, rec)
    assert not info.value.retryable and len(rec.requests) == 1 and not sleeps


def test_connect_error_retried_once(sleeps):
    rec = Recorder(lambda request: httpx.ConnectError("refused", request=request), ok_json())
    assert run({"api_key": KEY}, rec).image == PNG and len(rec.requests) == 2 and len(sleeps) == 1


# ---------------------------------------------------------------- 额外请求参数


def test_extra_body_merged_last():
    rec = Recorder(ok_json())
    extra = json.dumps({"quality": "high", "n": None, "background": "transparent", "size": "2048x2048"})
    result = run({"api_key": KEY, "quality": "low", "extra_body": extra}, rec)
    body = rec.body()
    assert body["quality"] == "high" and "n" not in body and body["background"] == "transparent"
    assert body["size"] == "2048x2048" and result.params["size"] == "2048x2048"
    assert result.params["background"] == "transparent"


def test_extra_body_as_dict():
    rec = Recorder(ok_json())
    run({"api_key": KEY, "extra_body": {"user": "u1"}}, rec)
    assert rec.body()["user"] == "u1"


@pytest.mark.parametrize("extra", ["{bad json", "[1, 2]", "42"])
def test_extra_body_invalid(extra):
    rec = Recorder(ok_json())
    with pytest.raises(DrawError, match="额外请求参数"):
        run({"api_key": KEY, "extra_body": extra}, rec)
    assert not rec.requests


# ---------------------------------------------------------------- 配置、状态与模型列表


@pytest.mark.parametrize("section", [None, {}, {"api_key": ""}, {"api_key": None}, {"api_key": "   "}])
def test_missing_api_key(section):
    assert OpenAIBackend(section).missing() == "管理员还没有配置 OpenAI API Key"
    with pytest.raises(DrawError, match="API Key"):
        run(section, Recorder(ok_json()))


def test_configured():
    backend = OpenAIBackend({"api_key": KEY})
    assert backend.missing() == ""
    assert backend.summary() == "OpenAI / gpt-image-2.5-flare / 竖图 1024x1536"
    assert backend.prompt_mode == "natural" and not backend.supports_negative and backend.supports_models
    assert backend.max_concurrency is None and backend.doc_url.startswith("https://developers.openai.com/")
    assert backend.timeout == 300.0
    assert OpenAIBackend({"api_key": KEY, "model": "dall-e-3", "portrait_size": "auto"}).summary() == \
        "OpenAI / dall-e-3 / 竖图 自动"
    assert backend.max_edit_images == 16 and not backend.supports_strength and backend.supports_balance
    assert backend.billing_url == "https://platform.openai.com/settings/organization/billing"
    assert backend.edit_missing() == "" and backend.edit_model == "gpt-image-2.5-flare"
    assert OpenAIBackend({"api_key": KEY, "edit_model": "gpt-image-2.5-sunburst"}).summary() == \
        "OpenAI / gpt-image-2.5-flare / 竖图 1024x1536 / 修改图片 gpt-image-2.5-sunburst"


def test_garbage_config_values_still_valid_request():
    section = {
        "api_key": KEY, "base_url": None, "proxy_url": None, "model": None, "portrait_size": "garbage",
        "landscape_size": 12345, "square_size": None, "quality": None, "output_format": 123, "background": [],
        "moderation": {"a": 1}, "response_format": None, "timeout": "abc", "extra_body": None,
    }
    backend = OpenAIBackend(section)
    assert backend.missing() == "" and backend.timeout == 300.0 and backend.summary()
    for orientation in ("portrait", "landscape", "square"):
        rec = Recorder(ok_json())
        run(section, rec, DrawRequest("x", orientation=orientation))
        assert rec.body() == {"model": "gpt-image-2.5-flare", "prompt": "x", "n": 1,
                              "size": oai.DEFAULT_SIZES[orientation], "quality": "medium"}
        assert str(rec.requests[0].url) == GEN_URL
    assert OpenAIBackend({"api_key": KEY, "timeout": 5}).timeout == 10.0
    assert OpenAIBackend({"api_key": KEY, "timeout": "99999"}).timeout == 1800.0


def test_list_models_filters_image_models():
    rec = Recorder(httpx.Response(200, json={"object": "list", "data": [
        {"id": "gpt-4.1"}, {"id": "gpt-image-2.5-flare"}, {"id": "chatgpt-image-latest"}, {"id": "dall-e-3"},
        {"id": "gpt-image-2"}, {"id": "whisper-1"}, {"id": "gpt-image-2"}, {"no_id": 1}, "junk",
    ]}))
    models = run({"api_key": KEY, "base_url": "https://relay.example.com"}, rec, action="models")
    assert models == ["chatgpt-image-latest", "dall-e-3", "gpt-image-2", "gpt-image-2.5-flare"]
    request = rec.requests[0]
    assert request.method == "GET" and str(request.url) == "https://relay.example.com/v1/models"
    assert request.headers["authorization"] == f"Bearer {KEY}"


def test_list_models_falls_back_to_all():
    rec = Recorder(httpx.Response(200, json={"data": [{"id": "flux-pro"}, {"id": "aaa"}, {"id": "sd3"}]}))
    assert run({"api_key": KEY}, rec, action="models") == ["aaa", "flux-pro", "sd3"]


def test_list_models_errors():
    rec = Recorder(httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(DrawError, match="API Key 无效"):
        run({"api_key": KEY}, rec, action="models")
    with pytest.raises(DrawError, match="API Key"):
        run({}, Recorder(ok_json()), action="models")
    assert run({"api_key": KEY}, Recorder(httpx.Response(200, json={"data": None})), action="models") == []


# ---------------------------------------------------------------- 配置段


def load_schema_section() -> dict:
    return json.loads((PLUGIN_DIR / "_conf_schema.json").read_text(encoding="utf-8"))["openai"]


def test_schema_section_shape_and_defaults_match_code():
    section = load_schema_section()
    items = section["items"]
    assert section["type"] == "object" and section["description"]
    assert list(items) == ["api_key", "base_url", "proxy_url", "model", "portrait_size", "landscape_size",
                           "square_size", "quality", "output_format", "background", "moderation", "response_format",
                           "edit_model", "edit_size", "timeout", "extra_body", "balance_mode", "balance_key"]
    for name, field in items.items():
        assert field["type"] in ("string", "text", "int", "float", "bool", "list"), name
        assert field["description"] and field["hint"], name
        if "options" in field:
            assert field["default"] in field["options"], name
            assert len(field.get("labels", field["options"])) == len(field["options"]), name
    assert items["api_key"]["secret"] is True
    assert "options" not in items["model"]
    assert "修改图片" in items["model"]["hint"] and "DALL·E 不支持" in items["model"]["hint"]
    # 修改图片
    assert items["edit_model"]["default"] == "" and "options" not in items["edit_model"]
    assert items["edit_size"]["default"] == oai.DEFAULT_EDIT_SIZE
    assert items["edit_size"]["options"] == list(oai.EDIT_SIZE_MODES)
    # 余额查询：管理密钥是密文字段，提示中说明只用于查询消费
    assert items["balance_mode"]["default"] == oai.DEFAULT_BALANCE_MODE
    assert items["balance_mode"]["options"] == list(oai.BALANCE_MODES)
    assert items["balance_key"]["secret"] is True and items["balance_key"]["default"] == ""
    assert "只用于" in items["balance_key"]["hint"] and "不能用于绘图" in items["balance_key"]["hint"]
    assert "secret" not in items["edit_model"] and "secret" not in items["balance_mode"]
    assert items["base_url"]["default"] == oai.DEFAULT_BASE_URL
    assert items["model"]["default"] == oai.DEFAULT_MODEL
    for orientation, size in oai.DEFAULT_SIZES.items():
        assert items[f"{orientation}_size"]["default"] == size
    assert items["quality"]["default"] == oai.DEFAULT_QUALITY
    assert items["quality"]["options"] == ["auto", *oai.QUALITIES["gpt25"]]
    assert items["timeout"]["default"] == oai.DEFAULT_TIMEOUT
    assert items["extra_body"]["type"] == "text" and items["extra_body"]["default"] == ""
    assert items["extra_body"]["editor_mode"] is True and items["extra_body"]["editor_language"] == "json"


@pytest.mark.parametrize("orientation", ["portrait", "landscape", "square"])
def test_schema_defaults_produce_valid_request(orientation):
    items = load_schema_section()["items"]
    section = {name: field["default"] for name, field in items.items()}
    section["api_key"] = KEY
    backend = OpenAIBackend(section)
    assert backend.missing() == ""
    rec = Recorder(ok_json())
    result = run(section, rec, DrawRequest("a cat", orientation=orientation))
    assert str(rec.requests[0].url) == GEN_URL
    assert rec.body() == {"model": "gpt-image-2.5-flare", "prompt": "a cat", "n": 1,
                          "size": oai.DEFAULT_SIZES[orientation], "quality": "medium"}
    assert result.params["size"] == oai.DEFAULT_SIZES[orientation]


@pytest.mark.parametrize("orientation", ["portrait", "landscape", "square"])
def test_schema_defaults_produce_valid_edit_request(orientation):
    items = load_schema_section()["items"]
    section = {name: field["default"] for name, field in items.items()}
    section["api_key"] = KEY
    backend = OpenAIBackend(section)
    assert backend.edit_missing() == ""
    # 默认的余额查询方式：官方地址、没有管理密钥 → 说明原因而不是报错
    assert "管理密钥" in backend.balance_missing()
    rec = Recorder(ok_json())
    size = {"portrait": (800, 1200), "landscape": (1200, 800), "square": (1000, 1000)}[orientation]
    req = DrawRequest("make it night", orientation=orientation, images=[InputImage(JPEG, "image/jpeg", *size)])
    result = run(section, rec, req)
    assert str(rec.requests[0].url) == EDIT_URL
    fields, files = parse_multipart(rec.requests[0])
    assert fields == {"model": "gpt-image-2.5-flare", "prompt": "make it night", "n": "1",
                      "size": oai.DEFAULT_SIZES[orientation], "quality": "medium"}
    assert [f[:3] for f in files] == [("image", "image_1.jpg", "image/jpeg")]
    assert result.params["mode"] == "edit"


# ---------------------------------------------------------------- 图片修改


EDIT_URL = "https://api.openai.com/v1/images/edits"
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 32
GIF = b"GIF89a" + b"\x00" * 32


def parse_multipart(request: httpx.Request) -> tuple[dict, list]:
    """解析 multipart 请求体，返回 (文本字段, [(字段名, 文件名, 类型, 内容)])。"""
    ctype = request.headers["content-type"]
    assert ctype.startswith("multipart/form-data; boundary=")
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + request.content)
    fields, files = {}, []
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        if part.get_filename():
            files.append((name, part.get_filename(), part.get_content_type(), payload))
        else:
            assert name not in fields, name
            fields[name] = payload.decode()
    return fields, files


def edit_req(*images, prompt="Make the sky purple", orientation="portrait", negative=""):
    return DrawRequest(prompt, negative=negative, orientation=orientation,
                       images=list(images) or [InputImage(JPEG, "image/jpeg", 1024, 1536)])


def test_edit_request_single_image():
    rec = Recorder(ok_json(size="1024x1536", quality="medium", output_format="png", usage={"total_tokens": 900}))
    req = edit_req(prompt="Make the sky purple", negative="text")
    result = run({"api_key": KEY}, rec, req)
    request = rec.requests[0]
    assert len(rec.requests) == 1 and request.method == "POST" and str(request.url) == EDIT_URL
    assert request.headers["authorization"] == f"Bearer {KEY}"
    fields, files = parse_multipart(request)
    assert fields == {"model": "gpt-image-2.5-flare", "prompt": "Make the sky purple\n\nAvoid: text", "n": "1",
                      "size": "1024x1536", "quality": "medium"}
    # 一张图时字段名为 image（与官方 SDK 一致），文件名和类型与内容一致
    assert files == [("image", "image_1.jpg", "image/jpeg", JPEG)]
    assert result.image == PNG and result.mime == "image/png"
    assert result.params == {"mode": "edit", "model": "gpt-image-2.5-flare", "size": "1024x1536",
                             "input_images": 1, "quality": "medium", "output_format": "png", "tokens": 900}


def test_edit_request_multiple_images_keep_order_and_detect_type():
    rec = Recorder(ok_json())
    # InputImage.mime 写错时以文件内容为准（中转服务按扩展名推断类型）
    req = edit_req(InputImage(JPEG, "image/jpeg", 1024, 1536), InputImage(PNG, "image/jpeg", 800, 600),
                   InputImage(WEBP, "image/webp", 512, 512))
    result = run({"api_key": KEY}, rec, req)
    _, files = parse_multipart(rec.requests[0])
    assert [f[:3] for f in files] == [("image[]", "image_1.jpg", "image/jpeg"), ("image[]", "image_2.png", "image/png"),
                                      ("image[]", "image_3.webp", "image/webp")]
    assert [f[3] for f in files] == [JPEG, PNG, WEBP]
    assert result.params["input_images"] == 3 and "ignored_images" not in result.params


def test_edit_more_than_16_images_trimmed():
    rec = Recorder(ok_json())
    images = [InputImage(PNG, "image/png", 1024, 1024) for _ in range(17)]
    result = run({"api_key": KEY}, rec, edit_req(*images, orientation="square"))
    _, files = parse_multipart(rec.requests[0])
    assert len(files) == 16 and {f[0] for f in files} == {"image[]"}
    assert files[-1][1] == "image_16.png"
    assert result.params["input_images"] == 16 and result.params["ignored_images"] == 1


@pytest.mark.parametrize(("data", "match"), [(GIF, "image/gif"), (b"not an image", "image/jpeg")])
def test_edit_unsupported_image_format_rejected(data, match):
    rec = Recorder(ok_json())
    with pytest.raises(DrawError, match=f"第 1 张图片的格式（{match}）.*PNG / JPEG / WebP"):
        run({"api_key": KEY}, rec, edit_req(InputImage(data, "image/jpeg", 100, 100)))
    assert not rec.requests


def test_edit_image_too_large_and_prompt_too_long(monkeypatch):
    rec = Recorder(ok_json())
    monkeypatch.setattr(oai, "MAX_EDIT_IMAGE_BYTES", 40)
    with pytest.raises(DrawError, match="第 2 张图片超过 50 MB"):
        run({"api_key": KEY}, rec, edit_req(InputImage(PNG[:30], "image/png"), InputImage(PNG, "image/png")))
    monkeypatch.undo()
    with pytest.raises(DrawError, match="修改说明超过 32000 个字符"):
        run({"api_key": KEY}, rec, edit_req(prompt="x" * 32001))
    assert not rec.requests


def test_edit_model_overrides_drawing_model_and_plan():
    section = {"api_key": KEY, "model": "gpt-image-2.5-flare", "edit_model": "gpt-image-2.5-sunburst"}
    req = edit_req(InputImage(JPEG, "image/jpeg", 1536, 2048))
    backend = OpenAIBackend(section)
    assert backend.plan(req) == {"mode": "edit", "model": "gpt-image-2.5-sunburst", "size": "1088x1456"}
    assert backend.plan(DrawRequest("x")) == {"model": "gpt-image-2.5-flare", "size": "1024x1536"}
    rec = Recorder(ok_json())
    result = run(section, rec, req)
    fields, _ = parse_multipart(rec.requests[0])
    assert fields["model"] == "gpt-image-2.5-sunburst" and result.params["model"] == "gpt-image-2.5-sunburst"
    # 文生图仍使用绘图模型
    rec = Recorder(ok_json())
    run(section, rec, DrawRequest("x"))
    assert rec.body()["model"] == "gpt-image-2.5-flare"


@pytest.mark.parametrize(("section", "model"), [
    ({"model": "dall-e-3"}, "dall-e-3"),
    ({"model": "gpt-image-2", "edit_model": "dall-e-2"}, "dall-e-2"),
    ({"model": "dall-e-2", "edit_model": "openai/dall-e-3"}, "openai/dall-e-3"),
])
def test_edit_missing_for_dalle(section, model):
    backend = OpenAIBackend({"api_key": KEY, **section})
    reason = backend.edit_missing()
    assert f"当前模型 {model} 不支持修改图片" in reason and "gpt-image-2.5-sunburst" in reason
    rec = Recorder(ok_json())
    with pytest.raises(DrawError, match="不支持修改图片"):
        run({"api_key": KEY, **section}, rec, edit_req())
    assert not rec.requests


@pytest.mark.parametrize("model", ["gpt-image-2.5-sunburst", "gpt-image-2", "gpt-image-1.5", "chatgpt-image-latest",
                                   "openai/gpt-image-2", "flux-kontext-pro"])
def test_edit_supported_models(model):
    assert OpenAIBackend({"api_key": KEY, "model": model}).edit_missing() == ""


@pytest.mark.parametrize(("section", "image", "orientation", "size"), [
    # gpt-image-2 / 2.5：保持原图比例，像素数与该方向的尺寸设置相同
    ({}, (1536, 2048), "portrait", "1088x1456"),
    ({}, (2048, 1152), "landscape", "1680x944"),
    ({}, (1000, 1100), "square", "976x1072"),
    ({}, (600, 2000), "portrait", "720x2160"),  # 超过 1:3 时限制在 1:3
    ({"portrait_size": "2K"}, (1536, 2048), "portrait", "1328x1776"),
    ({"portrait_size": "auto"}, (1536, 2048), "portrait", "1088x1456"),
    ({"model": "gpt-image-2"}, (1536, 2048), "portrait", "1088x1456"),
    # 用户要求换方向：使用该方向的尺寸
    ({}, (1536, 2048), "landscape", "1536x1024"),
    ({"landscape_size": "1920x1080"}, (1536, 2048), "landscape", "1936x1088"),
    # 不知道原图尺寸
    ({}, (0, 0), "portrait", "1024x1536"),
    # 旧 GPT Image 模型只有三种标准尺寸
    ({"model": "gpt-image-1.5"}, (1536, 2048), "portrait", "1024x1536"),
    ({"model": "chatgpt-image-latest"}, (1000, 1000), "square", "1024x1024"),
    # 其他设置
    ({"edit_size": "orientation"}, (1536, 2048), "portrait", "1024x1536"),
    ({"edit_size": "auto"}, (1536, 2048), "portrait", "auto"),
    ({"edit_size": "AUTO", "model": "gpt-image-1.5"}, (1536, 2048), "portrait", "auto"),
    ({"edit_size": "bogus"}, (1536, 2048), "portrait", "1088x1456"),
    # 中转服务上规则未知的模型：保持原图比例时不发送尺寸，按方向时与文生图相同
    ({"model": "flux-kontext-pro"}, (1536, 2048), "portrait", None),
    ({"model": "flux-kontext-pro", "edit_size": "auto"}, (1536, 2048), "portrait", None),
    ({"model": "flux-kontext-pro", "edit_size": "orientation"}, (1536, 2048), "portrait", "1024x1536"),
])
def test_edit_size(section, image, orientation, size):
    rec = Recorder(ok_json())
    req = edit_req(InputImage(JPEG, "image/jpeg", *image), orientation=orientation)
    result = run({"api_key": KEY, **section}, rec, req)
    fields, _ = parse_multipart(rec.requests[0])
    assert fields.get("size") == size
    assert result.params["size"] == (size or "auto")
    if size and size != "auto":
        w, h = map(int, size.split("x"))
        assert w % 16 == 0 and h % 16 == 0


@pytest.mark.parametrize(("model", "section", "expected", "absent"), [
    ("gpt-image-2.5-sunburst", {"quality": "max", "output_format": "jpg", "moderation": "low",
                                "background": "opaque", "response_format": "url"},
     {"quality": "max", "output_format": "jpeg", "moderation": "low", "background": "opaque"},
     ("response_format", "style", "input_fidelity")),
    ("gpt-image-2", {"quality": "xhigh", "output_format": "webp"}, {"quality": "high", "output_format": "webp"},
     ("response_format", "input_fidelity", "moderation", "background")),
    ("gpt-image-1.5", {"quality": "auto", "response_format": "b64_json"}, {},
     ("quality", "response_format", "input_fidelity")),
    # 中转服务上的其他模型：按配置发送 response_format，不发送 GPT Image 专用参数
    ("flux-kontext-pro", {"response_format": "b64_json", "quality": "high", "moderation": "low"},
     {"response_format": "b64_json"}, ("quality", "moderation", "output_format")),
    ("flux-kontext-pro", {}, {}, ("response_format",)),
])
def test_edit_fields_per_model_family(model, section, expected, absent):
    rec = Recorder(ok_json())
    result = run({"api_key": KEY, "model": model, **section}, rec, edit_req())
    fields, _ = parse_multipart(rec.requests[0])
    assert fields["model"] == model and fields["n"] == "1"
    for key, value in expected.items():
        assert fields[key] == value, key
        assert result.params[key] == value, key
    for key in absent:
        assert key not in fields, key
    assert not {"seed", "negative_prompt", "strength", "style"} & set(fields)


def test_edit_extra_body_as_form_strings():
    rec = Recorder(ok_json())
    extra = json.dumps({"output_compression": 80, "n": None, "stream": False, "nested": {"a": 1}, "list": [1],
                        "background": "transparent", "user": "u1"})
    result = run({"api_key": KEY, "output_format": "webp", "extra_body": extra}, rec, edit_req())
    fields, _ = parse_multipart(rec.requests[0])
    assert fields["output_compression"] == "80" and fields["stream"] == "false" and fields["user"] == "u1"
    assert fields["background"] == "transparent" and result.params["background"] == "transparent"
    assert "n" not in fields and "nested" not in fields and "list" not in fields


def test_edit_response_variants():
    def handler(request):
        if request.url.host == "cdn.example.com":
            return httpx.Response(200, content=JPEG, headers={"content-type": "image/jpeg"})
        return httpx.Response(200, json={"data": [{"url": "https://cdn.example.com/a.jpg"}]})

    rec = Recorder(handler)
    result = run({"api_key": KEY, "base_url": "https://relay.example.com", "model": "flux-kontext-pro"}, rec,
                 edit_req())
    assert result.image == JPEG and len(rec.requests) == 2
    assert str(rec.requests[0].url) == "https://relay.example.com/v1/images/edits"
    assert "authorization" not in rec.requests[1].headers
    # 中转服务直接返回图片
    rec = Recorder(httpx.Response(200, content=PNG, headers={"content-type": "image/png"}))
    assert run({"api_key": KEY}, rec, edit_req()).image == PNG
    with pytest.raises(DrawError, match="没有图片"):
        run({"api_key": KEY}, Recorder(httpx.Response(200, json={"data": []})), edit_req())


# ---------------------------------------------------------------- 图片修改的错误


@pytest.mark.parametrize(("status", "body"), [
    (501, {"error": {"message": "API not implemented", "type": "one_api_error", "code": "api_not_implemented"}}),
    (404, "<html><title>404 Not Found</title></html>"),
    (405, {"error": {"message": "Method Not Allowed"}}),
    (200, {"error": {"message": "API not implemented", "type": "one_api_error", "code": "api_not_implemented"}}),
])
def test_edit_endpoint_not_supported(status, body, sleeps):
    resp = httpx.Response(status, json=body) if isinstance(body, dict) else httpx.Response(status, text=body)
    rec = Recorder(resp)
    with pytest.raises(DrawError, match=r"不支持修改图片（/images/edits，HTTP \d+）") as info:
        run({"api_key": KEY, "base_url": "https://relay.example.com"}, rec, edit_req())
    assert not info.value.retryable and len(rec.requests) == 1 and not sleeps


def test_edit_model_not_found_404_keeps_generic_message():
    rec = Recorder(httpx.Response(404, json={"error": {"message": "The model `gpt-image-9` does not exist",
                                                       "code": "model_not_found"}}))
    with pytest.raises(DrawError, match="接口地址或模型不存在"):
        run({"api_key": KEY, "model": "gpt-image-9"}, rec, edit_req())


@pytest.mark.parametrize(("err", "match"), [
    ({"type": "invalid_request_error", "code": "unsupported_file_mimetype", "param": "image[3]",
      "message": "Invalid file 'image[3]': unsupported mimetype ('application/octet-stream')."},
     "不接受这张图片的格式（只支持 PNG / JPEG / WebP）"),
    ({"type": "invalid_request_error", "code": "unknown_parameter", "param": "moderation",
      "message": "Unknown parameter: 'moderation'."}, "不接受参数 moderation.*内容审核强度"),
    ({"type": "invalid_request_error", "code": "invalid_value", "param": "model",
      "message": "Invalid value: 'gpt-image-1.5'. Value must be 'dall-e-2'."}, "response_format / style"),
])
def test_edit_400_errors(err, match, sleeps):
    rec = Recorder(httpx.Response(400, json={"error": err}))
    with pytest.raises(DrawError, match=match) as info:
        run({"api_key": KEY, "moderation": "low"}, rec, edit_req())
    assert info.value.status == 400 and not info.value.retryable and len(rec.requests) == 1 and not sleeps


@pytest.mark.parametrize(("stage", "match"), [
    ("input", "提示词或输入图片被 OpenAI 的安全系统拦截了（类别：sexual）"),
    ("output", "生成的图片没有通过 OpenAI 的安全审核"),
])
def test_edit_moderation_blocked(stage, match, sleeps):
    err = {"type": "image_generation_user_error", "code": "moderation_blocked",
           "moderation_details": {"moderation_stage": stage, "categories": ["sexual"]}}
    rec = Recorder(httpx.Response(400, json={"error": err}))
    with pytest.raises(DrawError, match=match) as info:
        run({"api_key": KEY}, rec, edit_req())
    assert "换一种描述" in str(info.value) and not info.value.retryable and len(rec.requests) == 1 and not sleeps


def test_edit_moderation_details_at_top_level(sleeps):
    # openai_error 也读取响应顶层的 moderation_details；修改图片时同样要提到输入图片
    body = {"error": {"type": "image_generation_user_error", "code": "moderation_blocked"},
            "moderation_details": {"moderation_stage": "input", "categories": ["violence"]}}
    with pytest.raises(DrawError, match="^提示词或输入图片被 OpenAI 的安全系统拦截了（类别：violence）") as info:
        run({"api_key": KEY}, Recorder(httpx.Response(400, json=body)), edit_req())
    assert info.value.status == 400 and not info.value.retryable and not sleeps
    # 中转服务说明文字里的「提示词被」不改写
    body = {"error": {"message": "提示词被上游拒绝", "type": "new_api_error"}}
    with pytest.raises(DrawError, match="请求参数有误（HTTP 400）：提示词被上游拒绝"):
        run({"api_key": KEY}, Recorder(httpx.Response(400, json=body)), edit_req())


def test_edit_413_explains_upload_limit(sleeps):
    rec = Recorder(httpx.Response(413, text="<html><title>413 Request Entity Too Large</title></html>"))
    with pytest.raises(DrawError, match=r"拒绝了上传的图片.*HTTP 413.*413 Request Entity Too Large.*减少图片数量") as info:
        run({"api_key": KEY, "base_url": "https://relay.example.com"}, rec, edit_req())
    assert info.value.status == 413 and not info.value.retryable and len(rec.requests) == 1 and not sleeps


def test_edit_billing_429_not_retried(sleeps):
    rec = Recorder(httpx.Response(429, json={"error": {"code": "credit_balance_exhausted",
                                                       "type": "insufficient_quota"}}))
    with pytest.raises(DrawError, match="余额不足") as info:
        run({"api_key": KEY}, rec, edit_req())
    assert not info.value.retryable and len(rec.requests) == 1 and not sleeps


def test_edit_rate_limit_retried_with_identical_body(sleeps):
    limited = httpx.Response(429, json={"error": {"type": "requests", "code": "rate_limit_exceeded"}},
                             headers={"retry-after": "1"})
    rec = Recorder(limited, ok_json())
    req = edit_req(InputImage(JPEG, "image/jpeg", 1024, 1536), InputImage(PNG, "image/png", 512, 512))
    assert run({"api_key": KEY}, rec, req).image == PNG
    assert len(rec.requests) == 2 and sleeps == [1.0]
    assert parse_multipart(rec.requests[0]) == parse_multipart(rec.requests[1])
    assert len(parse_multipart(rec.requests[1])[1]) == 2


@pytest.mark.parametrize("status", [500, 502, 504])
def test_edit_5xx_not_retried(status, sleeps):
    rec = Recorder(httpx.Response(status, text="<html><title>Bad Gateway</title></html>"))
    with pytest.raises(DrawError, match="服务器错误") as info:
        run({"api_key": KEY}, rec, edit_req())
    assert not info.value.retryable and len(rec.requests) == 1 and not sleeps


def test_edit_read_timeout_not_retried_and_redirect_not_followed(sleeps):
    rec = Recorder(lambda request: httpx.ReadTimeout("slow", request=request))
    with pytest.raises(DrawError, match="超时") as info:
        run({"api_key": KEY}, rec, edit_req())
    assert not info.value.retryable and len(rec.requests) == 1 and not sleeps
    rec = Recorder(httpx.Response(302, headers={"location": "https://relay.example.com/v1/images/edits"}), ok_json())
    with pytest.raises(DrawError, match="跳转"):
        run({"api_key": KEY, "base_url": "http://relay.example.com"}, rec, edit_req())
    assert len(rec.requests) == 1


def test_edit_api_key_echoed_is_masked():
    secret = "sk-relay-secret-ABCDEFGH"
    rec = Recorder(httpx.Response(401, json={"error": {"message": f"invalid token {secret}"}}))
    with pytest.raises(DrawError, match="API Key 无效") as info:
        run({"api_key": secret}, rec, edit_req())
    assert secret not in str(info.value) and "EFGH" in str(info.value)


def test_edit_missing_api_key():
    rec = Recorder(ok_json())
    with pytest.raises(DrawError, match="API Key"):
        run({}, rec, edit_req())
    assert not rec.requests


# ---------------------------------------------------------------- 余额：配置检查


ADMIN_KEY = "sk-admin-secret-WXYZ9876"
RELAY = "https://relay.example.com"


@pytest.mark.parametrize(("section", "match"), [
    # 官方地址、没有管理密钥：说明官方没有余额接口并给出账单页面
    ({"api_key": KEY}, "OpenAI 官方 API 不能用普通 API Key 查询余额.*settings/organization/billing.*管理密钥"),
    ({"api_key": KEY, "balance_mode": "admin"}, "还没有在 OpenAI 设置中填写管理密钥"),
    ({"api_key": KEY, "balance_mode": "admin", "base_url": RELAY}, "还没有在 OpenAI 设置中填写管理密钥"),
    ({"api_key": KEY, "balance_mode": "off", "balance_key": ADMIN_KEY}, "已在 OpenAI 设置中关闭余额查询.*billing"),
    ({"api_key": KEY, "balance_mode": "off", "base_url": RELAY}, "已在 OpenAI 设置中关闭余额查询，请到 中转服务的后台"),
    ({"api_key": KEY, "balance_mode": "relay"}, "官方 API 没有中转服务的额度接口"),
    ({"api_key": KEY, "balance_key": "sk-admin-全角"}, "管理密钥.*重新复制"),
    ({"base_url": RELAY}, "管理员还没有配置 OpenAI API Key"),
    ({"api_key": KEY, "base_url": "ftp://relay.example.com"}, "API 地址无效"),
])
def test_balance_missing(section, match):
    backend = OpenAIBackend(section)
    assert backend.supports_balance
    reason = backend.balance_missing()
    assert re.search(match, reason), reason
    rec = Recorder(httpx.Response(200, json={}))
    with pytest.raises(DrawError) as info:
        run(section, rec, action="balance")
    assert str(info.value) == reason and not rec.requests


@pytest.mark.parametrize(("section", "source"), [
    ({"api_key": KEY, "balance_key": ADMIN_KEY}, "admin"),
    ({"balance_key": ADMIN_KEY}, "admin"),  # 管理密钥不需要绘图的 API Key
    ({"api_key": KEY, "base_url": "https://eu.api.openai.com/v1", "balance_key": f"Bearer {ADMIN_KEY}"}, "admin"),
    ({"api_key": KEY, "base_url": RELAY}, "relay"),
    ({"api_key": KEY, "base_url": RELAY, "balance_key": ADMIN_KEY}, "relay"),
    ({"api_key": KEY, "base_url": RELAY, "balance_mode": "admin", "balance_key": ADMIN_KEY}, "admin"),
    ({"api_key": KEY, "base_url": RELAY, "balance_mode": "relay"}, "relay"),
    ({"api_key": KEY, "base_url": RELAY, "balance_mode": "bogus"}, "relay"),
])
def test_balance_ready(section, source):
    backend = OpenAIBackend(section)
    assert backend.balance_missing() == "" and backend.balance_source() == source


# ---------------------------------------------------------------- 余额：官方管理密钥


def costs_page(values, has_more=False, next_page=None):
    return {"object": "page", "has_more": has_more, "next_page": next_page, "data": [
        {"object": "bucket", "start_time": 1, "end_time": 2, "results": [
            {"object": "organization.costs.result", "amount": {"value": v, "currency": "usd"}} for v in values]},
        {"object": "bucket", "start_time": 2, "end_time": 3, "results": []},
    ]}


def admin_handler(pages, limit=None, limit_status=200):
    pages = list(pages)

    def handler(request):
        if request.url.path == "/v1/organization/costs":
            return httpx.Response(200, json=pages.pop(0))
        if request.url.path == "/v1/organization/spend_limit":
            return httpx.Response(limit_status, json=limit if limit is not None else {"error": {"message": "no"}})
        return httpx.Response(404, json={"error": {"message": "unexpected"}})

    return handler


def test_balance_admin_costs_and_enforced_limit():
    pages = [costs_page([1.25, "2.5"], has_more=True, next_page="page_2"), costs_page([0.01, None, "bad"])]
    limit = {"object": "organization.spend_limit", "threshold_amount": 1000, "currency": "USD",
             "interval": "month", "enforcement": {"status": "enforcing"}}
    rec = Recorder(admin_handler(pages, limit))
    info = run({"api_key": KEY, "balance_key": ADMIN_KEY}, rec, action="balance")
    assert isinstance(info, BalanceInfo)
    assert info.items == [("本月已花费（整个组织）", "3.76 美元"), ("每月消费上限（已启用）", "10.00 美元"),
                          ("距离上限还可花费", "6.24 美元")]
    assert "不是余额" in info.note and "settings/organization/billing" in info.note
    assert [r.url.path for r in rec.requests] == ["/v1/organization/costs", "/v1/organization/costs",
                                                  "/v1/organization/spend_limit"]
    first, second = rec.requests[0].url.params, rec.requests[1].url.params
    assert first["start_time"] == str(oai.month_start(datetime.now(timezone.utc)))
    assert first["bucket_width"] == "1d" and first["limit"] == "31" and "page" not in first
    assert second["page"] == "page_2"
    # 只使用管理密钥，不发送绘图的 API Key
    assert {r.headers["authorization"] for r in rec.requests} == {f"Bearer {ADMIN_KEY}"}
    assert all(r.method == "GET" and r.url.host == "api.openai.com" for r in rec.requests)
    assert not any(KEY in str(v) or ADMIN_KEY in str(v) for item in info.items for v in item)


def test_month_start():
    assert oai.month_start(datetime(2026, 10, 25, 13, 5, tzinfo=timezone.utc)) == \
        int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp())


def test_balance_admin_limit_not_enforced_or_unavailable():
    limit = {"threshold_amount": 5000, "enforcement": {"status": "inactive"}}
    info = run({"balance_key": ADMIN_KEY}, Recorder(admin_handler([costs_page([12])], limit)), action="balance")
    assert info.items == [("本月已花费（整个组织）", "12.00 美元"), ("每月消费预算（未强制执行）", "50.00 美元")]
    # 没有设置上限
    no_limit = {"threshold_amount": 0, "enforcement": {"status": "inactive"}}
    info = run({"balance_key": ADMIN_KEY}, Recorder(admin_handler([costs_page([])], no_limit)), action="balance")
    assert info.items == [("本月已花费（整个组织）", "0.00 美元")]
    # 上限接口失败（403 / 500）不影响消费结果
    for status in (403, 500):
        info = run({"balance_key": ADMIN_KEY}, Recorder(admin_handler([costs_page([1])], None, status)),
                   action="balance")
        assert info.items == [("本月已花费（整个组织）", "1.00 美元")]


def test_balance_admin_key_never_sent_to_relay():
    rec = Recorder(admin_handler([costs_page([1])], {}))
    run({"api_key": KEY, "base_url": RELAY, "balance_mode": "admin", "balance_key": ADMIN_KEY}, rec, action="balance")
    assert {r.url.host for r in rec.requests} == {"api.openai.com"}
    # 区域官方地址
    rec = Recorder(lambda request: httpx.Response(200, json=costs_page([1]) if "costs" in request.url.path else {}))
    run({"base_url": "https://eu.api.openai.com/v1", "balance_key": ADMIN_KEY}, rec, action="balance")
    assert {r.url.host for r in rec.requests} == {"eu.api.openai.com"}
    # API 地址误写成 http:// 时，管理密钥仍只通过 https 发送
    rec = Recorder(lambda request: httpx.Response(200, json=costs_page([1]) if "costs" in request.url.path else {}))
    run({"base_url": "http://api.openai.com:80/v1", "balance_key": ADMIN_KEY}, rec, action="balance")
    assert {str(r.url).split("?")[0] for r in rec.requests} == {"https://api.openai.com/v1/organization/costs",
                                                                "https://api.openai.com/v1/organization/spend_limit"}


@pytest.mark.parametrize("status", [401, 403])
def test_balance_admin_key_rejected(status, sleeps):
    rec = Recorder(httpx.Response(status, json={"error": {"message": f"Invalid admin key {ADMIN_KEY}",
                                                          "code": "invalid_api_key"}}))
    with pytest.raises(DrawError, match=f"管理密钥（Admin API Key）无效或权限不足（HTTP {status}）.*sk-admin-") as info:
        run({"api_key": KEY, "balance_key": ADMIN_KEY}, rec, action="balance")
    assert ADMIN_KEY not in str(info.value) and "9876" in str(info.value)
    assert len(rec.requests) == 1 and not sleeps


def test_balance_admin_costs_retries_rate_limit(sleeps):
    rec = Recorder(httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}}),
                   lambda request: httpx.Response(200, json=costs_page([2]) if "costs" in request.url.path else {}))
    info = run({"balance_key": ADMIN_KEY}, rec, action="balance")
    assert info.items[0] == ("本月已花费（整个组织）", "2.00 美元") and len(sleeps) == 1


# ---------------------------------------------------------------- 余额：中转服务


def relay_handler(routes: dict):
    def handler(request):
        item = routes.get(request.url.path)
        if item is None:
            return httpx.Response(404, json={"error": {"message": "not found"}})
        return item(request) if callable(item) else item

    return handler


def test_balance_relay_new_api_cny():
    sub = {"object": "billing_subscription", "has_payment_method": True, "soft_limit_usd": 100.0,
           "hard_limit_usd": 100.0, "system_hard_limit_usd": 100.0, "access_until": 1798675200}
    rec = Recorder(relay_handler({
        "/v1/dashboard/billing/subscription": httpx.Response(200, json=sub),
        "/v1/dashboard/billing/usage": httpx.Response(200, json={"object": "list", "total_usage": 2534}),
        "/api/status": httpx.Response(200, json={"success": True, "data": {"quota_per_unit": 500000,
                                                                           "quota_display_type": "CNY"}}),
    }))
    info = run({"api_key": KEY, "base_url": f"{RELAY}/v1/chat/completions"}, rec, action="balance")
    assert info.items == [("剩余额度", "74.66 元"), ("已用", "25.34 元"), ("总额度", "100.00 元"),
                          ("令牌有效期至", "2026-12-31")]
    assert info.note == oai.RELAY_NOTE and "非 OpenAI 官方" in info.note
    paths = [r.url.path for r in rec.requests]
    assert paths == ["/v1/dashboard/billing/subscription", "/v1/dashboard/billing/usage", "/api/status"]
    assert rec.requests[0].headers["authorization"] == f"Bearer {KEY}"
    assert rec.requests[1].url.params["start_date"].endswith("-01")
    assert "authorization" not in rec.requests[2].headers  # 公开接口不带 Key


def test_balance_relay_unlimited_and_raw_points():
    routes = {
        "/v1/dashboard/billing/subscription": httpx.Response(200, json={"hard_limit_usd": 100000000,
                                                                         "access_until": 0}),
        "/v1/dashboard/billing/usage": httpx.Response(200, json={"total_usage": 1234}),
        "/api/status": httpx.Response(500, text="oops"),
    }
    info = run({"api_key": KEY, "base_url": RELAY}, Recorder(relay_handler(routes)), action="balance")
    # 不限额度只针对令牌本身，账户余额可能有限：不能显示成「剩余额度：不限额度」
    assert info.items == [("令牌额度", "不限额度"), ("已用", "12.34 美元")]
    assert "账户的剩余额度请到中转服务的后台查看" in info.note and oai.RELAY_NOTE in info.note
    # one-api 关闭 display_in_currency：返回原始额度点数
    routes = {
        "/v1/dashboard/billing/subscription": httpx.Response(200, json={"hard_limit_usd": 600000}),
        "/v1/dashboard/billing/usage": httpx.Response(200, json={"total_usage": 15000000}),
        "/api/status": httpx.Response(200, json={"success": True, "data": {"display_in_currency": False}}),
    }
    info = run({"api_key": KEY, "base_url": RELAY}, Recorder(relay_handler(routes)), action="balance")
    assert info.items == [("剩余额度", "450000 额度点数"), ("已用", "150000 额度点数"), ("总额度", "600000 额度点数")]


def test_balance_relay_credit_grants_fallback():
    routes = {"/v1/dashboard/billing/credit_grants": httpx.Response(200, json={
        "object": "credit_summary", "total_granted": 20, "total_used": 5.5, "total_available": 14.5})}
    rec = Recorder(relay_handler(routes))
    info = run({"api_key": KEY, "base_url": RELAY}, rec, action="balance")
    assert info.items == [("剩余额度", "14.50 美元"), ("已用", "5.50 美元"), ("总额度", "20.00 美元")]
    assert [r.url.path for r in rec.requests] == ["/v1/dashboard/billing/subscription",
                                                  "/v1/dashboard/billing/credit_grants"]


@pytest.mark.parametrize("resp", [
    httpx.Response(404, json={"error": {"message": "not found"}}),
    httpx.Response(501, json={"error": {"code": "api_not_implemented"}}),
    httpx.Response(200, text="<html><title>Relay Panel</title></html>", headers={"content-type": "text/html"}),
])
def test_balance_relay_without_endpoint(resp, sleeps):
    rec = Recorder(resp)
    with pytest.raises(DrawError, match="该 API 地址没有提供查询余额的接口"):
        run({"api_key": KEY, "base_url": RELAY}, rec, action="balance")
    assert len(rec.requests) == 2 and not sleeps


def test_balance_relay_errors(sleeps):
    # 中转服务以 HTTP 200 返回错误
    rec = Recorder(httpx.Response(200, json={"error": {"message": "令牌已过期", "type": "new_api_error"}}))
    with pytest.raises(DrawError, match="查询额度失败：令牌已过期"):
        run({"api_key": KEY, "base_url": RELAY}, rec, action="balance")
    # 用量接口出错
    routes = {"/v1/dashboard/billing/subscription": httpx.Response(200, json={"hard_limit_usd": 5}),
              "/v1/dashboard/billing/usage": httpx.Response(200, json={"error": {"message": "db error"}})}
    with pytest.raises(DrawError, match="查询用量失败：db error"):
        run({"api_key": KEY, "base_url": RELAY}, Recorder(relay_handler(routes)), action="balance")
    # Key 无效；中转服务原样带出的 Key 打码
    secret = "sk-relay-secret-ABCDEFGH"
    rec = Recorder(httpx.Response(401, json={"error": {"message": f"无效的令牌 {secret}"}}))
    with pytest.raises(DrawError, match="API Key 无效") as info:
        run({"api_key": secret, "base_url": RELAY}, rec, action="balance")
    assert secret not in str(info.value) and len(rec.requests) == 1
    # 反向代理到官方接口
    rec = Recorder(httpx.Response(403, json={"error": {"message": "Your request to GET /dashboard/billing/subscription "
                                                                  "must be made with a session key"}}))
    with pytest.raises(DrawError, match="转发到了 OpenAI 官方接口.*官方管理密钥"):
        run({"api_key": KEY, "base_url": "https://oai-proxy.example.com"}, rec, action="balance")
    assert not sleeps


def test_balance_relay_connect_error_retried(sleeps):
    routes = {"/v1/dashboard/billing/subscription": httpx.Response(200, json={"hard_limit_usd": 10}),
              "/v1/dashboard/billing/usage": httpx.Response(200, json={"total_usage": 100})}
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if len(calls) == 1:
            raise httpx.ConnectError("refused", request=request)
        return relay_handler(routes)(request)

    info = run({"api_key": KEY, "base_url": RELAY}, Recorder(handler), action="balance")
    assert info.items[0] == ("剩余额度", "9.00 美元") and len(sleeps) == 1
    assert calls[:2] == ["/v1/dashboard/billing/subscription"] * 2
