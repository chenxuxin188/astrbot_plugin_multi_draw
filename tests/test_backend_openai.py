"""OpenAI 后端的单元测试（不访问网络）：python -m pytest tests/test_backend_openai.py"""

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
from backends import openai as oai  # noqa: E402
from backends.base import Clients, DrawError, DrawRequest, HttpPool  # noqa: E402
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
                           "timeout", "extra_body"]
    for name, field in items.items():
        assert field["type"] in ("string", "text", "int", "float", "bool", "list"), name
        assert field["description"] and field["hint"], name
        if "options" in field:
            assert field["default"] in field["options"], name
            assert len(field.get("labels", field["options"])) == len(field["options"]), name
    assert items["api_key"]["secret"] is True
    assert "options" not in items["model"]
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
