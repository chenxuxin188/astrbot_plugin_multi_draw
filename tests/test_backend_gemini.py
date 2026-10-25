"""Gemini 后端的单元测试（不访问网络）：python -m pytest tests/test_backend_gemini.py"""

import asyncio
import base64
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import base as base_mod  # noqa: E402
from backends import gemini as gemini_mod  # noqa: E402
from backends.base import Clients, DrawError, DrawRequest, HttpPool, InputImage  # noqa: E402
from backends.gemini import (  # noqa: E402
    BILLING_URL,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SIZES,
    DEFAULT_TIMEOUT,
    MAX_EDIT_IMAGES,
    MODEL_CAPS,
    RATIOS,
    SPEND_URL,
    USAGE_URL,
    Caps,
    GeminiBackend,
    encode_inputs,
    gemini_base_url,
    gemini_error,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
B64 = base64.b64encode(PNG).decode()
KEY = "AIza-test-key"
GC_URL = f"{DEFAULT_BASE_URL}/models/{DEFAULT_MODEL}:generateContent"
IA_URL = f"{DEFAULT_BASE_URL}/interactions"

PLUGIN_SCHEMA = Path(__file__).resolve().parent.parent / "_conf_schema.json"


@pytest.fixture(autouse=True)
def no_env_proxy(monkeypatch):
    # 避免系统 / 环境变量中的代理绕过 MockTransport
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")


@pytest.fixture
def sleeps(monkeypatch):
    calls = []

    async def fake_sleep(delay):
        calls.append(delay)

    monkeypatch.setattr(base_mod.asyncio, "sleep", fake_sleep)
    return calls


class Recorder:
    """按顺序返回预设响应，并记录收到的请求。"""

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

    def body(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)


def gc_ok(*parts, finish="STOP") -> httpx.Response:
    parts = parts or ({"inlineData": {"mimeType": "image/png", "data": B64}},)
    return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": list(parts)},
                                                     "finishReason": finish}]})


def ia_ok(data: str = B64) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "v1_x", "status": "completed",
        "steps": [
            {"type": "thought", "summary": [{"type": "image", "data": base64.b64encode(JPEG).decode()}]},
            {"type": "model_output", "content": [{"type": "text", "text": "done"},
                                                  {"type": "image", "mime_type": "image/jpeg", "data": data}]},
        ],
    })


def section(**kw) -> dict:
    return {"api_key": KEY, **kw}


def run(backend: GeminiBackend, handler, req: DrawRequest | None = None, *, models: bool = False,
        balance: bool = False):
    async def main():
        pool = HttpPool(transport=httpx.MockTransport(handler))
        try:
            clients = Clients(http=pool)
            if models:
                return await backend.list_models(clients)
            if balance:
                return await backend.balance(clients)
            return await backend.generate(clients, req or DrawRequest("a cat"))
        finally:
            await pool.close()

    return asyncio.run(main())


def schema_section() -> dict:
    return json.loads(PLUGIN_SCHEMA.read_text(encoding="utf-8"))["gemini"]


def assert_valid_gc_body(body: dict, model: str):
    """请求体只包含目标模型接受的字段。"""
    caps = MODEL_CAPS.get(model)
    assert set(body) <= {"contents", "generationConfig"}
    config = body["generationConfig"]
    assert config["responseModalities"] == ["TEXT", "IMAGE"]
    assert "responseFormat" not in config and "candidateCount" not in config and "temperature" not in config
    image_config = config["imageConfig"]
    assert set(image_config) <= {"aspectRatio", "imageSize"}
    if caps:
        assert image_config["aspectRatio"] in caps.ratios
        if "imageSize" in image_config:
            assert image_config["imageSize"] in caps.sizes
        if "thinkingConfig" in config:
            assert config["thinkingConfig"]["thinkingLevel"].lower() in caps.thinking
    if "seed" in config:
        assert 0 <= config["seed"] < 2**31


# ---------------------------------------------------------------- 请求构造


@pytest.mark.parametrize(("orientation", "ratio"), [("portrait", "2:3"), ("landscape", "3:2"), ("square", "1:1"),
                                                    ("bogus", "2:3")])
def test_default_request_per_orientation(orientation, ratio):
    rec = Recorder(gc_ok())
    result = run(GeminiBackend(section()), rec, DrawRequest("a cat", orientation=orientation))
    req = rec.requests[0]
    assert req.method == "POST" and str(req.url) == GC_URL
    assert req.headers["x-goog-api-key"] == KEY
    assert req.headers["content-type"] == "application/json"
    assert "key=" not in str(req.url) and "api-revision" not in req.headers
    body = rec.body()
    assert body == {
        "contents": [{"role": "user", "parts": [{"text": "a cat"}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": {"aspectRatio": ratio}},
    }
    assert result.image == PNG and result.mime == "image/png"
    assert result.params == {"model": DEFAULT_MODEL, "size": ratio, "aspect_ratio": ratio, "api": "generateContent",
                             "usage": {"images": 1}}


@pytest.mark.parametrize(
    ("model", "image_size", "thinking", "portrait", "url_model", "image_config", "thinking_sent"),
    [
        ("gemini-nano-banana-2.1", "4K", "high", "1:4", "gemini-nano-banana-2.1",
         {"aspectRatio": "1:4", "imageSize": "4K"}, "HIGH"),
        ("gemini-nano-banana-2.1", "512", "medium", "2:3", "gemini-nano-banana-2.1",
         {"aspectRatio": "2:3", "imageSize": "1K"}, "MEDIUM"),
        ("gemini-3.1-flash-image", "512", "medium", "8:1", "gemini-3.1-flash-image",
         {"aspectRatio": "8:1", "imageSize": "512"}, None),
        ("gemini-3.1-flash-lite-image", "4k", "minimal", "1:4", "gemini-3.1-flash-lite-image",
         {"aspectRatio": "9:16", "imageSize": "1K"}, "MINIMAL"),
        ("gemini-3-pro-image", "2K", "high", "3:4", "gemini-3-pro-image",
         {"aspectRatio": "3:4", "imageSize": "2K"}, None),
        ("gemini-2.5-flash-image", "4K", "high", "4:5", "gemini-2.5-flash-image", {"aspectRatio": "4:5"}, None),
        ("gemini-3-pro-image-preview", "", "", "", "gemini-3-pro-image", {"aspectRatio": "2:3"}, None),
        ("models/gemini-3.1-flash-image", "", "high", "", "gemini-3.1-flash-image", {"aspectRatio": "2:3"}, "HIGH"),
        ("imagen-4.0-generate-001", "", "", "", "gemini-nano-banana-2.1", {"aspectRatio": "2:3"}, None),
        ("my-custom-image-model", "2048", "high", "1:8", "my-custom-image-model",
         {"aspectRatio": "9:16", "imageSize": "2K"}, None),
    ],
)
def test_model_families(model, image_size, thinking, portrait, url_model, image_config, thinking_sent):
    rec = Recorder(gc_ok())
    backend = GeminiBackend(section(model=model, image_size=image_size, thinking_level=thinking,
                                    portrait_size=portrait))
    result = run(backend, rec, DrawRequest("x", seed=42))
    assert str(rec.requests[0].url) == f"{DEFAULT_BASE_URL}/models/{url_model}:generateContent"
    config = rec.body()["generationConfig"]
    assert config["imageConfig"] == image_config
    assert config.get("thinkingConfig") == ({"thinkingLevel": thinking_sent} if thinking_sent else None)
    assert config["seed"] == 42
    assert_valid_gc_body(rec.body(), url_model)
    assert result.params["model"] == url_model
    assert result.params["size"] == " ".join(filter(None, [image_config["aspectRatio"],
                                                           image_config.get("imageSize")]))


@pytest.mark.parametrize(
    ("value", "image_size", "expected"),
    [
        ("2048x3072", "", {"aspectRatio": "2:3", "imageSize": "2K"}),
        ("1024x1536", "", {"aspectRatio": "2:3"}),  # 推断为 1K（默认档位）时不发送
        ("1366x768", "", {"aspectRatio": "16:9"}),
        ("832*1216", "4K", {"aspectRatio": "2:3", "imageSize": "4K"}),  # 配置的档位优先
        ("6000x8000", "", {"aspectRatio": "3:4", "imageSize": "4K"}),
        ("3：4", "", {"aspectRatio": "3:4"}),
        ("16/9", "", {"aspectRatio": "16:9"}),
        ("6:9", "", {"aspectRatio": "2:3"}),
        ("1:2", "", {"aspectRatio": "9:16"}),
        ("abc", "", {"aspectRatio": "2:3"}),
        ("0:3", "bogus", {"aspectRatio": "2:3"}),
    ],
)
def test_size_snapping(value, image_size, expected):
    rec = Recorder(gc_ok())
    run(GeminiBackend(section(portrait_size=value, image_size=image_size)), rec)
    assert rec.body()["generationConfig"]["imageConfig"] == expected


@pytest.mark.parametrize(
    ("model", "value", "image_size", "expected"),
    [
        (DEFAULT_MODEL, "16:9 2K", "", ("16:9", "2K")),
        (DEFAULT_MODEL, "2:3 2k", "4K", ("2:3", "2K")),  # 尺寸里的档位优先
        (DEFAULT_MODEL, "2K", "", ("2:3", "2K")),  # 只写档位时用默认比例
        (DEFAULT_MODEL, "1024x1536 4K", "", ("2:3", "4K")),
        (DEFAULT_MODEL, "2:3 512", "", ("2:3", "1K")),  # NB2.1 不支持 512
        ("gemini-3.1-flash-lite-image", "16:9 4K", "", ("16:9", "1K")),
        ("gemini-2.5-flash-image", "16:9 4K", "", ("16:9", "")),
        (DEFAULT_MODEL, "1024 x 512", "", ("16:9", "")),  # 像素尺寸，不是档位
        (DEFAULT_MODEL, "1024x512", "", ("16:9", "")),
        (DEFAULT_MODEL, "3:4K", "", ("2:3", "")),  # 无法识别，回到默认
    ],
)
def test_size_with_tier_suffix(model, value, image_size, expected):
    backend = GeminiBackend(section(model=model, portrait_size=value, image_size=image_size))
    assert backend.image_config("portrait") == expected


def test_huge_size_numbers_do_not_crash():
    # 超长数字视为无法识别，使用该方向的默认比例
    backend = GeminiBackend(section(portrait_size="1" + "0" * 400 + ":1"))
    assert backend.image_config("portrait") == ("2:3", "")
    backend = GeminiBackend(section(portrait_size="1x" + "9" * 400))
    assert backend.image_config("portrait") == ("2:3", "")


def test_size_512_inferred_only_where_supported():
    backend = GeminiBackend(section(model="gemini-3.1-flash-image", square_size="512x512"))
    assert backend.image_config("square") == ("1:1", "512")
    backend = GeminiBackend(section(square_size="512x512"))
    assert backend.image_config("square") == ("1:1", "")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", DEFAULT_BASE_URL),
        (None, DEFAULT_BASE_URL),
        ("generativelanguage.googleapis.com", DEFAULT_BASE_URL),
        ("https://generativelanguage.googleapis.com/", DEFAULT_BASE_URL),
        ("https://generativelanguage.googleapis.com/v1beta/openai/", DEFAULT_BASE_URL),
        ("https://generativelanguage.googleapis.com/v1beta/models", DEFAULT_BASE_URL),
        (f"{DEFAULT_BASE_URL}/models/gemini-3-pro-image:generateContent", DEFAULT_BASE_URL),
        (f"{DEFAULT_BASE_URL}/interactions", DEFAULT_BASE_URL),
        ("https://proxy.example.com/gemini", "https://proxy.example.com/gemini/v1beta"),
        ("http://127.0.0.1:8080/v1/", "http://127.0.0.1:8080/v1"),
        ("  https://proxy.example.com/v1alpha  ", "https://proxy.example.com/v1alpha"),
        # 误粘贴的查询参数（可能带着 Key）要去掉
        (f"{DEFAULT_BASE_URL}?key=AIzaSecret", DEFAULT_BASE_URL),
        (f"{DEFAULT_BASE_URL}/models/gemini-3-pro-image:generateContent?key=AIzaSecret#x", DEFAULT_BASE_URL),
        ("https://proxy.example.com/?key=AIzaSecret", "https://proxy.example.com/v1beta"),
    ],
)
def test_base_url_normalisation(value, expected):
    assert gemini_base_url(value) == expected
    assert GeminiBackend(section(base_url=value)).base_url == expected


def test_custom_base_url_used_for_request():
    rec = Recorder(gc_ok())
    run(GeminiBackend(section(base_url="proxy.example.com")), rec)
    assert str(rec.requests[0].url) == f"https://proxy.example.com/v1beta/models/{DEFAULT_MODEL}:generateContent"


def test_negative_prompt_and_seed():
    rec = Recorder(gc_ok())
    result = run(GeminiBackend(section()), rec, DrawRequest("a cat", negative="dogs", seed=2**32 + 5))
    body = rec.body()
    assert body["contents"][0]["parts"][0]["text"] == "a cat\n\nAvoid: dogs"
    assert body["generationConfig"]["seed"] == 5
    assert result.params["seed"] == 5
    rec = Recorder(gc_ok())
    run(GeminiBackend(section()), rec, DrawRequest("a cat", seed=-1))
    assert rec.body()["generationConfig"]["seed"] == 2**31 - 1


def test_extra_body_merged_last():
    extra = {
        "generationConfig": {"imageConfig": {"imageSize": "4K"}, "responseModalities": ["IMAGE"]},
        "safetySettings": [{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"}],
        "contents": None,
    }
    rec = Recorder(gc_ok())
    run(GeminiBackend(section(extra_body=json.dumps(extra), image_size="2K")), rec)
    body = rec.body()
    assert body["generationConfig"] == {"responseModalities": ["IMAGE"],
                                        "imageConfig": {"aspectRatio": "2:3", "imageSize": "4K"}}
    assert body["safetySettings"] == extra["safetySettings"]
    assert "contents" not in body
    # 已经是对象的配置值也可以
    rec = Recorder(gc_ok())
    run(GeminiBackend(section(extra_body={"labels": {"a": "b"}})), rec)
    assert rec.body()["labels"] == {"a": "b"}


def test_extra_body_null_in_new_object_and_config_untouched():
    extra = {"generationConfig": {"thinkingConfig": {"thinkingLevel": None, "includeThoughts": False}},
             "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"}],
             "toolConfig": {"a": {"b": None, "c": 1}}}
    snapshot = json.loads(json.dumps(extra))
    rec = Recorder(gc_ok())
    run(GeminiBackend(section(extra_body=extra)), rec)
    body = rec.body()
    assert body["generationConfig"]["thinkingConfig"] == {"includeThoughts": False}
    assert body["toolConfig"] == {"a": {"c": 1}}
    assert extra == snapshot  # 配置中的对象没有被改动


@pytest.mark.parametrize("extra", ["{bad json", "[1, 2]"])
def test_extra_body_invalid_fails_before_request(extra):
    rec = Recorder(gc_ok())
    with pytest.raises(DrawError, match="额外请求参数"):
        run(GeminiBackend(section(extra_body=extra)), rec)
    assert rec.requests == []


def test_interactions_request_and_response():
    rec = Recorder(ia_ok())
    backend = GeminiBackend(section(api="interactions", image_size="2K", thinking_level="HIGH"))
    result = run(backend, rec, DrawRequest("a cat", negative="dogs", orientation="landscape", seed=7))
    req = rec.requests[0]
    assert str(req.url) == IA_URL and req.headers["x-goog-api-key"] == KEY
    assert rec.body() == {
        "model": DEFAULT_MODEL,
        "input": "a cat\n\nAvoid: dogs",
        "response_format": {"type": "image", "aspect_ratio": "3:2", "image_size": "2K"},
        "store": False,
        "generation_config": {"thinking_level": "high", "seed": 7},
    }
    # 取最后一个 model_output 图片块，忽略 thought 步骤里的草图
    assert result.image == PNG and result.text == "done"
    assert result.params["api"] == "Interactions" and result.params["size"] == "3:2 2K"


def test_interactions_never_sends_mime_or_delivery():
    for model in MODEL_CAPS:
        rec = Recorder(ia_ok())
        run(GeminiBackend(section(api="interactions", model=model, image_size="4K", thinking_level="medium")), rec)
        fmt = rec.body()["response_format"]
        assert "mime_type" not in fmt and "delivery" not in fmt
        assert fmt["aspect_ratio"] in MODEL_CAPS[model].ratios
        assert fmt.get("image_size", "1K") in (MODEL_CAPS[model].sizes or ("1K",))
        assert "image_size" not in fmt or MODEL_CAPS[model].sizes
        assert rec.body()["store"] is False


def test_interactions_legacy_outputs_and_uri_download():
    rec = Recorder(httpx.Response(200, json={"outputs": [{"type": "image", "data": B64}]}))
    assert run(GeminiBackend(section(api="interactions")), rec).image == PNG

    def handler(request):
        if request.url.path.endswith("/interactions"):
            return httpx.Response(200, json={"status": "completed", "steps": [
                {"type": "model_output", "content": [{"type": "image", "uri": "https://cdn.example.com/a.jpg"}]}]})
        assert "x-goog-api-key" not in request.headers  # Key 不发给其他主机
        return httpx.Response(200, content=JPEG)

    rec = Recorder(handler)
    result = run(GeminiBackend(section(api="interactions")), rec)
    assert result.image == JPEG and result.mime == "image/jpeg"
    assert str(rec.requests[1].url) == "https://cdn.example.com/a.jpg"


@pytest.mark.parametrize(
    ("uri", "with_key"),
    [
        (f"{DEFAULT_BASE_URL}/files/a.jpg", True),
        ("http://generativelanguage.googleapis.com/v1beta/files/a.jpg", False),  # 明文 http 不带 Key
        ("https://generativelanguage.googleapis.com:8443/a.jpg", False),
        ("https://generativelanguage.googleapis.com.evil.example/a.jpg", False),
    ],
)
def test_interactions_uri_key_only_for_same_origin(uri, with_key):
    def handler(request):
        if request.url.path.endswith("/interactions"):
            return httpx.Response(200, json={"status": "completed", "steps": [
                {"type": "model_output", "content": [{"type": "image", "uri": uri}]}]})
        assert ("x-goog-api-key" in request.headers) is with_key
        return httpx.Response(200, content=JPEG)

    rec = Recorder(handler)
    assert run(GeminiBackend(section(api="interactions")), rec).image == JPEG
    assert len(rec.requests) == 2


def test_auto_falls_back_to_interactions_on_404():
    message = "models/x is not found for API version v1beta, or is not supported for generateContent."
    not_found = httpx.Response(404, json={"error": {"code": 404, "status": "NOT_FOUND", "message": message}})
    rec = Recorder(not_found, ia_ok())
    result = run(GeminiBackend(section(extra_body='{"safetySettings": []}')), rec)
    assert [str(r.url) for r in rec.requests] == [GC_URL, IA_URL]
    assert "safetySettings" in rec.body(0) and "safetySettings" not in rec.body(1)
    assert result.image == PNG and result.params["api"] == "Interactions"


def test_auto_fallback_both_404_reports_first_error():
    rec = Recorder(httpx.Response(404, json={"error": {"code": 404, "message": "gc missing", "status": "NOT_FOUND"}}),
                   httpx.Response(404, json={"error": {"code": "model_not_found", "message": "ia missing"}}))
    with pytest.raises(DrawError, match="gc missing") as info:
        run(GeminiBackend(section()), rec)
    assert info.value.status == 404 and len(rec.requests) == 2


@pytest.mark.parametrize("api", ["generate_content", "generateContent"])
def test_no_fallback_when_api_is_fixed(api):
    rec = Recorder(httpx.Response(404, json={"error": {"code": 404, "message": "nope", "status": "NOT_FOUND"}}))
    with pytest.raises(DrawError, match="HTTP 404"):
        run(GeminiBackend(section(api=api)), rec)
    assert len(rec.requests) == 1


def test_no_fallback_on_400():
    rec = Recorder(httpx.Response(400, json={"error": {"code": 400, "message": "Invalid value at 'x'",
                                                       "status": "INVALID_ARGUMENT"}}))
    with pytest.raises(DrawError, match="请求参数有误"):
        run(GeminiBackend(section()), rec)
    assert len(rec.requests) == 1


# ---------------------------------------------------------------- 响应解析


def test_response_skips_thoughts_and_takes_last_image():
    thought = {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(JPEG).decode()}, "thought": True}
    rec = Recorder(gc_ok({"text": "thinking", "thought": True}, thought, {"text": "Here you go"},
                         {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(JPEG).decode()}},
                         {"inline_data": {"mime_type": "image/png", "data": B64}, "thoughtSignature": "sig"}))
    result = run(GeminiBackend(section()), rec)
    assert result.image == PNG and result.text == "Here you go"


def test_response_urlsafe_base64_and_declared_mime_ignored():
    data = PNG + b"\xfb\xff\xfe"
    urlsafe = base64.urlsafe_b64encode(data).decode().rstrip("=")
    assert "-" in urlsafe or "_" in urlsafe
    rec = Recorder(gc_ok({"inlineData": {"mimeType": "image/jpeg", "data": urlsafe}}))
    result = run(GeminiBackend(section()), rec)
    assert result.image == data and result.mime == "image/png"


@pytest.mark.parametrize(
    ("payload", "pattern"),
    [
        ({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}, "提示词被安全策略拦截（PROHIBITED_CONTENT）"),
        ({"candidates": [{"finishReason": "IMAGE_SAFETY", "finishMessage": "unsafe"}]},
         "被安全策略拦截（IMAGE_SAFETY）.*unsafe"),
        ({"candidates": [{"content": {"parts": [{"text": "no"}]}, "finishReason": "SAFETY"}]}, "被安全策略拦截（SAFETY）"),
        ({"candidates": [{"finishReason": "NO_IMAGE"}]}, "没能生成图片（NO_IMAGE）"),
        ({"candidates": [{"finishReason": "MAX_TOKENS"}]}, "MAX_TOKENS"),
        ({"candidates": [{"content": {"parts": [{"text": "What style do you want?"}]}, "finishReason": "STOP"}]},
         "模型回复：What style do you want\\?"),
        ({"candidates": []}, "没有返回图片"),
        ({}, "没有返回图片"),
    ],
)
def test_no_image_responses(payload, pattern, sleeps):
    rec = Recorder(httpx.Response(200, json=payload))
    with pytest.raises(DrawError, match=pattern) as info:
        run(GeminiBackend(section()), rec)
    assert not info.value.retryable and len(rec.requests) == 1 and sleeps == []


def test_non_json_and_bad_image_data():
    rec = Recorder(httpx.Response(200, text="<html><title>Bad gateway page</title></html>"))
    with pytest.raises(DrawError, match="不是 JSON"):
        run(GeminiBackend(section()), rec)
    rec = Recorder(gc_ok({"inlineData": {"mimeType": "image/png", "data": base64.b64encode(b"hello").decode()}}))
    with pytest.raises(DrawError, match="不是可识别的图片"):
        run(GeminiBackend(section()), rec)


@pytest.mark.parametrize(
    ("api", "payload", "pattern", "status"),
    [
        ("generate_content", {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Resource exhausted"}},
         "HTTP 429.*Resource exhausted", 429),
        ("generate_content", {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                        "message": "API key not valid.", "details": [{"reason": "API_KEY_INVALID"}]}},
         "API Key 无效", 400),
        ("interactions", {"error": {"code": "image_safety", "message": "blocked"}}, "被安全策略拦截（image_safety）", 200),
        ("interactions", {"error": {"message": "upstream exploded"}}, "HTTP 200.*upstream exploded", 200),
    ],
)
def test_error_in_200_body(api, payload, pattern, status, sleeps):
    """部分中转服务出错时返回 HTTP 200 + error 对象：要给出错误说明，而不是「没有返回图片」。"""
    rec = Recorder(httpx.Response(200, json=payload), gc_ok())
    with pytest.raises(DrawError, match=pattern) as info:
        run(GeminiBackend(section(api=api)), rec)
    assert info.value.status == status and len(rec.requests) == 1 and sleeps == []


def test_error_in_200_with_404_falls_back_in_auto():
    rec = Recorder(httpx.Response(200, json={"error": {"code": 404, "status": "NOT_FOUND", "message": "no"}}), ia_ok())
    result = run(GeminiBackend(section()), rec)
    assert [str(r.url) for r in rec.requests] == [GC_URL, IA_URL] and result.params["api"] == "Interactions"


def test_interactions_refusals(sleeps):
    rec = Recorder(httpx.Response(400, json={"error": {"code": "image_safety", "message":
                                                       "Request blocked due to safety violations."}}))
    with pytest.raises(DrawError, match="被安全策略拦截（image_safety）") as info:
        run(GeminiBackend(section(api="interactions")), rec)
    assert info.value.status == 400 and not info.value.retryable and len(rec.requests) == 1

    rec = Recorder(httpx.Response(400, json={"error": {"code": "no_image", "message": "unable"}}))
    with pytest.raises(DrawError, match="没能生成图片"):
        run(GeminiBackend(section(api="interactions")), rec)

    rec = Recorder(httpx.Response(200, json={"status": "failed", "steps": [],
                                             "errors": [{"code": "prohibited_content", "message": "blocked"}]}))
    with pytest.raises(DrawError, match="被安全策略拦截（prohibited_content）.*blocked"):
        run(GeminiBackend(section(api="interactions")), rec)

    rec = Recorder(httpx.Response(200, json={"status": "incomplete", "steps": []}))
    with pytest.raises(DrawError, match="生成失败（incomplete）"):
        run(GeminiBackend(section(api="interactions")), rec)
    assert sleeps == []


# ---------------------------------------------------------------- 错误与重试


def test_invalid_key_400_and_list_wrapped():
    body = {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "API key not valid. Please pass a valid "
                      "API key.", "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                                               "reason": "API_KEY_INVALID"}]}}
    for api, payload in (("generate_content", body), ("interactions", [body])):
        rec = Recorder(httpx.Response(400, json=payload))
        with pytest.raises(DrawError, match="API Key 无效") as info:
            run(GeminiBackend(section(api=api)), rec)
        assert info.value.status == 400 and not info.value.retryable and len(rec.requests) == 1


@pytest.mark.parametrize("status", [400, 200])
def test_key_echoed_by_proxy_is_masked(status):
    body = {"error": {"code": "invalid_request", "message": f"bad header x-goog-api-key: {KEY}"}}
    rec = Recorder(httpx.Response(status, json=body))
    with pytest.raises(DrawError) as info:
        run(GeminiBackend(section(base_url="proxy.example.com")), rec)
    assert KEY not in str(info.value) and "••••" in str(info.value)
    rec = Recorder(httpx.Response(status, json=body))
    with pytest.raises(DrawError) as info:
        run(GeminiBackend(section(base_url="proxy.example.com")), rec, models=True)
    assert KEY not in str(info.value)


@pytest.mark.parametrize("key", ["AIzaSecretKey\nX", "AIzaSecretKey\tX", "AIzaＳｅｃｒｅｔ", "AIza​Secret"])
def test_key_with_bad_characters_rejected_before_request(key):
    """含换行时 httpx 的报错会带上整个请求头（即 Key）并发到聊天里；含全角字符时会抛出非 DrawError 的异常。"""
    backend = GeminiBackend({"api_key": key})
    assert "换行、全角或不可见字符" in backend.missing()
    rec = Recorder(gc_ok())
    for models in (False, True):
        with pytest.raises(DrawError, match="不可见字符") as info:
            run(backend, rec, models=models)
        assert key not in str(info.value)
    assert rec.requests == []


def test_401_not_retried(sleeps):
    rec = Recorder(httpx.Response(401, json={"error": {"code": "authentication", "message": "bad key"}}))
    with pytest.raises(DrawError, match="API Key 无效或已过期（HTTP 401）") as info:
        run(GeminiBackend(section()), rec)
    assert info.value.status == 401 and len(rec.requests) == 1 and sleeps == []


def test_region_not_supported():
    rec = Recorder(httpx.Response(400, json={"error": {"code": 400, "status": "FAILED_PRECONDITION",
                                                       "message": "User location is not supported for the API use."}}))
    with pytest.raises(DrawError, match="地区.*代理"):
        run(GeminiBackend(section()), rec)


@pytest.mark.parametrize(
    "payload",
    [
        {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "You exceeded your current quota.",
                   "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                       {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
                               {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "1s"}]}},
        {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                   "message": "Quota exceeded for metric: x, limit: 0, model: gemini-3-pro-image"}},
        {"error": {"code": "quota_exceeded", "message": "You have exceeded your daily quota."}},
    ],
)
def test_429_quota_exhausted_not_retried(payload, sleeps):
    rec = Recorder(httpx.Response(429, json=payload), gc_ok())
    with pytest.raises(DrawError, match="额度已用尽") as info:
        run(GeminiBackend(section()), rec)
    assert info.value.status == 429 and not info.value.retryable
    assert len(rec.requests) == 1 and sleeps == []


def test_402_not_retried(sleeps):
    rec = Recorder(httpx.Response(402, json={"error": {"code": "payment_required", "message": "depleted"}}), gc_ok())
    with pytest.raises(DrawError, match="余额不足"):
        run(GeminiBackend(section()), rec)
    assert len(rec.requests) == 1 and sleeps == []


def test_429_rate_limit_retried_once(sleeps):
    limited = httpx.Response(429, json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message":
                                                  "Quota exceeded for metric: x, limit: 10. Please retry in 2s.",
                                                  "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                                               "retryDelay": "2s"}]}})
    rec = Recorder(limited, gc_ok())
    assert run(GeminiBackend(section()), rec).image == PNG
    assert len(rec.requests) == 2 and sleeps == [2.0]

    rec = Recorder(limited, limited)
    with pytest.raises(DrawError, match="HTTP 429"):
        run(GeminiBackend(section()), rec)
    assert len(rec.requests) == 2

    ia_limited = httpx.Response(429, json={"error": {"code": "rate_limit_exceeded", "message": "slow down"}},
                                headers={"Retry-After": "1"})
    rec = Recorder(ia_limited, ia_ok())
    assert run(GeminiBackend(section(api="interactions")), rec).image == PNG
    assert len(rec.requests) == 2 and sleeps[-1] == 1.0


def test_429_long_retry_delay_not_retried(sleeps):
    rec = Recorder(httpx.Response(429, json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "x",
                                                       "details": [{"retryDelay": "300s"}]}}), gc_ok())
    with pytest.raises(DrawError, match="约 300 秒后再试"):
        run(GeminiBackend(section()), rec)
    assert len(rec.requests) == 1 and sleeps == []


@pytest.mark.parametrize(("status", "retried"), [(503, True), (502, True), (500, False), (504, False)])
def test_5xx_retry_policy(status, retried, sleeps):
    rec = Recorder(httpx.Response(status, json={"error": {"code": status, "message": "The model is overloaded."}}),
                   gc_ok())
    if retried:
        assert run(GeminiBackend(section()), rec).image == PNG
        assert len(rec.requests) == 2
    else:
        with pytest.raises(DrawError, match=f"HTTP {status}"):
            run(GeminiBackend(section()), rec)
        assert len(rec.requests) == 1


def test_timeout_not_retried(sleeps):
    rec = Recorder(httpx.ReadTimeout("slow"), gc_ok())
    with pytest.raises(DrawError, match="超时") as info:
        run(GeminiBackend(section()), rec)
    assert not info.value.retryable and len(rec.requests) == 1 and sleeps == []


def test_connect_error_retried_once(sleeps):
    rec = Recorder(httpx.ConnectError("refused"), gc_ok())
    assert run(GeminiBackend(section()), rec).image == PNG
    assert len(rec.requests) == 2 and len(sleeps) == 1


# ---------------------------------------------------------------- 配置与模型列表


def test_missing_and_class_attributes():
    assert GeminiBackend({}).missing() == "管理员还没有配置 Gemini API Key"
    assert GeminiBackend(None).missing() == "管理员还没有配置 Gemini API Key"
    assert GeminiBackend({"api_key": "  "}).missing() == "管理员还没有配置 Gemini API Key"
    assert GeminiBackend({"api_key": " k "}).missing() == ""
    assert GeminiBackend.key == "gemini" and GeminiBackend.label == "Gemini"
    assert GeminiBackend.default_prompt_mode == "natural" and GeminiBackend({}).prompt_mode == "natural"
    assert GeminiBackend.supports_negative is False and GeminiBackend.supports_models is True
    assert GeminiBackend.max_concurrency is None
    assert GeminiBackend.doc_url.startswith("https://ai.google.dev/")
    with pytest.raises(DrawError, match="API Key"):
        run(GeminiBackend({}), Recorder(gc_ok()))


def test_summary():
    assert GeminiBackend({}).summary() == f"Gemini / {DEFAULT_MODEL} / 竖图 2:3"
    assert GeminiBackend({"model": "gemini-3-pro-image", "image_size": "2K", "portrait_size": "9:16"}).summary() \
        == "Gemini / gemini-3-pro-image / 竖图 9:16 2K"
    assert GeminiBackend({"model": "gemini-3-pro-image-preview"}).summary() \
        == "Gemini / gemini-3-pro-image（gemini-3-pro-image-preview 已停用） / 竖图 2:3"


@pytest.mark.parametrize(
    "garbage",
    [
        {"api_key": KEY, "model": None, "portrait_size": None, "image_size": None, "thinking_level": None,
         "api": None, "timeout": None, "extra_body": None, "base_url": None, "proxy_url": None},
        {"api_key": KEY, "model": "", "portrait_size": "", "image_size": "", "thinking_level": "", "api": "",
         "timeout": "", "extra_body": "", "base_url": "", "proxy_url": ""},
        {"api_key": KEY, "model": "   ", "portrait_size": 123, "image_size": 7, "thinking_level": ["x"],
         "api": 5, "timeout": "abc", "extra_body": "   ", "base_url": "  ", "proxy_url": "  "},
    ],
)
def test_garbage_config_values(garbage):
    backend = GeminiBackend(garbage)
    assert backend.timeout == DEFAULT_TIMEOUT and backend.api == "auto" and backend.model == DEFAULT_MODEL
    rec = Recorder(gc_ok())
    run(backend, rec)
    assert str(rec.requests[0].url) == GC_URL
    assert_valid_gc_body(rec.body(), DEFAULT_MODEL)
    assert rec.body()["generationConfig"]["imageConfig"] == {"aspectRatio": "2:3"}
    assert backend.summary().startswith("Gemini / ")


def test_timeout_bounds():
    assert GeminiBackend({"timeout": 5}).timeout == 10.0
    assert GeminiBackend({"timeout": "600"}).timeout == 600.0
    assert GeminiBackend({"timeout": 99999}).timeout == 1800.0


def test_list_models_paginates_and_filters():
    pages = {
        None: {"models": [
            {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-nano-banana-2.1", "supportedGenerationMethods": ["generateContent", "countTokens"]},
            {"name": "models/imagen-4.0-generate-001", "supportedGenerationMethods": ["predict"]},
            {"name": "models/gemini-3-pro-image", "supportedGenerationMethods": ["generateContent"]},
            "garbage",
        ], "nextPageToken": "p2"},
        "p2": {"models": [
            {"name": "models/nano-banana-pro-preview", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.1-flash-lite-image"},
            {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/gemini-3-pro-image", "supportedGenerationMethods": ["generateContent"]},
        ]},
    }

    def handler(request):
        assert request.method == "GET" and request.url.path == "/v1beta/models"
        assert request.headers["x-goog-api-key"] == KEY
        assert request.url.params["pageSize"] == "1000"
        return httpx.Response(200, json=pages[request.url.params.get("pageToken")])

    rec = Recorder(handler)
    models = run(GeminiBackend(section()), rec, models=True)
    assert models == ["gemini-nano-banana-2.1", "gemini-3-pro-image", "nano-banana-pro-preview",
                      "gemini-3.1-flash-lite-image"]
    assert len(rec.requests) == 2 and "key=" not in str(rec.requests[0].url)


def test_list_models_errors():
    with pytest.raises(DrawError, match="API Key"):
        run(GeminiBackend({}), Recorder(gc_ok()), models=True)
    rec = Recorder(httpx.Response(403, json={"error": {"code": 403, "status": "PERMISSION_DENIED",
                                                       "message": "Method doesn't allow unregistered callers"}}))
    with pytest.raises(DrawError, match="HTTP 403"):
        run(GeminiBackend(section()), rec, models=True)


# ---------------------------------------------------------------- 配置结构


ALLOWED_TYPES = {"string", "text", "int", "float", "bool", "list"}


def test_schema_section_shape():
    sec = schema_section()
    assert sec["type"] == "object" and sec["description"].endswith("设置")
    items = sec["items"]
    assert list(items) == ["api_key", "base_url", "proxy_url", "model", "portrait_size", "landscape_size",
                           "square_size", "image_size", "thinking_level", "api", "timeout", "extra_body",
                           "track_usage", "price_per_image", "price_input_per_million",
                           "price_output_per_million", "price_currency"]
    for name, field in items.items():
        assert field["type"] in ALLOWED_TYPES, name
        assert field["description"] and field["hint"], name
        if "options" in field:
            assert field["default"] in field["options"], name
            assert len(field.get("labels", field["options"])) == len(field["options"]), name
    assert items["api_key"]["secret"] is True
    assert "options" not in items["model"]
    assert items["extra_body"]["type"] == "text" and items["extra_body"]["editor_mode"] is True
    assert items["extra_body"]["editor_language"] == "json" and items["extra_body"]["default"] == ""
    assert items["timeout"]["type"] == "int"


def test_schema_defaults_match_code():
    items = schema_section()["items"]
    assert items["base_url"]["default"] == DEFAULT_BASE_URL
    assert items["model"]["default"] == DEFAULT_MODEL
    assert items["timeout"]["default"] == DEFAULT_TIMEOUT
    for orientation, ratio in DEFAULT_SIZES.items():
        assert items[f"{orientation}_size"]["default"] == ratio
    assert items["api"]["default"] == GeminiBackend({}).api
    assert set(items["api"]["options"]) == {"auto", "generate_content", "interactions"}
    assert items["image_size"]["default"] == "" and items["thinking_level"]["default"] == ""
    # 选项都是代码能识别的值
    for value in items["image_size"]["options"][1:]:
        backend = GeminiBackend({"model": "gemini-3.1-flash-image", "image_size": value})
        assert backend.image_config("square")[1] == value
    for value in items["thinking_level"]["options"][1:]:
        assert GeminiBackend({"thinking_level": value}).thinking_level == value


@pytest.mark.parametrize("api", ["auto", "generate_content", "interactions"])
@pytest.mark.parametrize("orientation", ["portrait", "landscape", "square"])
def test_schema_defaults_produce_valid_request(api, orientation):
    defaults = {name: field["default"] for name, field in schema_section()["items"].items()}
    defaults.update(api_key=KEY, api=api)
    backend = GeminiBackend(defaults)
    rec = Recorder(ia_ok() if api == "interactions" else gc_ok())
    result = run(backend, rec, DrawRequest("a cat", orientation=orientation))
    assert result.image == PNG
    if api == "interactions":
        assert str(rec.requests[0].url) == IA_URL
        assert rec.body() == {"model": DEFAULT_MODEL, "input": "a cat", "store": False,
                              "response_format": {"type": "image", "aspect_ratio": DEFAULT_SIZES[orientation]}}
    else:
        assert str(rec.requests[0].url) == GC_URL
        assert_valid_gc_body(rec.body(), DEFAULT_MODEL)
        assert rec.body()["generationConfig"]["imageConfig"] == {"aspectRatio": DEFAULT_SIZES[orientation]}
    assert DEFAULT_SIZES[orientation] in RATIOS


# ---------------------------------------------------------------- 修改图片


WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 8
GIF = b"GIF89a" + b"\x00" * 16


def png_in(width: int = 900, height: int = 1200, mime: str = "image/png") -> InputImage:
    return InputImage(PNG, mime, width, height)


def jpeg_in(width: int = 900, height: int = 1200, mime: str = "image/jpeg") -> InputImage:
    return InputImage(JPEG, mime, width, height)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def edit_req(*images: InputImage, prompt: str = "make the sky red", orientation: str = "portrait",
             **kw) -> DrawRequest:
    return DrawRequest(prompt, orientation=orientation, images=list(images or (png_in(),)), **kw)


def test_edit_class_attributes():
    assert GeminiBackend.max_edit_images == MAX_EDIT_IMAGES == 14
    assert GeminiBackend.supports_strength is False and GeminiBackend.supports_balance is False
    assert GeminiBackend.billing_url == BILLING_URL == "https://aistudio.google.com/billing"
    # 每个已知模型都能修改图片，不超过类上声明的最大值
    for model, caps in MODEL_CAPS.items():
        backend = GeminiBackend(section(model=model))
        assert backend.edit_missing() == "", model
        assert 0 < backend.edit_limit == caps.edit_images <= GeminiBackend.max_edit_images, model
    assert GeminiBackend(section(model="gemini-2.5-flash-image")).edit_limit == 3
    assert GeminiBackend(section(model="gemini-3-pro-image-001")).edit_limit == 14  # 版本后缀
    assert GeminiBackend(section(model="some-relay-alias")).edit_limit == 14  # 未知模型
    assert GeminiBackend({}).edit_missing() == ""


@pytest.mark.parametrize(("model", "limit"), [*((m, c.edit_images) for m, c in MODEL_CAPS.items()),
                                              ("gemini-2.5-flash-image-preview", 14),  # 换成替代模型
                                              ("some-relay-alias", 14)])
def test_edit_instance_limit_follows_model(model, limit):
    """插件收集图片、网页试画限制图片数都读实例上的 max_edit_images：要按当前模型给出，不能超过类上的最大值。"""
    backend = GeminiBackend(section(model=model))
    assert backend.max_edit_images == backend.edit_limit == limit <= GeminiBackend.max_edit_images
    assert GeminiBackend.max_edit_images == MAX_EDIT_IMAGES  # 类属性不受实例影响
    assert GeminiBackend(None).max_edit_images == MAX_EDIT_IMAGES


def test_edit_generate_content_request_shape():
    rec = Recorder(gc_ok({"text": "Here it is"}, {"inlineData": {"mimeType": "image/jpeg", "data": b64(JPEG)}}))
    req = edit_req(png_in(900, 1200), jpeg_in(100, 100, mime="image/jpg"), seed=9)
    result = run(GeminiBackend(section()), rec, req)
    assert len(rec.requests) == 1
    request = rec.requests[0]
    assert request.method == "POST" and str(request.url) == GC_URL
    assert request.headers["content-type"] == "application/json"
    assert request.headers["x-goog-api-key"] == KEY and "key=" not in str(request.url)
    # 文字在前，图片按用户顺序在后；MIME 按文件头识别（image/jpg → image/jpeg）；base64 不带 data: 前缀
    assert rec.body() == {
        "contents": [{"role": "user", "parts": [
            {"text": "make the sky red"},
            {"inlineData": {"mimeType": "image/png", "data": b64(PNG)}},
            {"inlineData": {"mimeType": "image/jpeg", "data": b64(JPEG)}},
        ]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": {"aspectRatio": "3:4"},
                             "seed": 9},
    }
    assert result.image == JPEG and result.mime == "image/jpeg" and result.text == "Here it is"
    assert result.params == {"model": DEFAULT_MODEL, "size": "3:4", "aspect_ratio": "3:4", "api": "generateContent",
                             "mode": "edit", "input_images": 2, "seed": 9, "usage": {"images": 1}}


@pytest.mark.parametrize(
    ("model", "width", "height", "orientation", "extra", "image_config"),
    [
        (DEFAULT_MODEL, 900, 1200, "portrait", {}, {"aspectRatio": "3:4"}),  # 保持原图比例而不是配置的 2:3
        (DEFAULT_MODEL, 300, 1200, "portrait", {}, {"aspectRatio": "1:4"}),
        ("gemini-3-pro-image", 300, 1200, "portrait", {}, {"aspectRatio": "9:16"}),  # pro 没有 1:4
        (DEFAULT_MODEL, 1920, 1080, "landscape", {"image_size": "2K"}, {"aspectRatio": "16:9", "imageSize": "2K"}),
        (DEFAULT_MODEL, 1000, 1000, "square", {}, {"aspectRatio": "1:1"}),
        (DEFAULT_MODEL, 2048, 1800, "square", {}, {"aspectRatio": "5:4"}),  # 接近方图时也取最接近原图的比例
        # 用户要求换方向时用配置的比例
        (DEFAULT_MODEL, 900, 1200, "landscape", {"landscape_size": "16:9"}, {"aspectRatio": "16:9"}),
        (DEFAULT_MODEL, 1920, 1080, "portrait", {}, {"aspectRatio": "2:3"}),
        ("gemini-2.5-flash-image", 1080, 1920, "portrait", {"image_size": "4K"}, {"aspectRatio": "9:16"}),
    ],
)
def test_edit_keeps_input_aspect_ratio(model, width, height, orientation, extra, image_config):
    rec = Recorder(gc_ok())
    backend = GeminiBackend(section(model=model, **extra))
    result = run(backend, rec, edit_req(png_in(width, height), orientation=orientation))
    config = rec.body()["generationConfig"]
    assert config["imageConfig"] == image_config
    assert image_config["aspectRatio"] in MODEL_CAPS[model].ratios
    assert result.params["aspect_ratio"] == image_config["aspectRatio"]
    assert result.params["size"] == " ".join(filter(None, [image_config["aspectRatio"],
                                                           image_config.get("imageSize")]))
    # 记录中预计的尺寸与实际发送的一致
    plan = backend.plan(edit_req(png_in(width, height), orientation=orientation))
    assert plan == {"model": model, "size": result.params["size"], "mode": "edit", "input_images": 1}


def test_edit_unknown_input_size_follows_reference_image():
    """不知道原图尺寸时不发送宽高比，由接口按参考图决定；没有档位时不发送空的 imageConfig。"""
    rec = Recorder(gc_ok())
    result = run(GeminiBackend(section(model="gemini-2.5-flash-image")), rec, edit_req(png_in(0, 0)))
    assert rec.body()["generationConfig"] == {"responseModalities": ["TEXT", "IMAGE"]}
    assert result.params["size"] == "跟随原图" and result.params["aspect_ratio"] == "auto"

    rec = Recorder(gc_ok())
    result = run(GeminiBackend(section(image_size="2K")), rec, edit_req(png_in(0, 0)))
    assert rec.body()["generationConfig"]["imageConfig"] == {"imageSize": "2K"}
    assert result.params["size"] == "跟随原图 2K"

    rec = Recorder(ia_ok())
    run(GeminiBackend(section(api="interactions")), rec, edit_req(png_in(0, 0)))
    assert rec.body()["response_format"] == {"type": "image"}


@pytest.mark.parametrize(("width", "height"), [(-900, 1200), (900, -1200), (10**400, 1), (1, 10**7),
                                               ("abc", 1200), (None, None), (900, 0)])
def test_edit_invalid_input_size_treated_as_unknown(width, height):
    """尺寸无效时当作不知道尺寸（不发送宽高比），而不是在 math.log / 除法里抛出非 DrawError 的异常。"""
    rec = Recorder(gc_ok())
    req = edit_req(InputImage(PNG, "image/png", width, height))
    backend = GeminiBackend(section())
    assert backend.plan(req)["size"] == "跟随原图"
    result = run(backend, rec, req)
    assert "imageConfig" not in rec.body()["generationConfig"] and result.params["aspect_ratio"] == "auto"


def test_edit_unrecognized_orientation_keeps_input_shape():
    rec = Recorder(gc_ok())
    result = run(GeminiBackend(section()), rec, edit_req(png_in(1920, 1080), orientation="bogus"))
    assert rec.body()["generationConfig"]["imageConfig"] == {"aspectRatio": "16:9"}
    assert result.params["aspect_ratio"] == "16:9"


@pytest.mark.parametrize("api", ["generate_content", "interactions"])
@pytest.mark.parametrize("model", list(MODEL_CAPS))
def test_edit_every_model_family_sends_only_known_fields(model, api):
    """每个模型、两种接口：PNG（透明）+ JPEG、奇数尺寸、极端比例、strength 都不会带出该模型不接受的值或未知字段。"""
    caps = MODEL_CAPS[model]
    images = [png_in(333, 1999), jpeg_in(4001, 7), png_in(1, 1)]
    rec = Recorder(ia_ok() if api == "interactions" else gc_ok())
    backend = GeminiBackend(section(model=model, api=api, image_size="4K", thinking_level="medium"))
    result = run(backend, rec, edit_req(*images, strength=1.0, seed=3))
    body = rec.body()
    sent = min(len(images), caps.edit_images)
    assert "strength" not in json.dumps(body) and "mask" not in json.dumps(body)
    if api == "interactions":
        assert set(body) == {"model", "input", "response_format", "store", "generation_config"}
        assert body["input"][0] == {"type": "text", "text": "make the sky red"}
        assert [p["mime_type"] for p in body["input"][1:]] == ["image/png", "image/jpeg", "image/png"][:sent]
        assert all(set(p) == {"type", "mime_type", "data"} for p in body["input"][1:])
        fmt = body["response_format"]
        assert set(fmt) <= {"type", "aspect_ratio", "image_size"} and fmt["aspect_ratio"] in caps.ratios
        assert fmt.get("image_size", "1K") in (caps.sizes or ("1K",)) and ("image_size" in fmt) == bool(caps.sizes)
        assert body["store"] is False
        assert set(body["generation_config"]) <= {"seed", "thinking_level"}
        assert body["generation_config"].get("thinking_level") in (None, *caps.thinking)
    else:
        assert_valid_gc_body(body, model)
        parts = body["contents"][0]["parts"]
        assert parts[0] == {"text": "make the sky red"} and len(parts) == 1 + sent
        assert all(set(p) == {"inlineData"} and set(p["inlineData"]) == {"mimeType", "data"} for p in parts[1:])
        assert all(not p["inlineData"]["data"].startswith("data:") for p in parts[1:])
    assert result.params["input_images"] == sent and result.params["mode"] == "edit"


def test_edit_interactions_request_shape():
    rec = Recorder(ia_ok())
    backend = GeminiBackend(section(api="interactions", image_size="2K", thinking_level="high"))
    req = edit_req(jpeg_in(500, 500), png_in(10, 20), prompt="swap the hat", orientation="square", negative="text")
    result = run(backend, rec, req)
    assert str(rec.requests[0].url) == IA_URL and rec.requests[0].headers["x-goog-api-key"] == KEY
    assert rec.body() == {
        "model": DEFAULT_MODEL,
        "input": [
            {"type": "text", "text": "swap the hat\n\nAvoid: text"},
            {"type": "image", "mime_type": "image/jpeg", "data": b64(JPEG)},
            {"type": "image", "mime_type": "image/png", "data": b64(PNG)},
        ],
        "response_format": {"type": "image", "aspect_ratio": "1:1", "image_size": "2K"},
        "store": False,
        "generation_config": {"thinking_level": "high"},
    }
    assert result.image == PNG and result.text == "done"
    assert result.params == {"model": DEFAULT_MODEL, "size": "1:1 2K", "aspect_ratio": "1:1", "api": "Interactions",
                             "mode": "edit", "input_images": 2, "image_size": "2K", "thinking_level": "high",
                             "usage": {"images": 1}}


def test_edit_auto_fallback_resends_images():
    not_found = httpx.Response(404, json={"error": {"code": 404, "status": "NOT_FOUND", "message": "not found"}})
    rec = Recorder(not_found, ia_ok())
    backend = GeminiBackend(section(extra_body='{"safetySettings": []}'))
    result = run(backend, rec, edit_req(png_in(900, 1200)))
    assert [str(r.url) for r in rec.requests] == [GC_URL, IA_URL]
    assert rec.body(0)["contents"][0]["parts"][1] == {"inlineData": {"mimeType": "image/png", "data": b64(PNG)}}
    assert rec.body(1)["input"] == [{"type": "text", "text": "make the sky red"},
                                    {"type": "image", "mime_type": "image/png", "data": b64(PNG)}]
    assert rec.body(1)["response_format"] == {"type": "image", "aspect_ratio": "3:4"}
    assert "safetySettings" not in rec.body(1)
    assert result.params["api"] == "Interactions" and result.params["mode"] == "edit"


def test_edit_per_model_image_limit():
    # gemini-2.5-flash-image 只使用前 3 张，记录中注明
    images = [png_in(), jpeg_in(), png_in(), jpeg_in(), png_in()]
    rec = Recorder(gc_ok())
    backend = GeminiBackend(section(model="gemini-2.5-flash-image"))
    result = run(backend, rec, edit_req(*images))
    parts = rec.body()["contents"][0]["parts"]
    assert [p["inlineData"]["mimeType"] for p in parts[1:]] == ["image/png", "image/jpeg", "image/png"]
    assert result.params["input_images"] == 3 and result.params["input_images_dropped"] == 2
    assert backend.plan(edit_req(*images))["input_images"] == 3

    # 其他模型最多 14 张
    rec = Recorder(gc_ok())
    result = run(GeminiBackend(section()), rec, edit_req(*([png_in()] * 15)))
    assert len(rec.body()["contents"][0]["parts"]) == 1 + 14
    assert result.params["input_images"] == 14 and result.params["input_images_dropped"] == 1

    rec = Recorder(gc_ok())
    result = run(GeminiBackend(section()), rec, edit_req(*([png_in()] * 14)))
    assert len(rec.body()["contents"][0]["parts"]) == 15 and "input_images_dropped" not in result.params


def test_edit_missing_for_model_without_image_input(monkeypatch):
    monkeypatch.setitem(MODEL_CAPS, "text-only-image-model", Caps(RATIOS, (), (), 0))
    backend = GeminiBackend(section(model="text-only-image-model"))
    assert backend.edit_missing() == (f"当前模型 text-only-image-model 不支持修改图片，"
                                      f"请换用 {DEFAULT_MODEL} 等 Nano Banana 模型")
    rec = Recorder(gc_ok())
    with pytest.raises(DrawError, match="不支持修改图片"):
        run(backend, rec, edit_req())
    assert rec.requests == []
    # 文生图不受影响
    assert run(backend, rec).image == PNG


def test_edit_input_formats():
    # MIME 按文件头识别，声明的值不可信
    assert encode_inputs([InputImage(JPEG, "image/png"), InputImage(WEBP, "application/octet-stream")]) == [
        ("image/jpeg", b64(JPEG)), ("image/webp", b64(WEBP))]
    # 无法识别文件头时才使用声明的、官方支持的 MIME
    assert encode_inputs([InputImage(b"\x00\x00\x00\x18ftypheic", "IMAGE/HEIC")])[0][0] == "image/heic"
    for image, shown in ((InputImage(GIF, "image/gif"), "image/gif"),
                         (InputImage(b"\x00" * 10, "application/octet-stream"), "application/octet-stream"),
                         (InputImage(b"\x00" * 10, ""), "未知")):
        with pytest.raises(DrawError, match=f"不支持第 2 张图片的格式（{shown}）"):
            encode_inputs([png_in(), image])
    # 不支持的格式在发请求前报错
    rec = Recorder(gc_ok())
    with pytest.raises(DrawError, match="不支持第 1 张图片的格式"):
        run(GeminiBackend(section()), rec, edit_req(InputImage(GIF, "image/gif", 10, 10)))
    assert rec.requests == []
    # 声明为 PNG / JPEG / WebP 但文件头不符、或数据为空：不当作该格式发送（只会得到 400）
    for image in (InputImage(b"\x00" * 10, "image/png"), InputImage(b"garbage", "image/jpeg"),
                  InputImage(b"", "image/png"), InputImage(b"", "")):
        with pytest.raises(DrawError, match="第 2 张图片的数据无法识别"):
            encode_inputs([png_in(), image])
    rec = Recorder(gc_ok())
    with pytest.raises(DrawError, match="数据无法识别"):
        run(GeminiBackend(section()), rec, edit_req(InputImage(b"\x00" * 10, "image/png", 10, 10)))
    assert rec.requests == []
    # bytearray 也可以
    assert encode_inputs([InputImage(bytearray(PNG), "image/png")]) == [("image/png", b64(PNG))]


def test_edit_inline_size_limit(monkeypatch):
    assert gemini_mod.MAX_INLINE_B64 == 95_000_000
    monkeypatch.setattr(gemini_mod, "MAX_INLINE_B64", len(b64(PNG)) * 2 - 1)
    with pytest.raises(DrawError, match="合计过大"):
        encode_inputs([png_in(), png_in()])
    rec = Recorder(gc_ok())
    with pytest.raises(DrawError, match="合计过大"):
        run(GeminiBackend(section()), rec, edit_req(png_in(), png_in()))
    assert rec.requests == []
    assert run(GeminiBackend(section()), rec, edit_req(png_in())).image == PNG


def test_edit_ignores_strength_and_sends_no_unknown_fields():
    rec = Recorder(gc_ok())
    result = run(GeminiBackend(section(image_size="4K", thinking_level="medium")), rec,
                 edit_req(png_in(1200, 900), orientation="landscape", strength=0.7))
    body = rec.body()
    assert "strength" not in json.dumps(body).lower() and "mask" not in json.dumps(body).lower()
    assert set(body) == {"contents", "generationConfig"}
    config = body["generationConfig"]
    assert set(config) == {"responseModalities", "imageConfig", "thinkingConfig"}
    assert config["imageConfig"] == {"aspectRatio": "4:3", "imageSize": "4K"}
    assert len(body["contents"]) == 1 and body["contents"][0]["role"] == "user"  # 单个 user 回合
    assert all("thoughtSignature" not in part for part in body["contents"][0]["parts"])
    assert "strength" not in result.params
    # 参数里没有图片数据或 Key
    assert b64(PNG) not in json.dumps(result.params) and KEY not in json.dumps(result.params)


@pytest.mark.parametrize(
    ("payload", "pattern"),
    [
        ({"promptFeedback": {"blockReason": "IMAGE_SAFETY"}}, "拒绝修改：图片或修改要求被安全策略拦截（IMAGE_SAFETY）"),
        ({"candidates": [{"finishReason": "IMAGE_PROHIBITED_CONTENT"}]}, "被安全策略拦截（IMAGE_PROHIBITED_CONTENT）"),
        ({"candidates": [{"content": {"parts": [{"text": "I cannot edit photos of real people."}]},
                          "finishReason": "STOP"}]}, "模型回复：I cannot edit photos of real people"),
        ({"candidates": [{"finishReason": "NO_IMAGE"}]}, "没能生成图片"),
    ],
)
def test_edit_refusals(payload, pattern, sleeps):
    rec = Recorder(httpx.Response(200, json=payload), gc_ok())
    with pytest.raises(DrawError, match=pattern) as info:
        run(GeminiBackend(section()), rec, edit_req())
    assert not info.value.retryable and len(rec.requests) == 1 and sleeps == []


def test_edit_interactions_input_blocked(sleeps):
    rec = Recorder(httpx.Response(400, json={"error": {"code": "prohibited_content", "message": "Input blocked: x"}}))
    with pytest.raises(DrawError, match="被安全策略拦截（prohibited_content）") as info:
        run(GeminiBackend(section(api="interactions")), rec, edit_req())
    assert info.value.status == 400 and len(rec.requests) == 1 and sleeps == []


@pytest.mark.parametrize(("failure", "retried"), [
    (httpx.ReadTimeout("slow"), False),
    (httpx.Response(500, json={"error": {"code": 500, "message": "internal"}}), False),
    (httpx.Response(504, json={"error": {"code": 504, "message": "deadline"}}), False),
    (httpx.Response(503, json={"error": {"code": 503, "message": "overloaded"}}), True),
    (httpx.ConnectError("refused"), True),
])
def test_edit_retry_policy(failure, retried, sleeps):
    rec = Recorder(failure, gc_ok())
    if retried:
        assert run(GeminiBackend(section()), rec, edit_req()).image == PNG
        assert len(rec.requests) == 2 and rec.body(0) == rec.body(1)  # 重试时原样重发图片
    else:
        with pytest.raises(DrawError) as info:
            run(GeminiBackend(section()), rec, edit_req())
        assert not info.value.retryable and len(rec.requests) == 1 and sleeps == []


def test_edit_key_echo_masked():
    body = {"error": {"code": "invalid_request", "message": f"bad x-goog-api-key: {KEY}"}}
    rec = Recorder(httpx.Response(400, json=body))
    with pytest.raises(DrawError) as info:
        run(GeminiBackend(section(base_url="proxy.example.com")), rec, edit_req())
    assert KEY not in str(info.value) and "••••" in str(info.value)


# ---------------------------------------------------------------- 计费错误与余额


@pytest.mark.parametrize("status", [429, 402, 200])
def test_prepay_depleted_not_retried(status, sleeps):
    payload = {"error": {"code": 429 if status == 200 else status, "status": "RESOURCE_EXHAUSTED",
                         "message": "Your prepayment credits are depleted. Please go to AI Studio."}}
    rec = Recorder(httpx.Response(status, json=payload), gc_ok())
    with pytest.raises(DrawError, match="预付费余额不足") as info:
        run(GeminiBackend(section()), rec, edit_req())
    assert BILLING_URL in str(info.value) and not info.value.retryable
    assert len(rec.requests) == 1 and sleeps == []


def test_ordinary_429_still_retried(sleeps):
    limited = httpx.Response(429, json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "rate"}})
    rec = Recorder(limited, gc_ok())
    assert run(GeminiBackend(section()), rec, edit_req()).image == PNG
    assert len(rec.requests) == 2 and len(sleeps) == 1


def test_billing_error_mappings():
    e = gemini_error("Gemini", 402, {"error": {"code": "payment_required", "message": "depleted"}}, httpx.Headers())
    assert "预付费余额不足" in str(e) and BILLING_URL in str(e) and e.status == 402 and not e.retryable
    # 中转服务自己的 402
    e = gemini_error("Gemini", 402, {"error": {"message": "insufficient quota on relay"}}, httpx.Headers())
    assert "余额不足（HTTP 402）：insufficient quota on relay" in str(e) and "中转服务" in str(e) and not e.retryable
    e = gemini_error("Gemini", 403, {"error": {"code": 403, "status": "PERMISSION_DENIED",
                                               "message": "Your project has an outstanding balance."}}, httpx.Headers())
    assert "未付款项" in str(e) and "outstanding balance" in str(e) and not e.retryable
    e = gemini_error("Gemini", 400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message":
                                               "Request payload size exceeds the limit: 104857600 bytes."}},
                     httpx.Headers())
    assert "请求内容过大（HTTP 400）" in str(e) and "104857600" in str(e) and not e.retryable
    e = gemini_error("Gemini", 413, "<html><title>413 Request Entity Too Large</title></html>", httpx.Headers())
    assert "请求内容过大（HTTP 413）" in str(e) and "Request Entity Too Large" in str(e)
    # 其他 400 / 403 不受影响
    e = gemini_error("Gemini", 400, {"error": {"code": 400, "message": "Invalid value at 'x'"}}, httpx.Headers())
    assert "请求参数有误" in str(e)
    e = gemini_error("Gemini", 403, {"error": {"code": 403, "message": "Method doesn't allow"}}, httpx.Headers())
    assert "没有权限" in str(e)


@pytest.mark.parametrize("config", [section(), {}, section(model="gemini-3-pro-image", base_url="proxy.example.com")])
def test_balance_not_supported(config):
    backend = GeminiBackend(config)
    reason = backend.balance_missing()
    assert reason == (f"Gemini API 没有查询余额的接口：预付费余额和交易记录只能在 AI Studio 结算页 {BILLING_URL} 查看，"
                      f"用量见 {USAGE_URL}，项目支出上限见 {SPEND_URL}")
    assert KEY not in reason
    rec = Recorder(gc_ok())
    with pytest.raises(DrawError) as info:
        run(backend, rec, balance=True)
    assert str(info.value) == reason and rec.requests == []


# ---------------------------------------------------------------- 配置结构（修改图片 / 余额）


def test_schema_hints_cover_edit_and_balance():
    items = schema_section()["items"]
    hint = items["model"]["hint"]
    assert "修改图片" in hint and str(MAX_EDIT_IMAGES) in hint
    assert "gemini-2.5-flash-image" in hint and str(MODEL_CAPS["gemini-2.5-flash-image"].edit_images) in hint
    assert BILLING_URL in items["api_key"]["hint"]
    assert "修改图片" in items["portrait_size"]["hint"] and "修改图片" in items["api"]["hint"]
    # 没有只用于余额查询的凭据：Gemini 没有余额接口
    assert [name for name, field in items.items() if field.get("secret")] == ["api_key"]


@pytest.mark.parametrize("api", ["auto", "generate_content", "interactions"])
def test_schema_defaults_produce_valid_edit_request(api):
    defaults = {name: field["default"] for name, field in schema_section()["items"].items()}
    defaults.update(api_key=KEY, api=api)
    rec = Recorder(ia_ok() if api == "interactions" else gc_ok())
    result = run(GeminiBackend(defaults), rec, edit_req(jpeg_in(1200, 900), orientation="landscape"))
    assert result.image == PNG and result.params["mode"] == "edit" and result.params["input_images"] == 1
    if api == "interactions":
        assert rec.body()["input"][1] == {"type": "image", "mime_type": "image/jpeg", "data": b64(JPEG)}
        assert rec.body()["response_format"] == {"type": "image", "aspect_ratio": "4:3"}
    else:
        assert_valid_gc_body(rec.body(), DEFAULT_MODEL)
        assert rec.body()["generationConfig"]["imageConfig"] == {"aspectRatio": "4:3"}


def test_usage_from_both_apis():
    gemini = gemini_mod
    meta = {"usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 1290, "thoughtsTokenCount": 200,
                              "totalTokenCount": 1790}}
    # 思考 token 按输出计费，计入输出
    assert gemini.gemini_usage(meta) == {"images": 1, "input_tokens": 300, "output_tokens": 1490,
                                         "total_tokens": 1790}
    interactions = {"usage": {"total_input_tokens": 20, "total_output_tokens": 1120, "total_tokens": 1140}}
    assert gemini.gemini_usage(interactions) == {"images": 1, "input_tokens": 20, "output_tokens": 1120,
                                                 "total_tokens": 1140}
    assert gemini.gemini_usage({}) == {"images": 1}
    assert gemini.gemini_usage({"usageMetadata": {"promptTokenCount": "x"}}) == {"images": 1}

