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
from backends.base import Clients, DrawError, DrawRequest, HttpPool  # noqa: E402
from backends.gemini import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SIZES,
    DEFAULT_TIMEOUT,
    MODEL_CAPS,
    RATIOS,
    GeminiBackend,
    gemini_base_url,
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


def run(backend: GeminiBackend, handler, req: DrawRequest | None = None, *, models: bool = False):
    async def main():
        pool = HttpPool(transport=httpx.MockTransport(handler))
        try:
            clients = Clients(http=pool)
            if models:
                return await backend.list_models(clients)
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
    assert result.params == {"model": DEFAULT_MODEL, "size": ratio, "aspect_ratio": ratio, "api": "generateContent"}


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
                           "square_size", "image_size", "thinking_level", "api", "timeout", "extra_body"]
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
