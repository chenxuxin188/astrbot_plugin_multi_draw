import asyncio
import base64
import io
import json
import sys
import zipfile
from pathlib import Path

import httpx
import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

from backends.base import Clients, DrawError, DrawRequest, HttpPool, InputImage  # noqa: E402
from backends.gemini import GeminiBackend  # noqa: E402
from backends.novelai import REFERENCE_STRENGTH, NovelAIBackend, NovelAIClient  # noqa: E402
from backends.seedream import SeedreamBackend  # noqa: E402
from references import load_images, match_sources, parse_groups  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


def image(data=PNG):
    return InputImage(data, "image/png" if data.startswith(b"\x89PNG") else "image/jpeg")


def make_zip(data=PNG):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("image_0.png", data)
    return buf.getvalue()


def generate(backend, handler, req):
    async def main():
        transport = httpx.MockTransport(handler)
        pool = HttpPool(transport=transport)
        nai = NovelAIClient(transport=transport) if isinstance(backend, NovelAIBackend) else None
        try:
            return await backend.generate(Clients(http=pool, novelai=nai), req)
        finally:
            if nai is not None:
                await nai.close()
            await pool.close()

    return asyncio.run(main())


def test_parse_match_and_deduplicate_sources():
    groups = parse_groups(json.dumps([
        {"name": "角色一", "keywords": ["Alpha"], "images": ["a.png", "b.png"]},
        {"name": "角色二", "keywords": "BETA", "images": ["b.png", "c.png"]},
        {"name": "缺少图片，不参与匹配", "keywords": ["Alpha"]},
    ]))
    assert match_sources(groups, "draw ALPHA and beta together") == ["a.png", "b.png", "c.png"]
    assert match_sources(groups, "unrelated prompt") == []
    assert match_sources(groups, "") == []
    assert parse_groups("") == []
    assert parse_groups([{"keywords": "Solo", "images": "one.png"}]) == [
        {"name": "", "keywords": ["Solo"], "images": ["one.png"]}
    ]


@pytest.mark.parametrize("value", ["[", "{}", "[1]"])
def test_invalid_reference_config_raises(value):
    with pytest.raises(DrawError):
        parse_groups(value)


def test_load_local_data_url_and_remote_images(tmp_path):
    local = tmp_path / "local.png"
    local.write_bytes(PNG)
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("not an image", encoding="utf-8")
    missing = tmp_path / "missing.png"
    data_url = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")
    remote = "https://example.com/remote.jpg"

    def handler(request):
        assert str(request.url) == remote
        return httpx.Response(200, content=JPEG)

    async def main():
        pool = HttpPool(transport=httpx.MockTransport(handler))
        try:
            return await load_images(
                Clients(http=pool),
                [str(local), data_url, remote, str(invalid), str(missing)],
            )
        finally:
            await pool.close()

    images, skipped = asyncio.run(main())
    assert [item.mime for item in images] == ["image/png", "image/png", "image/jpeg"]
    assert [item.data for item in images] == [PNG, PNG, JPEG]
    assert len(skipped) == 2
    assert "不是可识别的图片" in skipped[0] and "文件不存在" in skipped[1]


def test_seedream_reference_body_and_edit_separation():
    backend = SeedreamBackend({"extra_body": json.dumps({"image": "fixed-image"})})
    single = backend.build_body(DrawRequest("a cat", references=[image()]))
    encoded = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")
    assert single["image"] == encoded  # 命中的参考图覆盖 extra_body 中的固定 image

    multiple = backend.build_body(DrawRequest("a cat", references=[image(), image()]))
    assert multiple["image"] == [encoded, encoded]

    edit = backend.build_edit_body(DrawRequest(
        "change the background",
        images=[image(JPEG)],
        references=[image()],
    ))
    assert edit["image"].startswith("data:image/jpeg;base64,")
    assert edit["image"] != encoded


def test_gemini_references_use_inline_data_without_edit_mode():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(PNG).decode("ascii")}}
                ]},
                "finishReason": "STOP",
            }]
        })

    result = generate(
        GeminiBackend({"api_key": "AIza-test-key"}),
        handler,
        DrawRequest("a cat", references=[image(), image(JPEG)]),
    )
    parts = seen["body"]["contents"][0]["parts"]
    assert parts[0] == {"text": "a cat"}
    assert [part["inlineData"]["mimeType"] for part in parts[1:]] == ["image/png", "image/jpeg"]
    assert result.params["references"] == 2
    assert "mode" not in result.params and "input_images" not in result.params


def test_novelai_references_use_vibe_transfer_and_png():
    pil_image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    pil_image.new("RGB", (32, 32), (10, 20, 30)).save(buf, "JPEG")
    jpeg = buf.getvalue()
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=make_zip())

    backend = NovelAIBackend({"token": "pst-test-token", "model": "nai-diffusion-4-5-full"})
    req = DrawRequest("a cat", references=[image(), image(jpeg)])
    assert backend.plan(req)["references"] == 2
    result = generate(backend, handler, req)

    body = seen["body"]
    assert body["action"] == "generate"
    params = body["parameters"]
    assert "image" not in params
    encoded = params["reference_image_multiple"]
    assert len(encoded) == 2
    assert base64.b64decode(encoded[0]) == PNG
    assert base64.b64decode(encoded[1]).startswith(b"\x89PNG\r\n\x1a\n")
    assert params["reference_information_extracted_multiple"] == [1.0, 1.0]
    assert params["reference_strength_multiple"] == [REFERENCE_STRENGTH, REFERENCE_STRENGTH]
    assert result.params["references"] == 2
    assert "mode" not in result.params and "references_ignored" not in result.params


def test_novelai_v3_explicitly_ignores_references():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=make_zip())

    backend = NovelAIBackend({"token": "pst-test-token", "model": "nai-diffusion-3"})
    req = DrawRequest("a cat", references=[image(), image()])
    planned = backend.plan(req)
    assert planned["references_ignored"] == 2 and "references" not in planned

    result = generate(backend, handler, req)
    params = seen["body"]["parameters"]
    assert params["reference_image_multiple"] == []
    assert params["reference_information_extracted_multiple"] == []
    assert params["reference_strength_multiple"] == []
    assert result.params["references_ignored"] == 2
    assert "references" not in result.params and "mode" not in result.params


def test_novelai_edit_does_not_turn_references_into_vibes():
    backend = NovelAIBackend({"token": "pst-test-token", "model": "nai-diffusion-4-5-full"})
    req = DrawRequest("change it", images=[image()], references=[image(), image()])
    image_req = backend.build_request(req)
    assert image_req.image == PNG
    assert image_req.references == []
