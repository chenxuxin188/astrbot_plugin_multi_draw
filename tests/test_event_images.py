"""event_images.py 的单元测试：用 astrbot_stub 模拟消息组件，下载走 httpx.MockTransport。"""

import asyncio
import base64
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PLUGIN_DIR.parent))

import astrbot_stub  # noqa: E402

astrbot_stub.install(HERE)

for _mod in [m for m in sys.modules if m.startswith(PLUGIN_DIR.name)]:
    del sys.modules[_mod]
event_images = __import__(f"{PLUGIN_DIR.name}.event_images", fromlist=["*"])

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x01" * 32
GIF = b"GIF89a" + b"\x02" * 32


def img(**attrs):
    comp = astrbot_stub.Image(attrs.pop("file", ""))
    comp.path = attrs.pop("path", "")
    comp.url = attrs.pop("url", "")
    for k, v in attrs.items():
        setattr(comp, k, v)
    return comp


def event_with(messages, platform="aiocqhttp", raw=None):
    event = astrbot_stub.AstrMessageEvent(platform=platform)
    event.messages = messages
    event.message_obj = type("M", (), {"raw_message": raw, "message": messages})()
    event.get_self_id = lambda: "10000"
    return event


def collect(event, handler=None, max_images=4, context=None):
    async def go():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler or (lambda r: httpx.Response(404))))
        try:
            collector = event_images.ImageCollector(client, max_images)
            await event_images.collect_event_images(event, context, collector)
            return collector.results, collector.seen
        finally:
            await client.aclose()

    return asyncio.run(go())


def test_current_message_first_then_quoted(tmp_path):
    local = tmp_path / "a.png"
    local.write_bytes(PNG)
    reply = astrbot_stub.Reply([img(url="https://q.example/reply.jpg")])
    event = event_with([reply, img(path=str(local))])

    def handler(request):
        assert request.url.host == "q.example"
        return httpx.Response(200, content=JPEG)

    images, seen = collect(event, handler, max_images=1)
    assert images == [PNG] and seen == 1  # 本条消息中的图片优先于引用的图片
    images, seen = collect(event, handler, max_images=4)
    assert images == [PNG, JPEG] and seen == 2


def test_file_uri_and_base64_refs(tmp_path):
    local = tmp_path / "b c.png"
    local.write_bytes(PNG)
    event = event_with([
        img(file=local.as_uri()),
        img(file="base64://" + base64.b64encode(GIF).decode()),
        img(file="data:image/jpeg;base64," + base64.b64encode(JPEG).decode()),
    ])
    images, seen = collect(event)
    assert images == [PNG, GIF, JPEG] and seen == 3


def test_falls_back_to_original_onebot_url_and_rejects_error_pages():
    # 4.26-4.28：组件里的临时文件已被删除，只能用原始事件中的 QQ 链接
    raw = {"message": [{"type": "image", "data": {"url": "https://multimedia.nt.qq.com.cn/download?rkey=SECRET"}}]}
    event = event_with([img(path="/tmp/deleted.jpg", url="/tmp/deleted.jpg")], raw=raw)
    images, seen = collect(event, lambda r: httpx.Response(200, content=JPEG))
    assert images == [JPEG] and seen == 1
    # 链接过期：返回 HTML 错误页，也算「看到了图片但读取失败」
    images, seen = collect(event, lambda r: httpx.Response(200, text="<html>expired</html>"))
    assert images == [] and seen == 1
    images, seen = collect(event, lambda r: httpx.Response(400, json={"retcode": -5503007}))
    assert images == [] and seen == 1


def test_no_images_at_all():
    event = event_with([astrbot_stub.Plain("把背景换成海边")])
    assert collect(event) == ([], 0)


def test_quoted_message_fetched_with_get_msg_when_chain_empty():
    calls = []

    class Bot:
        async def call_action(self, action, **params):
            calls.append((action, params))
            if action == "get_msg":
                return {"message": [{"type": "image", "data": {"url": "https://q.example/old.png"}}], "group_id": 1}
            return None

    reply = astrbot_stub.Reply([])
    reply.id = "12345"
    event = event_with([reply])
    event.bot = Bot()
    images, seen = collect(event, lambda r: httpx.Response(200, content=PNG))
    assert images == [PNG] and seen == 1
    assert calls[0] == ("get_msg", {"message_id": 12345, "self_id": 10000})


def test_webchat_quoted_image_from_database(tmp_path):
    local = tmp_path / "upload.jpg"
    local.write_bytes(JPEG)

    class DB:
        async def get_platform_message_history_by_id(self, record_id):
            assert record_id == 7
            return type("H", (), {"content": {"message": [{"type": "image", "attachment_id": "att1"}]}})()

        async def get_attachment_by_id(self, attachment_id):
            return type("A", (), {"path": str(local)})()

    class Ctx:
        def get_db(self):
            return DB()

    reply = astrbot_stub.Reply([astrbot_stub.Plain("quoted text")])
    reply.id = "7"
    event = event_with([reply], platform="webchat")
    images, seen = collect(event, context=Ctx())
    assert images == [JPEG] and seen == 1


def test_image_files_and_duplicates(tmp_path):
    local = tmp_path / "photo.webp"
    webp = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 16
    local.write_bytes(webp)

    class FakeFile:
        name = "photo.webp"
        file_ = str(local)
        url = ""

    original_file = event_images.File
    event_images.File = FakeFile
    try:
        event = event_with([FakeFile(), img(path=str(local))])
        images, seen = collect(event)
    finally:
        event_images.File = original_file
    assert images == [webp] and seen == 2  # 同一张图只返回一次


def test_log_never_contains_rkey(caplog):
    raw = {"message": [{"type": "image", "data": {"url": "https://multimedia.nt.qq.com.cn/download?rkey=SECRETKEY"}}]}
    event = event_with([img(url="https://multimedia.nt.qq.com.cn/download?rkey=SECRETKEY")], raw=raw)
    with caplog.at_level("WARNING"):
        collect(event, lambda r: httpx.Response(403))
    assert "SECRETKEY" not in caplog.text and "读取本条消息中的图片失败" in caplog.text


def test_discord_quoted_image_from_raw_message():
    class Attachment:
        def __init__(self, url, content_type="", filename=""):
            self.url, self.content_type, self.filename = url, content_type, filename

    quoted = type("Q", (), {"attachments": [Attachment("https://cdn.discordapp.com/a/cat.png?ex=1", "image/png"),
                                            Attachment("https://cdn.discordapp.com/a/notes.txt", "text/plain")]})()
    fetched = []

    class Channel:
        async def fetch_message(self, message_id):
            fetched.append(message_id)
            return quoted

    def make(resolved):
        reference = type("R", (), {"resolved": resolved, "message_id": 42})()
        raw = type("Raw", (), {"reference": reference, "channel": Channel()})()
        reply = astrbot_stub.Reply([astrbot_stub.Plain("quoted text")])  # AstrBot 只放了文字
        return event_with([reply], platform="discord", raw=raw)

    def handler(request):
        assert request.url.host == "cdn.discordapp.com" and request.url.path.endswith("cat.png")
        return httpx.Response(200, content=PNG)

    assert collect(make(quoted), handler) == ([PNG], 1)
    assert collect(make(None), handler) == ([PNG], 1) and fetched == [42]  # 没有随事件给出时再取一次


def test_describe_hides_telegram_bot_token():
    text = event_images._describe("https://api.telegram.org/file/bot123456:AAH-secret_token/photos/file_1.jpg")
    assert "secret" not in text and "/file/bot***/photos/file_1.jpg" in text
