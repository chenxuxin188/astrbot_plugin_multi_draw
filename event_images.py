"""读取用户要修改的图片：先取本条消息中的图片，再取引用（回复）的消息中的图片。

必须在 LLM 工具函数里、交给后台任务之前调用：AstrBot 4.26-4.28 会把消息中的图片下载成临时文件，
本轮事件结束后就删除；QQ 的图片链接带有签名（rkey），过一段时间会失效。
"""

import asyncio
import base64
import binascii
import hashlib
import html
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

from astrbot.api import logger
from astrbot.api import message_components as _mc

# 测试用的简化版 astrbot 可能没有 File
Image = getattr(_mc, "Image", None)
Reply = getattr(_mc, "Reply", None)
File = getattr(_mc, "File", None)

LOG_PREFIX = "[multi_draw]"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
DOWNLOAD_TIMEOUT = 30.0
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".heif", ".avif")


class _FetchError(Exception):
    pass


def _is(obj, cls) -> bool:
    return cls is not None and isinstance(obj, cls)


def sniff_mime(data: bytes) -> str:
    """按文件头识别图片格式；不是图片（例如链接过期后返回的错误页）时返回空串。"""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM" and len(data) > 26:
        return "image/bmp"
    if data[4:8] == b"ftyp" and data[8:12] in (b"avif", b"avis", b"heic", b"heix", b"mif1", b"msf1"):
        return "image/avif" if data[8:12] in (b"avif", b"avis") else "image/heic"
    return ""


def _describe(ref: str) -> str:
    """写日志用：不输出链接参数（QQ 的 rkey 相当于凭证）和 base64 内容。"""
    if ref.startswith(("base64://", "data:")):
        return f"<base64，{len(ref)} 字符>"
    if ref.lower().startswith(("http://", "https://")):
        parts = urlsplit(ref)
        return f"{parts.scheme}://{parts.netloc}{parts.path[:60]}"
    return ref if len(ref) <= 160 else ref[:157] + "..."


def _is_http(ref: str) -> bool:
    return ref.lower().startswith(("http://", "https://"))


def _is_inline(ref: str) -> bool:
    return ref.startswith("base64://") or ref[:5].lower() == "data:"


def _image_name(value) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = urlsplit(value).path if "://" in value else value
    return path.strip().lower().endswith(IMAGE_EXTS)


def _local_candidates(ref: str) -> list[str]:
    """file: URI 转换成本地路径，兼容 AstrBot 各版本生成的写法：
    file:///C:/a%20b.png（4.26 起）、file:///C:\\a b.png（4.25 及以前的 Windows）、file:////abs（4.25 及以前）。"""
    if ref[:5].lower() != "file:":
        return [ref]
    parts = urlsplit(ref)
    out = []
    for raw in (unquote(parts.path), parts.path):
        path, netloc = raw, parts.netloc
        if netloc and netloc.lower() != "localhost":
            path = netloc + path if (len(netloc) == 2 and netloc[1] == ":") else "//" + netloc + path
        if len(path) >= 3 and path[0] in "/\\" and path[1].isalpha() and path[2] == ":":
            path = path[1:]
        elif os.name != "nt" and path.startswith("//"):
            path = "/" + path.lstrip("/")
        if path and path not in out:
            out.append(path)
    return out


def _decode_inline(ref: str) -> bytes:
    payload = ref[len("base64://"):] if ref.startswith("base64://") else ref.partition(",")[2]
    payload = "".join(payload.split())
    if len(payload) * 3 // 4 > MAX_IMAGE_BYTES:
        raise _FetchError("图片超过 20 MB")
    try:
        return base64.b64decode(payload + "=" * (-len(payload) % 4), validate=True)
    except (binascii.Error, ValueError) as e:
        raise _FetchError("base64 数据无效") from e


async def _download(client, url: str) -> bytes:
    url = html.unescape(url) if "&amp;" in url else url  # 部分实现会把 & 转义成 &amp;
    async with client.stream("GET", url, timeout=DOWNLOAD_TIMEOUT) as resp:
        if resp.status_code != 200:
            raise _FetchError(f"HTTP {resp.status_code}（QQ 图片链接有有效期，过期后需要重新发送图片）")
        declared = resp.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_IMAGE_BYTES:
            raise _FetchError("图片超过 20 MB")
        buf = bytearray()
        async for chunk in resp.aiter_bytes():
            buf += chunk
            if len(buf) > MAX_IMAGE_BYTES:
                raise _FetchError("图片超过 20 MB")
        return bytes(buf)


async def _load(client, ref: str) -> bytes:
    ref = ref.strip()
    if _is_inline(ref):
        return _decode_inline(ref)
    if _is_http(ref):
        return await _download(client, ref)
    for candidate in _local_candidates(ref):
        try:
            path = Path(candidate)
            if not path.is_file():
                continue
        except (OSError, ValueError):
            continue
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise _FetchError("图片超过 20 MB")
        return await asyncio.to_thread(path.read_bytes)
    raise _FetchError("本地文件不存在（只有文件名、在其他容器中，或临时文件已被清理）")


def _rank(ref: str) -> int:
    """读取顺序：内嵌数据、本地文件（AstrBot 已下载好的）、网络链接。"""
    return 0 if _is_inline(ref) else 2 if _is_http(ref) else 1


def _refs(values) -> list[str]:
    out = []
    for value in values:
        if isinstance(value, str) and value.strip() and value.strip() not in out:
            out.append(value.strip())
    return sorted(out, key=_rank)


def _component_refs(comp) -> list[str]:
    if _is(comp, Image):
        return _refs([getattr(comp, "path", ""), getattr(comp, "url", ""), getattr(comp, "file", "")])
    if _is(comp, File):
        # 不能读取 comp.file：它会在异步环境中尝试同步下载
        name, local, url = getattr(comp, "name", ""), getattr(comp, "file_", ""), getattr(comp, "url", "")
        if any(_image_name(v) for v in (name, local, url)):
            return _refs([local, url])
    return []


def _onebot_urls(event) -> list[str]:
    """aiocqhttp 原始事件中各图片的 QQ 链接：AstrBot 4.26 起会把 Image 的地址改成临时文件，原链接只在这里。"""
    try:
        if event.get_platform_name() != "aiocqhttp":
            return []
    except Exception:
        return []
    raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
    segments = raw.get("message") if isinstance(raw, dict) else None
    urls = []
    for seg in segments if isinstance(segments, list) else []:
        if isinstance(seg, dict) and seg.get("type") == "image":
            url = (seg.get("data") or {}).get("url") if isinstance(seg.get("data"), dict) else None
            urls.append(url.strip() if isinstance(url, str) and _is_http(url.strip()) else "")
    return urls


async def _call_onebot(event, action: str, **params):
    call = getattr(getattr(event, "bot", None), "call_action", None)
    if not callable(call):
        return None
    self_id = str(event.get_self_id() or "")
    if self_id.isdigit():
        params.setdefault("self_id", int(self_id))  # 连接了多个 QQ 账号时指定用哪个账号调用
    ret = await call(action, **params)
    if isinstance(ret, dict) and isinstance(ret.get("data"), dict):
        return ret["data"]
    return ret if isinstance(ret, dict) else None


async def _onebot_quoted(event, reply) -> list[list[str]]:
    """aiocqhttp：AstrBot 没能取到被引用的消息内容时，自己调用 get_msg 再取一次。"""
    message_id = str(getattr(reply, "id", "") or "").strip()
    if not message_id:
        return []
    payload, error = None, None
    for value in ([int(message_id)] if message_id.lstrip("-").isdigit() else []) + [message_id]:
        try:
            payload = await _call_onebot(event, "get_msg", message_id=value)
        except Exception as e:
            error = e
            continue
        if payload:
            break
    segments = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(segments, list):
        logger.warning(f"{LOG_PREFIX} 获取被引用的消息失败（message_id={message_id}）：{error or '没有返回消息内容'}")
        return []
    group_id = str(payload.get("group_id") or event.get_group_id() or "")
    groups = []
    for seg in segments:
        data = seg.get("data") if isinstance(seg, dict) and isinstance(seg.get("data"), dict) else {}
        kind = seg.get("type") if isinstance(seg, dict) else ""
        if kind == "image":
            refs = _refs(data.get(k) for k in ("url", "path", "file"))
            if not any(_is_http(r) or _is_inline(r) for r in refs) and data.get("file"):
                try:  # 只给了文件 id 的实现：再问一次图片地址
                    info = await _call_onebot(event, "get_image", file=str(data["file"]))
                    if isinstance(info, dict):
                        refs = _refs(refs + [info.get("url"), info.get("file")])
                except Exception as e:
                    logger.debug(f"{LOG_PREFIX} get_image 失败：{e}")
            if refs:
                groups.append(refs)
        elif kind == "file" and _image_name(data.get("name") or data.get("file_name") or data.get("file")):
            if isinstance(data.get("url"), str) and _is_http(data["url"]):
                groups.append([data["url"]])
            elif data.get("file_id"):
                try:
                    if group_id.isdigit():
                        info = await _call_onebot(event, "get_group_file_url", group_id=int(group_id),
                                                  file_id=data["file_id"])
                    else:
                        info = await _call_onebot(event, "get_private_file_url", file_id=data["file_id"])
                    if isinstance(info, dict) and isinstance(info.get("url"), str) and info["url"]:
                        groups.append([info["url"]])
                except Exception as e:
                    logger.warning(f"{LOG_PREFIX} 获取被引用的图片文件链接失败：{e}")
    return groups


async def _webchat_quoted(context, reply) -> list[list[str]]:
    """WebUI 对话：历史记录不保存图片路径，Reply.chain 里没有图片，需要通过数据库找到附件。"""
    try:
        record_id = int(str(getattr(reply, "id", "")).strip())
        db = context.get_db()
        history = await db.get_platform_message_history_by_id(record_id)
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} 读取 WebUI 被引用的消息失败：{e}")
        return []
    content = getattr(history, "content", None) if history else None
    parts = content.get("message") if isinstance(content, dict) else None
    groups = []
    for part in parts if isinstance(parts, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "image" and not (part.get("type") == "file" and _image_name(part.get("filename"))):
            continue
        refs = [part["path"]] if isinstance(part.get("path"), str) and part["path"] else []
        if part.get("attachment_id"):
            try:
                attachment = await db.get_attachment_by_id(str(part["attachment_id"]))
                if attachment is not None and getattr(attachment, "path", ""):
                    refs.append(attachment.path)
            except Exception as e:
                logger.debug(f"{LOG_PREFIX} 读取 WebUI 附件失败：{e}")
        if refs:
            groups.append(refs)
    return groups


async def _quoted(event, context, reply) -> list[list[str]]:
    platform = event.get_platform_name()
    if platform == "aiocqhttp":
        return await _onebot_quoted(event, reply)
    if platform == "webchat":
        return await _webchat_quoted(context, reply)
    return []


async def _load_one(client, source: str, refs: list[str], comp=None) -> bytes | None:
    """依次尝试同一张图片的各个地址，返回第一个能读出图片的数据。"""
    errors = []
    for ref in refs:
        try:
            data = await _load(client, ref)
        except Exception as e:
            errors.append(f"{_describe(ref)} → {e}")
            continue
        if sniff_mime(data):
            return data
        errors.append(f"{_describe(ref)} → 内容不是图片（{len(data)} 字节）")
    if comp is not None and callable(getattr(comp, "convert_to_base64", None)) and not any(map(_is_http, refs)):
        # 最后才用 AstrBot 自带的转换：4.25 及以前它下载时不检查状态码，并会留下临时文件
        try:
            text = str(await comp.convert_to_base64() or "")
            data = _decode_inline(text if _is_inline(text) else "base64://" + text)
            if sniff_mime(data):
                return data
            errors.append("convert_to_base64() → 内容不是图片")
        except Exception as e:
            errors.append(f"convert_to_base64() → {e}")
    logger.warning(f"{LOG_PREFIX} 读取{source}中的图片失败：" + (" | ".join(errors) or "没有可用的图片地址"))
    return None


def _chain(event) -> list:
    try:
        return list(event.get_messages() or [])
    except Exception:
        return list(getattr(getattr(event, "message_obj", None), "message", None) or [])


def message_image_refs(event) -> list[list[str]]:
    """本条消息中每张图片（含图片文件）的候选地址，不下载。用于记住用户最近发送的图片。"""
    groups = []
    raw_urls = _onebot_urls(event)
    image_index = 0
    for comp in _chain(event):
        refs = _component_refs(comp)
        if _is(comp, Image):
            if image_index < len(raw_urls) and raw_urls[image_index]:
                refs = _refs(refs + [raw_urls[image_index]])
            image_index += 1
        if refs:
            groups.append(refs)
    return groups


def count_image_parts(event) -> int:
    """本条消息中图片、图片文件和引用的数量（不下载）。读取超时或出错时用来判断用户是否发了图。"""
    return sum(1 for comp in _chain(event) if _is(comp, Reply) or _component_refs(comp) or _is(comp, Image))


def _from_self(event, reply) -> bool:
    try:
        return str(getattr(reply, "sender_id", "") or "") == str(event.get_self_id() or "") != ""
    except Exception:
        return False


class ImageCollector:
    """读取过程中的结果：读取被超时打断时，调用方仍能拿到已经读到的图片和看到的图片数。"""

    def __init__(self, client, max_images: int):
        self.client = client
        self.max_images = max_images
        self.results: list[bytes] = []
        self.seen = 0
        self._digests: set[bytes] = set()

    @property
    def full(self) -> bool:
        return len(self.results) >= self.max_images

    async def take(self, source: str, refs: list[str], comp=None) -> bool:
        self.seen += 1
        data = await _load_one(self.client, source, refs, comp)
        if data is None:
            return False
        digest = hashlib.sha1(data).digest()
        if digest not in self._digests:
            self._digests.add(digest)
            self.results.append(data)
        return True


async def load_ref_groups(collector: ImageCollector, groups: list[list[str]], source: str) -> None:
    for refs in groups:
        if collector.full:
            break
        await collector.take(source, refs)


async def collect_event_images(event, context, collector: ImageCollector) -> None:
    """按顺序读取最多 collector.max_images 张图片：本条消息中的图片和图片文件，然后是引用的消息中的图片。

    结果写入 collector（results 为图片数据，seen 为消息中看到的图片数）。看到了图片却一张都读不出来时，
    调用方应提示用户重新发送，而不是改用别的图片。引用的消息读不到内容（且不是引用机器人自己的消息）时
    也计入 seen，以免改错图片。
    """
    if collector.max_images <= 0:
        return
    chain = _chain(event)
    raw_urls = _onebot_urls(event)
    image_index = 0
    for comp in chain:
        if collector.full:
            return
        refs = _component_refs(comp)
        if _is(comp, Image):
            if image_index < len(raw_urls) and raw_urls[image_index]:
                refs = _refs(refs + [raw_urls[image_index]])
            image_index += 1
            await collector.take("本条消息", refs, comp)
        elif refs:
            await collector.take("本条消息", refs)

    for reply in (c for c in chain if _is(c, Reply)):
        if collector.full:
            break
        quoted_chain = getattr(reply, "chain", None) or []
        got = False
        for comp in quoted_chain:
            refs = _component_refs(comp)
            if refs and not collector.full:
                got = await collector.take("引用的消息", refs, comp if _is(comp, Image) else None) or got
        if got or collector.full:
            continue
        try:
            groups = await _quoted(event, context, reply)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} 读取引用的消息中的图片失败：{e}")
            groups = []
        await load_ref_groups(collector, groups, "引用的消息")
        if not groups and not quoted_chain and not _from_self(event, reply):
            # 引用的消息完全读不到内容：可能是一张图，不能改用别的图片
            logger.info(f"{LOG_PREFIX} 读不到被引用的消息内容（平台 {event.get_platform_name()}）")
            collector.seen += 1
