"""图片转换：生成看图评价用的缩略图、压缩过大的图片、整理图片修改时用户发来的图片。

Pillow 是 AstrBot 的依赖；万一不可用，这些函数返回 None，调用方改用原图。
"""

import io

# 图片修改时输入图片的长边上限：足够各平台修改使用，又不会让请求过大
INPUT_MAX_EDGE = 2048
INPUT_MAX_PNG_BYTES = 6 * 1024 * 1024


def normalize_input(data: bytes, max_edge: int = INPUT_MAX_EDGE) -> tuple[bytes, str, int, int] | None:
    """把用户发来的图片整理成绘图平台都接受的格式，返回 (数据, MIME, 宽, 高)。

    动图取第一帧；有透明通道时保存为 PNG，否则保存为 JPEG；长边缩小到 max_edge 以内。
    不是图片或 Pillow 不可用时返回 None。
    """
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.seek(0)
            im.load()
            im = ImageOps.exif_transpose(im) or im  # 手机拍的照片按 EXIF 方向摆正
            alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
            im = im.convert("RGBA" if alpha else "RGB")
            if alpha and im.getchannel("A").getextrema() == (255, 255):
                im, alpha = im.convert("RGB"), False  # 透明通道全不透明，按普通图片处理
            if max(im.size) > max_edge:
                im.thumbnail((max_edge, max_edge))
            if alpha:
                # 透明 PNG 可能很大（多张图片时请求体容易超过平台上限），过大时先缩小
                png = _encode(im, "PNG")
                if len(png) > INPUT_MAX_PNG_BYTES and max(im.size) > 1536:
                    im.thumbnail((1536, 1536))
                    png = _encode(im, "PNG")
                return png, "image/png", im.size[0], im.size[1]
            return _encode(im, "JPEG"), "image/jpeg", im.size[0], im.size[1]
    except Exception:
        return None


def _encode(im, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "PNG":
        im.save(buf, "PNG", optimize=True)
    else:
        im.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def to_jpeg(data: bytes, max_edge: int = 0, quality: int = 90) -> bytes | None:
    """转换成 JPEG（透明背景铺白），max_edge > 0 时把长边缩小到不超过该值。失败时返回 None。"""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            if im.mode in ("RGBA", "LA", "P"):
                rgba = im.convert("RGBA")
                canvas = Image.new("RGB", rgba.size, (255, 255, 255))
                canvas.paste(rgba, mask=rgba.split()[-1])
                im = canvas
            else:
                im = im.convert("RGB")
            if max_edge > 0 and max(im.size) > max_edge:
                im.thumbnail((max_edge, max_edge))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality, optimize=True)
            return buf.getvalue()
    except Exception:
        return None
