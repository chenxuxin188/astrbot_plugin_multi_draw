"""图片转换：生成看图评价用的缩略图、压缩过大的图片。

Pillow 是 AstrBot 的依赖；万一不可用，这些函数返回 None，调用方改用原图。
"""

import io


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
