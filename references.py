"""角色参考图（references）：按提示词中的关键词命中，把参考图作为独立通道传给后端。

配置是顶层 references 字段（JSON 文本，或直接是列表）：
[
  {"name": "小汐", "keywords": ["xiaoxi-whale-girl"], "images": ["D:/img/xiaoxi.jpg"]}
]

- 提示词中命中某组的任一关键词（不区分大小写）时，该组所有图片都作为参考图
- 一组可放多张（如三视图），可配置多组角色；多组同时命中时按配置顺序合并、按来源去重
- images 支持本地文件路径、data: URL 和 http(s) URL
- 只用于文生图，与修改图片的 DrawRequest.images 完全独立；不支持的后端会忽略并记录日志

本模块不依赖 AstrBot，便于单独测试。
"""

import json
from pathlib import Path

try:
    from .backends.base import (
        MAX_IMAGE_BYTES,
        Clients,
        DrawError,
        InputImage,
        as_str,
        detect_mime,
        download_image,
    )
except ImportError:  # 单独测试时作为顶层模块导入
    from backends.base import (
        MAX_IMAGE_BYTES,
        Clients,
        DrawError,
        InputImage,
        as_str,
        detect_mime,
        download_image,
    )


def _str_list(value) -> list[str]:
    """配置中的关键词 / 图片列表：单个字符串按一项处理，其余类型返回空列表。"""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [as_str(item) for item in value if as_str(item)]


def parse_groups(value) -> list[dict]:
    """解析 references 配置，返回 [{"name", "keywords", "images"}]；缺关键词或图片的组不参与匹配。"""
    if isinstance(value, list):
        data = value
    else:
        text = as_str(value)
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise DrawError(f"角色参考图 references 不是有效的 JSON：{e}") from e
    if not isinstance(data, list):
        raise DrawError("角色参考图 references 必须是 JSON 数组（[...]）")
    groups = []
    for index, item in enumerate(data, 1):
        if not isinstance(item, dict):
            raise DrawError(f"角色参考图第 {index} 组必须是对象（{{...}}）")
        keywords, images = _str_list(item.get("keywords")), _str_list(item.get("images"))
        if not keywords or not images:
            continue
        groups.append({"name": as_str(item.get("name")), "keywords": keywords, "images": images})
    return groups


def match_sources(groups: list[dict], prompt: str) -> list[str]:
    """返回命中的参考图来源（本地路径 / URL），按配置顺序去重。"""
    text = as_str(prompt).lower()
    if not text:
        return []
    sources, seen = [], set()
    for group in groups:
        if not any(keyword.lower() in text for keyword in group["keywords"]):
            continue
        for source in group["images"]:
            if source not in seen:
                seen.add(source)
                sources.append(source)
    return sources


async def load_images(clients: Clients, sources: list[str], *, proxy: str = "",
                      timeout: float = 120.0) -> tuple[list[InputImage], list[str]]:
    """把来源读成 InputImage；读不到的项记入 skipped（由调用方写日志），不中断其余参考图。"""
    images, skipped = [], []
    for source in sources:
        try:
            if source.startswith("data:") or "://" in source:
                data = await download_image(clients.http, source, label="角色参考图", proxy=proxy, timeout=timeout)
            else:
                path = Path(source).expanduser()
                if not path.is_file():
                    skipped.append(f"{source}（文件不存在）")
                    continue
                data = path.read_bytes()
                if len(data) > MAX_IMAGE_BYTES:
                    skipped.append(f"{source}（超过 64 MiB）")
                    continue
            mime = detect_mime(data)
            if not mime:
                skipped.append(f"{source}（不是可识别的图片）")
                continue
            images.append(InputImage(data=data, mime=mime))
        except (DrawError, OSError) as e:
            skipped.append(f"{source}（{e}）")
    return images, skipped
