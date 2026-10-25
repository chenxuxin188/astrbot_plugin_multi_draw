"""生成图片的本地保存：按日期分目录保存在插件数据目录，超过数量上限时删除最旧的图片。"""

import asyncio
import os
import re
import secrets
from datetime import datetime, timedelta
from pathlib import Path

_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif", "image/bmp": "bmp"}
_MIME = {ext: mime for mime, ext in _EXT.items()} | {"jpeg": "image/jpeg"}
_NAME_RE = re.compile(r"^\d{8}/[A-Za-z0-9_.-]+\.(png|jpg|jpeg|webp|gif|bmp)$")


class Gallery:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = asyncio.Lock()
        self._last = datetime.min

    async def save(self, data: bytes, mime: str, backend: str, keep: int = 0) -> Path:
        """保存一张图片并返回文件路径；keep > 0 时只保留最新的 keep 张。"""
        async with self._lock:
            return await asyncio.to_thread(self._save_sync, data, mime, backend, keep)

    def _save_sync(self, data: bytes, mime: str, backend: str, keep: int) -> Path:
        # 文件名精确到微秒并保证严格递增：Windows 上系统时钟约 15 ms 才更新一次，
        # 连续保存的图片可能取到相同的时间，按文件名排序时会乱序
        now = datetime.now()
        if now <= self._last:
            now = self._last + timedelta(microseconds=1)
        self._last = now
        folder = self.root / now.strftime("%Y%m%d")
        folder.mkdir(parents=True, exist_ok=True)
        tag = re.sub(r"[^A-Za-z0-9]", "", backend)[:16] or "image"
        path = folder / f"{now.strftime('%H%M%S%f')}_{tag}_{secrets.token_hex(3)}.{_EXT.get(mime, 'png')}"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        if keep > 0:
            self._trim(keep)
        return path

    def _files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        files = [p for p in self.root.glob("*/*") if p.is_file() and self._valid(p)]
        # 文件名以时间开头，按 日期目录/文件名 排序即为时间顺序
        files.sort(key=lambda p: (p.parent.name, p.name))
        return files

    def _valid(self, path: Path) -> bool:
        return bool(_NAME_RE.match(f"{path.parent.name}/{path.name}"))

    def _trim(self, keep: int) -> None:
        files = self._files()
        for path in files[: max(0, len(files) - keep)]:
            try:
                path.unlink()
            except OSError:
                continue
        for folder in self.root.iterdir():
            if folder.is_dir():
                try:
                    folder.rmdir()  # 只会删除空目录
                except OSError:
                    pass

    async def recent(self, limit: int = 40) -> list[dict]:
        """从新到旧列出最近的图片。"""
        return await asyncio.to_thread(self._recent_sync, max(1, limit))

    def _recent_sync(self, limit: int) -> list[dict]:
        out = []
        for path in reversed(self._files()[-limit:]):
            try:
                stat = path.stat()
            except OSError:
                continue
            out.append({
                "name": f"{path.parent.name}/{path.name}",
                "time": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
                "size_bytes": stat.st_size,
            })
        return out

    def resolve(self, name: str) -> Path | None:
        """把列表中的图片名转换成文件路径；名字不合法或不在图库目录内时返回 None。"""
        name = str(name or "").strip().replace("\\", "/")
        if not _NAME_RE.match(name):
            return None
        path = (self.root / name).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError:
            return None
        return path if path.is_file() else None

    async def read(self, name: str) -> tuple[bytes, str] | None:
        path = self.resolve(name)
        if path is None:
            return None
        data = await asyncio.to_thread(path.read_bytes)
        return data, _MIME.get(path.suffix.lower().lstrip("."), "image/png")
