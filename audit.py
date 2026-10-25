"""绘图记录：以 JSON Lines 保存在插件数据目录，超过上限时自动裁剪旧记录。"""

import asyncio
import json
import math
import os
from pathlib import Path


class AuditLog:
    def __init__(self, path: Path, max_records: int = 2000):
        self.path = Path(path)
        self.max_records = max_records
        self._count: int | None = None
        self._lock = asyncio.Lock()

    async def append(self, record: dict) -> None:
        # 调用方被取消（如插件重载）时，既不丢记录，也不在写文件途中释放锁
        task = asyncio.ensure_future(self._append_locked(record))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.wait({task})
            raise

    async def _append_locked(self, record: dict) -> None:
        async with self._lock:
            await asyncio.to_thread(self._append_sync, record)

    async def page(self, page: int, page_size: int) -> tuple[list[dict], int, int, int]:
        """按从新到旧分页，返回 (本页记录, 实际页码, 总页数, 总条数)。每条记录带有序号 _no。"""
        async with self._lock:
            lines = await asyncio.to_thread(self._read_lines)
        records = []
        for no, line in enumerate(lines, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                record["_no"] = no
                records.append(record)
        page_size = max(1, page_size)
        total = len(records)
        pages = max(1, math.ceil(total / page_size))
        page = min(max(1, page), pages)
        records.reverse()
        start = (page - 1) * page_size
        return records[start:start + page_size], page, pages, total

    def _read_lines(self) -> list[str]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8", errors="replace") as f:
            return [line.strip() for line in f if line.strip()]

    def _append_sync(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        if self._count is None:
            self._count = len(self._read_lines())
        else:
            self._count += 1
        limit = max(1, int(self.max_records))
        # 留一点余量，避免每写一条都重写整个文件
        if self._count > limit + max(10, limit // 10):
            lines = self._read_lines()[-limit:]
            tmp = self.path.with_name(self.path.name + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(tmp, self.path)
            self._count = len(lines)


def _clip(text: object, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


_STAGE_NAMES = {"config": "检查配置", "prompt": "生成提示词", "generate": "生成图片", "send": "发送图片"}


def format_record(record: dict) -> str:
    """把一条记录格式化成多行文本。"""
    ok = record.get("status") == "success"
    when = str(record.get("time", ""))[:19].replace("T", " ")
    who = f"{record.get('user_name') or '?'}({record.get('user_id', '?')})"
    where = f"群 {record['group_id']}" if record.get("group_id") else "私聊"
    head = f"#{record.get('_no', '?')} {'✅' if ok else '❌'} {when} | {who} | {where}"

    lines = [head]
    if record.get("model") or record.get("backend"):
        parts = [str(record.get("backend_label") or record.get("backend") or ""), str(record.get("model") or "")]
        parts.append(str(record.get("size") or ""))
        if record.get("steps"):
            parts.append(f"{record['steps']}步")
        params = " ".join(p for p in parts if p)
        if record.get("seed"):
            params += f" seed {record['seed']}"
        params += f" | {record.get('duration', 0)}s"
        if record.get("anlas_cost") is not None:
            params += f" | 消耗 {record['anlas_cost']} Anlas"
        elif record.get("opus_free") is not None:
            params += " | 预计免费" if record["opus_free"] else " | 可能消耗 Anlas"
        lines.append("  " + params)
    if not ok:
        stage = _STAGE_NAMES.get(record.get("stage", ""), record.get("stage", ""))
        lines.append(f"  失败（{stage}）：{_clip(record.get('error'), 120)}")
    lines.append(f"  需求：{_clip(record.get('request'), 60)}")
    if record.get("prompt"):
        lines.append(f"  Prompt：{_clip(record.get('prompt'), 120)}")
    return "\n".join(lines)
