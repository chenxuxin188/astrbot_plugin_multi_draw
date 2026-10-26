"""使用范围与每日绘图次数：按 SID 白名单 / 黑名单判断会话能否画画，按会话和用户统计当天画了几张。"""

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def _entries(raw) -> set[str]:
    if isinstance(raw, str):
        raw = raw.replace("，", ",").replace("\n", ",").split(",")
    if not isinstance(raw, (list, tuple)):
        return set()
    return {str(x).strip() for x in raw if str(x).strip()}


def session_ids(umo: str, group_id: str = "") -> set[str]:
    """一个会话可以匹配的写法：完整的 SID（unified_msg_origin）、SID 最后一段（群号或私聊对方的 ID），以及群号。"""
    ids = {umo}
    if ":" in umo:
        ids.add(umo.rsplit(":", 1)[-1])
    if group_id:
        ids.add(str(group_id))
    return {i for i in ids if i}


def session_blocked(umo: str, group_id: str, whitelist, blacklist) -> str:
    """会话不能画画时返回原因（whitelist / blacklist），可以画画时返回空字符串。白名单不为空时不看黑名单。"""
    ids = session_ids(umo, group_id)
    allow = _entries(whitelist)
    if allow:
        return "" if ids & allow else "whitelist"
    return "blacklist" if ids & _entries(blacklist) else ""


@dataclass
class Reservation:
    """一次正在进行的绘图占用的次数：完成后成功则计入当天次数，失败则释放。"""

    session: str
    user: str
    done: bool = False


class DailyQuota:
    """每天的绘图次数。

    已完成的次数保存在 JSON 文件中（日期变化时清零）；正在进行的绘图也计入，避免同时发起的请求超出限制。
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.date = ""
        self.counts: dict[str, dict[str, int]] = {"session": {}, "user": {}}
        self.pending: dict[str, dict[str, int]] = {"session": {}, "user": {}}
        self._lock = asyncio.Lock()
        self._load()

    @staticmethod
    def today() -> str:
        return datetime.now().astimezone().date().isoformat()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict) and data.get("date") == self.today():
            self.date = data["date"]
            for kind in ("session", "user"):
                raw = data.get(kind)
                if isinstance(raw, dict):
                    self.counts[kind] = {str(k): v for k, v in raw.items() if isinstance(v, int) and v > 0}
        else:
            self.date = self.today()

    def _roll(self) -> None:
        today = self.today()
        if self.date != today:
            self.date = today
            self.counts = {"session": {}, "user": {}}

    def used(self, kind: str, key: str, pending: bool = True) -> int:
        """今天已经画了的张数；pending 为 True 时包括正在画的。"""
        self._roll()
        return self.counts[kind].get(key, 0) + (self.pending[kind].get(key, 0) if pending else 0)

    def reserve(self, session: str, user: str) -> Reservation:
        self._roll()
        for kind, key in (("session", session), ("user", user)):
            self.pending[kind][key] = self.pending[kind].get(key, 0) + 1
        return Reservation(session, user)

    async def release(self, reservation: Reservation, success: bool) -> None:
        """结束一次绘图：成功时计入当天的次数（跨过 0 点才完成的计入新的一天）并保存。"""
        if reservation.done:
            return
        reservation.done = True
        for kind, key in (("session", reservation.session), ("user", reservation.user)):
            left = self.pending[kind].get(key, 0) - 1
            if left > 0:
                self.pending[kind][key] = left
            else:
                self.pending[kind].pop(key, None)
        if not success:
            return
        self._roll()
        for kind, key in (("session", reservation.session), ("user", reservation.user)):
            self.counts[kind][key] = self.counts[kind].get(key, 0) + 1
        data = {"date": self.date, "session": dict(self.counts["session"]), "user": dict(self.counts["user"])}
        async with self._lock:
            await asyncio.to_thread(self._save, data)

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
