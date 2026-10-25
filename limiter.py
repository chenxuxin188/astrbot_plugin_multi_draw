"""按平台限制同时进行的绘图请求数：先来先服务，修改上限后立即按新值生效。"""

import asyncio
from collections import deque
from collections.abc import Callable


class Limiter:
    """可以随时调整上限的 FIFO 信号量。

    上限由 limit() 在每次分配名额时读取：调高后排队的请求立即开始（需要调用 dispatch()），
    调低后正在进行的请求不受影响，新的请求等到进行中的数量低于新上限后才开始。
    """

    def __init__(self, limit: Callable[[], int]):
        self._limit = limit
        self._waiters: deque[asyncio.Future] = deque()
        self.active = 0

    @property
    def waiting(self) -> int:
        return sum(1 for fut in self._waiters if not fut.done())

    def queued_ahead(self) -> int:
        """新请求前面还有几张：没有空闲名额时，正在进行和排队的都算在前面。"""
        if self.active < self._limit() and not self.waiting:
            return 0
        return self.active + self.waiting

    async def acquire(self) -> None:
        if self.active < self._limit() and not self._waiters:
            self.active += 1
            return
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()  # 刚分配到名额就被取消，把名额还回去
            else:
                try:
                    self._waiters.remove(fut)
                except ValueError:
                    pass
            raise

    def release(self) -> None:
        self.active -= 1
        self.dispatch()

    def dispatch(self) -> None:
        """按当前上限把空闲名额依次分给排队的请求。"""
        while self._waiters and self.active < self._limit():
            fut = self._waiters.popleft()
            if fut.done():
                continue
            self.active += 1
            fut.set_result(None)
