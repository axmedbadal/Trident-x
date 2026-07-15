import asyncio
from collections import defaultdict
from typing import Any, Callable, Dict, List


class EventBus:
    def __init__(self):
        self._subs: Dict[str, List[Callable]] = defaultdict(list)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task = None

    def subscribe(self, topic: str, handler: Callable):
        self._subs[topic].append(handler)

    def publish(self, topic: str, payload: Any):
        self._queue.put_nowait((topic, payload))

    async def start(self):
        self._task = asyncio.create_task(self._dispatch_loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _dispatch_loop(self):
        while True:
            topic, payload = await self._queue.get()
            for handler in self._subs.get(topic, []):
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(topic, payload))
                    else:
                        handler(topic, payload)
                except Exception:
                    pass


bus = EventBus()
