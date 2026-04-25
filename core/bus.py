from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Callable, Awaitable, Set, List, Optional
import asyncio
import fnmatch
import logging


__all__ = ["EventBus", "Event", "Subscriber", "bus"]

log = logging.getLogger(__name__)

@dataclass(slots=True, frozen=True)
class Event:
    topic: str
    payload: Dict[str, Any]

Subscriber = Callable[[Event], Awaitable[None]]

class EventBus:
    """Simple pub/sub bus built on asyncio.Queue with glob patterns."""
    def __init__(self) -> None:
        self._alive: bool = False
        # Allow None as a poison pill, so Optional[Event] is required.
        self._q: "asyncio.Queue[Optional[Event]]" = asyncio.Queue()
        self._subs: Dict[str, Set[Subscriber]] = {}
        self._task: Optional[asyncio.Task] = None

    async def publish(self, topic: str, **payload: Any) -> None:
        # Publishing after stop() is allowed; events are just not processed if router already stopped.
        await self._q.put(Event(topic=topic, payload=payload))

    def subscribe(self, pattern: str, handler: Subscriber) -> None:
        self._subs.setdefault(pattern, set()).add(handler)

    def unsubscribe(self, pattern: str, handler: Subscriber) -> None:
        handlers = self._subs.get(pattern)
        if handlers and handler in handlers:
            handlers.remove(handler)
            if not handlers:
                del self._subs[pattern]    

    async def _router(self) -> None:
        log.info("core.bus: router started")
        try:
            while self._alive:
                ev = await self._q.get()
                if ev is None:  # Poison pill: wake up and exit.
                    break
                # Collect matched targets.
                targets: List[Subscriber] = []
                for pattern, handlers in self._subs.items():
                    if fnmatch.fnmatch(ev.topic, pattern):
                        targets.extend(handlers)
                if targets:
                    await asyncio.gather(
                        *[self._safe_call(h, ev) for h in targets],
                        return_exceptions=True
                    )
        except asyncio.CancelledError:
            log.info("core.bus: router cancelled")
        finally:
            self._alive = False
            log.info("core.bus: router stopped")

    async def _safe_call(self, handler: Subscriber, ev: Event) -> None:
        try:
            await handler(ev)
        except Exception:
            log.exception("core.bus: handler %s failed on %s", handler, ev)

    def start(self) -> None:
        if self._task and not self._task.done():
            return  # Already running.
        self._alive = True
        self._task = asyncio.create_task(self._router(), name="BusRouter")

    async def stop(self) -> None:
        # Cooperative shutdown: ask router to exit and wake blocked .get().
        self._alive = False
        try:
            self._q.put_nowait(None)  # sentinel
        except Exception:
            pass
        # Also cancel in case .get() is stuck or handlers are blocked.
        task = self._task
        self._task = None
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        log.info("core.bus: router shutdown complete")

bus = EventBus()
