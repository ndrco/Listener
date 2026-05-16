"""Integrated OpenClaw response speaker runtime."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Callable

from core.config import cfg
from speaker.config import SpeakerConfig
from speaker.emoji import EmojiDisplayClient, extract_emoji_for_speech
from speaker.events import ChatSpeechRouter, SpeechSegment
from speaker.gateway import GatewayClient, GatewayError
from speaker.messages import ExtractedMessage, MessageDeduper, extract_latest_assistant_text
from speaker.tts import PiperSpeechEngine, SpeechEngine

log = logging.getLogger(__name__)


class SpeechPlaybackController:
    """Serial speech playback queue with interrupt support."""

    def __init__(
        self,
        *,
        speech: SpeechEngine,
        queue_size: int,
        enabled: bool = True,
        emoji_display: EmojiDisplayClient | None = None,
    ) -> None:
        self._speech = speech
        self._emoji_display = emoji_display
        self._queue: asyncio.Queue[SpeechSegment | None] = asyncio.Queue(
            maxsize=max(1, int(queue_size or 1))
        )
        self._enabled = bool(enabled)
        self._worker_task: asyncio.Task[None] | None = None
        self._current_task: asyncio.Task[None] | None = None
        self._current_segment: SpeechSegment | None = None
        self._closing = False
        self._last_interrupt_reason = ""

    async def start(self) -> None:
        if self._worker_task and not self._worker_task.done():
            return
        self._closing = False
        self._worker_task = asyncio.create_task(self._worker(), name="Speaker.playback")

    async def close(self) -> None:
        self._closing = True
        await self.interrupt(reason="shutdown")
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(None)
        task = self._worker_task
        self._worker_task = None
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def set_enabled(self, enabled: bool, *, reason: str = "") -> dict:
        self._enabled = bool(enabled)
        dropped = 0
        if not self._enabled:
            dropped = await self.interrupt(reason=reason or "disabled")
        return {**self.get_status(), "dropped": dropped}

    def enqueue(self, segment: SpeechSegment) -> bool:
        if not self._enabled:
            return False
        try:
            self._queue.put_nowait(segment)
            return True
        except asyncio.QueueFull:
            log.warning("SpeakerAgent: speech queue is full; dropping %s", segment.identifier)
            return False

    async def interrupt(self, *, reason: str, run_id: str | None = None) -> int:
        self._last_interrupt_reason = str(reason or "")
        dropped = self._drain_queue(run_id=run_id)
        current = self._current_segment
        task = self._current_task
        should_cancel_current = (
            task is not None
            and not task.done()
            and current is not None
            and (run_id is None or current.run_id == run_id)
        )
        if should_cancel_current:
            dropped += 1
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._emoji_display is not None:
            await self._emoji_display.clear(reason=str(reason or "interrupt"))
        if dropped:
            log.info(
                "SpeakerAgent: interrupted playback reason=%s run_id=%s dropped=%d",
                reason,
                run_id or "-",
                dropped,
            )
        return dropped

    def get_status(self) -> dict:
        current = self._current_segment
        return {
            "enabled": self._enabled,
            "queue_size": self._queue.qsize(),
            "current": current.identifier if current else None,
            "current_run_id": current.run_id if current else None,
            "last_interrupt_reason": self._last_interrupt_reason,
            "emoji_display": self._emoji_display.get_status()
            if self._emoji_display is not None
            else None,
        }

    def _drain_queue(self, *, run_id: str | None) -> int:
        dropped = 0
        kept: list[SpeechSegment | None] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                kept.append(item)
                self._queue.task_done()
                continue
            if run_id is None or item.run_id == run_id:
                dropped += 1
                self._queue.task_done()
                continue
            kept.append(item)
            self._queue.task_done()
        for item in kept:
            self._queue.put_nowait(item)
        return dropped

    async def _worker(self) -> None:
        while True:
            segment = await self._queue.get()
            if segment is None:
                self._queue.task_done()
                break
            if not self._enabled:
                self._queue.task_done()
                continue
            self._current_segment = segment
            parsed = extract_emoji_for_speech(segment.text)
            if parsed.tokens:
                log.debug(
                    "SpeakerAgent: extracted %d emoji(s) from segment id=%s symbols=%s",
                    len(parsed.tokens),
                    segment.identifier,
                    "".join(token.symbol for token in parsed.tokens),
                )
                if self._emoji_display is not None:
                    await self._emoji_display.show_tokens(
                        parsed.tokens,
                        run_id=segment.run_id,
                        segment_id=segment.identifier,
                    )
            if not parsed.speech_text:
                log.info(
                    "SpeakerAgent: skipped speech for emoji-only segment id=%s run_id=%s",
                    segment.identifier,
                    segment.run_id,
                )
                self._queue.task_done()
                self._current_segment = None
                continue
            self._current_task = asyncio.create_task(
                self._speech.speak(parsed.speech_text),
                name=f"Speaker.speak.{segment.identifier}",
            )
            try:
                log.info("SpeakerAgent: speaking assistant reply %s", segment.identifier)
                await self._current_task
            except asyncio.CancelledError:
                if self._closing:
                    raise
                log.debug("SpeakerAgent: speech task cancelled for %s", segment.identifier)
            except Exception as exc:  # noqa: BLE001 - speech errors should not stop the agent
                log.warning("SpeakerAgent: speech failed for %s: %s", segment.identifier, exc)
            finally:
                self._current_task = None
                self._current_segment = None
                self._queue.task_done()


class SpeakerAgent:
    """OpenClaw Gateway listener that voices assistant replies locally."""

    def __init__(
        self,
        *,
        config: SpeakerConfig | None = None,
        gateway_factory: Callable[[object], GatewayClient] | None = None,
        speech: SpeechEngine | None = None,
    ) -> None:
        self._config = config or cfg.speaker
        self._gateway_factory = gateway_factory or (lambda gateway_cfg: GatewayClient(gateway_cfg))
        self._speech = speech or PiperSpeechEngine(self._config.piper, self._config.playback)
        self._emoji_display = EmojiDisplayClient(self._config.emoji_display)
        self._playback = SpeechPlaybackController(
            speech=self._speech,
            queue_size=self._config.speaker.queue_size,
            enabled=bool(self._config.enabled),
            emoji_display=self._emoji_display,
        )
        self._running = False
        self._gateway_task: asyncio.Task[None] | None = None
        self._gateway: GatewayClient | None = None
        self._connected = False
        self._last_error = ""

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._playback.start()
        if self._config.enabled:
            self._ensure_gateway_task()
        log.info("SpeakerAgent: started (enabled=%s)", self._config.enabled)

    async def close(self) -> None:
        if not self._running:
            return
        self._running = False
        self._config.enabled = False
        await self._stop_gateway_task()
        await self._playback.close()
        log.info("SpeakerAgent: stopped")

    async def interrupt(self, *, reason: str = "api", run_id: str | None = None) -> int:
        if not self._running:
            return 0
        return await self._playback.interrupt(reason=reason, run_id=run_id)

    async def set_enabled(
        self,
        enabled: bool,
        *,
        source: str = "api",
        reason: str = "",
    ) -> dict:
        target = bool(enabled)
        self._config.enabled = target
        if target:
            await self._playback.set_enabled(True, reason=reason)
            if self._running:
                self._ensure_gateway_task()
        else:
            await self._stop_gateway_task()
            await self._playback.set_enabled(False, reason=reason or source)
        log.info("SpeakerAgent: enabled=%s source=%s reason=%s", target, source, reason)
        return self.get_status()

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "enabled": bool(self._config.enabled),
            "connected": self._connected,
            "mode": self._config.speaker.mode,
            "session_key": self._config.gateway.session_key,
            "gateway_url": self._config.gateway.url,
            "last_error": self._last_error or None,
            "emoji_display": self._emoji_display.get_status(),
            "playback": self._playback.get_status(),
        }

    def _ensure_gateway_task(self) -> None:
        if self._gateway_task and not self._gateway_task.done():
            return
        self._gateway_task = asyncio.create_task(self._run_forever(), name="Speaker.gateway")

    async def _stop_gateway_task(self) -> None:
        task = self._gateway_task
        self._gateway_task = None
        gateway = self._gateway
        if gateway is not None:
            with contextlib.suppress(Exception):
                await gateway.close()
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._connected = False

    async def _run_forever(self) -> None:
        backoff_s = 1.0
        while self._running and self._config.enabled:
            try:
                await self._run_until_disconnect()
                backoff_s = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - daemon must keep retrying
                self._last_error = str(exc)
                log.warning("SpeakerAgent: gateway loop failed: %s", exc)
                await asyncio.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 30.0)

    async def _run_until_disconnect(self) -> None:
        gateway = self._gateway_factory(self._config.gateway)
        self._gateway = gateway
        deduper = MessageDeduper()
        router = ChatSpeechRouter(self._config.gateway, self._config.speaker.streaming)
        try:
            await gateway.connect()
            self._connected = True
            self._last_error = ""
            log.info("SpeakerAgent: connected to OpenClaw Gateway")
            if not self._config.speaker.speak_existing_on_start:
                await self._mark_current_message_seen(gateway, deduper)
            async for event in gateway.events():
                if not self._running or not self._config.enabled:
                    break
                await self._handle_event(event, gateway, deduper, router)
        finally:
            self._connected = False
            if self._gateway is gateway:
                self._gateway = None
            await gateway.close()

    async def _mark_current_message_seen(
        self,
        gateway: GatewayClient,
        deduper: MessageDeduper,
    ) -> None:
        try:
            history = await self._load_history(gateway)
        except Exception as exc:  # noqa: BLE001
            log.debug("SpeakerAgent: initial history read failed: %s", exc)
            return
        message = extract_latest_assistant_text(history)
        if message:
            deduper.mark_seen(message)
            log.debug("SpeakerAgent: marked existing assistant reply as seen: %s", message.identifier)

    async def _handle_event(
        self,
        event: dict,
        gateway: GatewayClient,
        deduper: MessageDeduper,
        router: ChatSpeechRouter,
    ) -> None:
        if event.get("type") != "event" or event.get("event") != "chat":
            return
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return
        state = str(payload.get("state") or "")
        run_id = str(payload.get("runId") or "").strip()
        if state in {"aborted", "error"}:
            router.route(event)
            if run_id:
                await self.interrupt(reason=f"openclaw_{state}", run_id=run_id)
            return
        if self._config.speaker.mode == "streaming":
            await self._handle_streaming_event(payload, event, gateway, deduper, router)
            return
        await self._handle_final_event(payload, gateway, deduper)

    async def _handle_final_event(
        self,
        payload: dict,
        gateway: GatewayClient,
        deduper: MessageDeduper,
    ) -> None:
        if payload.get("state") != "final":
            return
        if not self._config.gateway.matches_session(str(payload.get("sessionKey") or "")):
            return
        try:
            history = await self._load_history(gateway)
        except GatewayError as exc:
            log.warning("SpeakerAgent: unable to load chat history: %s", exc)
            return
        message = extract_latest_assistant_text(history)
        if message is None or deduper.seen(message):
            return
        deduper.mark_seen(message)
        self._enqueue(_message_to_segment(message, str(payload.get("runId") or "final"), final=True))

    async def _handle_streaming_event(
        self,
        payload: dict,
        event: dict,
        gateway: GatewayClient,
        deduper: MessageDeduper,
        router: ChatSpeechRouter,
    ) -> None:
        result = router.route(event)
        for segment in result.segments:
            self._enqueue(segment)
        if not result.needs_history:
            return
        run_id = str(payload.get("runId") or "unknown")
        log.info(
            "SpeakerAgent: final event needs history check run_id=%s known_segments=%d",
            run_id,
            len(result.segments),
        )
        try:
            history = await self._load_history(gateway)
        except GatewayError as exc:
            router.discard(run_id)
            log.warning("SpeakerAgent: unable to load chat history: %s", exc)
            return
        message = extract_latest_assistant_text(history)
        if message is None:
            router.discard(run_id)
            log.debug("SpeakerAgent: history check found no assistant message run_id=%s", run_id)
            return
        if deduper.seen(message):
            router.discard(run_id)
            log.debug(
                "SpeakerAgent: history check skipped seen assistant message run_id=%s message=%s",
                run_id,
                message.identifier,
            )
            return
        deduper.mark_seen(message)
        history_result = router.route_final_text(run_id, message.text)
        log.info(
            "SpeakerAgent: history check produced %d final segment(s) run_id=%s message=%s",
            len(history_result.segments),
            run_id,
            message.identifier,
        )
        for segment in history_result.segments:
            self._enqueue(segment)

    def _enqueue(self, segment: SpeechSegment) -> None:
        enqueued = self._playback.enqueue(segment)
        if enqueued:
            log.debug(
                "SpeakerAgent: queued speech segment id=%s run_id=%s final=%s chars=%d text=%r",
                segment.identifier,
                segment.run_id,
                segment.final,
                len(segment.text),
                _preview(segment.text),
            )
        else:
            log.warning(
                "SpeakerAgent: dropped speech segment id=%s run_id=%s",
                segment.identifier,
                segment.run_id,
            )

    async def _load_history(self, gateway: GatewayClient) -> dict:
        return await gateway.request(
            "chat.history",
            {
                "sessionKey": self._config.gateway.session_key,
                "limit": self._config.gateway.history_limit,
                "maxChars": self._config.gateway.history_max_chars,
            },
            timeout_s=self._config.gateway.request_timeout_s,
        )


def _message_to_segment(message: ExtractedMessage, run_id: str, *, final: bool) -> SpeechSegment:
    return SpeechSegment(identifier=message.identifier, text=message.text, run_id=run_id, final=final)


def _preview(text: str, *, limit: int = 160) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return f"{value[: limit - 1]}..."


__all__ = ["SpeakerAgent", "SpeechPlaybackController"]
