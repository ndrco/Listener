from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .config import SpeakerConfig
from .events import ChatSpeechRouter, SpeechSegment
from .gateway import GatewayClient, GatewayError
from .messages import ExtractedMessage, MessageDeduper, extract_latest_assistant_text
from .tts import PiperSpeechEngine, SpeechEngine

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SpeakerService:
    config: SpeakerConfig
    gateway: GatewayClient | None = None
    speech: SpeechEngine | None = None

    async def run_forever(self) -> None:
        backoff_s = 1.0
        while True:
            try:
                await self.run_until_disconnect()
                backoff_s = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - daemon must keep running
                log.warning("Speaker loop failed: %s", exc)
                await asyncio.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 30.0)

    async def run_until_disconnect(self) -> None:
        gateway = self.gateway or GatewayClient(self.config.gateway)
        speech = self.speech or PiperSpeechEngine(self.config.piper, self.config.playback)
        deduper = MessageDeduper()
        router = ChatSpeechRouter(self.config.gateway, self.config.speaker.streaming)
        queue: asyncio.Queue[SpeechSegment] = asyncio.Queue(maxsize=self.config.speaker.queue_size)
        worker = asyncio.create_task(self._speech_worker(queue, speech))

        try:
            await gateway.connect()
            log.info("Connected to OpenClaw Gateway")
            if not self.config.speaker.speak_existing_on_start:
                await self._mark_current_message_seen(gateway, deduper)
            async for event in gateway.events():
                await self._handle_event(event, gateway, deduper, queue, router)
            await queue.join()
        finally:
            worker.cancel()
            await gateway.close()
            try:
                await worker
            except asyncio.CancelledError:
                pass

    async def _mark_current_message_seen(self, gateway: GatewayClient, deduper: MessageDeduper) -> None:
        try:
            history = await self._load_history(gateway)
        except Exception as exc:  # noqa: BLE001
            log.debug("Initial history read failed: %s", exc)
            return
        message = extract_latest_assistant_text(history)
        if message:
            deduper.mark_seen(message)
            log.debug("Marked existing assistant reply as seen: %s", message.identifier)

    async def _handle_event(
        self,
        event: dict,
        gateway: GatewayClient,
        deduper: MessageDeduper,
        queue: asyncio.Queue[SpeechSegment],
        router: ChatSpeechRouter,
    ) -> None:
        if event.get("type") != "event" or event.get("event") != "chat":
            return
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return
        if self.config.speaker.mode == "streaming":
            await self._handle_streaming_event(payload, event, gateway, deduper, queue, router)
            return
        await self._handle_final_event(payload, gateway, deduper, queue)

    async def _handle_final_event(
        self,
        payload: dict,
        gateway: GatewayClient,
        deduper: MessageDeduper,
        queue: asyncio.Queue[SpeechSegment],
    ) -> None:
        if payload.get("state") != "final":
            return
        if not self.config.gateway.matches_session(str(payload.get("sessionKey") or "")):
            return

        try:
            history = await self._load_history(gateway)
        except GatewayError as exc:
            log.warning("Unable to load chat history: %s", exc)
            return

        message = extract_latest_assistant_text(history)
        if message is None or deduper.seen(message):
            return
        deduper.mark_seen(message)
        self._enqueue(queue, _message_to_segment(message, str(payload.get("runId") or "final"), final=True))

    async def _handle_streaming_event(
        self,
        payload: dict,
        event: dict,
        gateway: GatewayClient,
        deduper: MessageDeduper,
        queue: asyncio.Queue[SpeechSegment],
        router: ChatSpeechRouter,
    ) -> None:
        result = router.route(event)
        for segment in result.segments:
            self._enqueue(queue, segment)
        if not result.needs_history:
            return
        run_id = str(payload.get("runId") or "unknown")
        log.info(
            "Streaming final event needs history check run_id=%s known_segments=%d",
            run_id,
            len(result.segments),
        )
        try:
            history = await self._load_history(gateway)
        except GatewayError as exc:
            router.discard(run_id)
            log.warning("Unable to load chat history: %s", exc)
            return
        message = extract_latest_assistant_text(history)
        if message is None:
            router.discard(run_id)
            log.debug("History check found no assistant message run_id=%s", run_id)
            return
        if deduper.seen(message):
            router.discard(run_id)
            log.debug(
                "History check skipped seen assistant message run_id=%s message=%s",
                run_id,
                message.identifier,
            )
            return
        deduper.mark_seen(message)
        history_result = router.route_final_text(run_id, message.text)
        log.info(
            "History check produced %d final segment(s) run_id=%s message=%s",
            len(history_result.segments),
            run_id,
            message.identifier,
        )
        for segment in history_result.segments:
            self._enqueue(queue, segment)

    def _enqueue(self, queue: asyncio.Queue[SpeechSegment], segment: SpeechSegment) -> None:
        try:
            queue.put_nowait(segment)
            log.debug(
                "Queued speech segment id=%s run_id=%s final=%s chars=%d text=%r",
                segment.identifier,
                segment.run_id,
                segment.final,
                len(segment.text),
                _preview(segment.text),
            )
        except asyncio.QueueFull:
            log.warning("Speech queue is full; dropping message %s", segment.identifier)

    async def _load_history(self, gateway: GatewayClient) -> dict:
        return await gateway.request(
            "chat.history",
            {
                "sessionKey": self.config.gateway.session_key,
                "limit": self.config.gateway.history_limit,
                "maxChars": self.config.gateway.history_max_chars,
            },
            timeout_s=self.config.gateway.request_timeout_s,
        )

    async def _speech_worker(self, queue: asyncio.Queue[SpeechSegment], speech: SpeechEngine) -> None:
        while True:
            segment = await queue.get()
            try:
                log.info("Speaking assistant reply %s", segment.identifier)
                await speech.speak(segment.text)
            except Exception as exc:  # noqa: BLE001 - speech errors should not stop listening
                log.warning("Speech failed for %s: %s", segment.identifier, exc)
            finally:
                queue.task_done()


def _message_to_segment(message: ExtractedMessage, run_id: str, *, final: bool) -> SpeechSegment:
    return SpeechSegment(identifier=message.identifier, text=message.text, run_id=run_id, final=final)


def _preview(text: str, *, limit: int = 160) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return f"{value[: limit - 1]}..."
