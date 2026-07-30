"""Integrated OpenClaw response speaker runtime."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Callable

from audio.ducking import PulseAudioDucker
from core.config import cfg
from core import perf
from core.runtime_state import RuntimeStateStore
from speaker.config import SpeakerConfig
from speaker.emoji import EmojiDisplayClient, EmojiToken, extract_emoji_for_speech
from speaker.events import ChatSpeechRouter, SpeechSegment
from speaker.file_renderer import TTSFileRenderer
from speaker.gateway import GatewayClient, GatewayError
from speaker.messages import ExtractedMessage, MessageDeduper, extract_latest_assistant_text
from speaker.style import EmojiStyleResolver
from speaker.tts import SpeechEngine, SpeechRequest, create_speech_engine

log = logging.getLogger(__name__)

_RECENT_FINALIZED_RUN_LIMIT = 512


@dataclass(frozen=True, slots=True)
class _PlaybackItem:
    segment: SpeechSegment
    request: SpeechRequest
    emoji_tokens: tuple[EmojiToken, ...]


@dataclass(frozen=True, slots=True)
class _FinishRun:
    run_id: str


class SpeechPlaybackController:
    """Serial speech playback queue with interrupt support."""

    def __init__(
        self,
        *,
        speech: SpeechEngine,
        queue_size: int,
        enabled: bool = True,
        emoji_display: EmojiDisplayClient | None = None,
        ducking_config: object | None = None,
        style_resolver: EmojiStyleResolver | None = None,
    ) -> None:
        self._speech = speech
        self._emoji_display = emoji_display
        self._ducking_config = ducking_config
        self._style_resolver = style_resolver or EmojiStyleResolver()
        self._run_ducker: PulseAudioDucker | None = None
        self._ducked_run_id: str | None = None
        self._queue: asyncio.Queue[_PlaybackItem | _FinishRun | None] = asyncio.Queue(
            maxsize=max(1, int(queue_size or 1))
        )
        self._enabled = bool(enabled)
        self._worker_task: asyncio.Task[None] | None = None
        self._current_task: asyncio.Task[None] | None = None
        self._current_segment: SpeechSegment | None = None
        self._closing = False
        self._last_interrupt_reason = ""
        self._interrupt_all_generation = 0
        self._interrupted_run_ids: set[str] = set()
        self._control_tasks: set[asyncio.Task[None]] = set()

    def _emoji_display_enabled(self) -> bool:
        emoji_display = self._emoji_display
        if emoji_display is None:
            return False
        return bool(getattr(getattr(emoji_display, "config", None), "enabled", False))

    def _accepts_segments(self) -> bool:
        return self._enabled or self._emoji_display_enabled()

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
        for control_task in list(self._control_tasks):
            control_task.cancel()
        if self._control_tasks:
            await asyncio.gather(*self._control_tasks, return_exceptions=True)
        self._control_tasks.clear()
        await _close_speech_engine(self._speech)
        self._style_resolver.clear()

    async def set_enabled(self, enabled: bool, *, reason: str = "") -> dict:
        self._enabled = bool(enabled)
        dropped = 0
        if not self._enabled:
            dropped = await self.interrupt(reason=reason or "disabled")
        return {**self.get_status(), "dropped": dropped}

    def enqueue(self, segment: SpeechSegment) -> bool:
        if not self._accepts_segments():
            return False
        if self._is_run_interrupted(segment.run_id):
            log.debug(
                "SpeakerAgent: dropped interrupted run segment id=%s run_id=%s",
                segment.identifier,
                segment.run_id,
            )
            return False
        parsed = extract_emoji_for_speech(segment.text)
        style = self._style_resolver.resolve(
            segment.text,
            parsed.tokens,
            run_id=segment.run_id,
        )
        request = SpeechRequest(
            text=parsed.speech_text,
            run_id=segment.run_id,
            segment_id=segment.identifier,
            style_id=style.style_id,
            instruction=style.instruction,
            emoji=style.emoji,
        )
        try:
            self._queue.put_nowait(_PlaybackItem(segment, request, parsed.tokens))
            return True
        except asyncio.QueueFull:
            log.warning("SpeakerAgent: speech queue is full; dropping %s", segment.identifier)
            return False

    def finish_run(self, run_id: str) -> None:
        """Forget inherited style after all requests for a run have been queued."""
        self._style_resolver.discard(run_id)
        marker = _FinishRun(str(run_id or ""))
        try:
            self._queue.put_nowait(marker)
        except asyncio.QueueFull:
            task = asyncio.create_task(
                self._queue.put(marker),
                name=f"Speaker.finish_run.{marker.run_id or 'unknown'}",
            )
            self._control_tasks.add(task)
            task.add_done_callback(self._control_tasks.discard)

    async def interrupt(self, *, reason: str, run_id: str | None = None) -> int:
        self._last_interrupt_reason = str(reason or "")
        if run_id is None:
            self._interrupt_all_generation += 1
            self._style_resolver.clear()
        else:
            self._remember_interrupted_run(run_id)
            self._style_resolver.discard(run_id)
        current = self._current_segment
        dropped, drained_run_ids = self._drain_queue(run_id=run_id)
        if run_id is None:
            if current is not None:
                self._remember_interrupted_run(current.run_id)
            for drained_run_id in drained_run_ids:
                self._remember_interrupted_run(drained_run_id)
        task = self._current_task
        current_affected = current is not None and (
            run_id is None or current.run_id == run_id
        )
        should_cancel_current = (
            task is not None
            and not task.done()
            and current is not None
            and (run_id is None or current.run_id == run_id)
        )
        if should_cancel_current:
            dropped += 1
            await _interrupt_speech_engine(self._speech, run_id=run_id)
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        elif current_affected:
            dropped += 1
        await self._restore_run_ducking()
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
            "style": self._style_resolver.get_status(),
            "tts": _speech_engine_status(self._speech),
        }

    def _drain_queue(self, *, run_id: str | None) -> tuple[int, set[str]]:
        dropped = 0
        dropped_run_ids: set[str] = set()
        kept: list[_PlaybackItem | _FinishRun | None] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                kept.append(item)
                self._queue.task_done()
                continue
            if isinstance(item, _FinishRun):
                if run_id is None or item.run_id == run_id:
                    self._queue.task_done()
                    continue
                kept.append(item)
                self._queue.task_done()
                continue
            if run_id is None or item.segment.run_id == run_id:
                dropped += 1
                if item.segment.run_id:
                    dropped_run_ids.add(item.segment.run_id)
                self._queue.task_done()
                continue
            kept.append(item)
            self._queue.task_done()
        for item in kept:
            self._queue.put_nowait(item)
        return dropped, dropped_run_ids

    async def _worker(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            if isinstance(item, _FinishRun):
                try:
                    await _finish_speech_run(self._speech, run_id=item.run_id)
                finally:
                    self._queue.task_done()
                    if self._ducked_run_id == item.run_id:
                        await self._restore_run_ducking()
                continue
            segment = item.segment
            request = item.request
            if not self._accepts_segments():
                self._queue.task_done()
                continue
            self._current_segment = segment
            interrupt_generation = self._interrupt_all_generation
            if self._is_segment_interrupted(segment, interrupt_generation):
                await self._skip_interrupted_segment(segment)
                continue
            should_speak = self._enabled
            if should_speak:
                await self._ensure_run_ducking(segment.run_id)
                if self._is_segment_interrupted(segment, interrupt_generation):
                    await self._skip_interrupted_segment(segment)
                    continue
            if item.emoji_tokens:
                log.debug(
                    "SpeakerAgent: extracted %d emoji(s) from segment id=%s symbols=%s",
                    len(item.emoji_tokens),
                    segment.identifier,
                    "".join(token.symbol for token in item.emoji_tokens),
                )
                if self._emoji_display is not None:
                    await self._emoji_display.show_tokens(
                        item.emoji_tokens,
                        run_id=segment.run_id,
                        segment_id=segment.identifier,
                    )
            if self._is_segment_interrupted(segment, interrupt_generation):
                await self._skip_interrupted_segment(segment)
                continue
            if not should_speak:
                self._queue.task_done()
                self._current_segment = None
                continue
            if not request.text:
                log.info(
                    "SpeakerAgent: skipped speech for emoji-only segment id=%s run_id=%s",
                    segment.identifier,
                    segment.run_id,
                )
                self._queue.task_done()
                self._current_segment = None
                if self._queue.empty():
                    await self._restore_run_ducking()
                continue
            self._current_task = asyncio.create_task(
                self._speech.speak(request),
                name=f"Speaker.speak.{segment.identifier}",
            )
            try:
                log.info("SpeakerAgent: speaking assistant reply %s", segment.identifier)
                speak_start_ns = perf.now_ns()
                perf.emit(
                    "speaker",
                    "segment_start",
                    run_id=segment.run_id,
                    segment_id=segment.identifier,
                    style_id=request.style_id,
                    text=perf.text_preview(request.text),
                )
                await self._current_task
                perf.emit(
                    "speaker",
                    "segment_done",
                    run_id=segment.run_id,
                    segment_id=segment.identifier,
                    duration_ms=perf.elapsed_ms(speak_start_ns),
                )
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
                if self._queue.empty():
                    await self._restore_run_ducking()

    async def _skip_interrupted_segment(self, segment: SpeechSegment) -> None:
        log.debug(
            "SpeakerAgent: skipped interrupted speech segment id=%s run_id=%s",
            segment.identifier,
            segment.run_id,
        )
        self._queue.task_done()
        self._current_segment = None
        if self._queue.empty():
            await self._restore_run_ducking()

    def _remember_interrupted_run(self, run_id: str) -> None:
        normalized = str(run_id or "").strip()
        if not normalized:
            return
        self._interrupted_run_ids.add(normalized)
        if len(self._interrupted_run_ids) > 512:
            for old_run_id in list(self._interrupted_run_ids)[:128]:
                self._interrupted_run_ids.discard(old_run_id)

    def _is_run_interrupted(self, run_id: str) -> bool:
        return str(run_id or "").strip() in self._interrupted_run_ids

    def _is_segment_interrupted(
        self,
        segment: SpeechSegment,
        interrupt_generation: int,
    ) -> bool:
        return (
            self._interrupt_all_generation != interrupt_generation
            or self._is_run_interrupted(segment.run_id)
        )

    async def _ensure_run_ducking(self, run_id: str) -> None:
        if self._ducking_config is None:
            return
        if self._ducked_run_id == run_id and self._run_ducker is not None:
            return
        await self._restore_run_ducking()
        ducker = PulseAudioDucker(self._ducking_config)
        duck_start_ns = perf.now_ns()
        await ducker.duck()
        self._run_ducker = ducker
        self._ducked_run_id = run_id
        perf.emit(
            "speaker",
            "ducking_start",
            run_id=run_id,
            duration_ms=perf.elapsed_ms(duck_start_ns),
        )

    async def _restore_run_ducking(self) -> None:
        ducker = self._run_ducker
        run_id = self._ducked_run_id
        self._run_ducker = None
        self._ducked_run_id = None
        if ducker is None:
            return
        restore_start_ns = perf.now_ns()
        await ducker.restore()
        perf.emit(
            "speaker",
            "ducking_restore",
            run_id=run_id,
            duration_ms=perf.elapsed_ms(restore_start_ns),
        )


class SpeakerAgent:
    """OpenClaw Gateway listener that voices assistant replies locally."""

    def __init__(
        self,
        *,
        config: SpeakerConfig | None = None,
        gateway_factory: Callable[[object], GatewayClient] | None = None,
        speech: SpeechEngine | None = None,
        state_store: RuntimeStateStore | None = None,
    ) -> None:
        self._config = config or cfg.speaker
        self._state_store = state_store or RuntimeStateStore.from_config()
        self._enabled_source = "config"
        self._enabled_reason = ""
        self._enabled_changed_at = time.time()
        self._restore_runtime_state()
        self._gateway_factory = gateway_factory or (lambda gateway_cfg: GatewayClient(gateway_cfg))
        tts_mode = str(getattr(self._config.speaker, "tts_mode", "persistent") or "persistent")
        persistent_tts = tts_mode == "persistent"
        neural_tts = str(self._config.tts.backend or "").strip().casefold() in {
            "voxcpm2",
            "cosyvoice3",
        }
        self._speech = speech or create_speech_engine(self._config)
        self._emoji_display = EmojiDisplayClient(self._config.emoji_display)
        self._playback = SpeechPlaybackController(
            speech=self._speech,
            queue_size=self._config.speaker.queue_size,
            enabled=bool(self._config.enabled),
            emoji_display=self._emoji_display,
            ducking_config=self._config.playback.ducking
            if persistent_tts and not neural_tts
            else None,
            style_resolver=EmojiStyleResolver(self._config.tts.style),
        )
        self._file_renderer = TTSFileRenderer(
            speech=self._speech,
            config=self._config.file_render,
            style_config=self._config.tts.style,
        )
        self._running = False
        self._gateway_task: asyncio.Task[None] | None = None
        self._gateway: GatewayClient | None = None
        self._connected = False
        self._last_error = ""
        self._seen_delta_runs: set[str] = set()
        self._finalized_runs: dict[str, None] = {}

    def _should_listen_to_gateway(self) -> bool:
        return bool(self._config.enabled or self._config.emoji_display.enabled)

    def _remember_finalized_run(self, run_id: str) -> None:
        normalized = str(run_id or "").strip()
        if not normalized:
            return
        self._finalized_runs[normalized] = None
        self._seen_delta_runs.discard(normalized)
        while len(self._finalized_runs) > _RECENT_FINALIZED_RUN_LIMIT:
            self._finalized_runs.pop(next(iter(self._finalized_runs)))

    def _finish_finalized_run(self, run_id: str) -> None:
        self._remember_finalized_run(run_id)
        self._playback.finish_run(run_id)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._playback.start()
        await self._file_renderer.start()
        if self._config.enabled:
            await _start_speech_engine(self._speech)
        if self._should_listen_to_gateway():
            self._ensure_gateway_task()
        log.info("SpeakerAgent: started (enabled=%s)", self._config.enabled)

    async def close(self) -> None:
        if not self._running:
            return
        self._running = False
        self._config.enabled = False
        await self._stop_gateway_task()
        await self._file_renderer.close()
        await self._playback.close()
        log.info("SpeakerAgent: stopped")

    async def interrupt(self, *, reason: str = "api", run_id: str | None = None) -> int:
        if not self._running:
            return 0
        return await self._playback.interrupt(reason=reason, run_id=run_id)

    async def create_tts_file(
        self,
        text: str,
        *,
        style: str | None = None,
        filename: str | None = None,
    ) -> dict:
        return await self._file_renderer.submit(text, style=style, filename=filename)

    def get_tts_file(self, job_id: str) -> dict:
        return self._file_renderer.get_job(job_id)

    def list_tts_files(self) -> list[dict]:
        return self._file_renderer.list_jobs()

    async def cancel_tts_file(self, job_id: str) -> dict:
        return await self._file_renderer.cancel(job_id)

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
            await _start_speech_engine(self._speech)
            await self._playback.set_enabled(True, reason=reason)
            if self._running:
                self._ensure_gateway_task()
        else:
            await self._playback.set_enabled(False, reason=reason or source)
            if not self._config.emoji_display.enabled:
                await self._stop_gateway_task()
            elif self._running:
                self._ensure_gateway_task()
        self._enabled_source = str(source or "api")
        self._enabled_reason = str(reason or "")
        self._enabled_changed_at = time.time()
        self._save_runtime_state()
        log.info("SpeakerAgent: enabled=%s source=%s reason=%s", target, source, reason)
        return self.get_status()

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "enabled": bool(self._config.enabled),
            "connected": self._connected,
            "mode": self._config.speaker.mode,
            "tts_mode": self._config.speaker.tts_mode,
            "tts_backend": self._config.tts.backend,
            "session_key": self._config.gateway.session_key,
            "gateway_url": self._config.gateway.url,
            "last_error": self._last_error or None,
            "changed_at": self._enabled_changed_at,
            "source": self._enabled_source,
            "reason": self._enabled_reason,
            "emoji_display": self._emoji_display.get_status(),
            "playback": self._playback.get_status(),
            "file_render": self._file_renderer.get_status(),
        }

    def _restore_runtime_state(self) -> None:
        section = self._state_store.get_section("speaker")
        if not section:
            return
        enabled = section.get("enabled")
        if enabled is not None:
            self._config.enabled = bool(enabled)
        self._enabled_source = str(section.get("source") or "runtime_state")
        self._enabled_reason = str(section.get("reason") or "")
        changed_at = section.get("changed_at")
        try:
            self._enabled_changed_at = float(changed_at)
        except (TypeError, ValueError):
            self._enabled_changed_at = time.time()
        log.info("SpeakerAgent: restored persisted enabled=%s", self._config.enabled)

    def _save_runtime_state(self) -> None:
        try:
            self._state_store.save_section(
                "speaker",
                {
                    "enabled": bool(self._config.enabled),
                    "changed_at": self._enabled_changed_at,
                    "source": self._enabled_source,
                    "reason": self._enabled_reason,
                },
            )
        except Exception as exc:  # noqa: BLE001 - persistence must stay best-effort
            log.warning("SpeakerAgent: failed to persist runtime state: %s", exc)

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
        while self._running and self._should_listen_to_gateway():
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
                if not self._running or not self._should_listen_to_gateway():
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
            if state == "error" and run_id in self._finalized_runs:
                error_message = _preview(str(payload.get("errorMessage") or ""))
                stop_reason = _preview(str(payload.get("stopReason") or ""))
                log.warning(
                    "SpeakerAgent: ignored stale OpenClaw error after final "
                    "run_id=%s seq=%s error=%s stop_reason=%s",
                    run_id,
                    payload.get("seq", "-"),
                    error_message or "-",
                    stop_reason or "-",
                )
                perf.emit(
                    "openclaw",
                    "stale_error_ignored",
                    run_id=run_id,
                    seq=payload.get("seq"),
                    error_message=error_message,
                    stop_reason=stop_reason,
                )
                return
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
        run_id = str(payload.get("runId") or "final")
        self._enqueue(_message_to_segment(message, run_id, final=True))
        self._finish_finalized_run(run_id)

    async def _handle_streaming_event(
        self,
        payload: dict,
        event: dict,
        gateway: GatewayClient,
        deduper: MessageDeduper,
        router: ChatSpeechRouter,
    ) -> None:
        result = router.route(event)
        state = str(payload.get("state") or "")
        run_id = str(payload.get("runId") or "unknown")
        if state == "delta" and run_id not in self._seen_delta_runs:
            self._seen_delta_runs.add(run_id)
            perf.emit("openclaw", "first_delta_seen", run_id=run_id)
        for segment in result.segments:
            self._enqueue(segment)
        if not result.needs_history:
            if state == "final":
                self._finish_finalized_run(run_id)
            return
        log.info(
            "SpeakerAgent: final event needs history check run_id=%s known_segments=%d",
            run_id,
            len(result.segments),
        )
        try:
            message = await self._load_final_history_message(
                gateway,
                run_id=run_id,
                router=router,
            )
        except GatewayError as exc:
            router.discard(run_id)
            self._playback.finish_run(run_id)
            log.warning("SpeakerAgent: unable to load chat history: %s", exc)
            return
        if message is None:
            router.discard(run_id)
            self._playback.finish_run(run_id)
            log.debug("SpeakerAgent: history check found no assistant message run_id=%s", run_id)
            return
        expected_text = router.emitted_text(run_id)
        if deduper.seen(message) and not expected_text:
            router.discard(run_id)
            self._finish_finalized_run(run_id)
            log.debug(
                "SpeakerAgent: history check skipped seen assistant message run_id=%s message=%s",
                run_id,
                message.identifier,
            )
            return
        history_result = router.route_final_text(run_id, message.text)
        if deduper.seen(message) and not history_result.segments:
            self._finish_finalized_run(run_id)
            log.debug(
                "SpeakerAgent: history check skipped seen assistant message run_id=%s message=%s",
                run_id,
                message.identifier,
            )
            return
        deduper.mark_seen(message)
        log.info(
            "SpeakerAgent: history check produced %d final segment(s) run_id=%s message=%s",
            len(history_result.segments),
            run_id,
            message.identifier,
        )
        for segment in history_result.segments:
            self._enqueue(segment)
        self._finish_finalized_run(run_id)

    async def _load_final_history_message(
        self,
        gateway: GatewayClient,
        *,
        run_id: str,
        router: ChatSpeechRouter,
    ) -> ExtractedMessage | None:
        streaming = self._config.speaker.streaming
        retries = max(1, int(getattr(streaming, "final_history_retries", 5) or 1))
        delay_s = max(
            0.0,
            float(getattr(streaming, "final_history_retry_delay_ms", 120) or 0) / 1000.0,
        )
        expected_text = router.emitted_text(run_id)
        last_message: ExtractedMessage | None = None
        for attempt in range(retries):
            if attempt and delay_s > 0:
                await asyncio.sleep(delay_s)
            history = await self._load_history(gateway)
            message = extract_latest_assistant_text(history)
            if message is None:
                continue
            last_message = message
            if not expected_text or message.text.startswith(expected_text):
                return message
            if attempt < retries - 1:
                log.debug(
                    "SpeakerAgent: history candidate does not match stream prefix "
                    "run_id=%s attempt=%d/%d expected_chars=%d message_chars=%d",
                    run_id,
                    attempt + 1,
                    retries,
                    len(expected_text),
                    len(message.text),
                )
        return last_message

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


async def _interrupt_speech_engine(speech: SpeechEngine, *, run_id: str | None) -> None:
    interrupt = getattr(speech, "interrupt", None)
    if not callable(interrupt):
        return
    try:
        await interrupt(run_id=run_id)
    except TypeError:
        await interrupt()
    except Exception as exc:  # noqa: BLE001 - interruption remains best-effort
        log.warning("SpeakerAgent: TTS interrupt failed: %s", exc)


async def _start_speech_engine(speech: SpeechEngine) -> None:
    start = getattr(speech, "start", None)
    if not callable(start):
        return
    await start()


async def _close_speech_engine(speech: SpeechEngine) -> None:
    close = getattr(speech, "close", None)
    if not callable(close):
        return
    try:
        await close()
    except Exception as exc:  # noqa: BLE001 - shutdown remains best-effort
        log.warning("SpeakerAgent: TTS close failed: %s", exc)


async def _finish_speech_run(speech: SpeechEngine, *, run_id: str) -> None:
    finish_run = getattr(speech, "finish_run", None)
    if not callable(finish_run):
        return
    try:
        await finish_run(run_id)
    except Exception as exc:  # noqa: BLE001 - next reply must remain playable
        log.warning("SpeakerAgent: TTS run drain failed run_id=%s: %s", run_id, exc)


def _speech_engine_status(speech: SpeechEngine) -> dict | None:
    get_status = getattr(speech, "get_status", None)
    if not callable(get_status):
        return None
    try:
        status = get_status()
    except Exception as exc:  # noqa: BLE001 - status must not break control API
        return {"last_error": str(exc)}
    return status if isinstance(status, dict) else None


__all__ = ["SpeakerAgent", "SpeechPlaybackController"]
