"""Asynchronous Whisper-based streaming speech-to-text pipeline."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from typing import Any, Awaitable, Callable, Dict, Iterable

from core.bus import EventBus, bus as default_bus
from core.config import WhisperSttCfg, cfg

from audio.writer import BufferedSpeechWriter, SpeechSegment

from .whisper_engine import WhisperEngine

log = logging.getLogger(__name__)

FinalCallback = Callable[[str, Dict[str, Any]], Awaitable[None] | None]

_SPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class StreamingTranscriberConfig:
    """Runtime configuration for :class:`WhisperStreamingTranscriber`."""

    partial_topic: str = "audio/stt/partial"
    final_topic: str = "audio/stt/final"
    min_confidence: float = 0.35
    stability_timeout_s: float = 1.2
    queue_wait_s: float = 0.2
    enable_punctuation: bool = True

    @classmethod
    def from_whisper_cfg(cls, stt_cfg: WhisperSttCfg) -> "StreamingTranscriberConfig":
        return cls(
            partial_topic=stt_cfg.partial_topic,
            final_topic=stt_cfg.final_topic,
            min_confidence=stt_cfg.min_confidence,
            stability_timeout_s=stt_cfg.stability_timeout_s,
            queue_wait_s=stt_cfg.queue_wait_s,
            enable_punctuation=stt_cfg.enable_punctuation,
        )


class WhisperStreamingTranscriber:
    """Consumes :class:`~audio.writer.SpeechSegment` objects and produces STT events."""

    def __init__(
        self,
        writer: BufferedSpeechWriter,
        *,
        stt_config: WhisperSttCfg | None = None,
        bus: EventBus | None = None,
        runtime_config: StreamingTranscriberConfig | None = None,
        llm_queue_maxsize: int = 0,
        on_final: FinalCallback | None = None,
        debug: bool | None = None,
    ) -> None:
        self._writer = writer
        self._bus = bus or default_bus

        self._stt_config = stt_config or cfg.audio.stt
        runtime_base = StreamingTranscriberConfig.from_whisper_cfg(self._stt_config)
        if runtime_config is None:
            self._runtime = runtime_base
        else:
            overrides = {
                field.name: getattr(runtime_config, field.name)
                for field in fields(StreamingTranscriberConfig)
            }
            self._runtime = replace(runtime_base, **overrides)

        self._engine_debug = bool(cfg.debug) if debug is None else bool(debug)
        self._engine: WhisperEngine | None = None
        self._executor: ThreadPoolExecutor | None = None

        self._llm_queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=llm_queue_maxsize)
        self._on_final = on_final

        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._segment_index = 0

        self._current_partial: str = ""
        self._current_raw_partial: str = ""
        self._last_update_ts: float | None = None
        self._current_metadata: Dict[str, Any] | None = None
        self._current_audio: bytes | None = None

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._running:
            return
        await self._ensure_engine()
        self._running = True
        self._task = asyncio.create_task(self._run(), name="WhisperStreamingTranscriber")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._flush_pending(reason="shutdown")
        self._shutdown_executor()

    async def __aenter__(self) -> "WhisperStreamingTranscriber":  # pragma: no cover - convenience
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - convenience
        await self.stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def llm_queue(self) -> "asyncio.Queue[str]":
        """Queue with final transcripts for downstream LLM pipeline."""

        return self._llm_queue

    async def _ensure_engine(self) -> None:
        if self._engine is not None:
            return
        executor = self._ensure_executor()
        loop = asyncio.get_running_loop()
        try:
            self._engine = await loop.run_in_executor(executor, self._create_engine)
        except Exception:
            self._shutdown_executor()
            raise

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="WhisperSTT",
            )
        return self._executor

    def _shutdown_executor(self) -> None:
        executor = self._executor
        self._executor = None
        self._engine = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _create_engine(self) -> WhisperEngine:
        return WhisperEngine(self._stt_config, debug=self._engine_debug)

    async def _transcribe_in_executor(self, data: bytes, *, sample_rate: int) -> list[str]:
        engine = self._engine
        if engine is None:
            raise RuntimeError("audio.stt.streaming: Whisper engine is not initialised")
        executor = self._ensure_executor()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            lambda: engine.transcribe(data, sample_rate=sample_rate),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _run(self) -> None:
        queue = self._writer.queue
        wait_timeout = max(0.05, float(self._runtime.queue_wait_s))
        try:
            while self._running:
                try:
                    segment = await asyncio.wait_for(queue.get(), timeout=wait_timeout)
                except asyncio.TimeoutError:
                    await self._finalise_if_stale(reason="timeout")
                    continue
                await self._process_segment(segment)
        except asyncio.CancelledError:
            log.debug("audio.stt.streaming: transcription loop cancelled")
        finally:
            self._running = False

    async def _process_segment(self, segment: SpeechSegment) -> None:
        self._segment_index += 1
        metadata = self._build_metadata(segment)
        self._current_metadata = metadata
        self._current_audio = segment.data
        try:
            results = await self._transcribe_in_executor(
                segment.data,
                sample_rate=segment.sample_rate,
            )
        except Exception:
            log.exception(
                "audio.stt.streaming: transcription failed for segment #%d "
                "(duration=%.2fs, sample_rate=%d, bytes=%d, model=%s, device=%s, "
                "compute_type=%s)",
                self._segment_index,
                segment.duration_ms / 1000.0,
                segment.sample_rate,
                len(segment.data),
                getattr(self._stt_config, "model", None),
                getattr(self._stt_config, "device", None),
                getattr(self._stt_config, "compute_type", None),
            )
            self._reset_state()
            return
        if not self._running:
            return

        if not results:
            log.debug(
                "audio.stt.streaming: empty transcription for segment #%d (duration=%.2fs)",
                self._segment_index,
                segment.duration_ms / 1000.0,
            )
            await self._finalise_if_stale(reason="empty")
            return

        for raw in results:
            cleaned = self._normalise_text(raw, final=False)
            if not cleaned:
                continue
            merged = self._merge_partial(self._current_partial, cleaned)
            if not merged:
                continue
            if merged == self._current_partial:
                continue
            self._current_partial = merged
            self._current_raw_partial = raw
            self._last_update_ts = time.monotonic()
            await self._publish_partial(self._current_partial, metadata)

        await self._finalise_if_stale(metadata=metadata, reason="segment_end", force=True)

    async def _finalise_if_stale(
        self,
        *,
        metadata: Dict[str, Any] | None = None,
        reason: str,
        force: bool = False,
    ) -> None:
        if not self._current_partial:
            return
        if not force and self._runtime.stability_timeout_s > 0.0:
            last_ts = self._last_update_ts
            if last_ts is None:
                return
            elapsed = time.monotonic() - last_ts
            if elapsed < self._runtime.stability_timeout_s:
                return

        metadata = metadata or self._current_metadata or {}
        confidence = self._estimate_confidence(metadata)
        if not force and confidence < self._runtime.min_confidence:
            log.debug(
                "audio.stt.streaming: confidence %.2f below threshold %.2f; waiting",
                confidence,
                self._runtime.min_confidence,
            )
            return

        final_text = self._normalise_text(self._current_partial, final=True)
        if not final_text:
            self._reset_state()
            return

        payload = dict(metadata)
        payload.update(
            {
                "text": final_text,
                "raw_text": self._current_raw_partial,
                "confidence": confidence,
                "reason": reason,
                "segment_index": self._segment_index,
                "timestamp": time.time(),
                "pcm_data": self._current_audio,
            }
        )

        await self._publish_final(payload)
        self._reset_state()

    async def _flush_pending(self, *, reason: str) -> None:
        if self._current_partial:
            await self._finalise_if_stale(reason=reason, force=True)

    def _reset_state(self) -> None:
        self._current_partial = ""
        self._current_raw_partial = ""
        self._current_metadata = None
        self._last_update_ts = None
        self._current_audio = None

    def _build_metadata(self, segment: SpeechSegment) -> Dict[str, Any]:
        metadata = dict(segment.metadata)
        metadata.update(
            {
                "start_timestamp": segment.start_timestamp,
                "end_timestamp": segment.end_timestamp,
                "duration_ms": segment.duration_ms,
                "sample_rate": segment.sample_rate,
                "channels": segment.channels,
                "frames": segment.frames,
                "voice_frames": segment.voice_frames,
            }
        )
        metadata.setdefault("confidence", self._estimate_confidence(metadata))
        return metadata

    @staticmethod
    def _merge_partial(previous: str, current: str) -> str:
        if not previous:
            return current
        if current.startswith(previous):
            return current
        if previous.startswith(current):
            return previous
        if previous in current:
            return current
        if current in previous:
            return previous

        overlap = 0
        max_len = min(len(previous), len(current))
        for size in range(max_len, 0, -1):
            if previous.endswith(current[:size]):
                overlap = size
                break
        if overlap:
            return previous + current[overlap:]
        if len(current) > len(previous):
            return current
        return previous

    def _normalise_text(self, text: str, *, final: bool) -> str:
        if not text:
            return ""
        cleaned = _SPACE_RE.sub(" ", text).strip()
        if not cleaned:
            return ""
        first = cleaned[0]
        if first.isalpha():
            cleaned = first.upper() + cleaned[1:]
        if final and self._runtime.enable_punctuation:
            if cleaned[-1] not in ".!?…":
                cleaned = f"{cleaned}."
        return cleaned

    @staticmethod
    def _estimate_confidence(metadata: Dict[str, Any]) -> float:
        candidates: Iterable[Any] = (
            metadata.get("confidence"),
            metadata.get("vad_probability_max"),
            metadata.get("vad_probability_avg"),
            metadata.get("webrtc_probability_max"),
            metadata.get("silero_probability_max"),
        )
        best = 0.0
        for value in candidates:
            try:
                if value is None:
                    continue
                best = max(best, float(value))
            except (TypeError, ValueError):
                continue
        return best

    async def _publish_partial(self, text: str, metadata: Dict[str, Any]) -> None:
        payload = dict(metadata)
        payload.update({"text": text, "raw_text": self._current_raw_partial, "is_final": False})
        await self._safe_publish(self._runtime.partial_topic, payload)

    async def _publish_final(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["is_final"] = True
        publish_payload = dict(payload)
        publish_payload.pop("pcm_data", None)
        await self._safe_publish(self._runtime.final_topic, publish_payload)

        try:
            self._llm_queue.put_nowait(payload["text"])
        except asyncio.QueueFull:
            log.warning("audio.stt.streaming: LLM queue full; dropping transcript")

        callback = self._on_final
        if callback is not None:
            try:
                result = callback(payload["text"], payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # pragma: no cover - defensive
                log.exception("audio.stt.streaming: final callback failed")

    async def _safe_publish(self, topic: str, payload: Dict[str, Any]) -> None:
        try:
            await self._bus.publish(topic, **payload)
        except Exception:
            log.exception("audio.stt.streaming: failed to publish %s", topic)


__all__ = ["WhisperStreamingTranscriber", "StreamingTranscriberConfig"]

