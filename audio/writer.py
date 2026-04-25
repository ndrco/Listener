"""Buffered speech writer for feeding STT pipelines."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, TYPE_CHECKING

from core.bus import Event, EventBus, bus as default_bus
from core.config import cfg

from .processing import ProcessedAudioFrame

log = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from core.config import AudioBufferCfg


_PCM_SAMPLE_WIDTH = 2  # bytes (16-bit little endian PCM)


def _frame_duration_ms(frame: ProcessedAudioFrame) -> float:
    """Return duration of *frame* in milliseconds."""

    if frame.sample_rate <= 0 or frame.channels <= 0:
        return 0.0
    samples = len(frame.data) / (_PCM_SAMPLE_WIDTH * frame.channels)
    if samples <= 0:
        return 0.0
    return (samples / frame.sample_rate) * 1000.0


@dataclass(slots=True)
class SpeechSegment:
    """Represents a chunk of PCM audio ready for STT."""

    data: bytes
    sample_rate: int
    channels: int
    start_timestamp: float
    end_timestamp: float
    duration_ms: float
    frames: int
    voice_frames: int
    metadata: dict[str, Any]


class BufferedSpeechWriter:
    """Collects PCM frames around VAD activity and exposes speech segments."""

    def __init__(
        self,
        *,
        pre_roll_ms: float = 300.0,
        post_roll_ms: float = 400.0,
        max_silence_ms: float = 1000.0,
        max_segment_duration_ms: float = 30_000.0,
        max_segment_bytes: int = 5 * 1024 * 1024,
        queue_maxsize: int = 0,
        bus: EventBus | None = None,
    ) -> None:
        self._bus = bus or default_bus
        self._pre_roll_ms = float(max(pre_roll_ms, 0.0))
        self._post_roll_ms = float(max(post_roll_ms, 0.0))
        self._max_silence_ms = float(max(max_silence_ms, 0.0))
        self._max_segment_duration_ms = float(max(max_segment_duration_ms, 0.0))
        self._max_segment_bytes = int(max(max_segment_bytes, 0))
        self._queue: "asyncio.Queue[SpeechSegment]" = asyncio.Queue(
            maxsize=queue_maxsize
        )

        self._pre_buffer: "Deque[ProcessedAudioFrame]" = deque()
        self._pre_buffer_duration_ms: float = 0.0

        self._segment_frames: list[ProcessedAudioFrame] = []
        self._segment_active: bool = False
        self._pending_stop: bool = False
        self._silence_started_ts: float | None = None
        self._segment_start_ts: float | None = None
        self._segment_end_ts: float | None = None
        self._segment_duration_ms: float = 0.0
        self._segment_total_bytes: int = 0
        self._segment_voice_frames: int = 0
        self._segment_vad_probability_sum: float = 0.0
        self._segment_vad_probability_max: float = 0.0
        self._segment_webrtc_probability_max: float = 0.0
        self._segment_silero_probability_max: float = 0.0

        self._voice_active: bool = False

        self._segments_emitted: int = 0

        self._subscriptions: list[tuple[str, Any]] = []
        self._subscribe(cfg.events.audio.processed_frame, self._on_processed_frame)
        self._subscribe(cfg.events.audio.voice_activity, self._on_voice_activity)

        log.info(
            "audio.writer: buffered writer initialised (pre_roll=%.0f ms, post_roll=%.0f ms)",
            self._pre_roll_ms,
            self._post_roll_ms,
        )

    @classmethod
    def from_config(
        cls,
        buffer_cfg: "AudioBufferCfg | None" = None,
        *,
        bus: EventBus | None = None,
    ) -> "BufferedSpeechWriter":
        """Instantiate writer using :mod:`core.config` settings."""

        if buffer_cfg is None:
            buffer_cfg = cfg.audio.buffer

        return cls(
            pre_roll_ms=buffer_cfg.pre_roll_ms,
            post_roll_ms=buffer_cfg.post_roll_ms,
            max_silence_ms=buffer_cfg.max_silence_ms,
            max_segment_duration_ms=buffer_cfg.max_segment_duration_ms,
            max_segment_bytes=buffer_cfg.max_segment_bytes,
            queue_maxsize=buffer_cfg.queue_maxsize,
            bus=bus,
        )

    @property
    def queue(self) -> "asyncio.Queue[SpeechSegment]":
        """Queue that yields :class:`SpeechSegment` objects for STT consumers."""

        return self._queue

    def _subscribe(self, pattern: str, handler: Any) -> None:
        self._bus.subscribe(pattern, handler)
        self._subscriptions.append((pattern, handler))

    async def close(self) -> None:
        """Unsubscribe from events and flush pending data."""

        await self._finalise_segment(force=True, reason="shutdown")
        for pattern, handler in self._subscriptions:
            try:
                self._bus.unsubscribe(pattern, handler)
            except Exception:  # pragma: no cover - defensive
                log.exception(
                    "audio.writer: failed to unsubscribe handler %s from %s",
                    handler,
                    pattern,
                )
        self._subscriptions.clear()

    async def _on_processed_frame(self, event: Event) -> None:
        payload = event.payload
        frame = ProcessedAudioFrame(
            data=payload["data"],
            sample_rate=int(payload["sample_rate"]),
            channels=int(payload["channels"]),
            voice_detected=bool(payload.get("voice_activity", False)),
            timestamp=float(payload.get("timestamp", 0.0)),
            vad_probability=float(payload.get("vad_probability", 0.0)),
            vad_speech_frames=int(payload.get("vad_speech_frames", 0)),
            vad_total_frames=int(payload.get("vad_total_frames", 0)),
            voice_active_duration=float(payload.get("voice_active_duration", 0.0)),
            webrtc_probability=float(payload.get("webrtc_probability", 0.0)),
            silero_probability=float(payload.get("silero_probability", 0.0)),
            silero_invocations=int(payload.get("silero_invocations", 0)),
        )

        self._update_prebuffer(frame)
        self._set_voice_active(frame.voice_detected, frame.timestamp, frame=frame)
        if self._segment_active:
            self._append_to_segment(frame)
            await self._check_segment_limits(frame.timestamp)

    async def _on_voice_activity(self, event: Event) -> None:
        payload = event.payload
        active = bool(payload.get("active", False))
        timestamp = float(payload.get("timestamp", 0.0))
        self._set_voice_active(active, timestamp)
        if not active and self._segment_active:
            self._pending_stop = True
            self._silence_started_ts = timestamp
            await self._check_segment_limits(timestamp)

    def _update_prebuffer(self, frame: ProcessedAudioFrame) -> None:
        duration_ms = _frame_duration_ms(frame)
        self._pre_buffer.append(frame)
        self._pre_buffer_duration_ms += duration_ms
        while self._pre_buffer and self._pre_buffer_duration_ms > self._pre_roll_ms:
            dropped = self._pre_buffer.popleft()
            self._pre_buffer_duration_ms -= _frame_duration_ms(dropped)
        if self._pre_buffer_duration_ms < 0:
            self._pre_buffer_duration_ms = 0.0
        #if log.isEnabledFor(logging.DEBUG):
        #    log.debug(
        #        "audio.writer: pre-buffer %.0f ms (%d frames)",
        #        self._pre_buffer_duration_ms,
        #        len(self._pre_buffer),
        #    )

    def _set_voice_active(
        self,
        active: bool,
        timestamp: float,
        *,
        frame: ProcessedAudioFrame | None = None,
    ) -> None:
        if active == self._voice_active:
            return
        self._voice_active = active
        if active:
            self._pending_stop = False
            self._silence_started_ts = None
            if not self._segment_active:
                self._start_segment(current_frame=frame)
        else:
            if self._segment_active:
                self._pending_stop = True
                self._silence_started_ts = timestamp

    def _start_segment(self, *, current_frame: ProcessedAudioFrame | None = None) -> None:
        if self._segment_active:
            return
        self._segment_frames = []
        self._segment_active = True
        self._segment_start_ts = None
        self._segment_end_ts = None
        self._segment_duration_ms = 0.0
        self._segment_total_bytes = 0
        self._segment_voice_frames = 0
        self._segment_vad_probability_sum = 0.0
        self._segment_vad_probability_max = 0.0
        self._segment_webrtc_probability_max = 0.0
        self._segment_silero_probability_max = 0.0

        if self._pre_buffer:
            buffered_frames = list(self._pre_buffer)
            if current_frame is not None and buffered_frames:
                # The newest frame in the pre-buffer is the current frame. Avoid
                # duplicating it because it will be appended by the caller.
                if buffered_frames[-1] is current_frame:
                    buffered_frames = buffered_frames[:-1]
            appended = 0
            for frame in buffered_frames:
                self._append_to_segment(frame)
                appended += 1
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "audio.writer: segment started with %.0f ms pre-roll (%d frames)",
                    self._pre_buffer_duration_ms,
                    appended,
                )

    def _append_to_segment(self, frame: ProcessedAudioFrame) -> None:
        duration_ms = _frame_duration_ms(frame)
        self._segment_frames.append(frame)
        if self._segment_start_ts is None:
            self._segment_start_ts = frame.timestamp
        end_ts = frame.timestamp + (duration_ms / 1000.0)
        if self._segment_end_ts is None or end_ts > self._segment_end_ts:
            self._segment_end_ts = end_ts
        self._segment_duration_ms += duration_ms
        self._segment_total_bytes += len(frame.data)
        if frame.voice_detected:
            self._segment_voice_frames += 1
        self._segment_vad_probability_sum += frame.vad_probability
        if frame.vad_probability > self._segment_vad_probability_max:
            self._segment_vad_probability_max = frame.vad_probability
        if frame.webrtc_probability > self._segment_webrtc_probability_max:
            self._segment_webrtc_probability_max = frame.webrtc_probability
        if frame.silero_probability > self._segment_silero_probability_max:
            self._segment_silero_probability_max = frame.silero_probability

    async def _check_segment_limits(self, timestamp: float) -> None:
        if not self._segment_active:
            return

        max_bytes_reached = (
            self._max_segment_bytes > 0
            and self._segment_total_bytes >= self._max_segment_bytes
        )
        max_duration_reached = (
            self._max_segment_duration_ms > 0
            and self._segment_duration_ms >= self._max_segment_duration_ms
        )
        if max_bytes_reached or max_duration_reached:
            reason = "max-bytes" if max_bytes_reached else "max-duration"
            await self._finalise_segment(reason=reason)
            return

        if self._pending_stop and self._silence_started_ts is not None:
            silence_ms = max(0.0, (timestamp - self._silence_started_ts) * 1000.0)
            if silence_ms >= self._post_roll_ms or silence_ms >= self._max_silence_ms:
                await self._finalise_segment(reason="silence")

    async def _finalise_segment(
        self, *, force: bool = False, reason: str | None = None
    ) -> None:
        if not self._segment_active:
            return

        if not self._segment_frames:
            self._reset_segment()
            return

        if force and reason is None:
            reason = "forced"

        data = b"".join(frame.data for frame in self._segment_frames)
        frames_count = len(self._segment_frames)
        sample_rate = self._segment_frames[0].sample_rate
        channels = self._segment_frames[0].channels
        start_ts = self._segment_start_ts or self._segment_frames[0].timestamp
        end_ts = self._segment_end_ts or start_ts
        duration_ms = self._segment_duration_ms

        avg_vad_probability = (
            self._segment_vad_probability_sum / frames_count
            if frames_count
            else 0.0
        )

        metadata = {
            "bytes": len(data),
            "frames": frames_count,
            "voice_frames": self._segment_voice_frames,
            "vad_probability_avg": avg_vad_probability,
            "vad_probability_max": self._segment_vad_probability_max,
            "webrtc_probability_max": self._segment_webrtc_probability_max,
            "silero_probability_max": self._segment_silero_probability_max,
        }

        segment = SpeechSegment(
            data=data,
            sample_rate=sample_rate,
            channels=channels,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            duration_ms=duration_ms,
            frames=frames_count,
            voice_frames=self._segment_voice_frames,
            metadata=metadata,
        )

        if metadata["bytes"] == 0:
            self._reset_segment()
            return

        await self._queue.put(segment)
        self._segments_emitted += 1

        log.info(
            "audio.writer: segment #%d ready (duration=%.2f s, frames=%d, bytes=%d, reason=%s)",
            self._segments_emitted,
            duration_ms / 1000.0,
            frames_count,
            metadata["bytes"],
            reason or "complete",
        )

        self._reset_segment()

    def _reset_segment(self) -> None:
        self._segment_frames = []
        self._segment_active = False
        self._pending_stop = False
        self._silence_started_ts = None
        self._segment_start_ts = None
        self._segment_end_ts = None
        self._segment_duration_ms = 0.0
        self._segment_total_bytes = 0
        self._segment_voice_frames = 0
        self._segment_vad_probability_sum = 0.0
        self._segment_vad_probability_max = 0.0
        self._segment_webrtc_probability_max = 0.0
        self._segment_silero_probability_max = 0.0


__all__ = ["BufferedSpeechWriter", "SpeechSegment"]
