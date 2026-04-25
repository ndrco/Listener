"""LiveKit Acoustic Echo Cancellation wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - optional dependency
    from livekit.rtc.apm import AudioProcessingModule
    from livekit.rtc.audio_frame import AudioFrame
except Exception:  # pragma: no cover - optional dependency missing or broken
    AudioProcessingModule = None  # type: ignore[assignment]
    AudioFrame = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


@dataclass(slots=True)
class AcousticEchoCancellationSettings:
    """AEC settings for LiveKit AudioProcessingModule."""

    enabled: bool = False
    frame_duration_ms: int = 10
    stream_delay_ms: int = 80
    noise_suppression: bool = False
    high_pass_filter: bool = False
    auto_gain_control: bool = False
    playback_event_topic: str | None = None
    playback_source: str = "event_bus"
    loopback_backend: str = "auto"
    loopback_device_index: int | None = None
    loopback_source_name: str | None = None
    loopback_device_name_contains: str | None = None
    loopback_frame_duration_ms: int | None = None


class AcousticEchoCanceller:
    """Wrapper around LiveKit AudioProcessingModule for acoustic echo cancellation."""

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        settings: AcousticEchoCancellationSettings,
        *,
        debug: bool = False,
    ) -> None:
        if AudioProcessingModule is None or AudioFrame is None:
            raise RuntimeError(
                "livekit.rtc.apm is not available; install 'livekit' to use AEC"
            )

        if channels != 1:
            raise ValueError("AcousticEchoCanceller currently supports only mono streams")

        frame_ms = max(5, int(settings.frame_duration_ms))
        frame_samples = int(sample_rate * frame_ms / 1000)
        if frame_samples <= 0 or frame_samples % channels != 0:
            raise ValueError(
                f"Invalid frame configuration for AEC: frame_ms={frame_ms}, "
                f"sample_rate={sample_rate}, channels={channels}"
            )

        self._debug = debug
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.frame_duration_ms = frame_ms
        self.frame_samples_per_channel = frame_samples // self.channels
        self.frame_samples_total = frame_samples
        self.frame_bytes = self.frame_samples_total * 2  # int16

        self._apm = AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=bool(settings.noise_suppression),
            high_pass_filter=bool(settings.high_pass_filter),
            auto_gain_control=bool(settings.auto_gain_control),
        )
        self._apm.set_stream_delay_ms(int(settings.stream_delay_ms))

        self._reverse_buffer = bytearray()
        self._near_buffer = bytearray()
        self._settings = settings

        if self._debug:
            log.debug(
                "audio.processing: AEC initialised (frame_ms=%d, delay_ms=%d)",
                self.frame_duration_ms,
                int(settings.stream_delay_ms),
            )

    @property
    def playback_event_topic(self) -> str | None:
        return self._settings.playback_event_topic

    @property
    def playback_source(self) -> str:
        return getattr(self._settings, "playback_source", "event_bus")

    @property
    def loopback_device_index(self) -> int | None:
        return getattr(self._settings, "loopback_device_index", None)

    def submit_farend(self, pcm: np.ndarray) -> None:
        """Add far-end audio (signal sent to speakers)."""
        if pcm.size == 0:
            return
        self._reverse_buffer.extend(np.asarray(pcm, dtype=np.int16).tobytes())
        self._drain_reverse()

    def process(self, pcm: np.ndarray) -> np.ndarray:
        """Apply AEC to near-end signal and return cleaned stream."""
        if pcm.size == 0:
            return pcm

        pcm_bytes = np.asarray(pcm, dtype=np.int16).tobytes()
        self._near_buffer.extend(pcm_bytes)

        frames: list[np.ndarray] = []
        while len(self._near_buffer) >= self.frame_bytes:
            self._process_reverse_frame()

            frame_bytes = bytes(self._near_buffer[: self.frame_bytes])
            del self._near_buffer[: self.frame_bytes]

            frame = AudioFrame(
                frame_bytes,
                self.sample_rate,
                self.channels,
                self.frame_samples_per_channel,
            )
            self._apm.process_stream(frame)
            cleaned = np.frombuffer(frame.data, dtype=np.int16)
            frames.append(cleaned.copy())

        if self._near_buffer:
            remainder = np.frombuffer(bytes(self._near_buffer), dtype=np.int16)
            frames.append(remainder.copy())
            self._near_buffer.clear()

        if not frames:
            return np.zeros(0, dtype=np.int16)
        if len(frames) == 1:
            return frames[0]
        return np.concatenate(frames)

    # === internal helpers ==================================================

    def _drain_reverse(self) -> None:
        while len(self._reverse_buffer) >= self.frame_bytes:
            self._process_reverse_frame()

    def _process_reverse_frame(self) -> None:
        if len(self._reverse_buffer) < self.frame_bytes:
            return
        frame_bytes = bytes(self._reverse_buffer[: self.frame_bytes])
        del self._reverse_buffer[: self.frame_bytes]
        frame = AudioFrame(
            frame_bytes,
            self.sample_rate,
            self.channels,
            self.frame_samples_per_channel,
        )
        self._apm.process_reverse_stream(frame)
