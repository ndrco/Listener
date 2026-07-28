"""Low-latency playback for PCM chunks produced by neural TTS workers."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Callable

from audio.ducking import normalize_listener_output_volume_state

try:  # pragma: no cover - availability depends on the host audio stack
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None  # type: ignore[assignment]


log = logging.getLogger(__name__)


class StreamingPlaybackError(RuntimeError):
    pass


class PCMStreamPlayer:
    """Own one RawOutputStream and use its blocking write as backpressure."""

    def __init__(
        self,
        *,
        device: int | str | None = None,
        latency: str | float = "low",
        stream_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.device = device
        self.latency = latency
        self._stream_factory = stream_factory
        self._stream: Any | None = None
        self._sample_rate: int | None = None
        self._channels: int | None = None
        self._lock = asyncio.Lock()
        self._bytes_written = 0
        self._normalized_stream_ids: list[int] = []
        self._normalization_error = ""

    async def start(self, *, sample_rate: int, channels: int = 1, sample_width: int = 2) -> None:
        if sample_rate <= 0:
            raise StreamingPlaybackError("sample_rate must be positive")
        if channels != 1 or sample_width != 2:
            raise StreamingPlaybackError("only mono PCM16 playback is supported")
        async with self._lock:
            if self._stream is not None:
                if self._sample_rate == sample_rate and self._channels == channels:
                    return
                await self._close_locked(abort=True)
            factory = self._stream_factory
            native_output = factory is None
            if factory is None:
                if sd is None:
                    raise StreamingPlaybackError("sounddevice is not installed")
                factory = sd.RawOutputStream
            try:
                stream = await asyncio.to_thread(
                    factory,
                    samplerate=int(sample_rate),
                    channels=int(channels),
                    dtype="int16",
                    device=self.device,
                    latency=self.latency,
                )
                await asyncio.to_thread(stream.start)
            except Exception as exc:  # noqa: BLE001 - normalized for backend fallback
                raise StreamingPlaybackError(f"unable to start PCM output: {exc}") from exc
            self._stream = stream
            self._sample_rate = int(sample_rate)
            self._channels = int(channels)
            self._bytes_written = 0
            self._normalized_stream_ids = []
            self._normalization_error = ""
            if native_output:
                try:
                    volume_state = await normalize_listener_output_volume_state()
                    self._normalized_stream_ids = [
                        int(value) for value in volume_state.get("stream_ids", [])
                    ]
                except Exception as exc:  # noqa: BLE001 - playback remains available
                    self._normalization_error = str(exc)
                    log.warning("PCM output volume normalization failed: %s", exc)

    async def write(self, pcm: bytes) -> None:
        payload = bytes(pcm)
        if not payload:
            return
        if len(payload) % 2:
            raise StreamingPlaybackError("PCM16 chunk length must be even")
        async with self._lock:
            stream = self._stream
            if stream is None:
                raise StreamingPlaybackError("PCM output is not started")
            try:
                # PortAudio is not safe when abort/close races a write from a
                # different executor thread. Serialize native stream calls;
                # worker cancellation still proceeds concurrently.
                await asyncio.to_thread(stream.write, payload)
            except Exception as exc:  # noqa: BLE001
                raise StreamingPlaybackError(f"PCM output failed: {exc}") from exc
            self._bytes_written += len(payload)

    async def finish(self) -> None:
        async with self._lock:
            await self._close_locked(abort=False)

    async def abort(self) -> None:
        async with self._lock:
            await self._close_locked(abort=True)

    def get_status(self) -> dict:
        return {
            "active": self._stream is not None,
            "sample_rate": self._sample_rate,
            "channels": self._channels,
            "bytes_written": self._bytes_written,
            "normalized_stream_ids": list(self._normalized_stream_ids),
            "normalization_error": self._normalization_error or None,
        }

    async def _close_locked(self, *, abort: bool) -> None:
        stream = self._stream
        self._stream = None
        self._sample_rate = None
        self._channels = None
        if stream is None:
            return
        if abort:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(stream.abort)
        else:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(stream.stop)
        with contextlib.suppress(Exception):
            await asyncio.to_thread(stream.close)


__all__ = ["PCMStreamPlayer", "StreamingPlaybackError"]
