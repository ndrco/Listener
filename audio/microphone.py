# audio/microphone.py
"""Asynchronous PCM stream reader for the selected microphone."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Optional


def _input_stream_accepts_kwarg(stream_cls: Any, kwarg: str) -> bool:
    """Return ``True`` if ``stream_cls`` constructor accepts ``kwarg``."""

    try:
        signature = inspect.signature(stream_cls)
    except (TypeError, ValueError):  # pragma: no cover - native implementations
        return True

    has_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if has_var_kwargs:
        return True

    return any(
        parameter.name == kwarg
        and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        for parameter in signature.parameters.values()
    )

from core.bus import EventBus, bus as default_bus
from core.config import cfg

log = logging.getLogger(__name__)


class MicrophoneStream:
    """Async context manager for reading PCM frames from a microphone.

    Uses ``sounddevice`` (or a compatible injected backend) via callback API.
    Incoming PCM blocks are queued asynchronously and yielded during iteration.
    Each frame is also published to EventBus as an ``audio/raw_frame`` event.
    """

    def __init__(
        self,
        *,
        sample_rate: Optional[int] = None,
        chunk_size: Optional[int] = None,
        channels: Optional[int] = None,
        device_index: Optional[int] = None,
        debug: Optional[bool] = None,
        queue_maxsize: int = 0,
        loop: asyncio.AbstractEventLoop | None = None,
        bus: EventBus | None = None,
        sounddevice_module: Any | None = None,
    ) -> None:
        # Take defaults from cfg.audio.input, but allow constructor overrides.
        audio_cfg = cfg.audio.input
        self.sample_rate = int(sample_rate or audio_cfg.input_sample_rate)
        self.chunk_size = int(chunk_size or audio_cfg.chunk_size)
        self.channels = int(channels or audio_cfg.channels)
        self.device_index = (
            int(device_index)
            if device_index is not None
            else (None if audio_cfg.device_index is None else int(audio_cfg.device_index))
        )

        # Queue for PCM data; callback pushes chunks here.
        self._queue: "asyncio.Queue[bytes | None]" = asyncio.Queue(maxsize=queue_maxsize)
        self._loop = loop
        self._bus = bus or default_bus
        self._sounddevice_module = sounddevice_module
        self._debug = cfg.debug if debug is None else bool(debug)

        # sounddevice stream object and state flags.
        self._stream: Any | None = None  # stream object
        self._running: bool = False      # capture state
        self._queue_closed: bool = False # queue state

    async def __aenter__(self) -> "MicrophoneStream":
        """Initialize and start the input audio stream."""
        self._loop = self._loop or asyncio.get_running_loop()
        module = self._load_backend()

        if self._debug:
            log.debug(
                "audio.microphone: opening input stream (rate=%d, chunk=%d, channels=%d, device=%s)",
                self.sample_rate,
                self.chunk_size,
                self.channels,
                "default" if self.device_index is None else self.device_index,
            )

        stream_kwargs = dict(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            channels=self.channels,
            dtype="int16",
            callback=self._on_chunk,
            start=False,
        )
        if not _input_stream_accepts_kwarg(module.InputStream, "start"):
            stream_kwargs.pop("start")
        if self.device_index is not None:
            stream_kwargs["device"] = int(self.device_index)

        # Create input stream.
        self._stream = module.InputStream(**stream_kwargs)

        # Start stream.
        self._running = True
        start_method = getattr(self._stream, "start", None)
        if callable(start_method):
            try:
                start_method()
                if self._debug:
                    log.debug("audio.microphone: input stream started successfully")
            except Exception:  # pragma: no cover - backend compatibility
                log.debug("audio.microphone: stream.start() failed; assuming auto-started", exc_info=True)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Gracefully stop stream on ``async with`` exit."""
        await self.stop()

    def __aiter__(self) -> "MicrophoneStream":
        """Enable ``async for chunk in mic`` iteration."""
        return self

    async def __anext__(self) -> bytes:
        """Return the next PCM block from the queue."""
        if self._queue_closed and self._queue.empty():
            raise StopAsyncIteration

        chunk = await self._queue.get()
        if chunk is None:  # Queue close signal.
            self._queue_closed = True
            raise StopAsyncIteration

        # Publish every emitted frame to EventBus.
        await self._bus.publish(
            cfg.events.audio.raw_frame,
            data=chunk,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        return chunk

    async def stop(self) -> None:
        """Stop stream and release resources."""
        if not self._running and self._stream is None:
            return

        self._running = False

        if self._debug:
            log.debug("audio.microphone: stopping input stream")

        # Stop and close sounddevice stream.
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception:  # pragma: no cover
                log.exception("audio.microphone: stop() failed")
            try:
                stream.close()
            except Exception:  # pragma: no cover
                log.exception("audio.microphone: close() failed")

        # Signal completion to queue consumers.
        await self._ensure_queue_closed()

    # === internal helpers ==================================================

    def _load_backend(self) -> Any:
        """Load sounddevice backend (or injected test module)."""
        if self._sounddevice_module is not None:
            if self._debug:
                log.debug(
                    "audio.microphone: using injected sounddevice backend %r",
                    type(self._sounddevice_module),
                )
            return self._sounddevice_module
        try:
            import sounddevice  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("sounddevice is required for MicrophoneStream") from exc
        if self._debug:
            backend_name = getattr(sounddevice, "__name__", "sounddevice")
            log.debug("audio.microphone: loaded backend %s", backend_name)
        return sounddevice

    def _on_chunk(self, indata, frames, _time, _status) -> None:
        """sounddevice callback fired for each incoming PCM block."""
        if not self._running:
            return
        if indata is None:
            return
        try:
            data = indata.tobytes()
        except AttributeError:  # pragma: no cover - defensive branch
            log.warning("audio.microphone: received unexpected chunk type %r", type(indata))
            return
        if data:
            self._schedule_chunk(data)

    def _schedule_chunk(self, data: bytes) -> None:
        """Put a PCM block into queue from callback thread."""
        if not self._loop:
            return

        def _safe_put(chunk: bytes) -> None:
            try:
                self._queue.put_nowait(chunk)
            except asyncio.QueueFull:
                if self._debug:
                    log.debug("audio.microphone: dropping input frame (queue full)")

        # Schedule queue write in the event loop from callback thread.
        (self._loop or asyncio.get_running_loop()).call_soon_threadsafe(_safe_put, data)

    async def _ensure_queue_closed(self) -> None:
        """Mark queue as closed and enqueue ``None`` to stop iteration."""
        if self._queue_closed:
            return
        self._queue_closed = True
        await self._queue.put(None)

        if self._debug:
            log.debug("audio.microphone: queue closed")


__all__ = ["MicrophoneStream"]
