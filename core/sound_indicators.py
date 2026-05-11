"""Short audio cues for SpeechGate/OpenClaw workflow events."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.config import cfg

log = logging.getLogger(__name__)

INDICATOR_REJECTED = "rejected"
INDICATOR_FORWARDED = "forwarded"
INDICATOR_LOCAL_HANDLED = "local_handled"
INDICATOR_INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class _ToneStep:
    frequency_hz: float
    duration_ms: int


@dataclass(frozen=True, slots=True)
class _TonePattern:
    steps: tuple[_ToneStep, ...]
    gap_ms: int = 24


_PATTERNS: dict[str, _TonePattern] = {
    INDICATOR_REJECTED: _TonePattern(
        steps=(_ToneStep(392.0, 55), _ToneStep(294.0, 85)),
        gap_ms=18,
    ),
    INDICATOR_FORWARDED: _TonePattern(steps=(_ToneStep(698.0, 72),), gap_ms=0),
    INDICATOR_LOCAL_HANDLED: _TonePattern(
        steps=(_ToneStep(523.25, 48), _ToneStep(659.25, 70)),
        gap_ms=20,
    ),
    INDICATOR_INTERRUPTED: _TonePattern(
        steps=(_ToneStep(784.0, 44), _ToneStep(622.0, 44), _ToneStep(523.25, 72)),
        gap_ms=16,
    ),
}

_CONFIG_FLAGS: dict[str, str] = {
    INDICATOR_REJECTED: "rejected",
    INDICATOR_FORWARDED: "forwarded",
    INDICATOR_LOCAL_HANDLED: "local_handled",
    INDICATOR_INTERRUPTED: "interrupted",
}


class SoundIndicatorPlayer:
    """Background player for short notification tones."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str | None] | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._backend_name: str | None = None
        self._backend_module: Any | None = None
        self._backend_warning_logged = False
        self._queue_warning_logged = False
        self._wave_cache: dict[tuple[str, int, float], np.ndarray] = {}

    async def start(self) -> None:
        if self._running:
            return
        if not getattr(cfg.indicators, "enabled", True):
            return
        queue_maxsize = int(getattr(cfg.indicators, "queue_maxsize", 8) or 8)
        if queue_maxsize < 1:
            queue_maxsize = 1
        self._queue = asyncio.Queue(maxsize=queue_maxsize)
        self._running = True
        self._queue_warning_logged = False
        self._task = asyncio.create_task(self._worker(), name="SoundIndicatorPlayer.worker")
        log.info(
            "sound_indicators: enabled (backend=%s, sample_rate=%s, device=%s, volume=%.2f)",
            getattr(cfg.indicators, "backend", "auto"),
            int(getattr(cfg.indicators, "sample_rate", 24000) or 24000),
            getattr(cfg.indicators, "output_device_index", None),
            float(getattr(cfg.indicators, "volume", 0.18) or 0.18),
        )

    async def close(self) -> None:
        if not self._running:
            return
        self._running = False
        queue = self._queue
        self._queue = None
        if queue is not None:
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(None)
        task = self._task
        self._task = None
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def emit(self, kind: str) -> bool:
        if kind not in _PATTERNS:
            return False
        if not getattr(cfg.indicators, "enabled", True):
            return False
        config_flag = _CONFIG_FLAGS.get(kind)
        if config_flag and not bool(getattr(cfg.indicators, config_flag, True)):
            return False
        if not self._running:
            await self.start()
        if not self._running or self._queue is None:
            return False
        try:
            self._queue.put_nowait(kind)
            return True
        except asyncio.QueueFull:
            if not self._queue_warning_logged:
                self._queue_warning_logged = True
                log.debug("sound_indicators: queue full, dropping notification tones")
            return False

    async def _worker(self) -> None:
        queue = self._queue
        if queue is None:
            return
        while self._running:
            kind = await queue.get()
            if kind is None:
                break
            try:
                await asyncio.to_thread(self._play_sync, kind)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("sound_indicators: failed to play %s tone", kind)

    def _play_sync(self, kind: str) -> None:
        backend_name, backend_module = self._get_backend()
        if backend_name == "sounddevice" and backend_module is not None:
            self._play_with_sounddevice(backend_module, kind)
            return
        if backend_name == "winsound" and backend_module is not None:
            self._play_with_winsound(backend_module, kind)
            return

    def _get_backend(self) -> tuple[str | None, Any | None]:
        if self._backend_name == "disabled":
            return None, None
        if self._backend_name is not None:
            return self._backend_name, self._backend_module

        requested = str(getattr(cfg.indicators, "backend", "auto") or "auto").strip().lower()
        if requested == "none":
            self._backend_name = "disabled"
            return None, None

        if requested in {"auto", "sounddevice"}:
            try:
                import sounddevice as sounddevice_backend  # type: ignore
            except Exception as exc:
                if requested == "sounddevice":
                    self._warn_backend_unavailable(
                        f"sounddevice backend unavailable for indicators: {exc}"
                    )
                else:
                    self._warn_backend_unavailable(
                        f"sounddevice backend unavailable for indicators, trying fallback: {exc}"
                    )
            else:
                self._backend_name = "sounddevice"
                self._backend_module = sounddevice_backend
                return self._backend_name, self._backend_module

        if requested in {"auto", "winsound"} and sys.platform.startswith("win"):
            try:
                import winsound  # type: ignore
            except Exception as exc:
                self._warn_backend_unavailable(
                    f"winsound backend unavailable for indicators: {exc}"
                )
            else:
                self._backend_name = "winsound"
                self._backend_module = winsound
                return self._backend_name, self._backend_module

        self._backend_name = "disabled"
        return None, None

    def _warn_backend_unavailable(self, message: str) -> None:
        if self._backend_warning_logged:
            return
        self._backend_warning_logged = True
        log.warning("sound_indicators: %s", message)

    def _play_with_sounddevice(self, backend: Any, kind: str) -> None:
        sample_rate = int(getattr(cfg.indicators, "sample_rate", 24000) or 24000)
        volume = float(getattr(cfg.indicators, "volume", 0.18) or 0.18)
        cache_key = (kind, sample_rate, round(volume, 4))
        wave = self._wave_cache.get(cache_key)
        if wave is None:
            wave = self._build_wave(kind, sample_rate, volume)
            self._wave_cache[cache_key] = wave
        device_index = getattr(cfg.indicators, "output_device_index", None)
        try:
            backend.play(wave, samplerate=sample_rate, device=device_index, blocking=True)
            backend.stop()
        except Exception as exc:
            self._warn_backend_unavailable(f"sounddevice playback failed for indicators: {exc}")
            if sys.platform.startswith("win"):
                self._backend_name = None
                self._backend_module = None
                backend_name, backend_module = self._get_backend()
                if backend_name == "winsound" and backend_module is not None:
                    self._play_with_winsound(backend_module, kind)

    def _play_with_winsound(self, backend: Any, kind: str) -> None:
        pattern = _PATTERNS.get(kind)
        if pattern is None:
            return
        for index, step in enumerate(pattern.steps):
            backend.Beep(max(37, int(round(step.frequency_hz))), max(20, int(step.duration_ms)))
            if index + 1 < len(pattern.steps) and pattern.gap_ms > 0:
                time.sleep(pattern.gap_ms / 1000.0)

    def _build_wave(self, kind: str, sample_rate: int, volume: float) -> np.ndarray:
        pattern = _PATTERNS[kind]
        pieces: list[np.ndarray] = []
        gap_samples = max(0, int(round(sample_rate * pattern.gap_ms / 1000.0)))
        fade_samples = max(1, int(round(sample_rate * 0.006)))
        silence = np.zeros(gap_samples, dtype=np.float32) if gap_samples > 0 else None

        for index, step in enumerate(pattern.steps):
            total_samples = max(1, int(round(sample_rate * step.duration_ms / 1000.0)))
            t = np.arange(total_samples, dtype=np.float32) / float(sample_rate)
            tone = np.sin(2.0 * math.pi * float(step.frequency_hz) * t).astype(
                np.float32, copy=False
            )
            if total_samples > 2 * fade_samples:
                fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                tone[:fade_samples] *= fade_in
                tone[-fade_samples:] *= fade_out
            tone *= max(0.0, min(1.0, volume))
            pieces.append(tone.astype(np.float32, copy=False))
            if silence is not None and index + 1 < len(pattern.steps):
                pieces.append(silence)

        if not pieces:
            return np.zeros(1, dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32, copy=False)


indicators = SoundIndicatorPlayer()


async def emit_indicator(kind: str) -> bool:
    return await indicators.emit(kind)


__all__ = [
    "INDICATOR_FORWARDED",
    "INDICATOR_INTERRUPTED",
    "INDICATOR_LOCAL_HANDLED",
    "INDICATOR_REJECTED",
    "SoundIndicatorPlayer",
    "emit_indicator",
    "indicators",
]
