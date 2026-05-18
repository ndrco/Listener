"""Audio processing components (VAD and related metrics)."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .agc import AGCSettings, AutomaticGainControl
from .highpass import DCBlockingHighPass
from .noise_suppression import NoiseSuppressor
from .resampler import StreamingResampler

__all__ = [
    "ProcessedAudioFrame",
    "AudioProcessor",
    "WindowsAudioProcessor",
    "SileroVADHelper",
    "NoiseSuppressor",
    "AutomaticGainControl",
    "AGCSettings",
    "DCBlockingHighPass",
    "StreamingResampler",
]


def __getattr__(name: str) -> Any:
    if name == "SileroVADHelper":
        return getattr(import_module(".silero_vad", __name__), name)
    if name in {"AudioProcessor", "ProcessedAudioFrame", "WindowsAudioProcessor"}:
        return getattr(import_module(".windows_processing", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
