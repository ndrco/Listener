"""Audio package exports for the voice-input runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "MicrophoneStream",
    "ProcessedAudioFrame",
    "AudioProcessor",
    "WindowsAudioProcessor",
    "BufferedSpeechWriter",
    "SpeechSegment",
    "WhisperEngine",
]


def __getattr__(name: str) -> Any:
    if name == "MicrophoneStream":
        return getattr(import_module(".microphone", __name__), name)
    if name in {"AudioProcessor", "ProcessedAudioFrame", "WindowsAudioProcessor"}:
        return getattr(import_module(".processing", __name__), name)
    if name == "WhisperEngine":
        return getattr(import_module(".stt", __name__), name)
    if name in {"BufferedSpeechWriter", "SpeechSegment"}:
        return getattr(import_module(".writer", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
