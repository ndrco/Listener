from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["GateDecision", "SpeechDirectionGate", "SpeechGateMode", "DirectedIntentClassifier"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(import_module(".speech_gate", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
