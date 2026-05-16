"""Lightweight performance logging helpers."""

from __future__ import annotations

import itertools
import logging
import time
from typing import Any

from core.config import cfg

log = logging.getLogger("perf")

_COUNTERS: dict[str, itertools.count] = {}


def enabled() -> bool:
    return bool(getattr(getattr(cfg, "performance", object()), "enabled", False))


def now_ns() -> int:
    return time.monotonic_ns()


def elapsed_ms(start_ns: int | None, end_ns: int | None = None) -> float | None:
    if start_ns is None:
        return None
    end = now_ns() if end_ns is None else end_ns
    return max(0.0, (end - start_ns) / 1_000_000.0)


def new_id(prefix: str) -> str:
    counter = _COUNTERS.setdefault(prefix, itertools.count(1))
    return f"{prefix}-{next(counter)}"


def text_preview(text: object) -> str | None:
    perf_cfg = getattr(cfg, "performance", None)
    if not bool(getattr(perf_cfg, "include_text_preview", True)):
        return None
    try:
        value = " ".join(str(text or "").split())
    except Exception:
        return None
    if not value:
        return None
    limit = int(getattr(perf_cfg, "text_preview_chars", 80) or 80)
    if limit <= 0:
        return None
    if len(value) > limit:
        return f"{value[: max(0, limit - 1)]}..."
    return value


def emit(namespace: str, stage: str, **fields: Any) -> None:
    if not enabled():
        return

    level_name = str(
        getattr(getattr(cfg, "performance", object()), "log_level", "info") or "info"
    ).upper()
    level = getattr(logging, level_name, logging.INFO)
    parts = [f"perf.{namespace}", f"stage={stage}"]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            value = round(value, 3)
        text = str(value).replace("\n", " ")
        parts.append(f"{key}={text}")
    log.log(level, " ".join(parts))


__all__ = ["elapsed_ms", "emit", "enabled", "new_id", "now_ns", "text_preview"]
