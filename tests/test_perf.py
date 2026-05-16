from __future__ import annotations

import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import perf  # noqa: E402
from core.config import cfg  # noqa: E402


def test_perf_ids_and_duration_are_stable():
    first = perf.new_id("test")
    second = perf.new_id("test")

    assert first.startswith("test-")
    assert second.startswith("test-")
    assert int(second.rsplit("-", 1)[1]) == int(first.rsplit("-", 1)[1]) + 1
    assert perf.elapsed_ms(1_000_000_000, 1_250_000_000) == 250.0


def test_perf_emit_respects_enabled_flag(caplog):
    old_enabled = cfg.performance.enabled
    old_log_level = cfg.performance.log_level
    try:
        cfg.performance.enabled = False
        cfg.performance.log_level = "info"
        caplog.set_level(logging.INFO, logger="perf")

        perf.emit("input", "disabled", segment_id="seg-1")
        assert not caplog.records

        cfg.performance.enabled = True
        perf.emit("input", "enabled", segment_id="seg-1", duration_ms=1.23456)

        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "perf.input" in message
        assert "stage=enabled" in message
        assert "segment_id=seg-1" in message
        assert "duration_ms=1.235" in message
    finally:
        cfg.performance.enabled = old_enabled
        cfg.performance.log_level = old_log_level
