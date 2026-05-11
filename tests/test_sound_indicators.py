from __future__ import annotations

import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import cfg  # noqa: E402
from core.sound_indicators import (  # noqa: E402
    INDICATOR_FORWARDED,
    INDICATOR_REJECTED,
    SoundIndicatorPlayer,
)


def test_sound_indicator_player_respects_per_event_toggles():
    async def _runner() -> None:
        player = SoundIndicatorPlayer()
        old_enabled = cfg.indicators.enabled
        old_rejected = cfg.indicators.rejected
        old_forwarded = cfg.indicators.forwarded
        try:
            cfg.indicators.enabled = True
            cfg.indicators.rejected = False
            cfg.indicators.forwarded = True

            rejected_result = await player.emit(INDICATOR_REJECTED)
            forwarded_result = await player.emit(INDICATOR_FORWARDED)

            assert rejected_result is False
            assert forwarded_result is True
        finally:
            await player.close()
            cfg.indicators.enabled = old_enabled
            cfg.indicators.rejected = old_rejected
            cfg.indicators.forwarded = old_forwarded

    asyncio.run(_runner())
