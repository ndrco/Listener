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


def test_sound_indicator_player_ducks_around_playback(monkeypatch):
    async def _runner() -> None:
        calls: list[tuple[str, object]] = []

        class FakeDucker:
            def __init__(self, config, *, exclude_speaker=True):
                calls.append(("init", exclude_speaker))

            async def duck(self):
                calls.append(("duck", None))

            async def restore(self):
                calls.append(("restore", None))

        player = SoundIndicatorPlayer()
        old_enabled = cfg.indicators.enabled
        old_forwarded = cfg.indicators.forwarded
        old_ducking_enabled = cfg.indicators.ducking.enabled
        try:
            cfg.indicators.enabled = True
            cfg.indicators.forwarded = True
            cfg.indicators.ducking.enabled = True
            monkeypatch.setattr("core.sound_indicators.PulseAudioDucker", FakeDucker)
            monkeypatch.setattr(
                player,
                "_play_sync",
                lambda kind: calls.append(("play", kind)),
            )

            assert await player.emit(INDICATOR_FORWARDED)
            for _ in range(100):
                if ("restore", None) in calls:
                    break
                await asyncio.sleep(0.01)
        finally:
            await player.close()
            cfg.indicators.enabled = old_enabled
            cfg.indicators.forwarded = old_forwarded
            cfg.indicators.ducking.enabled = old_ducking_enabled

        assert calls[:4] == [
            ("init", False),
            ("duck", None),
            ("play", INDICATOR_FORWARDED),
            ("restore", None),
        ]

    asyncio.run(_runner())
