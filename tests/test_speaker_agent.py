from __future__ import annotations

import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.speaker_agent import SpeechPlaybackController  # noqa: E402
from speaker.events import SpeechSegment  # noqa: E402


class BlockingSpeech:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    async def speak(self, text: str) -> None:
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class RecordingEmojiDisplay:
    def __init__(self) -> None:
        self.shown = []
        self.cleared = []

    async def show_tokens(self, tokens, *, run_id, segment_id):
        self.shown.append(([token.symbol for token in tokens], run_id, segment_id))
        return True

    async def clear(self, *, reason):
        self.cleared.append(reason)
        return True

    def get_status(self):
        return {"enabled": True}


def test_speech_playback_controller_interrupts_current_speech():
    async def _runner() -> None:
        speech = BlockingSpeech()
        controller = SpeechPlaybackController(speech=speech, queue_size=4, enabled=True)
        await controller.start()
        try:
            assert controller.enqueue(
                SpeechSegment(
                    identifier="seg-1",
                    text="Hello.",
                    run_id="run-1",
                    final=False,
                )
            )
            await asyncio.wait_for(speech.started.wait(), timeout=1.0)
            dropped = await controller.interrupt(reason="test")
            await asyncio.sleep(0)

            assert dropped == 1
            assert speech.cancelled is True
            assert controller.get_status()["current"] is None
            assert controller.get_status()["last_interrupt_reason"] == "test"
        finally:
            await controller.close()

    asyncio.run(_runner())


def test_speech_playback_controller_strips_and_forwards_emoji():
    async def _runner() -> None:
        class RecordingSpeech:
            def __init__(self) -> None:
                self.spoken = []
                self.done = asyncio.Event()

            async def speak(self, text: str) -> None:
                self.spoken.append(text)
                self.done.set()

        speech = RecordingSpeech()
        display = RecordingEmojiDisplay()
        controller = SpeechPlaybackController(
            speech=speech,
            queue_size=4,
            enabled=True,
            emoji_display=display,
        )
        await controller.start()
        try:
            assert controller.enqueue(SpeechSegment("seg-1", "Привет 🙂!", "run-1"))
            await asyncio.wait_for(speech.done.wait(), timeout=1.0)

            assert speech.spoken == ["Привет!"]
            assert display.shown == [(["🙂"], "run-1", "seg-1")]
        finally:
            await controller.close()

    asyncio.run(_runner())


def test_speech_playback_controller_skips_emoji_only_segment():
    async def _runner() -> None:
        class RecordingSpeech:
            def __init__(self) -> None:
                self.spoken = []

            async def speak(self, text: str) -> None:
                self.spoken.append(text)

        speech = RecordingSpeech()
        display = RecordingEmojiDisplay()
        controller = SpeechPlaybackController(
            speech=speech,
            queue_size=4,
            enabled=True,
            emoji_display=display,
        )
        await controller.start()
        try:
            assert controller.enqueue(SpeechSegment("seg-emoji", "✨", "run-1"))
            for _ in range(10):
                if display.shown:
                    break
                await asyncio.sleep(0.01)

            assert speech.spoken == []
            assert display.shown == [(["✨"], "run-1", "seg-emoji")]
        finally:
            await controller.close()

    asyncio.run(_runner())


def test_speech_playback_controller_disable_drops_queued_segments():
    async def _runner() -> None:
        class RecordingSpeech:
            async def speak(self, text: str) -> None:
                return None

        controller = SpeechPlaybackController(speech=RecordingSpeech(), queue_size=4, enabled=True)
        try:
            controller.enqueue(SpeechSegment("seg-1", "One.", "run-1"))
            controller.enqueue(SpeechSegment("seg-2", "Two.", "run-2"))
            status = await controller.set_enabled(False, reason="off")

            assert status["enabled"] is False
            assert status["dropped"] == 2
            assert controller.enqueue(SpeechSegment("seg-3", "Three.", "run-3")) is False
        finally:
            await controller.close()

    asyncio.run(_runner())
