from __future__ import annotations

import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.speaker_agent import SpeakerAgent, SpeechPlaybackController  # noqa: E402
from core.config import cfg  # noqa: E402
from core.runtime_state import RuntimeStateStore  # noqa: E402
from speaker.config import SpeakerConfig  # noqa: E402
from speaker.events import ChatSpeechRouter, SpeechSegment  # noqa: E402
from speaker.messages import MessageDeduper  # noqa: E402


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


class HistoryGateway:
    def __init__(self, history_text: str) -> None:
        self.history_text = history_text
        self.history_calls = 0

    async def request(self, method, params, timeout_s=10.0):
        self.history_calls += 1
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.history_text}],
                    "__openclaw": {"id": "reply-1"},
                }
            ]
        }


def chat_event(state, text=None, *, run_id="run-1", session_key="main"):
    payload = {
        "runId": run_id,
        "sessionKey": session_key,
        "state": state,
    }
    if text is not None:
        payload["message"] = {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        }
    return {"type": "event", "event": "chat", "payload": payload}


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


def test_speech_playback_controller_restores_ducking_when_interrupted_before_tts(
    monkeypatch,
):
    async def _runner() -> None:
        class RecordingSpeech:
            def __init__(self) -> None:
                self.spoken = []

            async def speak(self, text: str) -> None:
                self.spoken.append(text)

        class SlowDucker:
            def __init__(self, _config) -> None:
                self.duck_started = asyncio.Event()
                self.duck_continue = asyncio.Event()
                self.restored = asyncio.Event()

            async def duck(self) -> None:
                self.duck_started.set()
                await self.duck_continue.wait()

            async def restore(self) -> None:
                self.restored.set()

        speech = RecordingSpeech()
        ducker = SlowDucker(object())
        monkeypatch.setattr(
            "agents.speaker_agent.PulseAudioDucker",
            lambda config: ducker,
        )
        controller = SpeechPlaybackController(
            speech=speech,
            queue_size=4,
            enabled=True,
            ducking_config=object(),
        )
        await controller.start()
        try:
            assert controller.enqueue(SpeechSegment("seg-1", "Hello.", "run-1"))
            await asyncio.wait_for(ducker.duck_started.wait(), timeout=1.0)

            dropped = await controller.interrupt(reason="barge_in")
            ducker.duck_continue.set()
            await asyncio.wait_for(ducker.restored.wait(), timeout=1.0)

            assert dropped == 1
            assert speech.spoken == []
            assert controller.get_status()["current"] is None
        finally:
            await controller.close()

    asyncio.run(_runner())


def test_speech_playback_controller_drops_late_segments_for_interrupted_run():
    async def _runner() -> None:
        class RecordingSpeech:
            async def speak(self, text: str) -> None:
                return None

        controller = SpeechPlaybackController(
            speech=RecordingSpeech(),
            queue_size=4,
            enabled=True,
        )
        try:
            dropped = await controller.interrupt(reason="openclaw_aborted", run_id="run-1")

            assert dropped == 0
            assert controller.enqueue(SpeechSegment("seg-1", "Old.", "run-1")) is False
            assert controller.enqueue(SpeechSegment("seg-2", "New.", "run-2")) is True
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


def test_speaker_agent_restores_persisted_enabled_state(tmp_path):
    async def _runner() -> None:
        class RecordingSpeech:
            async def speak(self, text: str) -> None:
                return None

        old_state_path = cfg.control.state_path
        cfg.control.state_path = str(tmp_path / "runtime_state.json")
        try:
            first_config = SpeakerConfig()
            first = SpeakerAgent(config=first_config, speech=RecordingSpeech())
            await first.set_enabled(False, source="test", reason="quiet")
            await first.close()

            second_config = SpeakerConfig()
            second = SpeakerAgent(config=second_config, speech=RecordingSpeech())
            try:
                status = second.get_status()
                assert status["enabled"] is False
                assert status["source"] == "test"
                assert status["reason"] == "quiet"
            finally:
                await second.close()
        finally:
            cfg.control.state_path = old_state_path

    asyncio.run(_runner())


def test_speaker_agent_streaming_stale_final_history_queues_missing_tail():
    async def _runner() -> None:
        class RecordingSpeech:
            async def speak(self, text: str) -> None:
                return None

        config = SpeakerConfig()
        config.enabled = True
        config.speaker.mode = "streaming"
        config.speaker.speak_existing_on_start = True
        agent = SpeakerAgent(
            config=config,
            speech=RecordingSpeech(),
            state_store=RuntimeStateStore(None),
        )
        gateway = HistoryGateway("Первое предложение. Второе **предложение**.")
        router = ChatSpeechRouter(config.gateway, config.speaker.streaming)
        deduper = MessageDeduper()

        delta = chat_event("delta", "Первое предложение.")
        await agent._handle_streaming_event(delta["payload"], delta, gateway, deduper, router)
        final = chat_event("final", "Первое предложение.")
        await agent._handle_streaming_event(final["payload"], final, gateway, deduper, router)

        queued = []
        while not agent._playback._queue.empty():
            item = agent._playback._queue.get_nowait()
            queued.append(item.text)
            agent._playback._queue.task_done()

        assert gateway.history_calls == 1
        assert queued == ["Первое предложение.", "Второе предложение."]

    asyncio.run(_runner())
