from __future__ import annotations

import asyncio

from speaker.neural_tts import FallbackSpeechEngine, NeuralSpeechEngine
from speaker.streaming_playback import StreamingPlaybackInterrupted
from speaker.neural_worker_client import AudioChunk
from speaker.tts import SpeechRequest


class FakeClient:
    def __init__(self) -> None:
        self.cancelled = False
        self.closed = False
        self.requests = []

    async def generate(self, request, *, owner="default"):
        self.requests.append(request)
        yield AudioChunk("req-1", 0, b"\x00\x00", 48000)
        yield AudioChunk("req-1", 1, b"\x01\x00", 48000)

    async def cancel(self, *, owner=None):
        self.cancelled = True
        return True

    async def close(self):
        self.closed = True

    def get_status(self):
        return {"running": True}


class FakePlayer:
    def __init__(self) -> None:
        self.started = []
        self.writes = []
        self.finished = False
        self.run_finished = False
        self.aborted = False

    async def start(self, **audio_format):
        self.started.append(audio_format)

    async def write(self, pcm):
        self.writes.append(pcm)

    async def finish(self):
        self.finished = True

    async def finish_segment(self):
        self.finished = True

    async def finish_run(self, _run_id=None):
        self.run_finished = True

    async def abort(self):
        self.aborted = True

    def get_status(self):
        return {"active": bool(self.started) and not self.finished}


def test_neural_engine_plays_worker_chunks_directly():
    async def _run() -> None:
        client = FakeClient()
        player = FakePlayer()
        engine = NeuralSpeechEngine(backend="fake", client=client, player=player)
        request = SpeechRequest("Привет.", style_id="warm", instruction="Warm")

        await engine.speak(request)

        assert client.requests == [request]
        assert player.started == [
            {
                "sample_rate": 48000,
                "channels": 1,
                "sample_width": 2,
                "run_id": "",
            }
        ]
        assert player.writes == [b"\x00\x00", b"\x01\x00"]
        assert player.finished is True
        assert player.run_finished is True
        assert engine.get_status()["chunks_played"] == 2

    asyncio.run(_run())


def test_neural_engine_interrupts_generation_and_playback():
    async def _run() -> None:
        client = FakeClient()
        player = FakePlayer()
        engine = NeuralSpeechEngine(backend="fake", client=client, player=player)

        await engine.interrupt()

        assert client.cancelled is True
        assert player.aborted is True

    asyncio.run(_run())


def test_neural_engine_drains_late_audio_after_interrupt_without_fallback_error():
    class CancellingClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_requested = asyncio.Event()
            self.drained = asyncio.Event()

        async def generate(self, request, *, owner="default"):
            self.requests.append(request)
            yield AudioChunk("req-1", 0, b"\x00\x00", 48000)
            await self.cancel_requested.wait()
            yield AudioChunk("req-1", 1, b"\x01\x00", 48000)
            self.drained.set()

        async def cancel(self, *, owner=None):
            self.cancelled = True
            self.cancel_requested.set()
            await self.drained.wait()
            return True

    class BlockingPlayer(FakePlayer):
        def __init__(self) -> None:
            super().__init__()
            self.write_started = asyncio.Event()
            self.release_write = asyncio.Event()
            self.write_done = asyncio.Event()

        async def write(self, pcm):
            self.write_started.set()
            await self.release_write.wait()
            self.writes.append(pcm)
            self.write_done.set()

        async def abort(self):
            await self.write_done.wait()
            self.aborted = True

    async def _run() -> None:
        client = CancellingClient()
        player = BlockingPlayer()
        primary = NeuralSpeechEngine(backend="fake", client=client, player=player)

        speak_task = asyncio.create_task(primary.speak(SpeechRequest("Длинный.", run_id="run-1")))
        await asyncio.wait_for(player.write_started.wait(), timeout=1.0)
        interrupt_task = asyncio.create_task(primary.interrupt(run_id="run-1"))
        await asyncio.wait_for(client.cancel_requested.wait(), timeout=1.0)
        player.release_write.set()
        await asyncio.wait_for(interrupt_task, timeout=1.0)
        await asyncio.wait_for(speak_task, timeout=1.0)

        assert player.writes == [b"\x00\x00"]
        assert player.aborted is True
        assert primary.get_status()["last_error"] is None

    asyncio.run(_run())


def test_fallback_opens_circuit_after_consecutive_errors():
    class FailingEngine:
        def __init__(self):
            self.calls = 0

        async def speak(self, _request):
            self.calls += 1
            raise RuntimeError("GPU unavailable")

    class RecordingEngine:
        def __init__(self):
            self.requests = []

        async def speak(self, request):
            self.requests.append(request)

    async def _run() -> None:
        primary = FailingEngine()
        fallback = RecordingEngine()
        engine = FallbackSpeechEngine(primary, fallback, max_consecutive_errors=2)
        request = SpeechRequest("Тест.")

        await engine.speak(request)
        await engine.speak(request)
        await engine.speak(request)

        assert primary.calls == 2
        assert fallback.requests == [request, request, request]
        status = engine.get_status()
        assert status["backend"] == "fallback"
        assert status["active_backend"] == "fallback"
        assert status["using_fallback"] is True
        assert status["circuit_open"] is True

    asyncio.run(_run())


def test_playback_crash_does_not_replay_partially_audible_segment_with_fallback():
    class CrashingPlayer(FakePlayer):
        async def write(self, pcm):
            raise StreamingPlaybackInterrupted("player exited")

    class RecordingFallback:
        def __init__(self):
            self.requests = []

        async def speak(self, request):
            self.requests.append(request)

    async def _run() -> None:
        client = FakeClient()
        player = CrashingPlayer()
        primary = NeuralSpeechEngine(backend="fake", client=client, player=player)
        fallback = RecordingFallback()
        engine = FallbackSpeechEngine(primary, fallback)

        await engine.speak(SpeechRequest("Не повторять.", run_id="run-1"))

        assert fallback.requests == []
        assert primary.get_status()["last_error"] == "player exited"
        assert player.aborted is True

    asyncio.run(_run())
