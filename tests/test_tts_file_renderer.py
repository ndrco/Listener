from __future__ import annotations

import asyncio
import sys
import wave
from pathlib import Path

import pytest

from speaker.config import TTSFileRenderConfig
from speaker.file_renderer import TTSFileRenderUnavailable, TTSFileRenderer
from speaker.neural_tts import NeuralSpeechEngine
from speaker.neural_worker_client import AudioChunk, NeuralWorkerClient
from speaker.tts import SpeechRequest


ROOT = Path(__file__).resolve().parents[1]
FAKE_WORKER = ROOT / "speaker" / "workers" / "fake_worker.py"


class FakeClient:
    def __init__(self) -> None:
        self.requests = []
        self.owners = []
        self.cancel_owners = []

    async def generate(self, request, *, owner="default", cancellation_event=None):
        self.requests.append(request)
        self.owners.append(owner)
        yield AudioChunk("req", 0, b"\x00\x00" * 120, 24000)
        yield AudioChunk("req", 1, b"\x01\x00" * 120, 24000)

    async def cancel(self, *, owner=None):
        self.cancel_owners.append(owner)
        return False

    def get_status(self):
        return {"running": True}


def test_renderer_writes_atomic_wav_and_maps_leading_emoji(tmp_path: Path):
    async def _run() -> None:
        client = FakeClient()
        engine = NeuralSpeechEngine(backend="cosyvoice3", client=client)  # type: ignore[arg-type]
        renderer = TTSFileRenderer(
            speech=engine,
            config=TTSFileRenderConfig(output_dir=str(tmp_path), segment_chars=80),
        )
        try:
            await renderer.start()
            submitted = await renderer.submit(
                "😔 Первая фраза. Вторая фраза!",
                filename="../../story.wav",
            )
            job = await renderer.wait(submitted["id"], timeout_s=1.0)

            assert job["state"] == "completed"
            assert job["backend"] == "cosyvoice3"
            assert job["style_id"] == "sad"
            assert job["emoji"] == "😔"
            assert job["filename"].startswith("story-")
            output = Path(job["output_path"])
            assert output.parent == tmp_path.resolve()
            assert output.is_file()
            assert not list(tmp_path.glob("*.part"))
            with wave.open(str(output), "rb") as wav_file:
                assert wav_file.getframerate() == 24000
                assert wav_file.getnchannels() == 1
                assert wav_file.getsampwidth() == 2
                assert wav_file.getnframes() == 480

            assert len(client.requests) == 2
            assert all(request.style_id == "sad" for request in client.requests)
            assert all("sad" in request.instruction for request in client.requests)
            assert client.owners == [f"file:{submitted['id']}"] * 2
        finally:
            await renderer.close()

    asyncio.run(_run())


def test_renderer_cancels_only_its_active_owner(tmp_path: Path):
    class BlockingClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def generate(self, request, *, owner="default", cancellation_event=None):
            self.requests.append(request)
            self.owners.append(owner)
            self.started.set()
            yield AudioChunk("req", 0, b"\x00\x00" * 120, 24000)
            await self.release.wait()

        async def cancel(self, *, owner=None):
            self.cancel_owners.append(owner)
            self.release.set()
            return True

    async def _run() -> None:
        client = BlockingClient()
        engine = NeuralSpeechEngine(backend="voxcpm2", client=client)  # type: ignore[arg-type]
        renderer = TTSFileRenderer(
            speech=engine,
            config=TTSFileRenderConfig(output_dir=str(tmp_path)),
        )
        try:
            submitted = await renderer.submit("Длинный файловый тест.")
            await asyncio.wait_for(client.started.wait(), timeout=1.0)
            await renderer.cancel(submitted["id"])
            job = await renderer.wait(submitted["id"], timeout_s=1.0)

            assert job["state"] == "cancelled"
            assert job["output_path"] is None
            assert client.cancel_owners == [f"file:{submitted['id']}"]
            assert not list(tmp_path.iterdir())
        finally:
            await renderer.close()

    asyncio.run(_run())


def test_renderer_rejects_non_neural_engine(tmp_path: Path):
    class OtherEngine:
        pass

    async def _run() -> None:
        renderer = TTSFileRenderer(
            speech=OtherEngine(),  # type: ignore[arg-type]
            config=TTSFileRenderConfig(output_dir=str(tmp_path)),
        )
        with pytest.raises(TTSFileRenderUnavailable, match="VoxCPM2 or CosyVoice3"):
            await renderer.submit("Тест.")

    asyncio.run(_run())


def test_renderer_reuses_the_same_persistent_worker_process(tmp_path: Path):
    async def _run() -> None:
        client = NeuralWorkerClient(
            [sys.executable, str(FAKE_WORKER), "--chunks", "2", "--chunk-delay-ms", "0"]
        )
        engine = NeuralSpeechEngine(backend="cosyvoice3", client=client)
        renderer = TTSFileRenderer(
            speech=engine,
            config=TTSFileRenderConfig(output_dir=str(tmp_path)),
        )
        try:
            await client.start()
            worker_pid = client.get_status()["pid"]
            submitted = await renderer.submit("Файловая реплика.")
            completed = await renderer.wait(submitted["id"], timeout_s=1.0)
            assert completed["state"] == "completed"
            assert client.get_status()["pid"] == worker_pid

            playback_chunks = [
                chunk
                async for chunk in client.generate(
                    SpeechRequest("Обычная реплика."),
                    owner="playback",
                )
            ]
            assert len(playback_chunks) == 2
            assert client.get_status()["pid"] == worker_pid
        finally:
            await renderer.close()
            await client.close()

    asyncio.run(_run())
