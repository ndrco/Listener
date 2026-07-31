from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from speaker.neural_worker_client import NeuralWorkerClient, NeuralWorkerError
from speaker.tts import SpeechRequest


ROOT = Path(__file__).resolve().parents[1]
FAKE_WORKER = ROOT / "speaker" / "workers" / "fake_worker.py"


def _command(*args: str) -> list[str]:
    return [sys.executable, str(FAKE_WORKER), *args]


def test_fake_worker_streams_pcm_and_reports_health():
    async def _run() -> None:
        normalized: list[str] = []

        def normalize(text: str) -> str:
            normalized.append(text)
            return text.replace("Тест", "Нормализованный тест")

        client = NeuralWorkerClient(
            _command("--chunks", "3", "--chunk-delay-ms", "0"),
            text_normalizer=normalize,
        )
        try:
            ready = await client.start()
            health = await client.health()
            chunks = [
                chunk
                async for chunk in client.generate(
                    SpeechRequest("Тест.", run_id="run-1", segment_id="seg-1")
                )
            ]

            assert ready["backend"] == "fake"
            assert health["ok"] is True
            assert len(chunks) == 3
            assert [chunk.sequence for chunk in chunks] == [0, 1, 2]
            assert {chunk.sample_rate for chunk in chunks} == {24000}
            assert {len(chunk.pcm) for chunk in chunks} == {960}
            assert client.get_status()["running"] is True
            assert client.get_status()["text_normalization"] is True
            assert normalized == ["Тест."]
        finally:
            await client.close()

    asyncio.run(_run())


def test_cancel_keeps_persistent_worker_reusable():
    async def _run() -> None:
        client = NeuralWorkerClient(
            _command("--chunks", "100", "--chunk-delay-ms", "10"),
            cancel_timeout_s=1.0,
        )
        first_chunk = asyncio.Event()
        received = []

        async def _consume() -> None:
            async for chunk in client.generate(SpeechRequest("Длинный тест.", run_id="run-1")):
                received.append(chunk)
                first_chunk.set()

        try:
            task = asyncio.create_task(_consume())
            await asyncio.wait_for(first_chunk.wait(), timeout=1.0)
            assert await client.cancel() is True
            await asyncio.wait_for(task, timeout=1.0)

            assert 1 <= len(received) < 100
            assert client.get_status()["running"] is True

            second = [
                chunk
                async for chunk in client.generate(SpeechRequest("Ещё тест.", run_id="run-2"))
            ]
            assert len(second) == 100
        finally:
            await client.close()

    asyncio.run(_run())


def test_cancel_can_be_scoped_to_generation_owner():
    async def _run() -> None:
        client = NeuralWorkerClient(
            _command("--chunks", "100", "--chunk-delay-ms", "10"),
            cancel_timeout_s=1.0,
        )
        first_chunk = asyncio.Event()

        async def _consume() -> None:
            async for _chunk in client.generate(
                SpeechRequest("Файловый тест."),
                owner="file:job-1",
            ):
                first_chunk.set()

        try:
            task = asyncio.create_task(_consume())
            await asyncio.wait_for(first_chunk.wait(), timeout=1.0)
            assert client.get_status()["active_owner"] == "file:job-1"
            assert await client.cancel(owner="playback") is False
            assert not task.done()
            assert await client.cancel(owner="file:job-1") is True
            await asyncio.wait_for(task, timeout=1.0)
            assert client.get_status()["running"] is True
        finally:
            await client.close()

    asyncio.run(_run())


def test_cancelled_waiter_does_not_start_after_other_owner_finishes():
    async def _run() -> None:
        client = NeuralWorkerClient(
            _command("--chunks", "5", "--chunk-delay-ms", "10"),
            cancel_timeout_s=1.0,
        )
        playback_started = asyncio.Event()
        file_cancelled = asyncio.Event()
        file_chunks = []

        async def _playback() -> None:
            async for _chunk in client.generate(
                SpeechRequest("Обычная реплика."),
                owner="playback",
            ):
                playback_started.set()

        async def _file() -> None:
            async for chunk in client.generate(
                SpeechRequest("Отменённый файл."),
                owner="file:job-2",
                cancellation_event=file_cancelled,
            ):
                file_chunks.append(chunk)

        try:
            playback_task = asyncio.create_task(_playback())
            await asyncio.wait_for(playback_started.wait(), timeout=1.0)
            file_task = asyncio.create_task(_file())
            await asyncio.sleep(0)
            file_cancelled.set()
            assert await client.cancel(owner="file:job-2") is False
            await asyncio.wait_for(playback_task, timeout=1.0)
            await asyncio.wait_for(file_task, timeout=1.0)
            assert file_chunks == []

            reusable = [
                chunk
                async for chunk in client.generate(
                    SpeechRequest("Следующая реплика."),
                    owner="playback",
                )
            ]
            assert len(reusable) == 5
        finally:
            await client.close()

    asyncio.run(_run())


def test_late_cancel_does_not_poison_next_persistent_request():
    async def _run() -> None:
        client = NeuralWorkerClient(
            _command("--chunks", "1", "--chunk-delay-ms", "0"),
            cancel_timeout_s=1.0,
        )
        try:
            stream = client.generate(SpeechRequest("Первый.", run_id="run-1"))
            first = await anext(stream)
            assert first.sequence == 0

            # The worker has normally emitted done by now, while the client is
            # deliberately paused before consuming it.
            await asyncio.sleep(0.02)
            cancel_task = asyncio.create_task(client.cancel())
            await asyncio.sleep(0)
            remaining = [chunk async for chunk in stream]
            assert remaining == []
            assert await asyncio.wait_for(cancel_task, timeout=1.0) is True

            second = [
                chunk
                async for chunk in client.generate(SpeechRequest("Второй.", run_id="run-2"))
            ]
            assert len(second) == 1
            assert second[0].sequence == 0
        finally:
            await client.close()

    asyncio.run(_run())


def test_generation_error_is_propagated():
    async def _run() -> None:
        client = NeuralWorkerClient(_command("--fail-generate"))
        try:
            with pytest.raises(NeuralWorkerError, match="requested fake failure"):
                _ = [chunk async for chunk in client.generate(SpeechRequest("Ошибка."))]
        finally:
            await client.close()

    asyncio.run(_run())


def test_startup_timeout_terminates_worker():
    async def _run() -> None:
        client = NeuralWorkerClient(
            _command("--startup-delay-ms", "200"),
            startup_timeout_s=0.05,
        )
        with pytest.raises(NeuralWorkerError, match="failed to start"):
            await client.start()
        assert client.get_status()["running"] is False

    asyncio.run(_run())
