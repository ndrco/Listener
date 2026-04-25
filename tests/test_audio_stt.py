import asyncio
import logging
from pathlib import Path
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from audio.writer import SpeechSegment  # noqa: E402
import audio.stt.streaming as streaming  # noqa: E402
from audio.stt.streaming import WhisperStreamingTranscriber  # noqa: E402
from core.config import WhisperSttCfg  # noqa: E402


def test_whisper_streaming_transcriber_pipeline(monkeypatch):
    """Full pipeline integration test for :class:`WhisperStreamingTranscriber`."""

    async def runner() -> None:
        partial_event = asyncio.Event()
        final_event = asyncio.Event()

        class DummyBus:
            def __init__(self, partial_topic: str, final_topic: str) -> None:
                self.events: list[tuple[str, dict]] = []
                self.partial_topic = partial_topic
                self.final_topic = final_topic

            async def publish(self, topic: str, **payload):
                self.events.append((topic, payload))
                if topic == self.partial_topic:
                    partial_event.set()
                if topic == self.final_topic:
                    final_event.set()

        queue: asyncio.Queue[SpeechSegment] = asyncio.Queue()

        class DummyWriter:
            def __init__(self) -> None:
                self.queue = queue

        writer = DummyWriter()

        engine_calls: list[tuple[bytes, int]] = []

        class FakeWhisperEngine:
            def __init__(self, config, debug: bool = False) -> None:  # pragma: no cover - simple stub
                self.config = config
                self.debug = debug

            def transcribe(self, data: bytes, sample_rate: int):
                engine_calls.append((data, sample_rate))
                return ["hello", "hello world"]

        monkeypatch.setattr(streaming, "WhisperEngine", FakeWhisperEngine)

        stt_cfg = WhisperSttCfg(enabled=True, min_confidence=0.1, stability_timeout_s=0.0)

        bus = DummyBus(stt_cfg.partial_topic, stt_cfg.final_topic)

        transcriber = WhisperStreamingTranscriber(
            writer,
            stt_config=stt_cfg,
            bus=bus,
            llm_queue_maxsize=2,
        )

        await transcriber.start()

        segment = SpeechSegment(
            data=b"\x01\x02" * 16,
            sample_rate=16_000,
            channels=1,
            start_timestamp=0.0,
            end_timestamp=0.5,
            duration_ms=500.0,
            frames=50,
            voice_frames=45,
            metadata={"confidence": 0.95},
        )

        await queue.put(segment)

        await asyncio.wait_for(final_event.wait(), timeout=1.0)

        text_for_llm = await asyncio.wait_for(transcriber.llm_queue.get(), timeout=1.0)

        await transcriber.stop()

        assert engine_calls == [(segment.data, segment.sample_rate)]

        partial_payloads = [
            payload
            for topic, payload in bus.events
            if topic == stt_cfg.partial_topic
        ]
        final_payloads = [
            payload for topic, payload in bus.events if topic == stt_cfg.final_topic
        ]

        assert len(partial_payloads) == 2
        assert [payload["text"] for payload in partial_payloads] == ["Hello", "Hello world"]
        assert final_payloads and final_payloads[-1]["text"] == "Hello world."
        assert final_payloads[-1]["raw_text"] == "hello world"
        assert final_payloads[-1]["confidence"] == pytest.approx(0.95)
        assert final_payloads[-1]["is_final"] is True

        assert text_for_llm == "Hello world."

    asyncio.run(runner())


def test_whisper_streaming_transcriber_start_does_not_block_event_loop(monkeypatch):
    async def runner() -> None:
        queue: asyncio.Queue[SpeechSegment] = asyncio.Queue()

        class DummyWriter:
            def __init__(self) -> None:
                self.queue = queue

        class FakeWhisperEngine:
            def __init__(self, config, debug: bool = False) -> None:
                del config, debug
                time.sleep(0.25)

            def transcribe(self, data: bytes, sample_rate: int):
                del data, sample_rate
                return []

        monkeypatch.setattr(streaming, "WhisperEngine", FakeWhisperEngine)

        stt_cfg = WhisperSttCfg(enabled=True)
        transcriber = WhisperStreamingTranscriber(
            DummyWriter(),
            stt_config=stt_cfg,
        )

        started_at = time.monotonic()
        start_task = asyncio.create_task(transcriber.start())
        await asyncio.sleep(0.05)

        assert time.monotonic() - started_at < 0.15
        assert not start_task.done()

        await asyncio.wait_for(start_task, timeout=1.0)
        await transcriber.stop()

    asyncio.run(runner())


def test_whisper_streaming_transcriber_transcribe_does_not_block_event_loop(monkeypatch):
    async def runner() -> None:
        final_event = asyncio.Event()

        class DummyBus:
            def __init__(self, final_topic: str) -> None:
                self.final_topic = final_topic

            async def publish(self, topic: str, **payload):
                del payload
                if topic == self.final_topic:
                    final_event.set()

        queue: asyncio.Queue[SpeechSegment] = asyncio.Queue()

        class DummyWriter:
            def __init__(self) -> None:
                self.queue = queue

        class FakeWhisperEngine:
            def __init__(self, config, debug: bool = False) -> None:
                del config, debug

            def transcribe(self, data: bytes, sample_rate: int):
                del data, sample_rate
                time.sleep(0.25)
                return ["hello"]

        monkeypatch.setattr(streaming, "WhisperEngine", FakeWhisperEngine)

        stt_cfg = WhisperSttCfg(enabled=True, min_confidence=0.1, stability_timeout_s=0.0)
        transcriber = WhisperStreamingTranscriber(
            DummyWriter(),
            stt_config=stt_cfg,
            bus=DummyBus(stt_cfg.final_topic),
        )

        await transcriber.start()
        await queue.put(
            SpeechSegment(
                data=b"\x01\x02" * 16,
                sample_rate=16_000,
                channels=1,
                start_timestamp=0.0,
                end_timestamp=0.5,
                duration_ms=500.0,
                frames=50,
                voice_frames=45,
                metadata={"confidence": 0.95},
            )
        )

        started_at = time.monotonic()
        await asyncio.sleep(0.05)

        assert time.monotonic() - started_at < 0.15
        assert not final_event.is_set()

        await asyncio.wait_for(final_event.wait(), timeout=1.0)
        await transcriber.stop()

    asyncio.run(runner())


def test_whisper_streaming_transcriber_logs_transcribe_error_and_keeps_running(
    monkeypatch,
    caplog,
):
    async def runner() -> None:
        final_event = asyncio.Event()

        class DummyBus:
            def __init__(self, final_topic: str) -> None:
                self.final_topic = final_topic
                self.events: list[tuple[str, dict]] = []

            async def publish(self, topic: str, **payload):
                self.events.append((topic, payload))
                if topic == self.final_topic:
                    final_event.set()

        queue: asyncio.Queue[SpeechSegment] = asyncio.Queue()

        class DummyWriter:
            def __init__(self) -> None:
                self.queue = queue

        class FakeWhisperEngine:
            def __init__(self, config, debug: bool = False) -> None:
                del config, debug
                self.calls = 0

            def transcribe(self, data: bytes, sample_rate: int):
                del data, sample_rate
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
                return ["hello"]

        monkeypatch.setattr(streaming, "WhisperEngine", FakeWhisperEngine)

        stt_cfg = WhisperSttCfg(
            enabled=True,
            model="test-model",
            device="cuda",
            compute_type="int8",
            min_confidence=0.1,
            stability_timeout_s=0.0,
        )
        bus = DummyBus(stt_cfg.final_topic)
        transcriber = WhisperStreamingTranscriber(
            DummyWriter(),
            stt_config=stt_cfg,
            bus=bus,
        )

        await transcriber.start()
        for index in range(2):
            await queue.put(
                SpeechSegment(
                    data=b"\x01\x02" * 16,
                    sample_rate=16_000,
                    channels=1,
                    start_timestamp=float(index),
                    end_timestamp=float(index) + 0.5,
                    duration_ms=500.0,
                    frames=50,
                    voice_frames=45,
                    metadata={"confidence": 0.95},
                )
            )

        await asyncio.wait_for(final_event.wait(), timeout=1.0)
        await transcriber.stop()

        assert any(
            topic == stt_cfg.final_topic and payload["text"] == "Hello."
            for topic, payload in bus.events
        )

    caplog.set_level(logging.ERROR, logger=streaming.log.name)
    asyncio.run(runner())

    error_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == streaming.log.name
    ]
    assert any("transcription failed for segment #1" in message for message in error_messages)
    assert any("device=cuda" in message for message in error_messages)
    assert any("compute_type=int8" in message for message in error_messages)

