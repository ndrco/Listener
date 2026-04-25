import asyncio
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.processing import AudioProcessor, ProcessedAudioFrame, WindowsAudioProcessor
from core.config import AudioAgcCfg, AudioProcessingCfg, AudioVadCfg, cfg as global_cfg


class DummyBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def publish(self, topic: str, **payload):
        self.events.append((topic, payload))


def test_audio_processor_alias_is_backwards_compatible():
    assert WindowsAudioProcessor is AudioProcessor


def test_windows_processor_publishes_events(monkeypatch):
    bus = DummyBus()
    cfg = AudioProcessingCfg(
        enabled=True,
        vad=AudioVadCfg(
            enabled=True,
            mode=2,
            frame_duration_ms=30,
            energy_threshold_db=-60.0,
            hangover_ms=100,
            min_speech_duration_ms=30,
            min_silence_duration_ms=30,
            speech_pad_ms=0,
            publish_voice_activity=True,
        ),
    )
    cfg.highpass.enabled = False
    from audio.processing import windows_processing

    class FakeVad:
        result = True
        created_modes: list[int] = []
        calls: list[tuple[int, int]] = []

        def __init__(self, mode: int) -> None:
            type(self).created_modes.append(mode)

        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            type(self).calls.append((len(frame), sample_rate))
            return type(self).result

    FakeVad.result = True
    FakeVad.created_modes = []
    FakeVad.calls = []
    monkeypatch.setattr(
        windows_processing,
        "webrtcvad",
        types.SimpleNamespace(Vad=FakeVad),
    )

    frame_samples = int(16000 * cfg.vad.frame_duration_ms / 1000)
    pattern = np.array([0, 1000, -1000, 500], dtype=np.int16)
    pcm = np.tile(pattern, frame_samples // len(pattern))
    frame = pcm.tobytes()
    result: dict[str, ProcessedAudioFrame] = {}

    async def runner() -> None:
        async with WindowsAudioProcessor(
            sample_rate=16000,
            channels=1,
            config=cfg,
            bus=bus,
        ) as proc:
            await proc.submit(frame)
            result["frame"] = await proc.__anext__()
        with pytest.raises(StopAsyncIteration):
            await proc.__anext__()

    asyncio.run(runner())

    processed = result["frame"]
    assert isinstance(processed, ProcessedAudioFrame)
    assert processed.data == frame
    assert processed.voice_detected is True
    assert processed.vad_probability == pytest.approx(1.0)
    assert processed.webrtc_probability == pytest.approx(1.0)
    assert processed.silero_probability == pytest.approx(0.0)
    assert processed.silero_invocations == 0
    assert processed.vad_total_frames == 1
    assert processed.vad_speech_frames == 1
    assert processed.voice_active_duration >= 0.0

    assert FakeVad.created_modes == [cfg.vad.mode]
    expected_frame_bytes = frame_samples * 2
    assert FakeVad.calls and FakeVad.calls[0] == (expected_frame_bytes, 16000)

    processed_events = [
        payload
        for topic, payload in bus.events
        if topic == global_cfg.events.audio.processed_frame
    ]
    assert processed_events, "processed frame event not emitted"
    event = processed_events[-1]
    assert event["voice_activity"] is True
    assert event["vad_probability"] == pytest.approx(1.0)
    assert event["vad_speech_frames"] == 1
    assert event["vad_total_frames"] == 1
    assert event["webrtc_probability"] == pytest.approx(1.0)
    assert event["silero_probability"] == pytest.approx(0.0)
    assert event["silero_invocations"] == 0
    assert event["voice_active_duration"] >= 0.0
    assert event["segment_duration"] == pytest.approx(event["voice_active_duration"])

    vad_events = [
        payload
        for topic, payload in bus.events
        if topic == global_cfg.events.audio.voice_activity
    ]
    assert vad_events and vad_events[-1]["active"] is True
    vad_payload = vad_events[-1]
    assert "energy_db" not in vad_payload
    assert vad_payload["vad_probability"] == pytest.approx(1.0)
    assert vad_payload["vad_speech_frames"] == 1
    assert vad_payload["vad_total_frames"] == 1
    assert vad_payload["webrtc_probability"] == pytest.approx(1.0)
    assert vad_payload["silero_probability"] == pytest.approx(0.0)
    assert vad_payload["silero_invocations"] == 0
    assert vad_payload["voice_active_duration"] >= 0.0
    assert vad_payload["segment_duration"] == pytest.approx(vad_payload["voice_active_duration"])


def test_windows_processor_vad_threshold(monkeypatch):
    bus = DummyBus()
    cfg = AudioProcessingCfg(
        enabled=True,
        vad=AudioVadCfg(
            enabled=True,
            mode=2,
            frame_duration_ms=30,
            energy_threshold_db=-5.0,
            publish_voice_activity=True,
        ),
    )
    cfg.highpass.enabled = False
    from audio.processing import windows_processing

    class FakeVad:
        result = False
        created_modes: list[int] = []
        calls: list[tuple[int, int]] = []

        def __init__(self, mode: int) -> None:
            type(self).created_modes.append(mode)

        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            type(self).calls.append((len(frame), sample_rate))
            return type(self).result

    FakeVad.result = False
    FakeVad.created_modes = []
    FakeVad.calls = []
    monkeypatch.setattr(
        windows_processing,
        "webrtcvad",
        types.SimpleNamespace(Vad=FakeVad),
    )

    quiet_frame = (np.ones(160, dtype=np.int16) * 10).tobytes()
    result: dict[str, ProcessedAudioFrame] = {}

    async def runner() -> None:
        async with WindowsAudioProcessor(
            sample_rate=16000,
            channels=1,
            config=cfg,
            bus=bus,
        ) as proc:
            await proc.submit(quiet_frame)
            result["frame"] = await proc.__anext__()
        with pytest.raises(StopAsyncIteration):
            await proc.__anext__()

    asyncio.run(runner())

    processed = result["frame"]
    assert isinstance(processed, ProcessedAudioFrame)
    assert processed.voice_detected is False
    assert processed.vad_probability == pytest.approx(0.0)
    assert processed.webrtc_probability == pytest.approx(0.0)
    assert processed.silero_probability == pytest.approx(0.0)
    assert processed.silero_invocations == 0
    assert processed.vad_speech_frames == 0
    assert processed.vad_total_frames == 0
    assert processed.voice_active_duration == pytest.approx(0.0)

    expected_frame_bytes = int(16000 * cfg.vad.frame_duration_ms / 1000) * 2
    assert FakeVad.created_modes == [cfg.vad.mode]
    assert FakeVad.calls == []

    vad_events = [
        payload
        for topic, payload in bus.events
        if topic == global_cfg.events.audio.voice_activity
    ]
    assert vad_events and vad_events[-1]["active"] is False
    vad_payload = vad_events[-1]
    assert "energy_db" not in vad_payload
    assert vad_payload["vad_probability"] == pytest.approx(0.0)
    assert vad_payload["vad_speech_frames"] == 0
    assert vad_payload["vad_total_frames"] == 0
    assert vad_payload["webrtc_probability"] == pytest.approx(0.0)
    assert vad_payload["silero_probability"] == pytest.approx(0.0)
    assert vad_payload["silero_invocations"] == 0
    assert vad_payload["voice_active_duration"] == pytest.approx(0.0)
    assert vad_payload["segment_duration"] == pytest.approx(0.0)


def _setup_vad(monkeypatch, result: bool = True):
    from audio.processing import windows_processing

    class FakeVad:
        result_state = result
        created_modes: list[int] = []
        calls: list[tuple[int, int]] = []

        def __init__(self, mode: int) -> None:
            type(self).created_modes.append(mode)

        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            type(self).calls.append((len(frame), sample_rate))
            return bool(type(self).result_state)

    FakeVad.created_modes = []
    FakeVad.calls = []
    FakeVad.result_state = result

    monkeypatch.setattr(
        windows_processing,
        "webrtcvad",
        types.SimpleNamespace(Vad=FakeVad),
    )

    return FakeVad


def test_windows_processor_agc_raises_level(monkeypatch):
    bus = DummyBus()
    cfg = AudioProcessingCfg(
        enabled=True,
        vad=AudioVadCfg(
            enabled=True,
            mode=2,
            frame_duration_ms=30,
            energy_threshold_db=-60.0,
            min_speech_duration_ms=30,
            min_silence_duration_ms=30,
            speech_pad_ms=0,
            publish_voice_activity=True,
        ),
        agc=AudioAgcCfg(
            enabled=True,
            target_level_dbfs=-20.0,
            max_gain_db=20.0,
            attack_ms=1,
            release_ms=50,
        ),
    )
    cfg.highpass.enabled = False

    FakeVad = _setup_vad(monkeypatch, result=True)

    frame_samples = int(16000 * cfg.vad.frame_duration_ms / 1000)
    pcm = np.full(frame_samples, 500, dtype=np.int16)
    frame = pcm.tobytes()
    initial_rms = WindowsAudioProcessor._compute_rms(pcm)
    result: dict[str, ProcessedAudioFrame] = {}

    async def runner() -> None:
        async with WindowsAudioProcessor(
            sample_rate=16000,
            channels=1,
            config=cfg,
            bus=bus,
        ) as proc:
            await proc.submit(frame)
            result["frame"] = await proc.__anext__()
        with pytest.raises(StopAsyncIteration):
            await proc.__anext__()

    asyncio.run(runner())

    processed = result["frame"]
    assert isinstance(processed, ProcessedAudioFrame)
    processed_pcm = np.frombuffer(processed.data, dtype=np.int16)
    processed_rms = WindowsAudioProcessor._compute_rms(processed_pcm)

    target_rms = 32768.0 * (10.0 ** (cfg.agc.target_level_dbfs / 20.0))
    max_rms = initial_rms * (10.0 ** (cfg.agc.max_gain_db / 20.0))
    expected_rms = min(target_rms, max_rms)

    assert processed_rms == pytest.approx(expected_rms, rel=0.15)
    assert processed.voice_detected is True
    assert processed.vad_probability == pytest.approx(1.0)
    assert processed.vad_speech_frames == 1
    assert processed.vad_total_frames == 1

    assert FakeVad.created_modes == [cfg.vad.mode]
    assert FakeVad.calls and FakeVad.calls[0][1] == 16000


def test_windows_processor_agc_respects_max_gain(monkeypatch):
    bus = DummyBus()
    cfg = AudioProcessingCfg(
        enabled=True,
        vad=AudioVadCfg(
            enabled=True,
            mode=2,
            frame_duration_ms=30,
            energy_threshold_db=-60.0,
            min_speech_duration_ms=30,
            min_silence_duration_ms=30,
            speech_pad_ms=0,
            publish_voice_activity=True,
        ),
        agc=AudioAgcCfg(
            enabled=True,
            target_level_dbfs=-3.0,
            max_gain_db=6.0,
            attack_ms=1,
            release_ms=50,
        ),
    )
    cfg.highpass.enabled = False

    FakeVad = _setup_vad(monkeypatch, result=True)

    frame_samples = int(16000 * cfg.vad.frame_duration_ms / 1000)
    pcm = np.full(frame_samples, 2000, dtype=np.int16)
    frame = pcm.tobytes()
    initial_rms = WindowsAudioProcessor._compute_rms(pcm)
    result: dict[str, ProcessedAudioFrame] = {}

    async def runner() -> None:
        async with WindowsAudioProcessor(
            sample_rate=16000,
            channels=1,
            config=cfg,
            bus=bus,
        ) as proc:
            await proc.submit(frame)
            result["frame"] = await proc.__anext__()
        with pytest.raises(StopAsyncIteration):
            await proc.__anext__()

    asyncio.run(runner())

    processed = result["frame"]
    assert isinstance(processed, ProcessedAudioFrame)
    processed_pcm = np.frombuffer(processed.data, dtype=np.int16)
    processed_rms = WindowsAudioProcessor._compute_rms(processed_pcm)

    limited_rms = initial_rms * (10.0 ** (cfg.agc.max_gain_db / 20.0))
    assert processed_rms == pytest.approx(limited_rms, rel=0.15)
    assert processed.voice_detected is True
    assert processed.vad_probability == pytest.approx(1.0)
    assert processed.vad_speech_frames == 1
    assert processed.vad_total_frames == 1

    assert FakeVad.created_modes == [cfg.vad.mode]
    assert FakeVad.calls and FakeVad.calls[0][1] == 16000
