import asyncio
from pathlib import Path
import sys
import types
from typing import Mapping

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.processing import AudioProcessor, ProcessedAudioFrame, WindowsAudioProcessor
from audio.processing.resampler import StreamingResampler
from core.config import (
    AcousticEchoCancellationCfg,
    AudioAgcCfg,
    AudioHighpassCfg,
    AudioProcessingCfg,
    AudioVadCfg,
    NoiseSuppressionCfg,
    cfg as global_cfg,
)


def make_processing_cfg(
    *,
    vad: AudioVadCfg | Mapping[str, object] | None = None,
    agc: AudioAgcCfg | Mapping[str, object] | None = None,
    highpass: AudioHighpassCfg | Mapping[str, object] | None = None,
    **kwargs,
) -> AudioProcessingCfg:
    if vad is None:
        vad_cfg = AudioVadCfg()
    elif isinstance(vad, AudioVadCfg):
        vad_cfg = vad
    else:
        vad_cfg = AudioVadCfg(**dict(vad))

    if agc is None:
        agc_cfg = AudioAgcCfg()
    elif isinstance(agc, AudioAgcCfg):
        agc_cfg = agc
    else:
        agc_cfg = AudioAgcCfg(**dict(agc))

    if highpass is None:
        highpass_cfg = AudioHighpassCfg()
    elif isinstance(highpass, AudioHighpassCfg):
        highpass_cfg = highpass
    else:
        highpass_cfg = AudioHighpassCfg(**dict(highpass))

    return AudioProcessingCfg(vad=vad_cfg, agc=agc_cfg, highpass=highpass_cfg, **kwargs)


class DummyBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def publish(self, topic: str, **payload):
        self.events.append((topic, payload))


def test_audio_processor_alias_is_backwards_compatible():
    assert WindowsAudioProcessor is AudioProcessor


def _sine_wave(amplitude: int, samples: int, *, freq: float = 440.0, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32)
    wave = amplitude * np.sin(2.0 * np.pi * freq * t / float(sample_rate))
    return np.round(wave).astype(np.int16)


def install_fake_aec(monkeypatch):
    from audio.processing import echo_cancellation

    class FakeAudioFrame:
        def __init__(
            self, data: bytes, sample_rate: int, channels: int, samples_per_channel: int
        ) -> None:
            self.data = data
            self.sample_rate = sample_rate
            self.channels = channels
            self.samples_per_channel = samples_per_channel

    class FakeAudioProcessingModule:
        created: list["FakeAudioProcessingModule"] = []

        def __init__(
            self,
            *,
            echo_cancellation: bool,
            noise_suppression: bool,
            high_pass_filter: bool,
            auto_gain_control: bool,
        ) -> None:
            assert echo_cancellation is True
            FakeAudioProcessingModule.created.append(self)
            self.reverse_calls = 0
            self.stream_calls = 0
            self.last_reverse: bytes | None = None

        def set_stream_delay_ms(self, delay: float) -> None:
            self.delay = delay

        def process_reverse_stream(self, frame: FakeAudioFrame) -> None:
            self.reverse_calls += 1
            self.last_reverse = bytes(frame.data)

        def process_stream(self, frame: FakeAudioFrame) -> None:
            self.stream_calls += 1
            samples = np.frombuffer(frame.data, dtype=np.int16)
            frame.data = (samples // 2).astype(np.int16).tobytes()

    monkeypatch.setattr(echo_cancellation, "AudioFrame", FakeAudioFrame)
    monkeypatch.setattr(
        echo_cancellation,
        "AudioProcessingModule",
        FakeAudioProcessingModule,
    )
    return FakeAudioProcessingModule


def test_windows_processor_publishes_events(monkeypatch):
    bus = DummyBus()
    cfg = make_processing_cfg(
        enabled=True,
        vad=AudioVadCfg(
            enabled=True,
            mode=2,
            frame_duration_ms=30,
            energy_threshold_db=-60.0,
            hangover_ms=100,
            active_republish_interval_ms=100,
            min_speech_duration_ms=30,
            min_silence_duration_ms=30,
            speech_pad_ms=0,
            publish_voice_activity=True,
        ),
    )
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
    pcm = _sine_wave(2000, frame_samples)
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
    processed_pcm = np.frombuffer(processed.data, dtype=np.int16)
    assert processed_pcm.shape == pcm.shape
    assert WindowsAudioProcessor._compute_rms(processed_pcm) == pytest.approx(
        WindowsAudioProcessor._compute_rms(pcm), rel=0.2
    )
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
        payload for topic, payload in bus.events if topic == global_cfg.events.audio.processed_frame
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
        payload for topic, payload in bus.events if topic == global_cfg.events.audio.voice_activity
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
    cfg = make_processing_cfg(
        enabled=True,
        vad=AudioVadCfg(
            enabled=True,
            mode=2,
            frame_duration_ms=30,
            energy_threshold_db=-5.0,
            publish_voice_activity=True,
        ),
    )
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
        payload for topic, payload in bus.events if topic == global_cfg.events.audio.voice_activity
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
    cfg = make_processing_cfg(
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

    FakeVad = _setup_vad(monkeypatch, result=True)

    frame_samples = int(16000 * cfg.vad.frame_duration_ms / 1000)
    pcm = _sine_wave(500, frame_samples)
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


def _setup_sequence_vad(monkeypatch, sequence: list[bool]):
    from audio.processing import windows_processing

    class SequenceVad:
        created_modes: list[int] = []
        calls: list[tuple[int, int]] = []
        sequence: list[bool] = []
        index: int = 0

        def __init__(self, mode: int) -> None:
            type(self).created_modes.append(mode)

        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            type(self).calls.append((len(frame), sample_rate))
            if not type(self).sequence:
                return False
            if type(self).index < len(type(self).sequence):
                result = type(self).sequence[type(self).index]
            else:
                result = type(self).sequence[-1]
            type(self).index += 1
            return bool(result)

    SequenceVad.created_modes = []
    SequenceVad.calls = []
    SequenceVad.sequence = list(sequence)
    SequenceVad.index = 0

    monkeypatch.setattr(
        windows_processing,
        "webrtcvad",
        types.SimpleNamespace(Vad=SequenceVad),
    )

    return SequenceVad


def test_windows_processor_escalates_to_silero(monkeypatch):
    bus = DummyBus()
    cfg = make_processing_cfg(
        enabled=True,
        vad=AudioVadCfg(
            enabled=True,
            mode=2,
            frame_duration_ms=30,
            energy_threshold_db=-70.0,
            min_speech_duration_ms=30,
            min_silence_duration_ms=30,
            speech_pad_ms=0,
            publish_voice_activity=True,
            model_path="fake.pt",
            model_config_path="fake.json",
            webrtc_escalation_low_threshold=0.25,
            webrtc_escalation_high_threshold=0.75,
            silero_cadence_ms=60.0,
            silero_min_activation_duration_ms=60.0,
        ),
    )

    FakeVad = _setup_sequence_vad(monkeypatch, [True, False])
    from audio.processing import windows_processing

    class FakeSilero:
        probability: float = 0.92
        calls: list[tuple[np.ndarray, int]] = []

        def __init__(
            self,
            config: AudioProcessingCfg,
            *,
            debug: bool | None = None,
        ) -> None:
            self._config = config
            self._debug = debug

        def predict(self, pcm: np.ndarray, sample_rate: int) -> float:
            FakeSilero.calls.append((pcm.copy(), sample_rate))
            return float(type(self).probability)

    FakeSilero.calls = []
    monkeypatch.setattr(windows_processing, "SileroVADHelper", FakeSilero)

    frame_samples = int(16000 * cfg.vad.frame_duration_ms / 1000)
    pcm = _sine_wave(1200, frame_samples * 2)
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
    assert processed.voice_detected is True
    assert processed.vad_probability == pytest.approx(FakeSilero.probability)
    assert processed.webrtc_probability == pytest.approx(0.5)
    assert processed.silero_probability == pytest.approx(FakeSilero.probability)
    assert processed.silero_invocations == 1
    assert processed.vad_total_frames == 2
    assert processed.vad_speech_frames == 2

    assert FakeVad.calls and len(FakeVad.calls) == 2
    assert FakeSilero.calls and FakeSilero.calls[0][0].dtype == np.int16
    assert FakeSilero.calls[0][0].shape[0] == frame_samples * 2
    assert FakeSilero.calls[0][1] == 16000

    processed_events = [
        payload for topic, payload in bus.events if topic == global_cfg.events.audio.processed_frame
    ]
    assert processed_events, "processed frame event missing"
    event = processed_events[-1]
    assert event["silero_probability"] == pytest.approx(FakeSilero.probability)
    assert event["silero_invocations"] == 1
    assert event["webrtc_probability"] == pytest.approx(0.5)
    assert event["vad_probability"] == pytest.approx(FakeSilero.probability)


def test_windows_processor_uses_silero_when_webrtc_unavailable(monkeypatch):
    bus = DummyBus()
    cfg = make_processing_cfg(
        enabled=True,
        vad=AudioVadCfg(
            enabled=True,
            mode=2,
            frame_duration_ms=30,
            energy_threshold_db=-70.0,
            min_speech_duration_ms=30,
            min_silence_duration_ms=30,
            speech_pad_ms=0,
            publish_voice_activity=True,
            model_path="fake.pt",
            model_config_path="fake.json",
            silero_cadence_ms=30.0,
            silero_min_activation_duration_ms=30.0,
        ),
    )

    from audio.processing import windows_processing

    monkeypatch.setattr(windows_processing, "webrtcvad", None)

    class FakeSilero:
        probability: float = 0.8
        calls: list[tuple[np.ndarray, int]] = []

        def __init__(
            self,
            config: AudioProcessingCfg,
            *,
            debug: bool | None = None,
        ) -> None:
            self._config = config
            self._debug = debug

        def predict(self, pcm: np.ndarray, sample_rate: int) -> float:
            FakeSilero.calls.append((pcm.copy(), sample_rate))
            return float(type(self).probability)

    FakeSilero.calls = []
    monkeypatch.setattr(windows_processing, "SileroVADHelper", FakeSilero)

    frame_samples = int(16000 * cfg.vad.frame_duration_ms / 1000)
    pcm = _sine_wave(1400, frame_samples)
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
    assert processed.voice_detected is True
    assert processed.vad_probability == pytest.approx(FakeSilero.probability)
    assert processed.webrtc_probability == pytest.approx(0.0)
    assert processed.silero_probability == pytest.approx(FakeSilero.probability)
    assert processed.silero_invocations == 1
    assert processed.vad_total_frames == 1
    assert processed.vad_speech_frames == 1

    assert FakeSilero.calls and FakeSilero.calls[0][0].dtype == np.int16
    assert FakeSilero.calls[0][0].shape[0] == frame_samples
    assert FakeSilero.calls[0][1] == 16000

    processed_events = [
        payload for topic, payload in bus.events if topic == global_cfg.events.audio.processed_frame
    ]
    assert processed_events
    event = processed_events[-1]
    assert event["silero_probability"] == pytest.approx(FakeSilero.probability)
    assert event["silero_invocations"] == 1
    assert event["webrtc_probability"] == pytest.approx(0.0)
    assert event["vad_probability"] == pytest.approx(FakeSilero.probability)
def test_windows_processor_requires_min_speech(monkeypatch):
    bus = DummyBus()
    cfg = make_processing_cfg(
        enabled=True,
        vad=AudioVadCfg(
            enabled=True,
            mode=2,
            frame_duration_ms=30,
            energy_threshold_db=-60.0,
            min_speech_duration_ms=90,
            min_silence_duration_ms=150,
            speech_pad_ms=30,
            publish_voice_activity=True,
        ),
    )

    _setup_sequence_vad(monkeypatch, [True, False, False, False])

    frame_samples = int(16000 * cfg.vad.frame_duration_ms / 1000)
    pcm = _sine_wave(1000, frame_samples)
    frame = pcm.tobytes()
    outputs: list[ProcessedAudioFrame] = []

    async def runner() -> None:
        async with WindowsAudioProcessor(
            sample_rate=16000,
            channels=1,
            config=cfg,
            bus=bus,
        ) as proc:
            for _ in range(4):
                await proc.submit(frame)
                outputs.append(await proc.__anext__())
        with pytest.raises(StopAsyncIteration):
            await proc.__anext__()

    asyncio.run(runner())

    assert outputs, "no frames processed"
    assert all(not frame.voice_detected for frame in outputs)
    assert all(frame.voice_active_duration == pytest.approx(0.0) for frame in outputs)

    vad_events = [payload for topic, payload in bus.events if topic == global_cfg.events.audio.voice_activity]
    assert vad_events and all(event["active"] is False for event in vad_events)


def test_windows_processor_respects_min_silence(monkeypatch):
    bus = DummyBus()
    cfg = make_processing_cfg(
        enabled=True,
        vad=AudioVadCfg(
            enabled=True,
            mode=2,
            frame_duration_ms=30,
            energy_threshold_db=-60.0,
            min_speech_duration_ms=60,
            min_silence_duration_ms=90,
            speech_pad_ms=30,
            hangover_ms=0,
            active_republish_interval_ms=0,
            publish_voice_activity=True,
        ),
    )

    _setup_sequence_vad(monkeypatch, [True, True, True, False, False, False])

    frame_samples = int(16000 * cfg.vad.frame_duration_ms / 1000)
    pcm = _sine_wave(1200, frame_samples)
    frame = pcm.tobytes()
    outputs: list[ProcessedAudioFrame] = []

    async def runner() -> None:
        async with WindowsAudioProcessor(
            sample_rate=16000,
            channels=1,
            config=cfg,
            bus=bus,
        ) as proc:
            for _ in range(6):
                await proc.submit(frame)
                outputs.append(await proc.__anext__())
        with pytest.raises(StopAsyncIteration):
            await proc.__anext__()

    asyncio.run(runner())

    assert len(outputs) == 6
    assert outputs[0].voice_detected is False
    assert outputs[1].voice_detected is True
    assert outputs[2].voice_detected is True
    assert outputs[3].voice_detected is True
    assert outputs[4].voice_detected is True
    assert outputs[5].voice_detected is False
    assert outputs[3].voice_active_duration > 0.0
    assert outputs[4].voice_active_duration > outputs[3].voice_active_duration
    assert outputs[5].voice_active_duration == pytest.approx(0.12, rel=0.1)

    vad_events = [payload for topic, payload in bus.events if topic == global_cfg.events.audio.voice_activity]
    assert vad_events
    assert any(event["active"] is True for event in vad_events)
    last_event = vad_events[-1]
    assert last_event["active"] is False
    assert last_event["segment_duration"] == pytest.approx(0.12, rel=0.1)


def test_windows_processor_holds_state_during_hangover(monkeypatch):
    bus = DummyBus()
    cfg = make_processing_cfg(
        enabled=True,
        vad=AudioVadCfg(
            enabled=True,
            mode=2,
            frame_duration_ms=30,
            energy_threshold_db=-60.0,
            min_speech_duration_ms=60,
            min_silence_duration_ms=60,
            speech_pad_ms=0,
            hangover_ms=90,
            active_republish_interval_ms=90,
            publish_voice_activity=True,
        ),
    )

    sequence = [
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    _setup_sequence_vad(monkeypatch, sequence)

    frame_samples = int(16000 * cfg.vad.frame_duration_ms / 1000)
    pcm = _sine_wave(1400, frame_samples)
    frame = pcm.tobytes()
    outputs: list[ProcessedAudioFrame] = []

    async def runner() -> None:
        async with WindowsAudioProcessor(
            sample_rate=16000,
            channels=1,
            config=cfg,
            bus=bus,
        ) as proc:
            for _ in sequence:
                await proc.submit(frame)
                outputs.append(await proc.__anext__())
        with pytest.raises(StopAsyncIteration):
            await proc.__anext__()

    asyncio.run(runner())

    assert len(outputs) == len(sequence)
    active_indices = [idx for idx, frame in enumerate(outputs) if frame.voice_detected]
    assert active_indices, "voice never detected"
    first_active = active_indices[0]
    assert first_active <= 1
    assert all(frame.voice_detected for frame in outputs[first_active:6])
    assert outputs[3].voice_active_duration > 0.0
    assert outputs[-1].voice_detected is False

    voice_events = [payload for topic, payload in bus.events if topic == global_cfg.events.audio.voice_activity]
    assert voice_events
    active_indices = [idx for idx, event in enumerate(voice_events) if event["active"]]
    assert active_indices, "no active events"
    first_active_event = active_indices[0]
    assert all(
        event["active"] is True for event in voice_events[first_active_event:-1]
    )
    final_event = voice_events[-1]
    assert final_event["active"] is False
    assert final_event["voice_active_duration"] == pytest.approx(
        final_event["segment_duration"], rel=0.05
    )
    final_frame = next(
        frame
        for frame in outputs
        if not frame.voice_detected and frame.voice_active_duration > 0.0
    )
    assert final_event["voice_active_duration"] == pytest.approx(
        final_frame.voice_active_duration, rel=0.05
    )


def test_windows_processor_active_republish_interval(monkeypatch):
    bus = DummyBus()
    hangover_ms = 300
    republish_ms = 120
    cfg = make_processing_cfg(
        enabled=True,
        vad=AudioVadCfg(
            enabled=True,
            mode=2,
            frame_duration_ms=30,
            energy_threshold_db=-60.0,
            probability_threshold=0.0,
            min_speech_duration_ms=30,
            min_silence_duration_ms=30,
            speech_pad_ms=0,
            hangover_ms=hangover_ms,
            active_republish_interval_ms=republish_ms,
            publish_voice_activity=True,
        ),
    )

    sequence = [True, True, True, True] + [False] * 12
    _setup_sequence_vad(monkeypatch, sequence)

    from audio.processing import windows_processing

    class FakeTime:
        def __init__(self, start: float = 0.0) -> None:
            self.current = float(start)

        def set(self, value: float) -> None:
            self.current = float(value)

        def time(self) -> float:
            return self.current

    fake_time = FakeTime()
    monkeypatch.setattr(windows_processing.time, "time", fake_time.time)

    frame_samples = int(16000 * cfg.vad.frame_duration_ms / 1000)
    pcm = _sine_wave(1500, frame_samples)
    frame = pcm.tobytes()
    timestamps = [idx * 0.06 for idx in range(len(sequence) + 4)]
    outputs: list[ProcessedAudioFrame] = []

    async def runner() -> None:
        async with WindowsAudioProcessor(
            sample_rate=16000,
            channels=1,
            config=cfg,
            bus=bus,
        ) as proc:
            for ts in timestamps:
                fake_time.set(ts)
                await proc.submit(frame)
                outputs.append(await proc.__anext__())
        with pytest.raises(StopAsyncIteration):
            await proc.__anext__()

    asyncio.run(runner())

    voice_events = [
        payload for topic, payload in bus.events if topic == global_cfg.events.audio.voice_activity
    ]
    assert voice_events, "voice activity events not published"

    active_events = [event for event in voice_events if event["active"]]
    assert len(active_events) >= 3

    republish_interval_s = republish_ms / 1000.0
    first_interval = active_events[1]["timestamp"] - active_events[0]["timestamp"]
    second_interval = active_events[2]["timestamp"] - active_events[1]["timestamp"]
    assert first_interval == pytest.approx(republish_interval_s, rel=0.05)
    assert second_interval == pytest.approx(republish_interval_s, rel=0.05)

    silence_start_idx = sequence.index(False)
    silence_start_ts = timestamps[silence_start_idx]
    assert any(
        event["active"] and event["timestamp"] > silence_start_ts
        for event in voice_events
    ), "hangover state not retained after silence"

    final_event = voice_events[-1]
    assert final_event["active"] is False
    hangover_s = hangover_ms / 1000.0
    assert final_event["timestamp"] - silence_start_ts >= hangover_s


def test_windows_processor_agc_respects_max_gain(monkeypatch):
    bus = DummyBus()
    cfg = make_processing_cfg(
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

    FakeVad = _setup_vad(monkeypatch, result=True)

    frame_samples = int(16000 * cfg.vad.frame_duration_ms / 1000)
    pcm = _sine_wave(2000, frame_samples)
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

def test_windows_processor_aec_stub(monkeypatch):
    from audio.processing import echo_cancellation

    class FakeAudioFrame:
        def __init__(self, data: bytes, sample_rate: int, channels: int, samples_per_channel: int) -> None:
            self.data = data
            self.sample_rate = sample_rate
            self.channels = channels
            self.samples_per_channel = samples_per_channel

    class FakeAudioProcessingModule:
        created: list["FakeAudioProcessingModule"] = []

        def __init__(
            self,
            *,
            echo_cancellation: bool,
            noise_suppression: bool,
            high_pass_filter: bool,
            auto_gain_control: bool,
        ) -> None:
            assert echo_cancellation is True
            FakeAudioProcessingModule.created.append(self)
            self.reverse_calls = 0
            self.stream_calls = 0

        def set_stream_delay_ms(self, delay: float) -> None:
            self.delay = delay

        def process_reverse_stream(self, frame: FakeAudioFrame) -> None:
            self.reverse_calls += 1

        def process_stream(self, frame: FakeAudioFrame) -> None:
            self.stream_calls += 1
            samples = np.frombuffer(frame.data, dtype=np.int16)
            frame.data = (samples // 2).astype(np.int16).tobytes()

    monkeypatch.setattr(echo_cancellation, "AudioFrame", FakeAudioFrame)
    monkeypatch.setattr(echo_cancellation, "AudioProcessingModule", FakeAudioProcessingModule)

    bus = DummyBus()
    cfg = make_processing_cfg(
        enabled=True,
        vad=AudioVadCfg(enabled=False, publish_voice_activity=False),
    )
    cfg.highpass.enabled = False
    cfg.agc.enabled = False
    cfg.noise_suppression.enabled = False
    cfg.aec.enabled = True
    cfg.aec.playback_event_topic = None
    cfg.aec.stream_delay_ms = 0.0

    frame_samples = 160
    near_pcm = (np.ones(frame_samples, dtype=np.int16) * 2000).astype(np.int16)
    far_pcm = (np.ones(frame_samples, dtype=np.int16) * 1000).astype(np.int16)

    result: dict[str, ProcessedAudioFrame] = {}

    async def runner() -> None:
        async with WindowsAudioProcessor(
            sample_rate=16000,
            channels=1,
            config=cfg,
            bus=bus,
        ) as proc:
            proc.submit_playback(far_pcm.tobytes())
            await proc.submit(near_pcm.tobytes())
            result["frame"] = await proc.__anext__()
        with pytest.raises(StopAsyncIteration):
            await proc.__anext__()

    asyncio.run(runner())

    assert FakeAudioProcessingModule.created, "AEC module was not initialised"
    instance = FakeAudioProcessingModule.created[-1]
    assert instance.reverse_calls >= 1
    assert instance.stream_calls >= 1

    processed = result["frame"]
    processed_pcm = np.frombuffer(processed.data, dtype=np.int16)
    assert np.all(processed_pcm == near_pcm // 2)


def test_windows_processor_aec_loopback(monkeypatch):
    from audio.processing import echo_cancellation
    from audio.processing import windows_processing

    class FakeAudioFrame:
        def __init__(self, data: bytes, sample_rate: int, channels: int, samples_per_channel: int) -> None:
            self.data = data
            self.sample_rate = sample_rate
            self.channels = channels
            self.samples_per_channel = samples_per_channel

    class FakeAudioProcessingModule:
        created: list["FakeAudioProcessingModule"] = []

        def __init__(
            self,
            *,
            echo_cancellation: bool,
            noise_suppression: bool,
            high_pass_filter: bool,
            auto_gain_control: bool,
        ) -> None:
            assert echo_cancellation is True
            FakeAudioProcessingModule.created.append(self)
            self.reverse_calls = 0
            self.stream_calls = 0
            self.last_reverse: bytes | None = None

        def set_stream_delay_ms(self, delay: float) -> None:
            self.delay = delay

        def process_reverse_stream(self, frame: FakeAudioFrame) -> None:
            self.reverse_calls += 1
            self.last_reverse = bytes(frame.data)

        def process_stream(self, frame: FakeAudioFrame) -> None:
            self.stream_calls += 1
            samples = np.frombuffer(frame.data, dtype=np.int16)
            frame.data = (samples // 2).astype(np.int16).tobytes()

    monkeypatch.setattr(echo_cancellation, "AudioFrame", FakeAudioFrame)
    monkeypatch.setattr(echo_cancellation, "AudioProcessingModule", FakeAudioProcessingModule)

    class FakeSoundDevice:
        class WasapiSettings:
            def __init__(self, *, loopback: bool = False) -> None:
                self.loopback = loopback

        class InputStream:
            created: list["FakeSoundDevice.InputStream"] = []

            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.callback = kwargs["callback"]
                self.started = False
                self.stopped = False
                self.closed = False
                FakeSoundDevice.InputStream.created.append(self)

            def start(self) -> None:
                self.started = True
                frames = int(self.kwargs["blocksize"])
                channels = int(self.kwargs["channels"])
                data = (np.ones(frames * channels, dtype=np.int16) * 500).reshape(frames, channels)
                self.callback(data, frames, None, None)

            def stop(self) -> None:
                self.stopped = True

            def close(self) -> None:
                self.closed = True

    monkeypatch.setattr(
        windows_processing.WindowsAudioProcessor,
        "_load_sounddevice_backend",
        lambda self: FakeSoundDevice,
    )

    FakeSoundDevice.InputStream.created.clear()

    bus = DummyBus()
    cfg = make_processing_cfg(
        enabled=True,
        vad=AudioVadCfg(enabled=False, publish_voice_activity=False),
    )
    cfg.highpass.enabled = False
    cfg.agc.enabled = False
    cfg.noise_suppression.enabled = False
    cfg.aec.enabled = True
    cfg.aec.playback_source = "loopback"
    cfg.aec.loopback_device_index = 7
    cfg.aec.stream_delay_ms = 0.0

    frame_samples = 160
    near_pcm = (np.ones(frame_samples, dtype=np.int16) * 2000).astype(np.int16)

    result: dict[str, ProcessedAudioFrame] = {}

    async def runner() -> None:
        async with WindowsAudioProcessor(
            sample_rate=16000,
            channels=1,
            config=cfg,
            bus=bus,
        ) as proc:
            await asyncio.sleep(0)
            await proc.submit(near_pcm.tobytes())
            result["frame"] = await proc.__anext__()
        with pytest.raises(StopAsyncIteration):
            await proc.__anext__()

    asyncio.run(runner())

    assert FakeAudioProcessingModule.created, "AEC module was not initialised"
    module = FakeAudioProcessingModule.created[-1]
    assert module.reverse_calls >= 1
    assert module.last_reverse is not None and len(module.last_reverse) > 0

    processed = result["frame"]
    processed_pcm = np.frombuffer(processed.data, dtype=np.int16)
    assert np.all(processed_pcm == near_pcm // 2)

    assert FakeSoundDevice.InputStream.created, "Loopback stream not created"
    stream = FakeSoundDevice.InputStream.created[-1]
    assert stream.started is True
    assert stream.stopped is True
    assert stream.closed is True
    assert stream.kwargs.get("device") == cfg.aec.loopback_device_index


def test_audio_processor_linux_auto_loopback_selects_monitor(monkeypatch):
    from audio.processing import windows_processing

    FakeAudioProcessingModule = install_fake_aec(monkeypatch)
    monkeypatch.setattr(windows_processing.platform, "system", lambda: "Linux")

    class FakeSoundDevice:
        devices = [
            {
                "name": "Built-in Microphone",
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000,
            },
            {
                "name": "alsa_output.pci Speaker Monitor",
                "hostapi": 1,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000,
            },
        ]
        hostapis = [{"name": "ALSA"}, {"name": "PulseAudio"}]

        @classmethod
        def query_devices(cls):
            return cls.devices

        @classmethod
        def query_hostapis(cls):
            return cls.hostapis

        class InputStream:
            created: list["FakeSoundDevice.InputStream"] = []

            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.callback = kwargs["callback"]
                self.started = False
                self.stopped = False
                self.closed = False
                FakeSoundDevice.InputStream.created.append(self)

            def start(self) -> None:
                self.started = True
                frames = int(self.kwargs["blocksize"])
                channels = int(self.kwargs["channels"])
                data = (np.ones(frames * channels, dtype=np.int16) * 500).reshape(frames, channels)
                self.callback(data, frames, None, None)

            def stop(self) -> None:
                self.stopped = True

            def close(self) -> None:
                self.closed = True

    monkeypatch.setattr(
        windows_processing.AudioProcessor,
        "_load_sounddevice_backend",
        lambda self: FakeSoundDevice,
    )

    bus = DummyBus()
    cfg = make_processing_cfg(
        enabled=True,
        vad=AudioVadCfg(enabled=False, publish_voice_activity=False),
    )
    cfg.highpass.enabled = False
    cfg.agc.enabled = False
    cfg.noise_suppression.enabled = False
    cfg.aec.enabled = True
    cfg.aec.playback_source = "loopback"
    cfg.aec.loopback_backend = "auto"
    cfg.aec.loopback_device_index = None
    cfg.aec.stream_delay_ms = 0.0

    near_pcm = (np.ones(160, dtype=np.int16) * 2000).astype(np.int16)
    result: dict[str, ProcessedAudioFrame] = {}

    async def runner() -> None:
        async with AudioProcessor(
            sample_rate=16000,
            channels=1,
            config=cfg,
            bus=bus,
        ) as proc:
            await asyncio.sleep(0)
            await proc.submit(near_pcm.tobytes())
            result["frame"] = await proc.__anext__()

    asyncio.run(runner())

    assert FakeAudioProcessingModule.created
    assert FakeAudioProcessingModule.created[-1].reverse_calls >= 1
    assert FakeSoundDevice.InputStream.created
    stream = FakeSoundDevice.InputStream.created[-1]
    assert stream.kwargs.get("device") == 1
    assert "extra_settings" not in stream.kwargs
    processed_pcm = np.frombuffer(result["frame"].data, dtype=np.int16)
    assert np.all(processed_pcm == near_pcm // 2)


def test_audio_processor_linux_loopback_uses_explicit_device_without_wasapi(monkeypatch):
    from audio.processing import windows_processing

    install_fake_aec(monkeypatch)
    monkeypatch.setattr(windows_processing.platform, "system", lambda: "Linux")

    class FakeSoundDevice:
        class InputStream:
            created: list["FakeSoundDevice.InputStream"] = []

            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.callback = kwargs["callback"]
                FakeSoundDevice.InputStream.created.append(self)

            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

            def close(self) -> None:
                pass

    monkeypatch.setattr(
        windows_processing.AudioProcessor,
        "_load_sounddevice_backend",
        lambda self: FakeSoundDevice,
    )

    bus = DummyBus()
    cfg = make_processing_cfg(
        enabled=True,
        vad=AudioVadCfg(enabled=False, publish_voice_activity=False),
    )
    cfg.highpass.enabled = False
    cfg.agc.enabled = False
    cfg.noise_suppression.enabled = False
    cfg.aec.enabled = True
    cfg.aec.playback_source = "loopback"
    cfg.aec.loopback_backend = "pulse"
    cfg.aec.loopback_device_index = 42

    async def runner() -> None:
        async with AudioProcessor(
            sample_rate=16000,
            channels=1,
            config=cfg,
            bus=bus,
        ):
            pass

    asyncio.run(runner())

    assert FakeSoundDevice.InputStream.created
    stream = FakeSoundDevice.InputStream.created[-1]
    assert stream.kwargs.get("device") == 42
    assert "extra_settings" not in stream.kwargs


def test_audio_processor_linux_loopback_missing_monitor_does_not_fail(monkeypatch):
    from audio.processing import windows_processing

    install_fake_aec(monkeypatch)
    monkeypatch.setattr(windows_processing.platform, "system", lambda: "Linux")

    class FakeSoundDevice:
        @staticmethod
        def query_devices():
            return [
                {
                    "name": "Built-in Microphone",
                    "hostapi": 0,
                    "max_input_channels": 1,
                    "max_output_channels": 0,
                    "default_samplerate": 48000,
                }
            ]

        @staticmethod
        def query_hostapis():
            return [{"name": "ALSA"}]

        class InputStream:
            created: list["FakeSoundDevice.InputStream"] = []

            def __init__(self, **kwargs) -> None:
                FakeSoundDevice.InputStream.created.append(self)

    monkeypatch.setattr(
        windows_processing.AudioProcessor,
        "_load_sounddevice_backend",
        lambda self: FakeSoundDevice,
    )

    bus = DummyBus()
    cfg = make_processing_cfg(
        enabled=True,
        vad=AudioVadCfg(enabled=False, publish_voice_activity=False),
    )
    cfg.highpass.enabled = False
    cfg.agc.enabled = False
    cfg.noise_suppression.enabled = False
    cfg.aec.enabled = True
    cfg.aec.playback_source = "loopback"
    cfg.aec.loopback_backend = "auto"
    cfg.aec.loopback_device_index = None

    near_pcm = (np.ones(160, dtype=np.int16) * 2000).astype(np.int16)
    result: dict[str, ProcessedAudioFrame] = {}

    async def runner() -> None:
        async with AudioProcessor(
            sample_rate=16000,
            channels=1,
            config=cfg,
            bus=bus,
        ) as proc:
            await proc.submit(near_pcm.tobytes())
            result["frame"] = await proc.__anext__()

    asyncio.run(runner())

    assert not FakeSoundDevice.InputStream.created
    assert isinstance(result["frame"], ProcessedAudioFrame)


def test_windows_processor_resamples_to_output_rate():
    bus = DummyBus()
    processing_cfg = make_processing_cfg(
        enabled=True,
        vad=AudioVadCfg(enabled=False),
        agc=AudioAgcCfg(enabled=False),
        highpass=AudioHighpassCfg(enabled=False),
        noise_suppression=NoiseSuppressionCfg(enabled=False),
        aec=AcousticEchoCancellationCfg(enabled=False),
    )

    input_cfg = global_cfg.audio.input
    prev_input_rate = input_cfg.input_sample_rate
    prev_output_rate = input_cfg.output_sample_rate

    try:
        input_cfg.input_sample_rate = 48000
        input_cfg.output_sample_rate = 16000

        frame_samples = 4800
        pcm = _sine_wave(8000, frame_samples, freq=1000.0, sample_rate=48000)
        frame = pcm.tobytes()
        result: dict[str, ProcessedAudioFrame] = {}

        async def runner() -> None:
            async with WindowsAudioProcessor(
                sample_rate=48000,
                channels=1,
                config=processing_cfg,
                bus=bus,
            ) as proc:
                await proc.submit(frame)
                result["frame"] = await proc.__anext__()
                result["processor_rate"] = proc.output_sample_rate

        asyncio.run(runner())
    finally:
        input_cfg.input_sample_rate = prev_input_rate
        input_cfg.output_sample_rate = prev_output_rate

    processed = result["frame"]
    assert result["processor_rate"] == 16000
    assert processed.sample_rate == 16000

    processed_pcm = np.frombuffer(processed.data, dtype=np.int16)
    resampler = StreamingResampler(48000, 16000)
    expected_pcm = resampler.process(pcm)

    assert processed_pcm.size == expected_pcm.size
    np.testing.assert_array_equal(processed_pcm, expected_pcm)
