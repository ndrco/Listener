# tests/test_silero_vad_helper.py
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.processing.silero_vad import SileroVADHelper
from core.config import AudioProcessingCfg, AudioVadCfg


def test_silero_vad_helper_streaming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "silero_vad.pt"
    model_path.write_bytes(b"")
    config_path = tmp_path / "silero_vad.json"
    config_data = {
        "model_id": "test-silero-vad",
        "default_device": "cpu",
        "pcm": {
            "dtype": "int16",
            "num_channels": 1,
            "sample_rates": [8000, 16000],
            "normalization_factor": 0.01,
        },
        "frame": {
            "duration_ms": 30,
            "samples_per_rate": {
                "8000": 240,
                "16000": 480,
            },
        },
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    fake_time = {"value": 0.0}

    def fake_monotonic() -> float:
        return fake_time["value"]

    monkeypatch.setattr("audio.processing.silero_vad.time.monotonic", fake_monotonic)

    expected_probability = 0.73

    class FakeSileroModel:
        def __init__(self) -> None:
            self.call_count = 0
            self.reset_count = 0
            self.last_call_args: list[tuple[torch.Tensor, int]] = []
            self.devices: list[torch.device] = []

        def __call__(self, frame: torch.Tensor, sample_rate: int) -> torch.Tensor:
            self.call_count += 1
            self.last_call_args.append((frame, sample_rate))
            return torch.tensor([expected_probability], dtype=torch.float32, device=frame.device)

        def eval(self) -> FakeSileroModel:
            return self

        def to(self, device: torch.device) -> FakeSileroModel:
            self.devices.append(device)
            return self

        def reset_states(self) -> None:
            self.reset_count += 1

    load_calls: list[tuple[str, torch.device | None]] = []
    models: list[FakeSileroModel] = []

    def fake_jit_load(path: str, map_location: torch.device | None = None) -> FakeSileroModel:
        load_calls.append((path, map_location))
        model = FakeSileroModel()
        models.append(model)
        return model

    monkeypatch.setattr("audio.processing.silero_vad.torch.jit.load", fake_jit_load)

    original_run_inference = SileroVADHelper._run_inference
    run_inference_calls: list[int] = []

    def run_inference_spy(self: SileroVADHelper, sample_rate: int) -> float:
        run_inference_calls.append(sample_rate)
        return original_run_inference(self, sample_rate)

    monkeypatch.setattr(SileroVADHelper, "_run_inference", run_inference_spy)

    cfg = AudioProcessingCfg(
        vad=AudioVadCfg(
            frame_duration_ms=30,
            model_path=str(model_path),
            model_config_path=str(config_path),
        )
    )

    helper = SileroVADHelper(cfg)

    assert helper._supported_sample_rates == (8000, 16000)

    empty_chunk = np.zeros(0, dtype=np.int16)
    initial_result = helper.predict(empty_chunk, 16_000)
    assert initial_result == pytest.approx(0.0)
    assert helper._buffer_fill == 0

    assert helper._frame_buffer is not None
    initial_buffer_id = id(helper._frame_buffer)
    assert helper._frame_samples == 480
    assert load_calls == []

    chunk = np.ones(160, dtype=np.int16)

    result1 = helper.predict(chunk, 16_000)
    assert result1 == pytest.approx(0.0)
    assert helper._buffer_fill == 160
    assert id(helper._frame_buffer) == initial_buffer_id
    assert len(load_calls) == 1
    assert len(models) == 1

    model = models[0]
    assert model.call_count == 0
    assert model.reset_count == 1
    assert run_inference_calls == []

    config_scale = float(config_data["pcm"]["normalization_factor"])

    result2 = helper.predict(chunk, 16_000)
    assert result2 == pytest.approx(0.0)
    assert helper._buffer_fill == 320
    assert id(helper._frame_buffer) == initial_buffer_id
    assert model.call_count == 0
    assert run_inference_calls == []

    result3 = helper.predict(chunk, 16_000)
    assert result3 == pytest.approx(expected_probability)
    assert helper._buffer_fill == 0
    assert id(helper._frame_buffer) == initial_buffer_id
    assert run_inference_calls == [16_000]
    assert model.call_count == 1
    assert helper._last_probability == pytest.approx(expected_probability)

    assert model.last_call_args
    frame_tensor, frame_rate = model.last_call_args[-1]
    assert frame_rate == 16_000
    assert frame_tensor.shape == (1, helper._frame_samples)
    assert frame_tensor.dtype == torch.float32
    assert frame_tensor.device == helper.device
    assert torch.allclose(
        frame_tensor.squeeze(0),
        torch.full_like(frame_tensor.squeeze(0), config_scale),
        atol=1e-6,
    )

    frame_duration = helper._frame_duration_sec
    assert frame_duration > 0

    fake_time["value"] += frame_duration / 2
    cached_from_empty = helper.predict(empty_chunk, 16_000)
    assert cached_from_empty == pytest.approx(expected_probability)
    assert helper._buffer_fill == 0
    assert model.call_count == 1
    assert run_inference_calls == [16_000]

    fake_time["value"] += frame_duration / 4
    tiny_chunk = np.ones(10, dtype=np.int16)
    cached_from_tiny = helper.predict(tiny_chunk, 16_000)
    assert cached_from_tiny == pytest.approx(expected_probability)
    assert helper._buffer_fill == 10
    assert model.call_count == 1
    assert run_inference_calls == [16_000]
    assert id(helper._frame_buffer) == initial_buffer_id

    fake_time["value"] += frame_duration / 4
    cached_still = helper.predict(empty_chunk, 16_000)
    assert cached_still == pytest.approx(expected_probability)
    assert helper._buffer_fill == 10
    assert model.call_count == 1
    assert run_inference_calls == [16_000]

    reset_before = model.reset_count
    fake_time["value"] += frame_duration
    reset_result = helper.predict(empty_chunk, 8_000)
    assert reset_result == pytest.approx(0.0)
    assert helper._buffer_fill == 0
    assert helper._frame_samples == 240
    assert model.reset_count == reset_before + 1
    buffer_id_after_reset = id(helper._frame_buffer)
    assert buffer_id_after_reset != initial_buffer_id

    reset_before_second = model.reset_count
    fake_time["value"] += frame_duration
    reset_back_result = helper.predict(empty_chunk, 16_000)
    assert reset_back_result == pytest.approx(0.0)
    assert helper._buffer_fill == 0
    assert helper._frame_samples == 480
    assert model.reset_count == reset_before_second + 1

    assert len(load_calls) == 1
    assert model.call_count == 1
