from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import audio.stt.whisper_engine as whisper_engine  # noqa: E402
from audio.stt.whisper_engine import WhisperEngine  # noqa: E402
from core.config import WhisperSttCfg  # noqa: E402


def test_whisper_engine_resolves_project_relative_model_path(monkeypatch, tmp_path):
    model_dir = tmp_path / "local-whisper"
    model_dir.mkdir()
    captured: dict[str, object] = {}

    class FakeWhisperModel:
        def __init__(self, model_name: str, **kwargs) -> None:
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

    monkeypatch.setattr(whisper_engine, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(whisper_engine, "WhisperModel", FakeWhisperModel)

    WhisperEngine(
        WhisperSttCfg(
            enabled=True,
            model="local-whisper",
            device="cuda",
            compute_type="int8",
            download_root="models/whisper",
            local_files_only=True,
        )
    )

    assert captured["model_name"] == str(model_dir.resolve())
    assert captured["kwargs"] == {
        "device": "cuda",
        "compute_type": "int8",
        "download_root": "models/whisper",
        "local_files_only": True,
    }


def test_whisper_engine_leaves_huggingface_repo_id_unchanged(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class FakeWhisperModel:
        def __init__(self, model_name: str, **kwargs) -> None:
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(whisper_engine, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(whisper_engine, "WhisperModel", FakeWhisperModel)

    WhisperEngine(
        WhisperSttCfg(
            enabled=True,
            model="avazir/faster-distil-whisper-large-v3-ru",
            device="cuda",
            compute_type="int8",
            local_files_only=True,
        )
    )

    assert captured["model_name"] == "avazir/faster-distil-whisper-large-v3-ru"


def test_whisper_engine_falls_back_to_cpu_on_cuda_oom(monkeypatch, caplog, tmp_path):
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeWhisperModel:
        def __init__(self, model_name: str, **kwargs) -> None:
            calls.append((model_name, kwargs))
            if kwargs.get("device") == "cuda":
                raise RuntimeError("CUDA failed with error out of memory")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(whisper_engine, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(whisper_engine, "WhisperModel", FakeWhisperModel)

    caplog.set_level("WARNING", logger=whisper_engine.log.name)
    engine = WhisperEngine(
        WhisperSttCfg(
            enabled=True,
            model="avazir/faster-distil-whisper-large-v3-ru",
            device="cuda",
            compute_type="int8",
            local_files_only=True,
        )
    )

    assert len(calls) == 2
    assert calls[0][1]["device"] == "cuda"
    assert calls[1][1]["device"] == "cpu"
    assert engine.active_device == "cpu"
    assert any(
        "CUDA model load ran out of memory; retrying on CPU" in record.getMessage()
        for record in caplog.records
    )
