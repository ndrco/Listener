from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Paths, cfg, load  # noqa: E402


def test_load_resolves_project_relative_runtime_paths(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_dir = project_root / "config"
    models_dir = project_root / "models"
    whisper_dir = models_dir / "whisper"
    stt_snapshot = whisper_dir / "local-model"
    speech_gate_dir = models_dir / "gate"
    silero_path = models_dir / "silero.jit"

    config_dir.mkdir()
    stt_snapshot.mkdir(parents=True)
    speech_gate_dir.mkdir(parents=True)
    whisper_dir.mkdir(exist_ok=True)
    silero_path.write_bytes(b"")
    (config_dir / "silero.json").write_text("{}", encoding="utf-8")
    (config_dir / "speech_gate_patterns.json").write_text("{}", encoding="utf-8")

    config_path = config_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "speech_gate": {
                    "patterns_file": "config/speech_gate_patterns.json",
                    "identity_file": ".openclaw/workspace/IDENTITY.md",
                    "model": {"path": "models/gate"},
                },
                "audio": {
                    "processing": {
                        "vad": {
                            "model_path": "models/silero.jit",
                            "model_config_path": "config/silero.json",
                        }
                    },
                    "stt": {
                        "model": "models/whisper/local-model",
                        "download_root": "models/whisper",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    real_paths = cfg.paths
    monkeypatch.chdir(tmp_path)
    try:
        monkeypatch.setattr(
            cfg,
            "paths",
            Paths(
                root=project_root,
                profiles_json=project_root / "data" / "profiles.json",
                voice_profiles_json=project_root / "data" / "voice_profiles.json",
            ),
        )
        load(str(config_path))

        assert cfg.speech_gate.patterns_file == str(
            (config_dir / "speech_gate_patterns.json").resolve()
        )
        assert cfg.speech_gate.identity_file == str(
            (project_root / ".openclaw" / "workspace" / "IDENTITY.md").resolve()
        )
        assert cfg.speech_gate.model.path == str(speech_gate_dir.resolve())
        assert cfg.audio.processing.vad.model_path == str(silero_path.resolve())
        assert cfg.audio.processing.vad.model_config_path == str(
            (config_dir / "silero.json").resolve()
        )
        assert cfg.audio.stt.model == str(stt_snapshot.resolve())
        assert cfg.audio.stt.download_root == str(whisper_dir.resolve())
    finally:
        cfg.paths = real_paths
        load(str(ROOT / "config" / "config.json"))
