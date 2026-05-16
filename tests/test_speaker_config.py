import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from speaker.config import SpeakerConfig, default_piper_model


class ConfigTests(unittest.TestCase):
    def test_loads_project_config_by_default(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text('{"speaker": {"mode": "final"}}', encoding="utf-8")

            with patch("speaker.config.DEFAULT_CONFIG_PATH", config_path):
                config = SpeakerConfig.load()

        self.assertEqual(config.speaker.mode, "final")

    def test_explicit_config_replaces_project_default(self):
        with TemporaryDirectory() as tmp:
            default_path = Path(tmp) / "speaker.json"
            explicit_path = Path(tmp) / "explicit.json"
            default_path.write_text('{"gateway": {"session_key": "project"}}', encoding="utf-8")
            explicit_path.write_text('{"gateway": {"session_key": "explicit"}}', encoding="utf-8")

            with patch("speaker.config.DEFAULT_CONFIG_PATH", default_path):
                config = SpeakerConfig.load(str(explicit_path))

        self.assertEqual(config.gateway.session_key, "explicit")

    def test_env_overrides_mode_and_commands(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text('{"speaker": {"mode": "final"}}', encoding="utf-8")

            with (
                patch("speaker.config.DEFAULT_CONFIG_PATH", config_path),
                patch.dict(
                    os.environ,
                    {
                        "SPEAKER_MODE": "streaming",
                        "SPEAKER_PIPER_COMMAND": "/tmp/python3",
                        "SPEAKER_PIPER_MODEL": "/tmp/model.onnx",
                        "SPEAKER_PIPER_VOLUME": "0.65",
                        "SPEAKER_PLAYER_COMMAND": "/bin/true",
                        "SPEAKER_DUCKING_FADE_IN_MS": "35",
                        "SPEAKER_DUCKING_FADE_OUT_MS": "90",
                        "SPEAKER_DUCKING_ENABLED": "true",
                        "SPEAKER_DUCKING_VOLUME_SCALE": "0.4",
                    },
                    clear=False,
                ),
            ):
                config = SpeakerConfig.load()

        self.assertEqual(config.speaker.mode, "streaming")
        self.assertEqual(config.piper.command, "/tmp/python3")
        self.assertEqual(config.piper.model, "/tmp/model.onnx")
        self.assertEqual(config.piper.volume, 0.65)
        self.assertEqual(config.playback.command, "/bin/true")
        self.assertEqual(config.playback.ducking.fade_in_ms, 35)
        self.assertEqual(config.playback.ducking.fade_out_ms, 90)
        self.assertTrue(config.playback.ducking.enabled)
        self.assertEqual(config.playback.ducking.volume_scale, 0.4)

    def test_rejects_invalid_mode(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text('{"speaker": {"mode": "loudly"}}', encoding="utf-8")

            with patch("speaker.config.DEFAULT_CONFIG_PATH", config_path):
                with self.assertRaises(ValueError):
                    SpeakerConfig.load()

    def test_default_piper_model_prefers_models_directory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            legacy_dir = root / "piper"
            models_dir.mkdir()
            legacy_dir.mkdir()
            (models_dir / "ru_RU-irina-medium.onnx").write_text("new", encoding="utf-8")
            (legacy_dir / "ru_RU-irina-medium.onnx").write_text("old", encoding="utf-8")

            with (
                patch("speaker.config.DEFAULT_MODELS_DIR", models_dir),
                patch("speaker.config.LEGACY_PIPER_DIR", legacy_dir),
            ):
                model = default_piper_model()

        self.assertEqual(model, str(models_dir / "ru_RU-irina-medium.onnx"))

    def test_default_piper_model_falls_back_to_legacy_location(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            legacy_dir = root / "piper"
            legacy_dir.mkdir()
            (legacy_dir / "ru_RU-irina-medium.onnx").write_text("old", encoding="utf-8")

            with (
                patch("speaker.config.DEFAULT_MODELS_DIR", models_dir),
                patch("speaker.config.LEGACY_PIPER_DIR", legacy_dir),
            ):
                model = default_piper_model()

        self.assertEqual(model, str(legacy_dir / "ru_RU-irina-medium.onnx"))

    def test_json_volume_is_normalized(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text('{"piper": {"volume": -2}}', encoding="utf-8")

            with patch("speaker.config.DEFAULT_CONFIG_PATH", config_path):
                config = SpeakerConfig.load()

        self.assertEqual(config.piper.volume, 0.0)

    def test_json_ducking_values_are_normalized(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                '{"playback": {"ducking": {"enabled": true, "volume_scale": 2.5, "fade_in_ms": -5, "fade_out_ms": -20}}}',
                encoding="utf-8",
            )

            with patch("speaker.config.DEFAULT_CONFIG_PATH", config_path):
                config = SpeakerConfig.load()

        self.assertEqual(config.playback.ducking.fade_in_ms, 0)
        self.assertEqual(config.playback.ducking.fade_out_ms, 0)
        self.assertTrue(config.playback.ducking.enabled)
        self.assertEqual(config.playback.ducking.volume_scale, 1.0)

    def test_legacy_playback_fade_keys_are_mapped_to_ducking(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                '{"playback": {"fade_in_ms": 15, "fade_out_ms": 45, "ducking": {"enabled": true}}}',
                encoding="utf-8",
            )

            with patch("speaker.config.DEFAULT_CONFIG_PATH", config_path):
                config = SpeakerConfig.load()

        self.assertEqual(config.playback.ducking.fade_in_ms, 15)
        self.assertEqual(config.playback.ducking.fade_out_ms, 45)
        self.assertTrue(config.playback.ducking.enabled)


if __name__ == "__main__":
    unittest.main()
