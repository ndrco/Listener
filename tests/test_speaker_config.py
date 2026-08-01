import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from speaker.config import PROJECT_ROOT, SpeakerConfig, default_piper_model


class ConfigTests(unittest.TestCase):
    def test_missing_tts_section_keeps_piper_backend(self):
        config = SpeakerConfig().merge_dict({"enabled": True})

        self.assertEqual(config.tts.backend, "piper")

    def test_neural_workers_default_to_listener_python(self):
        config = SpeakerConfig()

        self.assertEqual(config.voxcpm2.python, sys.executable)
        self.assertEqual(config.cosyvoice3.python, sys.executable)

    def test_neural_assets_default_to_project_directories(self):
        config = SpeakerConfig()

        self.assertEqual(
            config.voxcpm2.model_path,
            str(PROJECT_ROOT / "models" / "tts" / "voxcpm2" / "model"),
        )
        self.assertEqual(
            config.voxcpm2.reference_wav_path,
            str(PROJECT_ROOT / "references" / "voxcpm2" / "Nata.wav"),
        )
        self.assertEqual(
            config.cosyvoice3.repo_path,
            str(PROJECT_ROOT / "models" / "tts" / "cosyvoice3" / "CosyVoice"),
        )
        self.assertEqual(
            config.cosyvoice3.wetext_path,
            str(PROJECT_ROOT / "models" / "tts" / "cosyvoice3" / "wetext"),
        )

    def test_relative_neural_asset_paths_are_resolved_from_project_root(self):
        config = SpeakerConfig().merge_dict(
            {
                "voxcpm2": {"model_path": "models/custom-vox"},
                "cosyvoice3": {"repo_path": "models/custom-cosy"},
            }
        )

        self.assertEqual(
            config.voxcpm2.model_path,
            str((PROJECT_ROOT / "models/custom-vox").resolve()),
        )
        self.assertEqual(
            config.cosyvoice3.repo_path,
            str((PROJECT_ROOT / "models/custom-cosy").resolve()),
        )

    def test_blank_neural_worker_python_falls_back_to_listener_python(self):
        config = SpeakerConfig().merge_dict(
            {
                "voxcpm2": {"python": "  "},
                "cosyvoice3": {"python": ""},
            }
        )

        self.assertEqual(config.voxcpm2.python, sys.executable)
        self.assertEqual(config.cosyvoice3.python, sys.executable)

    def test_integrated_listener_speaker_section_is_supported(self):
        config = SpeakerConfig().merge_dict(
            {
                "version": "1.0",
                "speaker": {
                    "enabled": True,
                    "mode": "streaming",
                    "tts": {"backend": "cosyvoice3"},
                    "cosyvoice3": {"model_path": "/tmp/cosy"},
                },
            }
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.speaker.mode, "streaming")
        self.assertEqual(config.tts.backend, "cosyvoice3")
        self.assertEqual(config.cosyvoice3.model_path, "/tmp/cosy")

    def test_shared_tts_number_normalization_can_be_disabled(self):
        config = SpeakerConfig().merge_dict({"tts": {"normalize_numbers": "false"}})

        self.assertFalse(config.tts.normalize_numbers)

    def test_legacy_cosyvoice_number_normalization_setting_is_migrated(self):
        config = SpeakerConfig().merge_dict(
            {"cosyvoice3": {"normalize_numbers": "false"}}
        )

        self.assertFalse(config.tts.normalize_numbers)

    def test_neural_tts_sections_are_merged_and_normalized(self):
        config = SpeakerConfig().merge_dict(
            {
                "tts": {
                    "backend": "VOXCPM2",
                    "startup_timeout_s": 0,
                    "style": {"leading_emoji_only": "false"},
                },
                "voxcpm2": {
                    "python": " /tmp/vox-python ",
                    "model_path": " /tmp/vox-model ",
                    "optimize": "false",
                    "load_denoiser": "true",
                },
            }
        )

        self.assertEqual(config.tts.backend, "voxcpm2")
        self.assertEqual(config.tts.startup_timeout_s, 1.0)
        self.assertFalse(config.tts.style.leading_emoji_only)
        self.assertEqual(config.voxcpm2.python, "/tmp/vox-python")
        self.assertEqual(config.voxcpm2.model_path, "/tmp/vox-model")
        self.assertFalse(config.voxcpm2.optimize)
        self.assertTrue(config.voxcpm2.load_denoiser)

    def test_file_render_config_is_merged_and_normalized(self):
        config = SpeakerConfig().merge_dict(
            {
                "file_render": {
                    "enabled": "false",
                    "output_dir": " custom/tts ",
                    "max_text_chars": 0,
                    "max_pending_jobs": 999,
                    "max_completed_jobs": 0,
                    "segment_chars": 5,
                }
            }
        )

        self.assertFalse(config.file_render.enabled)
        self.assertEqual(config.file_render.output_dir, "custom/tts")
        self.assertEqual(config.file_render.max_text_chars, 1)
        self.assertEqual(config.file_render.max_pending_jobs, 128)
        self.assertEqual(config.file_render.max_completed_jobs, 1)
        self.assertEqual(config.file_render.segment_chars, 40)

    def test_rejects_unknown_tts_backend(self):
        with self.assertRaises(ValueError):
            SpeakerConfig().merge_dict({"tts": {"backend": "unknown"}})

    def test_loads_project_config_by_default(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text('{"speaker": {"mode": "final"}}', encoding="utf-8")

            with patch("speaker.config.DEFAULT_CONFIG_PATH", config_path):
                config = SpeakerConfig.load()

        self.assertEqual(config.speaker.mode, "final")

    def test_loads_config_with_utf8_bom(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                '\ufeff{"speaker": {"mode": "final"}}',
                encoding="utf-8",
            )

            config = SpeakerConfig.load(str(config_path))

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
                        "SPEAKER_PLAYBACK_BACKEND": "paplay",
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
        self.assertEqual(config.playback.backend, "paplay")
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

    def test_streaming_playback_values_are_normalized(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                '{"playback": {"streaming_backend": "PACAT", "prebuffer_ms": -1, '
                '"latency_ms": 2, "queue_ms": 1, "restart_attempts": -2, '
                '"write_timeout_s": 0}}',
                encoding="utf-8",
            )

            with patch("speaker.config.DEFAULT_CONFIG_PATH", config_path):
                config = SpeakerConfig.load()

        self.assertEqual(config.playback.streaming_backend, "pacat")
        self.assertEqual(config.playback.prebuffer_ms, 0)
        self.assertEqual(config.playback.latency_ms, 10)
        self.assertEqual(config.playback.queue_ms, 20)
        self.assertEqual(config.playback.restart_attempts, 0)
        self.assertEqual(config.playback.write_timeout_s, 0.1)

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

    def test_emoji_display_config_is_normalized(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                """
                {
                  "emoji_display": {
                    "enabled": "yes",
                    "url": "http://127.0.0.1:18791/",
                    "token": " secret ",
                    "timeout_s": 0,
                    "hold_ms": -5,
                    "mode": "QUEUE",
                    "source": " test ",
                    "send": "FIRST",
                    "clear_on_interrupt": "false"
                  }
                }
                """,
                encoding="utf-8",
            )

            with patch("speaker.config.DEFAULT_CONFIG_PATH", config_path):
                config = SpeakerConfig.load()

        self.assertTrue(config.emoji_display.enabled)
        self.assertEqual(config.emoji_display.url, "http://127.0.0.1:18791")
        self.assertEqual(config.emoji_display.token, "secret")
        self.assertEqual(config.emoji_display.timeout_s, 0.05)
        self.assertEqual(config.emoji_display.hold_ms, 0)
        self.assertEqual(config.emoji_display.mode, "queue")
        self.assertEqual(config.emoji_display.source, "test")
        self.assertEqual(config.emoji_display.send, "first")
        self.assertFalse(config.emoji_display.clear_on_interrupt)

    def test_redacts_emoji_display_token(self):
        config = SpeakerConfig()
        config.emoji_display.token = "secret"

        self.assertEqual(config.to_redacted_dict()["emoji_display"]["token"], "<redacted>")


if __name__ == "__main__":
    unittest.main()
