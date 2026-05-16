import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from speaker.config import PiperConfig
from speaker.tts import (
    build_piper_args,
    build_playback_args,
    build_ducking_steps,
    parse_sink_input_volumes,
    resolve_piper_command,
    SinkInputVolume,
    split_complete_speech_units,
    split_speech_units,
)


class SpeechUnitTests(unittest.TestCase):
    def test_split_speech_units_keeps_trailing_emoji_separate(self):
        text = "Сисадмин в раю просит доступ root. Ему говорят: «Зачем?» Он: «Просто посмотреть». 😼"

        self.assertEqual(
            split_speech_units(text),
            [
                "Сисадмин в раю просит доступ root.",
                "Ему говорят: «Зачем?»",
                "Он: «Просто посмотреть».",
                "😼",
            ],
        )

    def test_split_speech_units_collapses_whitespace(self):
        self.assertEqual(split_speech_units("  One.\n\nTwo  "), ["One.", "Two"])

    def test_split_complete_speech_units_excludes_incomplete_tail(self):
        self.assertEqual(split_complete_speech_units("One. Two"), ["One."])

    def test_split_speech_units_keeps_ellipsis_with_sentence(self):
        text = "Ангелы говорят: «Не трогай ничего важного...» 😼"

        self.assertEqual(
            split_speech_units(text),
            ["Ангелы говорят: «Не трогай ничего важного...»", "😼"],
        )

    def test_split_speech_units_skips_punctuation_only(self):
        self.assertEqual(split_speech_units(". .» 😼"), ["😼"])

    def test_resolve_piper_command_prefers_venv_python_for_entrypoint(self):
        with TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / ".venv" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "python3").touch()
            (bin_dir / "piper").touch()

            self.assertEqual(
                resolve_piper_command(str(bin_dir / "piper")),
                [str(bin_dir / "python3"), "-m", "piper"],
            )

    def test_resolve_piper_command_uses_python_module_mode(self):
        self.assertEqual(
            resolve_piper_command("/tmp/venv/bin/python3"),
            ["/tmp/venv/bin/python3", "-m", "piper"],
        )

    def test_build_piper_args_includes_volume(self):
        config = PiperConfig(
            command="/tmp/venv/bin/python3",
            model="/tmp/model.onnx",
            volume=0.7,
            sentence_silence=0.4,
            extra_args=["--noise-scale", "0.5"],
        )

        self.assertEqual(
            build_piper_args(config, Path("/tmp/out.wav")),
            [
                "/tmp/venv/bin/python3",
                "-m",
                "piper",
                "--model",
                "/tmp/model.onnx",
                "--output-file",
                "/tmp/out.wav",
                "--sentence-silence",
                "0.4",
                "--volume",
                "0.7",
                "--noise-scale",
                "0.5",
            ],
        )

    def test_build_playback_args_sets_speaker_properties(self):
        class PlaybackConfigStub:
            command = "/usr/bin/paplay"
            client_name = "Speaker"
            stream_name = "Speaker TTS"

        self.assertEqual(
            build_playback_args(PlaybackConfigStub(), Path("/tmp/out.wav")),
            [
                "/usr/bin/paplay",
                "--client-name",
                "Speaker",
                "--stream-name",
                "Speaker TTS",
                "--property=application.id=speaker",
                "--property=media.role=a11y",
                "/tmp/out.wav",
            ],
        )

    def test_parse_sink_input_volumes_extracts_channel_values(self):
        payload = [
            {
                "index": 12,
                "volume": {
                    "front-left": {"value": 65536},
                    "front-right": {"value": 60000},
                },
            },
            {"index": 13, "volume": {}},
            {"index": "bad", "volume": {"front-left": {"value": 1}}},
        ]

        snapshot = parse_sink_input_volumes(payload)

        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0].sink_input_id, 12)
        self.assertEqual(snapshot[0].volumes, [65536, 60000])

    def test_parse_sink_input_volumes_skips_speaker_streams(self):
        payload = [
            {
                "index": 20,
                "volume": {"mono": {"value": 16384}},
                "properties": {
                    "application.id": "speaker",
                    "application.name": "Speaker",
                    "media.name": "Speaker TTS",
                },
            },
            {
                "index": 21,
                "volume": {"mono": {"value": 65536}},
                "properties": {
                    "application.name": "Speaker",
                    "media.name": "Speaker TTS",
                },
            },
            {
                "index": 22,
                "volume": {"mono": {"value": 65536}},
                "properties": {"application.name": "Music"},
            },
        ]

        snapshot = parse_sink_input_volumes(payload)

        self.assertEqual(
            snapshot,
            [SinkInputVolume(sink_input_id=22, volumes=[65536], application_name="Music")],
        )

    def test_build_ducking_steps_ramps_down_to_target_scale(self):
        snapshot = [SinkInputVolume(sink_input_id=7, volumes=[100, 80])]

        steps = build_ducking_steps(snapshot, 0.5, 20, step_ms=20)

        self.assertEqual(
            steps,
            [
                [SinkInputVolume(sink_input_id=7, volumes=[75, 60])],
                [SinkInputVolume(sink_input_id=7, volumes=[50, 40])],
            ],
        )

    def test_build_ducking_steps_restore_returns_to_original_volume(self):
        snapshot = [SinkInputVolume(sink_input_id=7, volumes=[100, 80])]

        steps = build_ducking_steps(snapshot, 0.5, 60, restore=True, step_ms=20)

        self.assertEqual(steps[0], [SinkInputVolume(sink_input_id=7, volumes=[62, 50])])
        self.assertEqual(steps[-1], [SinkInputVolume(sink_input_id=7, volumes=[100, 80])])

    def test_build_ducking_steps_without_duration_jumps_to_final_state(self):
        snapshot = [SinkInputVolume(sink_input_id=9, volumes=[120])]

        self.assertEqual(
            build_ducking_steps(snapshot, 0.25, 0),
            [[SinkInputVolume(sink_input_id=9, volumes=[30])]],
        )


if __name__ == "__main__":
    unittest.main()
