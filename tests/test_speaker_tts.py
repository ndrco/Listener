import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import audio.ducking as ducking_module
from speaker.config import PiperConfig
from speaker.tts import (
    PulseAudioDucker,
    build_piper_args,
    build_playback_args,
    build_ducking_steps,
    build_stream_restore_id,
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
                "--property=state.restore-props=false",
                "--property=state.restore-target=false",
                "--property=module-stream-restore.id="
                f"{build_stream_restore_id(Path('/tmp/out.wav'))}",
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
            [
                SinkInputVolume(
                    sink_input_id=22,
                    volumes=[65536],
                    channel_names=["mono"],
                    application_name="Music",
                )
            ],
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

    def test_nested_ducking_restores_original_volume(self):
        class DuckerConfig:
            enabled = True
            fade_in_ms = 0
            fade_out_ms = 0

            def __init__(self, volume_scale):
                self.volume_scale = volume_scale

        async def _runner():
            state = {"volumes": [100]}

            async def fake_list_sink_inputs(*, exclude_speaker=True):
                return [SinkInputVolume(sink_input_id=7, volumes=list(state["volumes"]))]

            async def fake_run_pactl(*args):
                self.assertEqual(args[0], "set-sink-input-volume")
                self.assertEqual(args[1], "7")
                state["volumes"] = [int(value) for value in args[2:]]

            ducking_module._ACTIVE_DUCKERS.clear()
            ducking_module._DUCKING_ACTIVE_SCALES.clear()
            ducking_module._DUCKING_ORIGINALS.clear()
            with (
                patch.object(ducking_module, "_list_sink_inputs", fake_list_sink_inputs),
                patch.object(ducking_module, "_run_pactl", fake_run_pactl),
            ):
                first = PulseAudioDucker(DuckerConfig(0.6), exclude_speaker=False)
                second = PulseAudioDucker(DuckerConfig(0.35), exclude_speaker=False)

                await first.duck()
                self.assertEqual(state["volumes"], [60])
                await second.duck()
                self.assertEqual(state["volumes"], [35])
                await first.restore()
                self.assertEqual(state["volumes"], [35])
                await second.restore()
                self.assertEqual(state["volumes"], [100])

            self.assertEqual(ducking_module._DUCKING_ACTIVE_SCALES, {})
            self.assertEqual(ducking_module._DUCKING_ORIGINALS, {})

        asyncio.run(_runner())

    def test_ducking_restores_wireplumber_route_settings_when_stream_disappears(self):
        class DuckerConfig:
            enabled = True
            fade_in_ms = 0
            fade_out_ms = 0
            volume_scale = 0.35

        async def _runner():
            original = SinkInputVolume(
                sink_input_id=42,
                volumes=[65536, 65536],
                channel_names=["front-left", "front-right"],
                application_name="Google Chrome",
            )
            state = {"current": [original]}
            metadata_calls: list[tuple[str, ...]] = []

            async def fake_list_sink_inputs(*, exclude_speaker=True):
                return list(state["current"])

            async def fake_run_pactl(*args):
                state["current"] = [
                    SinkInputVolume(
                        sink_input_id=42,
                        volumes=[int(value) for value in args[2:]],
                        channel_names=["front-left", "front-right"],
                        application_name="Google Chrome",
                    )
                ]

            async def fake_run_pw_metadata(*args):
                metadata_calls.append(tuple(args))

            ducking_module._ACTIVE_DUCKERS.clear()
            ducking_module._DUCKING_ACTIVE_SCALES.clear()
            ducking_module._DUCKING_ORIGINALS.clear()
            with (
                patch.object(ducking_module, "_list_sink_inputs", fake_list_sink_inputs),
                patch.object(ducking_module, "_run_pactl", fake_run_pactl),
                patch.object(ducking_module, "_run_pw_metadata", fake_run_pw_metadata),
            ):
                ducker = PulseAudioDucker(DuckerConfig(), exclude_speaker=False)

                await ducker.duck()
                self.assertEqual(state["current"][0].volumes, [22938, 22938])
                state["current"] = []
                await ducker.restore()

            self.assertTrue(metadata_calls)
            self.assertEqual(
                metadata_calls[-1][1],
                "restore.stream.Output/Audio.application.name:Google Chrome",
            )
            self.assertIn('"volumes":[1.0,1.0]', metadata_calls[-1][2])
            self.assertIn('"channels":["FL","FR"]', metadata_calls[-1][2])

        asyncio.run(_runner())


if __name__ == "__main__":
    unittest.main()
