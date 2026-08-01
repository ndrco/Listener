import json
import sys
from unittest.mock import patch

from speaker.cli import main
from speaker.config import SpeakerConfig
from speaker.tts import SpeechRequest


class _SpeechEngine:
    def __init__(self) -> None:
        self.requests: list[SpeechRequest] = []
        self.closed = False

    async def speak(self, request: SpeechRequest) -> None:
        self.requests.append(request)

    async def close(self) -> None:
        self.closed = True


def test_print_config_reports_shared_listener_python(capsys):
    config = SpeakerConfig()
    config.gateway.token = "secret"

    with patch("speaker.cli.SpeakerConfig.load", return_value=config):
        result = main(["print-config"])

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["gateway"]["token"] == "<redacted>"
    assert output["voxcpm2"]["python"] == sys.executable
    assert output["cosyvoice3"]["python"] == sys.executable


def test_say_resolves_leading_emoji_style_for_configured_backend():
    engine = _SpeechEngine()
    with (
        patch("speaker.cli.SpeakerConfig.load", return_value=SpeakerConfig()),
        patch("speaker.cli.create_speech_engine", return_value=engine),
    ):
        result = main(["say", "😔 Иногда тишина всё понимает."])

    assert result == 0
    assert engine.closed is True
    assert len(engine.requests) == 1
    assert engine.requests[0].text == "Иногда тишина всё понимает."
    assert engine.requests[0].style_id == "sad"
