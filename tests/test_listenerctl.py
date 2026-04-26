from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from utils.listenerctl import build_parser, build_set_mode_payload, format_speech_gate_status  # noqa: E402


def test_listenerctl_set_mode_payload_shape():
    parser = build_parser()
    args = parser.parse_args(
        [
            "speech-gate",
            "set-mode",
            "chatty",
            "--ttl",
            "600",
            "--reason",
            "conversation mode",
            "--source",
            "openclaw",
        ]
    )

    assert build_set_mode_payload(args) == {
        "mode": "chatty",
        "ttl_seconds": 600.0,
        "source": "openclaw",
        "reason": "conversation mode",
    }


def test_listenerctl_status_json_parse():
    parser = build_parser()
    args = parser.parse_args(["speech-gate", "status", "--json"])

    assert args.resource == "speech-gate"
    assert args.action == "status"
    assert args.json is True


def test_listenerctl_status_shortcut_parse():
    parser = build_parser()
    args = parser.parse_args(["status", "--json"])

    assert args.resource == "speech-gate"
    assert args.action == "status"
    assert args.json is True


def test_listenerctl_mode_shortcut_payload_shape():
    parser = build_parser()
    args = parser.parse_args(
        [
            "chatty",
            "--ttl",
            "600",
            "--reason",
            "conversation mode",
            "--source",
            "openclaw",
        ]
    )

    assert args.resource == "speech-gate"
    assert args.action == "set-mode"
    assert build_set_mode_payload(args) == {
        "mode": "chatty",
        "ttl_seconds": 600.0,
        "source": "openclaw",
        "reason": "conversation mode",
    }


def test_listenerctl_formats_permanent_status():
    assert format_speech_gate_status(
        {
            "mode": "mute",
            "temporary": False,
            "source": "openclaw",
        }
    ) == "speech_gate mode=mute state=permanent expires_in=- expires_at=- restore=- source=openclaw"


def test_listenerctl_formats_temporary_status():
    text = format_speech_gate_status(
        {
            "mode": "standby",
            "temporary": True,
            "expires_in_seconds": 299.94,
            "expires_at": 1800000000,
            "restore_mode": "normal",
            "source": "openclaw",
            "reason": "quiet, please",
        }
    )

    assert text.startswith("speech_gate mode=standby state=temporary expires_in=299.9s ")
    assert "expires_at=" in text
    assert "restore=normal" in text
    assert "source=openclaw" in text
    assert 'reason="quiet, please"' in text
