from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from utils.listenerctl import (  # noqa: E402
    build_parser,
    build_set_mode_payload,
    format_ready_status,
    format_speaker_status,
    format_speech_gate_status,
)


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


def test_listenerctl_speaker_commands_parse():
    parser = build_parser()
    status = parser.parse_args(["speaker", "status", "--json"])
    off = parser.parse_args(["speaker", "off", "--reason", "quiet"])
    alias = parser.parse_args(["voice-on", "--source", "openclaw"])

    assert status.resource == "speaker"
    assert status.action == "status"
    assert status.json is True
    assert off.resource == "speaker"
    assert off.action == "off"
    assert off.enabled is False
    assert off.reason == "quiet"
    assert alias.resource == "speaker"
    assert alias.action == "enabled"
    assert alias.enabled is True
    assert alias.source == "openclaw"


def test_listenerctl_service_commands_parse():
    parser = build_parser()
    health = parser.parse_args(["health", "--json"])
    ready = parser.parse_args(["ready"])
    stop = parser.parse_args(["stop", "--reason", "systemd"])

    assert health.resource == "service"
    assert health.action == "health"
    assert health.json is True
    assert ready.resource == "service"
    assert ready.action == "ready"
    assert stop.resource == "service"
    assert stop.action == "stop"
    assert stop.reason == "systemd"


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


def test_listenerctl_formats_speaker_status():
    text = format_speaker_status(
        {
            "enabled": False,
            "running": True,
            "connected": False,
            "mode": "streaming",
            "session_key": "main",
            "playback": {
                "queue_size": 2,
                "current": "seg-1",
                "last_interrupt_reason": "voice command",
            },
        }
    )

    assert text.startswith("speaker=off agent=running gateway=disconnected mode=streaming")
    assert "queue=2" in text
    assert "current=seg-1" in text
    assert 'last_interrupt="voice command"' in text


def test_listenerctl_formats_ready_status():
    text = format_ready_status(
        {
            "ready": False,
            "components": {
                "audio": {
                    "state": "failed",
                    "ok": False,
                    "critical": True,
                },
                "speaker": {
                    "state": "started",
                    "ok": True,
                    "critical": False,
                },
            },
            "last_error": "failed to start AudioAgent",
        }
    )

    assert text.startswith("listener=not_ready")
    assert "audio=failed!" in text
    assert "speaker=started" in text
    assert 'last_error="failed to start AudioAgent"' in text
