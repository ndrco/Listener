from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from utils.listenerctl import build_parser, build_set_mode_payload  # noqa: E402


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
