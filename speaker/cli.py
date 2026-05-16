from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .app import SpeakerService
from .config import SpeakerConfig
from .messages import clean_for_speech
from .tts import PiperSpeechEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speaker")
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Run the OpenClaw speaker sidecar")
    sub.add_parser("print-config", help="Print effective config with secrets redacted")

    say = sub.add_parser("say", help="Speak one text string through Piper")
    say.add_argument("text", help="Text to synthesize and play")
    return parser


def configure_logging(level: str) -> None:
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    config = SpeakerConfig.load(args.config)

    if args.command == "print-config":
        print(json.dumps(config.to_redacted_dict(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "say":
        engine = PiperSpeechEngine(config.piper, config.playback)
        asyncio.run(engine.speak(clean_for_speech(args.text)))
        return 0
    if args.command == "run":
        try:
            asyncio.run(SpeakerService(config).run_forever())
        except KeyboardInterrupt:
            return 130
        return 0

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
