#!/usr/bin/env python3
"""Dependency-free worker used to validate framing, streaming and cancellation."""

from __future__ import annotations

import argparse
import collections
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from speaker.neural_protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    ProtocolError,
    encode_audio,
    encode_metadata,
    read_command,
)


_EOF = object()


class _CommandReader(threading.Thread):
    def __init__(self, output: queue.Queue[object]) -> None:
        super().__init__(name="fake-tts-command-reader", daemon=True)
        self.output = output

    def run(self) -> None:
        try:
            while True:
                command = read_command(sys.stdin.buffer)
                if command is None:
                    self.output.put(_EOF)
                    return
                self.output.put(command)
        except Exception as exc:  # noqa: BLE001 - surfaced over the protocol
            self.output.put({"command": "__reader_error__", "error": str(exc)})


class FakeWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.commands: queue.Queue[object] = queue.Queue()
        self.pending: collections.deque[dict[str, Any]] = collections.deque()
        self.shutting_down = False

    def run(self) -> int:
        _CommandReader(self.commands).start()
        if self.args.startup_delay_ms:
            time.sleep(self.args.startup_delay_ms / 1000.0)
        self._metadata(
            event="ready",
            protocol_version=PROTOCOL_VERSION,
            backend="fake",
            sample_rate=self.args.sample_rate,
        )
        if self.args.exit_after_ready:
            return 3

        while not self.shutting_down:
            command = self._next_command()
            if command is None:
                break
            name = str(command.get("command") or "")
            if name == "generate":
                self._generate(command)
            elif name == "cancel":
                # Late cancels are already satisfied by the preceding terminal
                # frame.  Emitting another frame would poison the next request.
                continue
            elif name == "health":
                self._metadata(event="health", ok=True, backend="fake")
            elif name == "shutdown":
                self._metadata(event="shutdown", ok=True)
                self.shutting_down = True
            elif name == "__reader_error__":
                self._metadata(event="error", error=str(command.get("error") or "reader failed"))
                return 2
            else:
                self._metadata(event="error", error=f"unknown command: {name}")
        return 0

    def _generate(self, command: dict[str, Any]) -> None:
        request_id = str(command.get("request_id") or "")
        if not request_id:
            self._metadata(event="error", request_id="", error="missing request_id")
            return
        if self.args.fail_generate:
            self._metadata(event="error", request_id=request_id, error="requested fake failure")
            return

        self._metadata(
            event="start",
            request_id=request_id,
            sample_rate=self.args.sample_rate,
            channels=1,
            sample_width=2,
            encoding="pcm_s16le",
        )
        samples = max(1, self.args.sample_rate * self.args.chunk_ms // 1000)
        pcm = b"\x00\x00" * samples
        cancelled = False
        for sequence in range(self.args.chunks):
            cancelled = self._poll_control(request_id)
            if cancelled or self.shutting_down:
                break
            if self.args.chunk_delay_ms:
                time.sleep(self.args.chunk_delay_ms / 1000.0)
            cancelled = self._poll_control(request_id)
            if cancelled or self.shutting_down:
                break
            sys.stdout.buffer.write(encode_audio(pcm))
            sys.stdout.buffer.flush()
            print(
                f"fake worker emitted request={request_id} sequence={sequence}",
                file=sys.stderr,
                flush=True,
            )

        if cancelled:
            self._metadata(event="cancelled", request_id=request_id)
        elif not self.shutting_down:
            self._metadata(event="done", request_id=request_id)

    def _poll_control(self, request_id: str) -> bool:
        cancelled = False
        while True:
            try:
                raw = self.commands.get_nowait()
            except queue.Empty:
                return cancelled
            if raw is _EOF:
                self.shutting_down = True
                return True
            command = raw if isinstance(raw, dict) else {}
            name = str(command.get("command") or "")
            if name == "cancel" and str(command.get("request_id") or "") == request_id:
                cancelled = True
            elif name == "health":
                self._metadata(event="health", ok=True, backend="fake")
            elif name == "shutdown":
                self.shutting_down = True
                return True
            else:
                self.pending.append(command)

    def _next_command(self) -> dict[str, Any] | None:
        if self.pending:
            return self.pending.popleft()
        raw = self.commands.get()
        if raw is _EOF:
            return None
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _metadata(**fields: Any) -> None:
        sys.stdout.buffer.write(encode_metadata(fields))
        sys.stdout.buffer.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--chunk-ms", type=int, default=20)
    parser.add_argument("--chunk-delay-ms", type=int, default=5)
    parser.add_argument("--startup-delay-ms", type=int, default=0)
    parser.add_argument("--fail-generate", action="store_true")
    parser.add_argument("--exit-after-ready", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        return FakeWorker(build_parser().parse_args(argv)).run()
    except (BrokenPipeError, ProtocolError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
