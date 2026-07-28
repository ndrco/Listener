"""Small runtime helpers shared by model-specific worker entry points."""

from __future__ import annotations

import collections
import os
import queue
import sys
import threading
from typing import Any

from speaker.neural_protocol import encode_audio, encode_metadata, read_command


_EOF = object()


class _CommandReader(threading.Thread):
    def __init__(self, output: queue.Queue[object]) -> None:
        super().__init__(name="tts-command-reader", daemon=True)
        self.output = output

    def run(self) -> None:
        try:
            while True:
                command = read_command(sys.stdin.buffer)
                if command is None:
                    self.output.put(_EOF)
                    return
                self.output.put(command)
        except Exception as exc:  # noqa: BLE001 - consumed by the main worker thread
            self.output.put({"command": "__reader_error__", "error": str(exc)})


class WorkerRuntime:
    """Keep protocol output isolated while libraries log to stdout/stderr."""

    def __init__(self, *, redirect_stdout: bool = True) -> None:
        if redirect_stdout:
            protocol_fd = os.dup(sys.stdout.fileno())
            self.output = os.fdopen(protocol_fd, "wb", buffering=0)
            os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
            sys.stdout = sys.stderr
        else:
            self.output = sys.stdout.buffer
        self.commands: queue.Queue[object] = queue.Queue()
        self.pending: collections.deque[dict[str, Any]] = collections.deque()
        self.shutting_down = False

    def start(self) -> None:
        _CommandReader(self.commands).start()

    def metadata(self, **fields: Any) -> None:
        self.output.write(encode_metadata(fields))
        self.output.flush()

    def audio(self, pcm: bytes) -> None:
        self.output.write(encode_audio(pcm))
        self.output.flush()

    def next_command(self) -> dict[str, Any] | None:
        if self.pending:
            return self.pending.popleft()
        raw = self.commands.get()
        if raw is _EOF:
            return None
        return raw if isinstance(raw, dict) else {}

    def poll_control(self, request_id: str) -> bool:
        """Return True when active generation should be cancelled."""
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
                self.metadata(event="health", ok=True)
            elif name == "shutdown":
                self.shutting_down = True
                return True
            elif name == "__reader_error__":
                self.metadata(event="error", request_id=request_id, error=command.get("error"))
                self.shutting_down = True
                return True
            else:
                self.pending.append(command)


__all__ = ["WorkerRuntime"]
