"""Async client for a persistent TTS process running in an isolated environment."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from typing import AsyncIterator, Mapping, Sequence

from .neural_protocol import (
    FrameKind,
    MAX_AUDIO_BYTES,
    MAX_METADATA_BYTES,
    OutputFrame,
    PROTOCOL_VERSION,
    ProtocolError,
    decode_output_header,
    encode_command,
)
from .text_normalization import TextNormalizer
from .tts import SpeechRequest, normalize_speech_request


log = logging.getLogger(__name__)
_OUTPUT_HEADER_SIZE = 5


class NeuralWorkerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AudioChunk:
    request_id: str
    sequence: int
    pcm: bytes
    sample_rate: int
    channels: int = 1
    sample_width: int = 2


class NeuralWorkerClient:
    def __init__(
        self,
        command: Sequence[str],
        *,
        startup_timeout_s: float = 90.0,
        generation_timeout_s: float = 120.0,
        cancel_timeout_s: float = 1.0,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        text_normalizer: TextNormalizer | None = None,
    ) -> None:
        if not command:
            raise ValueError("worker command must not be empty")
        self.command = tuple(str(part) for part in command)
        self.startup_timeout_s = max(0.05, float(startup_timeout_s))
        self.generation_timeout_s = max(0.05, float(generation_timeout_s))
        self.cancel_timeout_s = max(0.05, float(cancel_timeout_s))
        self.env = dict(env) if env is not None else None
        self.cwd = cwd
        self.text_normalizer = text_normalizer
        self._proc: asyncio.subprocess.Process | None = None
        self._start_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._generation_lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._active_request_id: str | None = None
        self._active_owner: str | None = None
        self._active_cancelled: asyncio.Event | None = None
        self._active_cancel_command_sent = False
        self._ready_metadata: dict | None = None
        self._last_error = ""
        self._started_at: float | None = None

    async def start(self) -> dict:
        async with self._start_lock:
            if self._is_running() and self._ready_metadata is not None:
                return dict(self._ready_metadata)
            await self._close_process(send_shutdown=False)
            child_env = None
            if self.env is not None:
                child_env = os.environ.copy()
                child_env.update(self.env)
            self._proc = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=child_env,
                cwd=self.cwd,
            )
            self._stderr_task = asyncio.create_task(
                self._drain_stderr(), name="Speaker.neural_worker.stderr"
            )
            try:
                frame = await asyncio.wait_for(
                    self._read_frame(), timeout=self.startup_timeout_s
                )
                metadata = _metadata_frame(frame)
                if metadata.get("event") == "error":
                    raise NeuralWorkerError(str(metadata.get("error") or "worker startup failed"))
                if metadata.get("event") != "ready":
                    raise NeuralWorkerError(f"worker returned invalid startup event: {metadata!r}")
                version = int(metadata.get("protocol_version") or 0)
                if version != PROTOCOL_VERSION:
                    raise NeuralWorkerError(
                        f"worker protocol version {version} does not match {PROTOCOL_VERSION}"
                    )
            except Exception as exc:
                self._last_error = str(exc)
                await self._close_process(send_shutdown=False)
                raise NeuralWorkerError(self._format_error(f"worker failed to start: {exc}")) from exc
            self._ready_metadata = metadata
            self._started_at = time.time()
            self._last_error = ""
            return dict(metadata)

    async def generate(
        self,
        request: SpeechRequest,
        *,
        owner: str = "default",
        cancellation_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AudioChunk]:
        speech_request = normalize_speech_request(
            SpeechRequest.coerce(request),
            self.text_normalizer,
        )
        if not speech_request.text:
            return
        generation_owner = str(owner or "default").strip() or "default"
        async with self._generation_lock:
            if cancellation_event is not None and cancellation_event.is_set():
                return
            await self.start()
            if cancellation_event is not None and cancellation_event.is_set():
                return
            request_id = uuid.uuid4().hex
            cancel_event = asyncio.Event()
            self._active_request_id = request_id
            self._active_owner = generation_owner
            self._active_cancelled = cancel_event
            self._active_cancel_command_sent = False
            terminal_received = False
            sequence = 0
            sample_rate = 0
            channels = 1
            sample_width = 2
            deadline = asyncio.get_running_loop().time() + self.generation_timeout_s
            try:
                await self._send(
                    {
                        "command": "generate",
                        "request_id": request_id,
                        "request": asdict(speech_request),
                    }
                )
                if (
                    cancellation_event is not None
                    and cancellation_event.is_set()
                    and not self._active_cancel_command_sent
                ):
                    self._active_cancel_command_sent = True
                    await self._send({"command": "cancel", "request_id": request_id})
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    frame = await asyncio.wait_for(self._read_frame(), timeout=remaining)
                    if frame.kind is FrameKind.AUDIO:
                        if sample_rate <= 0:
                            raise NeuralWorkerError("worker sent audio before start metadata")
                        yield AudioChunk(
                            request_id=request_id,
                            sequence=sequence,
                            pcm=frame.payload,
                            sample_rate=sample_rate,
                            channels=channels,
                            sample_width=sample_width,
                        )
                        sequence += 1
                        continue

                    metadata = frame.metadata()
                    event = str(metadata.get("event") or "")
                    event_request_id = str(metadata.get("request_id") or "")
                    if event_request_id and event_request_id != request_id:
                        raise NeuralWorkerError(
                            f"worker response belongs to unexpected request {event_request_id}"
                        )
                    if event == "start":
                        sample_rate = int(metadata.get("sample_rate") or 0)
                        channels = int(metadata.get("channels") or 1)
                        sample_width = int(metadata.get("sample_width") or 2)
                        if sample_rate <= 0 or channels != 1 or sample_width != 2:
                            raise NeuralWorkerError(f"unsupported audio format: {metadata!r}")
                    elif event == "done":
                        terminal_received = True
                        cancel_event.set()
                        self._last_error = ""
                        return
                    elif event == "cancelled":
                        terminal_received = True
                        cancel_event.set()
                        self._last_error = ""
                        return
                    elif event == "error":
                        raise NeuralWorkerError(str(metadata.get("error") or "worker failed"))
                    elif event == "health":
                        continue
                    else:
                        raise NeuralWorkerError(f"unknown worker event: {metadata!r}")
            except asyncio.TimeoutError as exc:
                self._last_error = f"generation timed out after {self.generation_timeout_s:.2f}s"
                await self._close_process(send_shutdown=False)
                raise NeuralWorkerError(self._last_error) from exc
            except asyncio.CancelledError:
                await self._close_process(send_shutdown=False)
                raise
            except Exception as exc:
                self._last_error = str(exc)
                raise
            finally:
                # A cancel acknowledgement is only complete after the terminal
                # frame has been consumed.  Waking cancel() merely because the
                # consumer stopped would leave audio/metadata from this request
                # in stdout, corrupting the next request on the persistent pipe.
                if terminal_received:
                    cancel_event.set()
                self._active_request_id = None
                self._active_owner = None
                self._active_cancelled = None
                self._active_cancel_command_sent = False

    async def cancel(self, *, owner: str | None = None) -> bool:
        request_id = self._active_request_id
        active_owner = self._active_owner
        event = self._active_cancelled
        if not request_id or event is None or not self._is_running():
            return False
        if owner is not None and str(owner) != active_owner:
            return False
        if not self._active_cancel_command_sent:
            self._active_cancel_command_sent = True
            await self._send({"command": "cancel", "request_id": request_id})
        try:
            await asyncio.wait_for(event.wait(), timeout=self.cancel_timeout_s)
        except asyncio.TimeoutError as exc:
            self._last_error = f"worker cancel timed out after {self.cancel_timeout_s:.2f}s"
            await self._close_process(send_shutdown=False)
            raise NeuralWorkerError(self._last_error) from exc
        return True

    async def health(self) -> dict:
        async with self._generation_lock:
            await self.start()
            await self._send({"command": "health"})
            frame = await asyncio.wait_for(self._read_frame(), timeout=self.cancel_timeout_s)
            metadata = _metadata_frame(frame)
            if metadata.get("event") != "health":
                raise NeuralWorkerError(f"worker returned invalid health event: {metadata!r}")
            return metadata

    async def close(self) -> None:
        await self._close_process(send_shutdown=True)

    def get_status(self) -> dict:
        proc = self._proc
        return {
            "running": self._is_running(),
            "pid": proc.pid if proc is not None and proc.returncode is None else None,
            "backend": (self._ready_metadata or {}).get("backend"),
            "active_request_id": self._active_request_id,
            "active_owner": self._active_owner,
            "started_at": self._started_at,
            "last_error": self._last_error or None,
            "stderr_tail": list(self._stderr_tail),
            "text_normalization": self.text_normalizer is not None,
        }

    async def _send(self, command: dict) -> None:
        async with self._write_lock:
            proc = self._proc
            if proc is None or proc.returncode is not None or proc.stdin is None:
                raise NeuralWorkerError("worker is not running")
            try:
                proc.stdin.write(encode_command(command))
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise NeuralWorkerError(self._format_error("worker input pipe closed")) from exc

    async def _read_frame(self) -> OutputFrame:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise NeuralWorkerError("worker is not running")
        try:
            header = await proc.stdout.readexactly(_OUTPUT_HEADER_SIZE)
            kind, length = decode_output_header(header)
            limit = MAX_METADATA_BYTES if kind is FrameKind.METADATA else MAX_AUDIO_BYTES
            if length > limit:
                raise ProtocolError(f"{kind.name.lower()} frame exceeds {limit} bytes")
            payload = await proc.stdout.readexactly(length)
        except asyncio.IncompleteReadError as exc:
            raise NeuralWorkerError(self._format_error("worker output pipe closed")) from exc
        return OutputFrame(kind=kind, payload=payload)

    async def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while True:
            line = await proc.stderr.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                self._stderr_tail.append(text)
                log.debug("Neural TTS worker: %s", text)

    async def _close_process(self, *, send_shutdown: bool) -> None:
        proc = self._proc
        self._proc = None
        self._ready_metadata = None
        self._active_request_id = None
        self._active_owner = None
        self._active_cancel_command_sent = False
        event = self._active_cancelled
        self._active_cancelled = None
        if event is not None:
            event.set()
        if proc is not None:
            if send_shutdown and proc.returncode is None and proc.stdin is not None:
                with contextlib.suppress(Exception):
                    proc.stdin.write(encode_command({"command": "shutdown"}))
                    await proc.stdin.drain()
            if proc.stdin is not None:
                with contextlib.suppress(Exception):
                    proc.stdin.close()
                    await proc.stdin.wait_closed()
            if proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        with contextlib.suppress(ProcessLookupError):
                            proc.kill()
                        with contextlib.suppress(Exception):
                            await proc.wait()
        stderr_task = self._stderr_task
        self._stderr_task = None
        if stderr_task is not None:
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task

    def _is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def _format_error(self, message: str) -> str:
        if not self._stderr_tail:
            return message
        return f"{message}: {self._stderr_tail[-1]}"


def _metadata_frame(frame: OutputFrame) -> dict:
    if frame.kind is not FrameKind.METADATA:
        raise NeuralWorkerError("worker sent audio where metadata was expected")
    return frame.metadata()


__all__ = ["AudioChunk", "NeuralWorkerClient", "NeuralWorkerError"]
