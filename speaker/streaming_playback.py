"""Crash-isolated streaming playback for neural TTS PCM chunks."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import shutil
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from audio.ducking import normalize_listener_output_volume_state

try:  # pragma: no cover - availability depends on the host audio stack
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None  # type: ignore[assignment]


log = logging.getLogger(__name__)

_PCM_BLOCK_MS = 20
_STDERR_TAIL_BYTES = 8192


class StreamingPlaybackError(RuntimeError):
    pass


class StreamingPlaybackInterrupted(StreamingPlaybackError):
    """Playback failed after audio delivery started; replay could duplicate speech."""


def resolve_pcm_backend(requested: str = "auto", command: str = "") -> tuple[str, str]:
    backend = str(requested or "auto").strip().casefold()
    configured_command = str(command or "").strip()
    if backend not in {"auto", "pacat", "pwcat", "sounddevice"}:
        raise StreamingPlaybackError(f"unsupported PCM playback backend: {backend!r}")

    if backend == "sounddevice":
        if sd is None:
            raise StreamingPlaybackError("sounddevice is not installed")
        return backend, ""

    if configured_command:
        if backend == "auto":
            name = configured_command.rsplit("/", 1)[-1].casefold()
            backend = "pwcat" if name in {"pw-cat", "pw-play"} else "pacat"
        return backend, configured_command

    candidates = (
        (("pacat", "pacat"), ("pwcat", "pw-cat"))
        if backend == "auto"
        else ((backend, "pacat" if backend == "pacat" else "pw-cat"),)
    )
    for candidate_backend, executable in candidates:
        resolved = shutil.which(executable)
        if resolved:
            return candidate_backend, resolved

    if backend == "auto" and not sys.platform.startswith("linux") and sd is not None:
        return "sounddevice", ""
    raise StreamingPlaybackError(
        "no crash-isolated PCM player found (install pacat or pw-cat)"
    )


def build_pcm_command(
    backend: str,
    command: str,
    *,
    sample_rate: int,
    channels: int,
    latency_ms: int,
    client_name: str,
    stream_name: str,
) -> list[str]:
    if backend == "pacat":
        return [
            command,
            "--playback",
            "--raw",
            "--format=s16le",
            f"--rate={sample_rate}",
            f"--channels={channels}",
            f"--latency-msec={latency_ms}",
            f"--client-name={client_name}",
            f"--stream-name={stream_name}",
            "--volume=65536",
            "--property=application.id=speaker",
            "--property=media.role=production",
            "--property=state.restore-props=false",
            "--property=state.restore-target=false",
        ]
    if backend == "pwcat":
        properties = json.dumps(
            {
                "application.id": "speaker",
                "application.name": client_name,
                "media.name": stream_name,
                "media.role": "production",
                "state.restore-props": False,
                "state.restore-target": False,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return [
            command,
            "--playback",
            "--format=s16",
            f"--rate={sample_rate}",
            f"--channels={channels}",
            f"--latency={latency_ms}ms",
            f"--properties={properties}",
            "-",
        ]
    raise StreamingPlaybackError(f"backend {backend!r} does not use a subprocess")


async def play_pcm_once(
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
    backend: str = "auto",
    command: str = "",
    latency_ms: int = 50,
    timeout_s: float = 5.0,
    client_name: str = "Listener",
    stream_name: str = "Listener audio",
    device: int | str | None = None,
) -> str:
    payload = bytes(pcm)
    if not payload:
        return "none"
    if sample_rate <= 0 or channels <= 0 or len(payload) % (2 * channels):
        raise StreamingPlaybackError("PCM playback requires aligned signed 16-bit samples")
    resolved_backend, resolved_command = resolve_pcm_backend(backend, command)
    if resolved_backend == "sounddevice":
        if sd is None:  # pragma: no cover - guarded by resolve_pcm_backend
            raise StreamingPlaybackError("sounddevice is not installed")

        def _play() -> None:
            stream = sd.RawOutputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                device=device,
                latency=max(0.01, latency_ms / 1000.0),
            )
            try:
                stream.start()
                stream.write(payload)
                stream.stop()
            finally:
                stream.close()

        await asyncio.wait_for(asyncio.to_thread(_play), timeout=max(0.1, timeout_s))
        return resolved_backend

    args = build_pcm_command(
        resolved_backend,
        resolved_command,
        sample_rate=sample_rate,
        channels=channels,
        latency_ms=max(10, int(latency_ms)),
        client_name=client_name,
        stream_name=stream_name,
    )
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(
            proc.communicate(payload),
            timeout=max(0.1, float(timeout_s)),
        )
    except asyncio.TimeoutError as exc:
        await _terminate_process(proc)
        raise StreamingPlaybackError(
            f"{resolved_backend} playback timed out after {timeout_s:.1f}s"
        ) from exc
    if proc.returncode != 0:
        detail = bytes(stderr or b"").decode("utf-8", errors="replace").strip()
        raise StreamingPlaybackError(
            f"{resolved_backend} exited with code {proc.returncode}: {detail or 'no stderr'}"
        )
    return resolved_backend


class PCMStreamPlayer:
    """Buffer PCM and stream it to a crash-isolated player for one reply run."""

    def __init__(
        self,
        *,
        backend: str = "auto",
        command: str = "",
        prebuffer_ms: int = 150,
        latency_ms: int = 100,
        queue_ms: int = 2000,
        restart_attempts: int = 1,
        write_timeout_s: float = 5.0,
        timeout_s: float = 120.0,
        client_name: str = "Speaker",
        stream_name: str = "Speaker TTS",
        device: int | str | None = None,
        latency: str | float = "low",
        stream_factory: Callable[..., Any] | None = None,
        process_factory: Callable[..., Awaitable[Any]] | None = None,
        on_playback_start: Callable[[], Awaitable[None]] | None = None,
        on_playback_stop: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.requested_backend = "sounddevice" if stream_factory is not None else backend
        self.command = str(command or "")
        self.prebuffer_ms = max(0, int(prebuffer_ms))
        self.latency_ms = max(10, int(latency_ms))
        self.queue_ms = max(self.prebuffer_ms + _PCM_BLOCK_MS, int(queue_ms))
        self.restart_attempts = max(0, int(restart_attempts))
        self.write_timeout_s = max(0.1, float(write_timeout_s))
        self.timeout_s = max(1.0, float(timeout_s))
        self.client_name = str(client_name or "Speaker")
        self.stream_name = str(stream_name or "Speaker TTS")
        self.device = device
        self.latency = latency
        self._stream_factory = stream_factory
        self._process_factory = process_factory or asyncio.create_subprocess_exec
        self._on_playback_start = on_playback_start
        self._on_playback_stop = on_playback_stop

        self._lock = asyncio.Lock()
        self._run_id = ""
        self._sample_rate: int | None = None
        self._channels: int | None = None
        self._sample_width: int | None = None
        self._backend_name: str | None = None
        self._backend_command = ""
        self._process: Any | None = None
        self._stream: Any | None = None
        self._queue: asyncio.Queue[bytes | None] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._prebuffer = bytearray()
        self._generation = 0
        self._closing = False
        self._playback_notified = False
        self._writer_error: StreamingPlaybackInterrupted | None = None
        self._bytes_enqueued = 0
        self._bytes_written = 0
        self._last_bytes_written = 0
        self._stderr_tail = bytearray()
        self._last_error = ""
        self._last_exit_code: int | None = None
        self._restart_count = 0
        self._failed_run_id = ""
        self._normalized_stream_ids: list[int] = []
        self._normalization_error = ""
        self._native_io_task: asyncio.Task[Any] | None = None

    def set_lifecycle_callbacks(
        self,
        *,
        on_start: Callable[[], Awaitable[None]] | None,
        on_stop: Callable[[], Awaitable[None]] | None,
    ) -> None:
        self._on_playback_start = on_start
        self._on_playback_stop = on_stop

    async def start(
        self,
        *,
        sample_rate: int,
        channels: int = 1,
        sample_width: int = 2,
        run_id: str = "",
    ) -> None:
        if sample_rate <= 0:
            raise StreamingPlaybackError("sample_rate must be positive")
        if channels != 1 or sample_width != 2:
            raise StreamingPlaybackError("only mono PCM16 playback is supported")
        normalized_run_id = str(run_id or "")

        needs_finish = False
        async with self._lock:
            if self._sample_rate is not None:
                same_format = (
                    self._sample_rate == sample_rate
                    and self._channels == channels
                    and self._sample_width == sample_width
                )
                same_run = self._run_id == normalized_run_id
                if same_format and same_run:
                    self._raise_writer_error_locked()
                    return
                needs_finish = True
        if needs_finish:
            await self.finish_run()

        async with self._lock:
            if self._failed_run_id == normalized_run_id:
                if self._restart_count > self.restart_attempts:
                    raise StreamingPlaybackError(
                        f"PCM player restart limit exceeded for run {normalized_run_id or '-'}"
                    )
            else:
                self._failed_run_id = ""
                self._restart_count = 0
            self._run_id = normalized_run_id
            self._sample_rate = int(sample_rate)
            self._channels = int(channels)
            self._sample_width = int(sample_width)
            self._prebuffer.clear()
            self._writer_error = None
            self._bytes_enqueued = 0
            self._bytes_written = 0
            self._stderr_tail.clear()
            self._last_error = ""
            self._last_exit_code = None
            self._normalized_stream_ids = []
            self._normalization_error = ""
            self._closing = False
            self._generation += 1

    async def write(self, pcm: bytes) -> None:
        payload = bytes(pcm)
        if not payload:
            return
        async with self._lock:
            if self._sample_rate is None or self._channels is None or self._sample_width is None:
                raise StreamingPlaybackError("PCM output is not started")
            if len(payload) % (self._channels * self._sample_width):
                raise StreamingPlaybackError("PCM16 chunk length must be frame-aligned")
            self._raise_writer_error_locked()
            if self._queue is None:
                self._prebuffer.extend(payload)
                if len(self._prebuffer) < self._prebuffer_bytes_locked():
                    return
                payload = bytes(self._prebuffer)
                self._prebuffer.clear()
                await self._start_backend_locked()
            queue = self._queue
            generation = self._generation
        if queue is None:  # pragma: no cover - guarded by _start_backend_locked
            raise StreamingPlaybackError("PCM writer queue was not created")
        await self._enqueue_payload(queue, payload, generation)

    async def finish_segment(self) -> None:
        payload = b""
        async with self._lock:
            if self._sample_rate is None:
                return
            self._raise_writer_error_locked()
            if self._queue is None:
                payload = bytes(self._prebuffer)
                self._prebuffer.clear()
                if not payload:
                    return
                await self._start_backend_locked()
            queue = self._queue
            generation = self._generation
        if payload and queue is not None:
            await self._enqueue_payload(queue, payload, generation)

    async def finish(self) -> None:
        """Backward-compatible alias: finish and close the current playback run."""
        await self.finish_run()

    async def finish_run(self, run_id: str | None = None) -> None:
        async with self._lock:
            if run_id is not None and self._run_id and str(run_id) != self._run_id:
                return
        await self.finish_segment()

        async with self._lock:
            queue = self._queue
            writer_task = self._writer_task
            if queue is None or writer_task is None:
                await self._reset_session_locked(notify_stop=True)
                return
            if writer_task.done():
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await writer_task
                error = self._writer_error
                await self._reset_session_locked(notify_stop=True)
                if error is not None:
                    raise error
                return
            self._closing = True
        try:
            await asyncio.wait_for(queue.put(None), timeout=self.write_timeout_s)
        except asyncio.TimeoutError as exc:
            error = StreamingPlaybackInterrupted(
                "PCM playback queue did not accept the end-of-run marker within "
                f"{self.write_timeout_s:.1f}s"
            )
            self._record_failure(error)
            await self.abort()
            raise error from exc
        try:
            await asyncio.wait_for(asyncio.shield(writer_task), timeout=self.timeout_s)
        except asyncio.TimeoutError as exc:
            error = StreamingPlaybackInterrupted(
                f"PCM playback did not finish within {self.timeout_s:.1f}s"
            )
            self._record_failure(error)
            await self.abort()
            raise error from exc

        async with self._lock:
            error = self._writer_error
            await self._reset_session_locked(notify_stop=True)
        if error is not None:
            raise error

    async def abort(self) -> None:
        async with self._lock:
            writer_task = self._writer_task
            stderr_task = self._stderr_task
            process = self._process
            stream = self._stream
            native_io_task = self._native_io_task
            queue = self._queue
            self._generation += 1
            self._writer_task = None
            self._stderr_task = None
            self._process = None
            self._stream = None
            self._native_io_task = None
            self._queue = None
            self._prebuffer.clear()
            self._closing = True
            if queue is not None:
                _drain_queue(queue)
        for task in (writer_task, stderr_task):
            if task is not None and not task.done():
                task.cancel()
        if writer_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await writer_task
        if stderr_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stderr_task
        if native_io_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(native_io_task)
        if stream is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(stream.abort)
            with contextlib.suppress(Exception):
                await asyncio.to_thread(stream.close)
        if process is not None:
            await _terminate_process(process)
            self._last_exit_code = process.returncode
        async with self._lock:
            await self._reset_session_locked(notify_stop=True)

    def get_status(self) -> dict[str, Any]:
        process = self._process
        queue = self._queue
        queued_blocks = queue.qsize() if queue is not None else 0
        bytes_per_second = self._bytes_per_second()
        return {
            "active": self._sample_rate is not None,
            "backend": self._backend_name,
            "pid": getattr(process, "pid", None),
            "run_id": self._run_id or None,
            "sample_rate": self._sample_rate,
            "channels": self._channels,
            "prebuffer_ms": self.prebuffer_ms,
            "prebuffered_ms": round(len(self._prebuffer) * 1000 / bytes_per_second, 1)
            if bytes_per_second
            else 0.0,
            "queue_blocks": queued_blocks,
            "queue_ms": self.queue_ms,
            "bytes_enqueued": self._bytes_enqueued,
            "bytes_written": self._bytes_written,
            "last_bytes_written": self._last_bytes_written,
            "restart_count": self._restart_count,
            "last_exit_code": self._last_exit_code,
            "last_error": self._last_error or None,
            "stderr_tail": bytes(self._stderr_tail).decode("utf-8", errors="replace").strip()
            or None,
            "normalized_stream_ids": list(self._normalized_stream_ids),
            "normalization_error": self._normalization_error or None,
        }

    async def _start_backend_locked(self) -> None:
        if self._queue is not None:
            return
        if self._sample_rate is None or self._channels is None:
            raise StreamingPlaybackError("PCM format is not configured")
        if self._stream_factory is not None:
            backend, command = "sounddevice", ""
        else:
            backend, command = resolve_pcm_backend(self.requested_backend, self.command)
        self._backend_name = backend
        self._backend_command = command
        try:
            await self._notify_start_locked()
            if backend == "sounddevice":
                factory = self._stream_factory
                native_output = factory is None
                if factory is None:
                    if sd is None:  # pragma: no cover - guarded by resolver
                        raise StreamingPlaybackError("sounddevice is not installed")
                    factory = sd.RawOutputStream
                stream = await asyncio.to_thread(
                    factory,
                    samplerate=self._sample_rate,
                    channels=self._channels,
                    dtype="int16",
                    device=self.device,
                    latency=self.latency,
                )
                await asyncio.to_thread(stream.start)
                self._stream = stream
                if native_output:
                    try:
                        state = await normalize_listener_output_volume_state()
                        self._normalized_stream_ids = [
                            int(value) for value in state.get("stream_ids", [])
                        ]
                    except Exception as exc:  # noqa: BLE001
                        self._normalization_error = str(exc)
                        log.warning("PCM output volume normalization failed: %s", exc)
            else:
                args = build_pcm_command(
                    backend,
                    command,
                    sample_rate=self._sample_rate,
                    channels=self._channels,
                    latency_ms=self.latency_ms,
                    client_name=self.client_name,
                    stream_name=self.stream_name,
                )
                self._process = await self._process_factory(
                    *args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                if getattr(self._process, "stdin", None) is None:
                    raise StreamingPlaybackError(f"{backend} stdin pipe is unavailable")
                if getattr(self._process, "stderr", None) is not None:
                    self._stderr_task = asyncio.create_task(
                        self._capture_stderr(self._process.stderr),
                        name="PCMStreamPlayer.stderr",
                    )

            max_blocks = max(2, math.ceil(self.queue_ms / _PCM_BLOCK_MS))
            self._queue = asyncio.Queue(maxsize=max_blocks)
            self._writer_task = asyncio.create_task(
                self._writer_loop(self._queue, self._generation),
                name="PCMStreamPlayer.writer",
            )
        except Exception:
            await self._notify_stop_locked()
            raise

    async def _enqueue_payload(
        self,
        queue: asyncio.Queue[bytes | None],
        payload: bytes,
        generation: int,
    ) -> None:
        block_bytes = max(2, self._block_bytes())
        for offset in range(0, len(payload), block_bytes):
            block = payload[offset : offset + block_bytes]
            try:
                await asyncio.wait_for(queue.put(block), timeout=self.write_timeout_s)
            except asyncio.TimeoutError as exc:
                error = StreamingPlaybackInterrupted(
                    f"PCM playback queue blocked for {self.write_timeout_s:.1f}s"
                )
                self._record_failure(error)
                raise error from exc
            if generation != self._generation:
                raise StreamingPlaybackInterrupted("PCM playback was interrupted")
            self._bytes_enqueued += len(block)
            if self._writer_error is not None:
                raise self._writer_error

    async def _writer_loop(
        self,
        queue: asyncio.Queue[bytes | None],
        generation: int,
    ) -> None:
        try:
            while True:
                payload = await queue.get()
                try:
                    if payload is None:
                        await self._finish_backend()
                        return
                    if generation != self._generation:
                        return
                    await self._write_backend(payload)
                    self._bytes_written += len(payload)
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - converted to a recoverable playback failure
            error = (
                exc
                if isinstance(exc, StreamingPlaybackInterrupted)
                else StreamingPlaybackInterrupted(str(exc))
            )
            self._record_failure(error)
            _drain_queue(queue)

    async def _write_backend(self, payload: bytes) -> None:
        if self._backend_name == "sounddevice":
            if self._stream is None:
                raise StreamingPlaybackInterrupted("sounddevice stream disappeared")
            io_task = asyncio.create_task(asyncio.to_thread(self._stream.write, payload))
            self._native_io_task = io_task
            try:
                await asyncio.shield(io_task)
            finally:
                if io_task.done() and self._native_io_task is io_task:
                    self._native_io_task = None
            return
        process = self._process
        if process is None or process.returncode is not None:
            code = None if process is None else process.returncode
            raise StreamingPlaybackInterrupted(
                f"{self._backend_name or 'PCM player'} exited during playback (code={code})"
            )
        process.stdin.write(payload)
        try:
            await asyncio.wait_for(process.stdin.drain(), timeout=self.write_timeout_s)
        except (BrokenPipeError, ConnectionResetError, asyncio.TimeoutError) as exc:
            raise StreamingPlaybackInterrupted(
                f"{self._backend_name} stopped accepting PCM: {exc}"
            ) from exc

    async def _finish_backend(self) -> None:
        if self._backend_name == "sounddevice":
            stream = self._stream
            if stream is None:
                return
            await asyncio.to_thread(stream.stop)
            await asyncio.to_thread(stream.close)
            self._stream = None
            return
        process = self._process
        if process is None:
            return
        stdin = getattr(process, "stdin", None)
        if stdin is not None:
            stdin.close()
            wait_closed = getattr(stdin, "wait_closed", None)
            if callable(wait_closed):
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    await wait_closed()
        try:
            await asyncio.wait_for(process.wait(), timeout=self.timeout_s)
        except asyncio.TimeoutError as exc:
            await _terminate_process(process)
            raise StreamingPlaybackInterrupted(
                f"{self._backend_name} did not drain within {self.timeout_s:.1f}s"
            ) from exc
        self._last_exit_code = process.returncode
        if process.returncode != 0:
            detail = bytes(self._stderr_tail).decode("utf-8", errors="replace").strip()
            raise StreamingPlaybackInterrupted(
                f"{self._backend_name} exited with code {process.returncode}: "
                f"{detail or 'no stderr'}"
            )

    async def _capture_stderr(self, stream: Any) -> None:
        try:
            while True:
                chunk = await stream.read(2048)
                if not chunk:
                    return
                self._stderr_tail.extend(chunk)
                if len(self._stderr_tail) > _STDERR_TAIL_BYTES:
                    del self._stderr_tail[:-_STDERR_TAIL_BYTES]
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - diagnostics must not affect playback
            return

    async def _reset_session_locked(self, *, notify_stop: bool) -> None:
        stderr_task = self._stderr_task
        self._stderr_task = None
        if stderr_task is not None and not stderr_task.done():
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stderr_task
        self._last_bytes_written = self._bytes_written
        self._run_id = ""
        self._sample_rate = None
        self._channels = None
        self._sample_width = None
        self._process = None
        self._stream = None
        self._queue = None
        self._writer_task = None
        self._prebuffer.clear()
        self._closing = False
        self._generation += 1
        if notify_stop:
            await self._notify_stop_locked()

    async def _notify_start_locked(self) -> None:
        if self._playback_notified:
            return
        callback = self._on_playback_start
        if callback is not None:
            await callback()
        self._playback_notified = True

    async def _notify_stop_locked(self) -> None:
        if not self._playback_notified:
            return
        self._playback_notified = False
        callback = self._on_playback_stop
        if callback is not None:
            with contextlib.suppress(Exception):
                await callback()

    def _record_failure(self, error: StreamingPlaybackInterrupted) -> None:
        self._writer_error = error
        self._last_error = str(error)
        self._failed_run_id = self._run_id
        self._restart_count += 1
        process = self._process
        if process is not None and process.returncode is not None:
            self._last_exit_code = process.returncode

    def _raise_writer_error_locked(self) -> None:
        if self._writer_error is not None:
            raise self._writer_error

    def _bytes_per_second(self) -> int:
        if self._sample_rate is None or self._channels is None or self._sample_width is None:
            return 0
        return self._sample_rate * self._channels * self._sample_width

    def _prebuffer_bytes_locked(self) -> int:
        return max(0, math.ceil(self._bytes_per_second() * self.prebuffer_ms / 1000))

    def _block_bytes(self) -> int:
        frame_bytes = max(2, (self._channels or 1) * (self._sample_width or 2))
        wanted = max(frame_bytes, self._bytes_per_second() * _PCM_BLOCK_MS // 1000)
        return max(frame_bytes, wanted - (wanted % frame_bytes))


def _drain_queue(queue: asyncio.Queue[bytes | None]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        else:
            queue.task_done()


async def _terminate_process(process: Any) -> None:
    if getattr(process, "returncode", None) is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
        return
    except (asyncio.TimeoutError, ProcessLookupError):
        pass
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
        await asyncio.wait_for(process.wait(), timeout=1.0)


__all__ = [
    "PCMStreamPlayer",
    "StreamingPlaybackError",
    "StreamingPlaybackInterrupted",
    "build_pcm_command",
    "play_pcm_once",
    "resolve_pcm_backend",
]
