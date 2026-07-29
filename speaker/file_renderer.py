"""Asynchronous WAV rendering through Listener's existing neural TTS worker."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, SpeechStyleConfig, TTSFileRenderConfig
from .emoji import extract_emoji_for_speech
from .neural_tts import FallbackSpeechEngine, NeuralSpeechEngine
from .style import EmojiStyleResolver, STYLE_DEFINITIONS
from .tts import SpeechEngine, SpeechRequest, split_speech_units


class TTSFileRenderError(RuntimeError):
    """Base error returned by the file-render service."""


class TTSFileRenderUnavailable(TTSFileRenderError):
    """Raised when no persistent neural backend is available."""


class TTSFileRenderBusy(TTSFileRenderError):
    """Raised when the bounded render queue is full."""


class TTSFileRenderNotFound(TTSFileRenderError):
    """Raised for an unknown render job identifier."""


_TERMINAL_STATES = {"completed", "failed", "cancelled"}
_SAFE_FILENAME_RE = re.compile(r"[^\w.-]+", flags=re.UNICODE)


@dataclass(slots=True)
class _RenderJob:
    identifier: str
    text: str
    speech_text: str
    style_id: str
    instruction: str
    emoji: str | None
    filename: str
    backend: str
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    state: str = "queued"
    output_path: str | None = None
    error: str = ""
    segment_count: int = 0
    audio_seconds: float = 0.0
    cancel_requested: bool = False
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def owner(self) -> str:
        return f"file:{self.identifier}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "state": self.state,
            "backend": self.backend,
            "format": "wav",
            "filename": self.filename,
            "output_path": self.output_path,
            "style_id": self.style_id,
            "emoji": self.emoji,
            "text_chars": len(self.text),
            "text_preview": _preview(self.text),
            "segment_count": self.segment_count,
            "audio_seconds": round(self.audio_seconds, 3),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error or None,
        }


class TTSFileRenderer:
    """Queue WAV jobs on the same client used for spoken reply playback."""

    def __init__(
        self,
        *,
        speech: SpeechEngine,
        config: TTSFileRenderConfig,
        style_config: SpeechStyleConfig | None = None,
    ) -> None:
        self.config = config
        self._engine = _find_neural_engine(speech)
        self._style_resolver = EmojiStyleResolver(style_config)
        self._output_dir = _resolve_output_dir(config.output_dir)
        self._queue: asyncio.Queue[_RenderJob | None] = asyncio.Queue(
            maxsize=max(1, int(config.max_pending_jobs))
        )
        self._jobs: dict[str, _RenderJob] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._current_job: _RenderJob | None = None
        self._closing = False

    async def start(self) -> None:
        if not self.config.enabled or self._engine is None:
            return
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._closing = False
        self._worker_task = asyncio.create_task(self._worker(), name="Speaker.file_render")

    async def close(self) -> None:
        self._closing = True
        for job in self._jobs.values():
            if job.state not in _TERMINAL_STATES:
                job.cancel_requested = True
                job.cancel_event.set()
        current = self._current_job
        if current is not None and self._engine is not None:
            with contextlib.suppress(Exception):
                await self._engine.client.cancel(owner=current.owner)
        task = self._worker_task
        if task is None:
            return
        await self._queue.put(None)
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._worker_task = None
        self._style_resolver.clear()

    async def submit(
        self,
        text: str,
        *,
        style: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_available()
        value = str(text or "").strip()
        if not value:
            raise ValueError("text must not be empty")
        if len(value) > self.config.max_text_chars:
            raise ValueError(
                f"text exceeds max_text_chars={self.config.max_text_chars}"
            )
        self._prune_completed_jobs()
        if self._pending_count() >= self.config.max_pending_jobs:
            raise TTSFileRenderBusy("TTS file render queue is full")

        identifier = uuid.uuid4().hex
        parsed = extract_emoji_for_speech(value)
        if not parsed.speech_text or not _split_render_units(
            parsed.speech_text,
            max_chars=self.config.segment_chars,
        ):
            raise ValueError("text contains no speakable content")
        style_id, instruction, emoji = self._resolve_style(
            value,
            parsed.tokens,
            identifier=identifier,
            explicit_style=style,
        )
        output_name = _safe_output_name(filename, identifier=identifier)
        engine = self._engine
        if engine is None:  # Kept explicit for type narrowing after _ensure_available().
            raise TTSFileRenderUnavailable("persistent neural TTS is unavailable")
        job = _RenderJob(
            identifier=identifier,
            text=value,
            speech_text=parsed.speech_text,
            style_id=style_id,
            instruction=instruction,
            emoji=emoji,
            filename=output_name,
            backend=engine.backend,
        )
        self._jobs[identifier] = job
        await self.start()
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            self._jobs.pop(identifier, None)
            raise TTSFileRenderBusy("TTS file render queue is full") from exc
        return job.as_dict()

    def get_job(self, identifier: str) -> dict[str, Any]:
        return self._get_job(identifier).as_dict()

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
        return [job.as_dict() for job in jobs]

    async def wait(self, identifier: str, *, timeout_s: float | None = None) -> dict[str, Any]:
        job = self._get_job(identifier)
        if timeout_s is None:
            await job.done.wait()
        else:
            await asyncio.wait_for(job.done.wait(), timeout=max(0.01, float(timeout_s)))
        return job.as_dict()

    async def cancel(self, identifier: str) -> dict[str, Any]:
        job = self._get_job(identifier)
        if job.state in _TERMINAL_STATES:
            return job.as_dict()
        job.cancel_requested = True
        job.cancel_event.set()
        if job.state == "running" and self._engine is not None:
            await self._engine.client.cancel(owner=job.owner)
        return job.as_dict()

    def get_status(self) -> dict[str, Any]:
        engine = self._engine
        current = self._current_job
        return {
            "enabled": bool(self.config.enabled),
            "available": bool(self.config.enabled and engine is not None),
            "backend": engine.backend if engine is not None else None,
            "output_dir": str(self._output_dir),
            "queued": sum(job.state == "queued" for job in self._jobs.values()),
            "current_job_id": current.identifier if current is not None else None,
            "jobs": len(self._jobs),
            "max_text_chars": self.config.max_text_chars,
            "max_pending_jobs": self.config.max_pending_jobs,
            "max_completed_jobs": self.config.max_completed_jobs,
        }

    def _ensure_available(self) -> None:
        if not self.config.enabled:
            raise TTSFileRenderUnavailable("TTS file rendering is disabled")
        if self._engine is None:
            raise TTSFileRenderUnavailable(
                "TTS file rendering requires persistent VoxCPM2 or CosyVoice3"
            )
        if self._closing:
            raise TTSFileRenderUnavailable("TTS file renderer is shutting down")

    def _pending_count(self) -> int:
        return sum(job.state not in _TERMINAL_STATES for job in self._jobs.values())

    def _prune_completed_jobs(self) -> None:
        terminal = sorted(
            (job for job in self._jobs.values() if job.state in _TERMINAL_STATES),
            key=lambda item: item.completed_at or item.created_at,
            reverse=True,
        )
        for job in terminal[self.config.max_completed_jobs :]:
            self._jobs.pop(job.identifier, None)

    def _get_job(self, identifier: str) -> _RenderJob:
        job_id = str(identifier or "").strip()
        job = self._jobs.get(job_id)
        if job is None:
            raise TTSFileRenderNotFound(f"unknown TTS file render job: {job_id}")
        return job

    def _resolve_style(
        self,
        text: str,
        tokens: tuple,
        *,
        identifier: str,
        explicit_style: str | None,
    ) -> tuple[str, str, str | None]:
        if explicit_style not in (None, ""):
            style_id = str(explicit_style).strip().casefold()
            definition = STYLE_DEFINITIONS.get(style_id)
            if definition is None:
                allowed = ", ".join(sorted(STYLE_DEFINITIONS))
                raise ValueError(f"unknown style {style_id!r}; allowed: {allowed}")
            return definition.identifier, definition.instruction, None
        resolved = self._style_resolver.resolve(text, tokens, run_id=identifier)
        return resolved.style_id, resolved.instruction, resolved.emoji

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job is None:
                    return
                if job.cancel_requested or self._closing:
                    self._finish_cancelled(job)
                    continue
                self._current_job = job
                await self._render(job)
            finally:
                if job is not None and self._current_job is job:
                    self._current_job = None
                if job is not None and job.state in _TERMINAL_STATES:
                    self._style_resolver.discard(job.identifier)
                    self._prune_completed_jobs()
                self._queue.task_done()

    async def _render(self, job: _RenderJob) -> None:
        engine = self._engine
        if engine is None:
            self._finish_failed(job, "persistent neural TTS is unavailable")
            return
        output = self._output_dir / job.filename
        partial = output.with_suffix(output.suffix + ".part")
        job.state = "running"
        job.started_at = time.time()
        units = _split_render_units(job.speech_text, max_chars=self.config.segment_chars)
        audio_bytes = 0
        audio_format: tuple[int, int, int] | None = None
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            with wave.open(str(partial), "wb") as wav_file:
                for index, unit in enumerate(units):
                    if job.cancel_requested or self._closing:
                        break
                    request = SpeechRequest(
                        text=unit,
                        run_id=job.identifier,
                        segment_id=f"{job.identifier}:{index}",
                        style_id=job.style_id,
                        instruction=job.instruction,
                        emoji=job.emoji,
                    )
                    async for chunk in engine.client.generate(
                        request,
                        owner=job.owner,
                        cancellation_event=job.cancel_event,
                    ):
                        chunk_format = (
                            chunk.sample_rate,
                            chunk.channels,
                            chunk.sample_width,
                        )
                        if audio_format is None:
                            audio_format = chunk_format
                            wav_file.setframerate(chunk.sample_rate)
                            wav_file.setnchannels(chunk.channels)
                            wav_file.setsampwidth(chunk.sample_width)
                        elif chunk_format != audio_format:
                            raise TTSFileRenderError(
                                f"worker changed audio format from {audio_format} to {chunk_format}"
                            )
                        wav_file.writeframesraw(chunk.pcm)
                        audio_bytes += len(chunk.pcm)
                    job.segment_count = index + 1
            if job.cancel_requested or self._closing:
                self._finish_cancelled(job)
                return
            if audio_format is None or audio_bytes == 0:
                raise TTSFileRenderError("TTS worker produced no audio")
            os.replace(partial, output)
            rate, channels, width = audio_format
            job.audio_seconds = audio_bytes / (rate * channels * width)
            job.output_path = str(output)
            job.state = "completed"
            job.completed_at = time.time()
            job.error = ""
            job.done.set()
        except asyncio.CancelledError:
            self._finish_cancelled(job)
            raise
        except Exception as exc:  # noqa: BLE001 - job errors belong in status
            if job.cancel_requested or self._closing:
                self._finish_cancelled(job)
            else:
                self._finish_failed(job, str(exc))
        finally:
            self._style_resolver.discard(job.identifier)
            if partial.exists():
                with contextlib.suppress(OSError):
                    partial.unlink()

    @staticmethod
    def _finish_cancelled(job: _RenderJob) -> None:
        job.state = "cancelled"
        job.completed_at = time.time()
        job.output_path = None
        job.error = ""
        job.done.set()

    @staticmethod
    def _finish_failed(job: _RenderJob, error: str) -> None:
        job.state = "failed"
        job.completed_at = time.time()
        job.output_path = None
        job.error = str(error or "TTS file render failed")
        job.done.set()


def _find_neural_engine(speech: SpeechEngine) -> NeuralSpeechEngine | None:
    if isinstance(speech, NeuralSpeechEngine):
        return speech
    if isinstance(speech, FallbackSpeechEngine) and isinstance(
        speech.primary, NeuralSpeechEngine
    ):
        return speech.primary
    return None


def _resolve_output_dir(value: str) -> Path:
    path = Path(str(value or "state/tts-files")).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _safe_output_name(value: str | None, *, identifier: str) -> str:
    if value not in (None, ""):
        raw = Path(str(value)).name
        if raw.casefold().endswith(".wav"):
            raw = raw[:-4]
        stem = _SAFE_FILENAME_RE.sub("-", raw).strip("-._")[:80]
    else:
        stem = time.strftime("tts-%Y%m%d-%H%M%S")
    if not stem:
        stem = "tts"
    return f"{stem}-{identifier[:8]}.wav"


def _split_render_units(text: str, *, max_chars: int) -> list[str]:
    result: list[str] = []
    for sentence in split_speech_units(text):
        remainder = sentence.strip()
        while len(remainder) > max_chars:
            split_at = remainder.rfind(" ", 0, max_chars + 1)
            if split_at < max_chars // 2:
                split_at = max_chars
            result.append(remainder[:split_at].strip())
            remainder = remainder[split_at:].strip()
        if remainder:
            result.append(remainder)
    return result


def _preview(text: str, *, limit: int = 160) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else f"{value[: limit - 1]}…"


__all__ = [
    "TTSFileRenderBusy",
    "TTSFileRenderError",
    "TTSFileRenderNotFound",
    "TTSFileRenderer",
    "TTSFileRenderUnavailable",
]
