"""SpeechEngine adapters shared by neural TTS backends."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from audio.ducking import PulseAudioDucker
from core import perf

from .neural_worker_client import NeuralWorkerClient
from .streaming_playback import PCMStreamPlayer, StreamingPlaybackInterrupted
from .tts import SpeechEngine, SpeechRequest


log = logging.getLogger(__name__)


class NeuralSpeechEngine:
    def __init__(
        self,
        *,
        backend: str,
        client: NeuralWorkerClient,
        player: PCMStreamPlayer | None = None,
        ducking_config: object | None = None,
    ) -> None:
        self.backend = str(backend)
        self.client = client
        self.player = player or PCMStreamPlayer()
        self._active_run_id: str | None = None
        self._last_error = ""
        self._chunks_played = 0
        self._last_ttfa_ms: float | None = None
        self._last_generation_ms: float | None = None
        self._last_audio_ms: float | None = None
        self._active_cancel: asyncio.Event | None = None
        self._ducking_config = ducking_config
        self._ducker: PulseAudioDucker | None = None
        set_callbacks = getattr(self.player, "set_lifecycle_callbacks", None)
        if callable(set_callbacks):
            set_callbacks(
                on_start=self._on_playback_start,
                on_stop=self._on_playback_stop,
            )

    async def start(self) -> None:
        await self.client.start()

    async def speak(self, request: SpeechRequest | str) -> None:
        speech_request = SpeechRequest.coerce(request)
        if not speech_request.text:
            return
        self._active_run_id = speech_request.run_id or None
        self._chunks_played = 0
        self._last_ttfa_ms = None
        self._last_generation_ms = None
        self._last_audio_ms = None
        started = time.perf_counter()
        audio_seconds = 0.0
        completed = False
        playback_failed = False
        cancel_requested = asyncio.Event()
        self._active_cancel = cancel_requested
        try:
            async for chunk in self.client.generate(speech_request, owner="playback"):
                if cancel_requested.is_set():
                    # Keep consuming until the worker's terminal cancellation
                    # frame so its persistent stdout remains aligned.
                    continue
                if self._chunks_played == 0:
                    self._last_ttfa_ms = (time.perf_counter() - started) * 1000.0
                    perf.emit(
                        "speaker",
                        "neural_first_audio",
                        backend=self.backend,
                        run_id=speech_request.run_id,
                        segment_id=speech_request.segment_id,
                        style_id=speech_request.style_id,
                        ttfa_ms=self._last_ttfa_ms,
                    )
                    await self.player.start(
                        sample_rate=chunk.sample_rate,
                        channels=chunk.channels,
                        sample_width=chunk.sample_width,
                        run_id=speech_request.run_id,
                    )
                if cancel_requested.is_set():
                    continue
                if playback_failed:
                    continue
                try:
                    await self.player.write(chunk.pcm)
                except StreamingPlaybackInterrupted as exc:
                    # Do not replay the whole segment through Piper: part of it
                    # may already have been audible. Drain the neural worker so
                    # its protocol stays aligned, then resume on the next
                    # segment with a fresh playback process.
                    playback_failed = True
                    self._last_error = str(exc)
                    log.error("Neural PCM playback interrupted: %s", exc)
                    perf.emit(
                        "speaker",
                        "neural_playback_interrupted",
                        backend=self.backend,
                        run_id=speech_request.run_id,
                        segment_id=speech_request.segment_id,
                        error=exc,
                    )
                    await self.player.abort()
                    continue
                except Exception:
                    # abort() can close the stream after the check above but
                    # before write() acquires the player lock.  That is an
                    # expected cancellation race, not a backend failure.
                    if cancel_requested.is_set():
                        continue
                    raise
                self._chunks_played += 1
                audio_seconds += len(chunk.pcm) / (
                    chunk.sample_rate * chunk.channels * chunk.sample_width
                )
            if not playback_failed:
                try:
                    finish_segment = getattr(self.player, "finish_segment", None)
                    if callable(finish_segment):
                        await finish_segment()
                    else:
                        await self.player.finish()
                except StreamingPlaybackInterrupted as exc:
                    playback_failed = True
                    self._last_error = str(exc)
                    log.error("Neural PCM playback interrupted at segment end: %s", exc)
                    await self.player.abort()
            if not speech_request.run_id and not playback_failed:
                finish_run = getattr(self.player, "finish_run", None)
                if callable(finish_run):
                    await finish_run()
            completed = True
            self._last_generation_ms = (time.perf_counter() - started) * 1000.0
            self._last_audio_ms = audio_seconds * 1000.0
            perf.emit(
                "speaker",
                "neural_segment_done",
                backend=self.backend,
                run_id=speech_request.run_id,
                segment_id=speech_request.segment_id,
                chunks=self._chunks_played,
                generation_ms=self._last_generation_ms,
                audio_ms=self._last_audio_ms,
                rtf=(self._last_generation_ms / self._last_audio_ms)
                if self._last_audio_ms
                else None,
            )
            if not playback_failed:
                self._last_error = ""
        except asyncio.CancelledError:
            await self.player.abort()
            raise
        except Exception as exc:
            self._last_error = str(exc)
            await self.player.abort()
            raise
        finally:
            if not completed:
                await self.player.abort()
            if self._active_cancel is cancel_requested:
                self._active_cancel = None
            self._active_run_id = None

    async def interrupt(self, *, run_id: str | None = None) -> None:
        active_run_id = self._active_run_id
        if run_id is not None and active_run_id is not None and run_id != active_run_id:
            return
        cancel_requested = self._active_cancel
        if cancel_requested is not None:
            cancel_requested.set()
        await asyncio.gather(self.player.abort(), self.client.cancel(owner="playback"))

    async def close(self) -> None:
        await self.player.abort()
        await self.client.close()

    async def finish_run(self, run_id: str) -> None:
        try:
            await self.player.finish_run(run_id)
        except StreamingPlaybackInterrupted as exc:
            self._last_error = str(exc)
            log.error("Neural PCM playback failed while draining run %s: %s", run_id, exc)
            await self.player.abort()

    async def _on_playback_start(self) -> None:
        if self._ducking_config is None or self._ducker is not None:
            return
        ducker = PulseAudioDucker(self._ducking_config)
        await ducker.duck()
        self._ducker = ducker

    async def _on_playback_stop(self) -> None:
        ducker = self._ducker
        self._ducker = None
        if ducker is not None:
            await ducker.restore()

    def get_status(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "active_run_id": self._active_run_id,
            "chunks_played": self._chunks_played,
            "last_ttfa_ms": self._last_ttfa_ms,
            "last_generation_ms": self._last_generation_ms,
            "last_audio_ms": self._last_audio_ms,
            "last_error": self._last_error or None,
            "worker": self.client.get_status(),
            "playback": self.player.get_status(),
        }


class FallbackSpeechEngine:
    """Use fallback for failed utterances and open a circuit after repeated errors."""

    def __init__(
        self,
        primary: SpeechEngine,
        fallback: SpeechEngine,
        *,
        max_consecutive_errors: int = 3,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.max_consecutive_errors = max(1, int(max_consecutive_errors))
        self._consecutive_errors = 0
        self._circuit_open = False
        self._last_error = ""

    async def start(self) -> None:
        method = getattr(self.primary, "start", None)
        if not callable(method):
            return
        try:
            await method()
        except Exception as exc:  # noqa: BLE001 - startup must leave Piper operational
            self._last_error = str(exc)
            self._consecutive_errors = self.max_consecutive_errors
            self._circuit_open = True
            log.error("Neural TTS failed to start; using fallback: %s", exc)
        else:
            self._consecutive_errors = 0
            self._circuit_open = False
            self._last_error = ""

    async def speak(self, request: SpeechRequest | str) -> None:
        speech_request = SpeechRequest.coerce(request)
        if self._circuit_open:
            await self.fallback.speak(speech_request)
            return
        try:
            await self.primary.speak(speech_request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - fallback deliberately catches backend failures
            self._last_error = str(exc)
            self._consecutive_errors += 1
            if self._consecutive_errors >= self.max_consecutive_errors:
                self._circuit_open = True
                log.error("Neural TTS circuit opened after %d errors", self._consecutive_errors)
            else:
                log.warning("Neural TTS failed; using fallback: %s", exc)
            await self.fallback.speak(speech_request)
        else:
            self._consecutive_errors = 0
            self._last_error = ""

    async def interrupt(self, *, run_id: str | None = None) -> None:
        await asyncio.gather(
            _maybe_interrupt(self.primary, run_id=run_id),
            _maybe_interrupt(self.fallback, run_id=run_id),
            return_exceptions=True,
        )

    async def close(self) -> None:
        await asyncio.gather(
            _maybe_close(self.primary),
            _maybe_close(self.fallback),
            return_exceptions=True,
        )

    async def finish_run(self, run_id: str) -> None:
        await asyncio.gather(
            _maybe_finish_run(self.primary, run_id=run_id),
            _maybe_finish_run(self.fallback, run_id=run_id),
            return_exceptions=True,
        )

    def get_status(self) -> dict[str, Any]:
        primary = _maybe_status(self.primary)
        fallback = _maybe_status(self.fallback)
        primary_backend = _status_backend(primary, default="primary")
        fallback_backend = _status_backend(fallback, default="fallback")
        return {
            # Retained for API compatibility: this engine wraps the selected
            # neural backend with a fallback backend.
            "backend": "fallback",
            "active_backend": fallback_backend if self._circuit_open else primary_backend,
            "using_fallback": self._circuit_open,
            "circuit_open": self._circuit_open,
            "consecutive_errors": self._consecutive_errors,
            "max_consecutive_errors": self.max_consecutive_errors,
            "last_error": self._last_error or None,
            "primary": primary,
            "fallback": fallback,
        }


async def _maybe_interrupt(engine: SpeechEngine, *, run_id: str | None) -> None:
    method = getattr(engine, "interrupt", None)
    if callable(method):
        await method(run_id=run_id)


async def _maybe_close(engine: SpeechEngine) -> None:
    method = getattr(engine, "close", None)
    if callable(method):
        await method()


async def _maybe_finish_run(engine: SpeechEngine, *, run_id: str) -> None:
    method = getattr(engine, "finish_run", None)
    if callable(method):
        await method(run_id)


def _maybe_status(engine: SpeechEngine) -> dict | None:
    method = getattr(engine, "get_status", None)
    return method() if callable(method) else None


def _status_backend(status: dict | None, *, default: str) -> str:
    if not status:
        return default
    return str(status.get("backend") or default)


__all__ = ["FallbackSpeechEngine", "NeuralSpeechEngine"]
