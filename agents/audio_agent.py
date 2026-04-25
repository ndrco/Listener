"""Audio agent orchestrating microphone → processing → STT pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import logging

from audio.microphone import MicrophoneStream
from audio.processing import AudioProcessor
from audio.writer import BufferedSpeechWriter
from audio.stt.streaming import WhisperStreamingTranscriber
from core.bus import EventBus, bus as default_bus
from core.config import cfg

log = logging.getLogger(__name__)


class AudioAgent:
    """High level coordinator for the audio capture and STT pipeline."""

    def __init__(self, *, bus: EventBus | None = None) -> None:
        self._bus = bus or default_bus

        self._microphone: MicrophoneStream | None = None
        self._processor: AudioProcessor | None = None
        self._writer: BufferedSpeechWriter | None = None
        self._transcriber: WhisperStreamingTranscriber | None = None

        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._microphone_task: asyncio.Task[None] | None = None

        self._running: bool = False
        self._paused: bool = False
        self._stopping: bool = False

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._running:
            return

        self._paused = False
        self._exit_stack = contextlib.AsyncExitStack()

        try:
            self._microphone = await self._exit_stack.enter_async_context(
                MicrophoneStream(bus=self._bus)
            )
            self._processor = await self._exit_stack.enter_async_context(
                AudioProcessor(bus=self._bus)
            )

            self._writer = BufferedSpeechWriter.from_config(bus=self._bus)
            self._transcriber = WhisperStreamingTranscriber(
                self._writer,
                bus=self._bus,
                on_final=self._on_final_transcript,
            )
            await self._transcriber.start()

            self._running = True
            self._microphone_task = asyncio.create_task(
                self._run_microphone_loop(), name="AudioAgent.microphone"
            )
            log.info("AudioAgent: pipeline started")
        except Exception:
            log.exception("AudioAgent: failed to start pipeline")
            await self._shutdown_resources()
            raise

    async def pause(self) -> None:
        if self._paused:
            return
        self._paused = True
        await self._shutdown_resources()
        log.info("AudioAgent: paused")

    async def resume(self) -> None:
        if not self._paused:
            return
        await self.start()

    async def close(self) -> None:
        self._paused = False
        await self._shutdown_resources()
        log.info("AudioAgent: stopped")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _shutdown_resources(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        try:
            if (
                not self._running
                and self._microphone_task is None
                and self._exit_stack is None
                and self._transcriber is None
                and self._writer is None
            ):
                return

            self._running = False

            task = self._microphone_task
            self._microphone_task = None
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            if self._transcriber is not None:
                try:
                    await self._transcriber.stop()
                except Exception:
                    log.exception("AudioAgent: failed to stop transcriber")
                self._transcriber = None

            if self._writer is not None:
                try:
                    await self._writer.close()
                except Exception:
                    log.exception("AudioAgent: failed to close writer")
                self._writer = None

            if self._exit_stack is not None:
                try:
                    await self._exit_stack.aclose()
                except Exception:
                    log.exception("AudioAgent: failed to close audio contexts")
                self._exit_stack = None

            self._microphone = None
            self._processor = None
        finally:
            self._stopping = False

    async def _run_microphone_loop(self) -> None:
        processor = self._processor
        microphone = self._microphone
        if processor is None or microphone is None:
            return

        try:
            async for chunk in microphone:
                if not self._running:
                    break
                if not chunk:
                    continue
                try:
                    await processor.submit(chunk)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("AudioAgent: failed to submit audio frame")
                    break
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("AudioAgent: microphone loop failed")
        finally:
            self._running = False
            if not self._stopping:
                asyncio.create_task(self._shutdown_resources())

    async def _on_final_transcript(self, text: str, payload: dict[str, object]) -> None:
        enriched = dict(payload)
        enriched.setdefault("text", text)
        enriched.setdefault("source", "stt")
        enriched.pop("pcm_data", None)
        try:
            await self._bus.publish(cfg.events.llm.input_text, **enriched)
        except Exception:
            log.exception("AudioAgent: failed to publish llm/input_text event")


__all__ = ["AudioAgent"]
