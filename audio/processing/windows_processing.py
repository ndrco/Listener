# audio/processing/windows_processing.py
"""Asynchronous audio frame processing for Listener voice input."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import webrtcvad
except ImportError:  # pragma: no cover - dependency should be installed
    webrtcvad = None  # type: ignore[assignment]

from .agc import AGCSettings, AutomaticGainControl
from .echo_cancellation import (
    AcousticEchoCancellationSettings,
    AcousticEchoCanceller,
)
from .highpass import DCBlockingHighPass
from .noise_suppression import NoiseSuppressor
from .resampler import StreamingResampler
from .silero_vad import SileroVADHelper

from core.bus import Event, EventBus, bus as default_bus
from core.config import AudioProcessingCfg, cfg
from core import perf

log = logging.getLogger(__name__)


DEFAULT_ACTIVE_REPUBLISH_INTERVAL_MS = 250.0


class _ParecLoopbackStream:
    """Capture PipeWire/Pulse monitor audio via parec."""

    def __init__(
        self,
        *,
        source_name: str,
        sample_rate: int,
        channels: int,
        blocksize: int,
        on_chunk: Any,
        debug: bool = False,
    ) -> None:
        self.source_name = str(source_name)
        self.sample_rate = max(1, int(sample_rate))
        self.channels = max(1, int(channels))
        self.blocksize = max(1, int(blocksize))
        self._on_chunk = on_chunk
        self._debug = bool(debug)
        self._stop_event = threading.Event()
        self._proc: subprocess.Popen[bytes] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: list[str] = []

    def start(self) -> None:
        if shutil.which("parec") is None:
            raise RuntimeError("parec is not available")
        cmd = [
            "parec",
            "--raw",
            "--format=s16le",
            f"--rate={self.sample_rate}",
            f"--channels={self.channels}",
            f"--device={self.source_name}",
            "--latency-msec=20",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            name="ParecLoopback.stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            name="ParecLoopback.stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    def close(self) -> None:
        self.stop()

    def error_tail(self) -> str:
        return "\n".join(self._stderr_tail[-6:]).strip()

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        chunk_bytes = self.blocksize * self.channels * 2
        while not self._stop_event.is_set():
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break
            if len(data) % 2:
                data = data[:-1]
            if data:
                self._on_chunk(data)

    def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw_line in iter(proc.stderr.readline, b""):
            if self._stop_event.is_set():
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            self._stderr_tail.append(line)
            if len(self._stderr_tail) > 20:
                self._stderr_tail = self._stderr_tail[-20:]
            if self._debug:
                log.debug("audio.processing: parec stderr: %s", line)


@dataclass(slots=True)
class ProcessedAudioFrame:
    """Result of processing an audio frame."""

    data: bytes
    sample_rate: int
    channels: int
    voice_detected: bool
    timestamp: float
    vad_probability: float
    vad_speech_frames: int
    vad_total_frames: int
    voice_active_duration: float
    webrtc_probability: float
    silero_probability: float
    silero_invocations: int


class AudioProcessor:
    """Asynchronous audio stream processor with built-in VAD."""

    def __init__(
        self,
        *,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        config: Optional[AudioProcessingCfg] = None,
        bus: Optional[EventBus] = None,
        debug: Optional[bool] = None,
        queue_maxsize: int = 0,
    ) -> None:
        audio_cfg = cfg.audio
        input_cfg = audio_cfg.input
        self.channels = int(channels or input_cfg.channels)
        self.config = config or audio_cfg.processing
        self._bus = bus or default_bus
        self._debug = cfg.debug if debug is None else bool(debug)

        self.input_sample_rate = int(sample_rate or input_cfg.input_sample_rate)
        output_rate_source = getattr(
            input_cfg, "output_sample_rate", self.input_sample_rate
        )
        try:
            output_rate = int(output_rate_source)
        except (TypeError, ValueError):
            output_rate = int(self.input_sample_rate)
        if output_rate <= 0:
            output_rate = int(self.input_sample_rate)
        self.output_sample_rate = int(output_rate)
        self._resampler: StreamingResampler | None = None
        if self.output_sample_rate < self.input_sample_rate:
            try:
                self._resampler = StreamingResampler(
                    self.input_sample_rate, self.output_sample_rate
                )
            except Exception:
                self._resampler = None
                self.output_sample_rate = int(self.input_sample_rate)
                if self._debug:
                    log.exception("audio.processing: failed to initialise resampler")
                else:
                    log.warning(
                        "audio.processing: failed to initialise resampler", exc_info=False
                    )

        self._input_queue: "asyncio.Queue[bytes | None]" = asyncio.Queue(
            maxsize=queue_maxsize
        )
        self._output_queue: "asyncio.Queue[ProcessedAudioFrame | None]" = (
            asyncio.Queue(maxsize=queue_maxsize)
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._running: bool = False
        self._closed: bool = False

        self._last_voice_state: Optional[bool] = None
        self._last_voice_state_change_ts: float = 0.0
        self._last_active_republish_ts: float = 0.0
        self._vad: Any | None = None
        self._vad_frame_samples: int = 0
        self._vad_frame_bytes: int = 0
        self._vad_frame_duration_ms: int = 0
        self._vad_downmix: bool = self.channels != 1
        self._silero_vad: SileroVADHelper | None = None
        self._silero_pending_audio = bytearray()
        self._silero_pending_ms: float = 0.0
        self._silero_last_probability: float = 0.0
        self._vad_pipeline: str = "hybrid"
        self._allow_webrtc: bool = True
        self._allow_silero: bool = True
        self._noise_suppressor: NoiseSuppressor | None = None
        self._aec: AcousticEchoCanceller | None = None
        self._aec_subscription_pattern: str | None = None
        self._aec_subscription_handler: Any | None = None
        self._aec_playback_source: str | None = None
        self._aec_loopback_params: dict[str, Any] | None = None
        self._aec_loopback_stream: Any | None = None
        self._aec_loopback_task: asyncio.Task[None] | None = None
        self._aec_loopback_queue: "asyncio.Queue[bytes]" | None = None
        self._aec_loopback_dropped_frames: int = 0
        self._aec_loopback_last_drop_log_ts: float = 0.0
        self._sounddevice_backend: Any | None = None
        self._sounddevice_backend_error: bool = False
        self._vad_tail = bytearray()

        self._initialize_vad()
        self._initialize_noise_suppression()
        self._initialize_aec()
        if self._debug:
            log.debug(
                "audio.processing: processor initialised (rate=%d, channels=%d)",
                self.output_sample_rate,
                self.channels,
            )
        highpass_cfg = getattr(self.config, "highpass", None)
        cutoff_source = getattr(highpass_cfg, "cutoff_hz", 100.0) if highpass_cfg else 100.0
        try:
            cutoff_hz = float(cutoff_source)
        except (TypeError, ValueError):
            cutoff_hz = 100.0
        self._dc_filter = DCBlockingHighPass(
            self.output_sample_rate,
            self.channels,
            cutoff_hz=cutoff_hz,
        )
        agc_cfg = getattr(self.config, "agc", None)
        agc_headroom = float(getattr(agc_cfg, "headroom_db", 0.8)) if agc_cfg else 0.8
        agc_limiter_attack = float(
            getattr(agc_cfg, "limiter_attack_ms", 0.75)
        ) if agc_cfg else 0.75
        agc_limiter_release = float(
            getattr(agc_cfg, "limiter_release_ms", 60.0)
        ) if agc_cfg else 60.0
        self._agc = AutomaticGainControl(
            headroom_db=agc_headroom,
            limiter_attack_ms=agc_limiter_attack,
            limiter_release_ms=agc_limiter_release,
        )
        self._voice_active: bool = False
        self._speech_run_ms: float = 0.0
        self._silence_run_ms: float = 0.0
        self._stream_time_ms: float = 0.0
        self._segment_start_ms: float | None = None
        self._segment_padded_start_ms: float | None = None
        self._last_speech_time_ms: float | None = None
        self._last_segment_padded_duration_ms: float = 0.0
        self._segment_ended_in_current_chunk: bool = False
        self._last_positive_vad_ms: float | None = None
        self._hangover_expires_ms: float | None = None
        self._pending_segment_end: bool = False

    @property
    def sample_rate(self) -> int:
        """Backward-compatible alias for output sample rate."""

        return self.output_sample_rate

    async def __aenter__(self) -> "AudioProcessor":
        if not self.config.enabled:
            log.warning("audio.processing: started while configuration is disabled")
        self._running = True
        await self._warmup_silero_vad()
        if self._debug:
            log.debug("audio.processing: starting processor worker task")
        self._worker_task = asyncio.create_task(
            self._worker(), name="AudioProcessing"
        )
        await self._start_aec_loopback_capture()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    def __aiter__(self) -> "AudioProcessor":
        return self

    async def __anext__(self) -> ProcessedAudioFrame:
        if self._closed and self._output_queue.empty():
            raise StopAsyncIteration
        item = await self._output_queue.get()
        if item is None:
            self._closed = True
            raise StopAsyncIteration
        return item

    async def submit(self, data: bytes) -> None:
        if not self._running:
            raise RuntimeError("AudioProcessor is not running")
        await self._input_queue.put(bytes(data))

    def submit_playback(self, data: bytes) -> None:
        """Send playback data into AEC (far-end reference)."""
        if not data or self._aec is None:
            return
        try:
            pcm = np.frombuffer(bytes(data), dtype=np.int16)
        except Exception:
            if self._debug:
                log.exception("audio.processing: failed to parse playback chunk for AEC")
            return
        self._aec.submit_farend(pcm)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._debug:
            log.debug("audio.processing: stopping processor")
        await self._stop_aec_loopback_capture()
        await self._input_queue.put(None)
        task = self._worker_task
        self._worker_task = None
        if task is not None:
            await task
            if self._debug:
                log.debug("audio.processing: processor worker stopped")
        self._teardown_aec_subscription()

    # === internal ==========================================================

    async def _warmup_silero_vad(self) -> None:
        helper = self._silero_vad
        warmup = getattr(helper, "warmup", None)
        if helper is None or not callable(warmup):
            return
        start_ns = perf.now_ns()
        try:
            await asyncio.to_thread(warmup, int(self.output_sample_rate))
        except Exception as exc:  # noqa: BLE001 - warmup must not prevent startup
            log.warning("audio.processing: Silero VAD warmup failed: %s", exc)
            return
        perf.emit(
            "input",
            "silero_warmup",
            duration_ms=perf.elapsed_ms(start_ns),
        )

    def _initialize_vad(self) -> None:
        """Create and configure VAD instance."""

        self._vad = None
        self._vad_frame_samples = 0
        self._vad_frame_bytes = 0
        self._vad_frame_duration_ms = 0
        self._vad_downmix = self.channels != 1
        self._silero_vad = None
        self._silero_pending_audio.clear()
        self._silero_pending_ms = 0.0
        self._silero_last_probability = 0.0
        self._vad_tail.clear()

        vad_cfg = getattr(self.config, "vad", None)
        pipeline_raw = (
            getattr(vad_cfg, "pipeline", "hybrid") if vad_cfg is not None else "hybrid"
        )
        pipeline = str(pipeline_raw).strip().lower()
        if pipeline not in {"hybrid", "webrtc", "silero"}:
            pipeline = "hybrid"
        self._vad_pipeline = pipeline
        self._allow_webrtc = pipeline in {"hybrid", "webrtc"}
        self._allow_silero = pipeline in {"hybrid", "silero"}
        if vad_cfg is not None and pipeline != getattr(vad_cfg, "pipeline", None):
            try:
                vad_cfg.pipeline = pipeline
            except Exception:  # pragma: no cover - defensive assignment guard
                pass
        if self._debug:
            log.debug(
                "audio.processing: VAD pipeline initialised (mode=%s)",
                self._vad_pipeline,
            )

        if not (self.config.enabled and getattr(vad_cfg, "enabled", True)):
            if self._debug:
                log.debug("audio.processing: VAD disabled in configuration")
            return

        frame_duration_value = getattr(vad_cfg, "frame_duration_ms", 30)
        try:
            frame_ms = int(frame_duration_value)
        except (TypeError, ValueError):
            frame_ms = 30
        if frame_ms not in (10, 20, 30):
            log.warning(
                "audio.processing: unsupported VAD frame size %s ms, falling back to 30 ms",
                frame_ms,
            )
            frame_ms = 30
        if vad_cfg and getattr(vad_cfg, "frame_duration_ms", None) != frame_ms:
            try:
                vad_cfg.frame_duration_ms = frame_ms
            except Exception:  # pragma: no cover - defensive assignment guard
                pass

        vad_model_path = getattr(vad_cfg, "model_path", None)
        vad_model_config_path = getattr(vad_cfg, "model_config_path", None)
        if self._allow_silero:
            if vad_model_path and vad_model_config_path:
                try:
                    self._silero_vad = SileroVADHelper(self.config, debug=self._debug)
                    if self._debug:
                        log.debug(
                            "audio.processing: Silero VAD helper initialised (model_path=%s)",
                            vad_model_path,
                        )
                except Exception:  # pragma: no cover - Silero helper initialization should not crash
                    if self._debug:
                        log.exception(
                            "audio.processing: Silero VAD helper initialisation failed"
                        )
                    else:
                        log.exception("audio.processing: failed to initialize Silero VAD helper")
                    self._silero_vad = None
            elif self._debug:
                log.debug(
                    "audio.processing: Silero VAD helper not configured (missing paths)"
                )
        else:
            self._silero_vad = None
            if self._debug:
                log.debug(
                    "audio.processing: Silero VAD disabled by pipeline mode '%s'",
                    self._vad_pipeline,
                )

        sample_rate = int(self.output_sample_rate)
        frame_samples = 0
        if sample_rate not in (8000, 16000, 32000, 48000):
            log.warning(
                "audio.processing: sample rate %s Hz is not supported by VAD",
                sample_rate,
            )
        else:
            frame_samples = int(sample_rate * frame_ms / 1000)
            if frame_samples <= 0:
                log.warning(
                    "audio.processing: computed invalid VAD frame sample count: %s",
                    frame_samples,
                )
                frame_samples = 0

        self._vad_frame_samples = frame_samples
        self._vad_frame_bytes = frame_samples * 2 if frame_samples > 0 else 0
        self._vad_frame_duration_ms = frame_ms

        if not self._allow_webrtc:
            self._vad = None
            if self._debug:
                log.debug(
                    "audio.processing: WebRTC VAD disabled by pipeline mode '%s'",
                    self._vad_pipeline,
                )
            return

        if webrtcvad is None:
            if self._debug:
                log.error("audio.processing: webrtcvad library is not installed")
            return

        if frame_samples <= 0:
            return

        mode = int(getattr(vad_cfg, "mode", 0))
        mode = max(0, min(3, mode))

        try:
            vad = webrtcvad.Vad(mode)
        except Exception:  # pragma: no cover - VAD initialization must not crash
            log.exception("audio.processing: failed to initialize webrtcvad")
            return

        self._vad = vad
        if self._debug:
            log.debug(
                "audio.processing: webrtcvad initialised (mode=%d, frame_ms=%d, frame_samples=%d)",
                mode,
                self._vad_frame_duration_ms,
                self._vad_frame_samples,
            )

    def _initialize_noise_suppression(self) -> None:
        cfg_ns = getattr(self.config, "noise_suppression", None)
        if not (cfg_ns and getattr(cfg_ns, "enabled", False)):
            if self._noise_suppressor is not None and self._debug:
                log.debug("audio.processing: noise suppression disabled")
            self._noise_suppressor = None
            return

        try:
            self._noise_suppressor = NoiseSuppressor(
                sample_rate=self.output_sample_rate,
                channels=self.channels,
                config=cfg_ns,
            )
            if self._debug:
                log.debug(
                    "audio.processing: noise suppression enabled (frame_ms=%s)",
                    getattr(cfg_ns, "frame_duration_ms", "?"),
                )
        except Exception:
            self._noise_suppressor = None
            if self._debug:
                log.exception("audio.processing: failed to initialise noise suppressor")
            else:
                log.warning(
                    "audio.processing: failed to initialise noise suppressor", exc_info=False
                )

    def _initialize_aec(self) -> None:
        self._aec = None
        self._teardown_aec_subscription()
        self._aec_playback_source = None
        self._aec_loopback_params = None

        cfg_aec = getattr(self.config, "aec", None)
        if not (cfg_aec and getattr(cfg_aec, "enabled", False)):
            if self._debug:
                log.debug("audio.processing: AEC disabled in configuration")
            return

        if self.channels != 1:
            log.warning("audio.processing: AEC currently supports only mono input")
            return

        frame_ms_raw = getattr(cfg_aec, "frame_duration_ms", 10)
        try:
            frame_ms = int(frame_ms_raw)
        except (TypeError, ValueError):
            frame_ms = 10

        source_raw = getattr(cfg_aec, "playback_source", None)
        source_normalized = ""
        if isinstance(source_raw, str):
            source_normalized = source_raw.strip().lower()
        topic_raw = getattr(cfg_aec, "playback_event_topic", None)
        if not source_normalized:
            source_normalized = "event_bus" if topic_raw else "manual"

        event_bus_aliases = {"event_bus", "bus", "topic"}
        default_playback_alias = cfg.events.audio.playback_frame.strip().lower()
        event_bus_aliases.add(default_playback_alias)

        if source_normalized in {"loopback", "submit_playback", "windows_loopback"}:
            playback_source = "loopback"
        elif source_normalized in event_bus_aliases:
            playback_source = "event_bus"
        elif source_normalized in {"manual", "none", "disabled"}:
            playback_source = "manual"
        else:
            if source_normalized:
                log.warning(
                    "audio.processing: unknown AEC playback source '%s', defaulting to event bus",
                    source_raw,
                )
            playback_source = "event_bus" if topic_raw else "manual"

        if playback_source == "event_bus":
            if topic_raw:
                topic = topic_raw
            elif source_raw is None:
                topic = cfg.events.audio.playback_frame
            else:
                topic = None
                playback_source = "manual"
        else:
            topic = topic_raw

        settings = AcousticEchoCancellationSettings(
            enabled=True,
            frame_duration_ms=frame_ms,
            stream_delay_ms=float(getattr(cfg_aec, "stream_delay_ms", 80)),
            noise_suppression=bool(getattr(cfg_aec, "noise_suppression", False)),
            high_pass_filter=bool(getattr(cfg_aec, "high_pass_filter", False)),
            auto_gain_control=bool(getattr(cfg_aec, "auto_gain_control", False)),
            playback_event_topic=topic,
            playback_source=playback_source,
            loopback_backend=str(getattr(cfg_aec, "loopback_backend", "auto") or "auto"),
            loopback_device_index=getattr(cfg_aec, "loopback_device_index", None),
            loopback_source_name=getattr(cfg_aec, "loopback_source_name", None),
            loopback_device_name_contains=getattr(
                cfg_aec, "loopback_device_name_contains", None
            ),
            loopback_frame_duration_ms=getattr(
                cfg_aec, "loopback_frame_duration_ms", None
            ),
        )

        try:
            self._aec = AcousticEchoCanceller(
                sample_rate=self.input_sample_rate,
                channels=self.channels,
                settings=settings,
                debug=self._debug,
            )
        except Exception:
            self._aec = None
            if self._debug:
                log.exception("audio.processing: failed to initialise AEC")
            else:
                log.warning("audio.processing: failed to initialise AEC", exc_info=False)
            return

        self._aec_playback_source = playback_source
        if playback_source == "event_bus" and topic:
            async def _on_playback(event: Event) -> None:
                data = event.payload.get("data")
                if not data:
                    return
                sr = event.payload.get("sample_rate")
                ch = event.payload.get("channels")
                if sr is not None and int(sr) != int(self.input_sample_rate):
                    if self._debug:
                        log.debug(
                            "audio.processing: skipping AEC playback frame due to sample rate mismatch (%s != %s)",
                            sr,
                            self.input_sample_rate,
                        )
                    return
                if ch is not None and int(ch) != int(self.channels):
                    if self._debug:
                        log.debug(
                            "audio.processing: skipping AEC playback frame due to channel mismatch (%s != %s)",
                            ch,
                            self.channels,
                        )
                    return
                try:
                    pcm = np.frombuffer(bytes(data), dtype=np.int16)
                except Exception:
                    if self._debug:
                        log.exception("audio.processing: failed to parse playback frame for AEC")
                    return
                if self._aec is not None:
                    self._aec.submit_farend(pcm)

            self._bus.subscribe(topic, _on_playback)
            self._aec_subscription_pattern = topic
            self._aec_subscription_handler = _on_playback
        elif playback_source == "loopback":
            self._aec_loopback_params = {
                "device_index": getattr(cfg_aec, "loopback_device_index", None),
                "source_name": getattr(cfg_aec, "loopback_source_name", None),
                "backend": getattr(cfg_aec, "loopback_backend", "auto"),
                "device_name_contains": getattr(
                    cfg_aec, "loopback_device_name_contains", None
                ),
                "frame_duration_ms": getattr(
                    cfg_aec, "loopback_frame_duration_ms", None
                ),
            }

        if self._debug:
            log.debug(
                "audio.processing: AEC enabled (frame_ms=%d, delay_ms=%s, source=%s, topic=%s)",
                self._aec.frame_duration_ms if self._aec else frame_ms,
                getattr(cfg_aec, "stream_delay_ms", 80),
                playback_source,
                topic,
            )

    def _teardown_aec_subscription(self) -> None:
        pattern = self._aec_subscription_pattern
        handler = self._aec_subscription_handler
        if pattern and handler:
            try:
                self._bus.unsubscribe(pattern, handler)
            except Exception:
                if self._debug:
                    log.exception("audio.processing: failed to unsubscribe AEC handler")
            finally:
                self._aec_subscription_pattern = None
                self._aec_subscription_handler = None
        else:
            self._aec_subscription_pattern = None
            self._aec_subscription_handler = None

    async def _start_aec_loopback_capture(self) -> None:
        if not self._running:
            return
        if self._aec is None or self._aec_loopback_params is None:
            return
        if self._aec_loopback_task is not None:
            return
        loopback_backend = self._resolve_loopback_backend(
            self._aec_loopback_params.get("backend")
        )

        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=64)
        self._aec_loopback_queue = queue
        self._aec_loopback_dropped_frames = 0
        self._aec_loopback_last_drop_log_ts = 0.0

        frame_ms = self._aec_loopback_params.get("frame_duration_ms")
        if frame_ms is None:
            frame_ms = getattr(self._aec, "frame_duration_ms", None)
        if frame_ms is None:
            vad_cfg = getattr(self.config, "vad", None)
            frame_ms = getattr(vad_cfg, "frame_duration_ms", None)
        try:
            blocksize = int(self.input_sample_rate * int(frame_ms) / 1000)
        except Exception:
            blocksize = 0
        if blocksize <= 0:
            blocksize = int(self.input_sample_rate * self.channels / 100)
        if blocksize <= 0:
            blocksize = 1024

        def _safe_put(chunk: bytes) -> None:
            try:
                queue.put_nowait(chunk)
            except asyncio.QueueFull:
                self._note_loopback_queue_full(queue)

        async def _drain() -> None:
            try:
                while self._running and self._aec is not None:
                    data = await queue.get()
                    if data is None:
                        break
                    self.submit_playback(data)
            except asyncio.CancelledError:  # pragma: no cover - cancellation is expected
                pass

        if self._should_prefer_pulse_loopback(loopback_backend):
            pulse_source = self._resolve_pulse_loopback_source_name()
            if pulse_source:
                try:
                    stream = self._create_pulse_loopback_stream(
                        source_name=pulse_source,
                        blocksize=blocksize,
                        on_chunk=lambda data: loop.call_soon_threadsafe(_safe_put, data),
                    )
                    start_method = getattr(stream, "start", None)
                    if callable(start_method):
                        start_method()
                except Exception:
                    if self._debug:
                        log.exception(
                            "audio.processing: failed to start Pulse/PipeWire loopback source"
                        )
                    else:
                        log.warning(
                            "audio.processing: failed to start Pulse/PipeWire loopback source",
                            exc_info=False,
                        )
                else:
                    self._aec_loopback_stream = stream
                    self._aec_loopback_task = asyncio.create_task(
                        _drain(), name="AudioProcessingLoopback"
                    )
                    log.info(
                        "audio.processing: AEC loopback capture started "
                        "(backend=pulse, source=%s, sample_rate=%d, blocksize=%d)",
                        pulse_source,
                        self.input_sample_rate,
                        blocksize,
                    )
                    return

        backend = self._load_sounddevice_backend()
        if backend is None:
            self._aec_loopback_queue = None
            return

        device_index = self._aec_loopback_params.get("device_index")
        if device_index is None and loopback_backend in {
            "pipewire",
            "pulse",
            "sounddevice_monitor",
        }:
            device_index = self._find_loopback_monitor_device(
                backend,
                name_contains=self._aec_loopback_params.get("device_name_contains"),
                backend_hint=loopback_backend,
            )
            if device_index is None:
                log.warning(
                    "audio.processing: no Linux loopback monitor device found; "
                    "AEC loopback capture is disabled. Run "
                    "`python utils/list_devices.py --monitors` and set "
                    "audio.processing.aec.loopback_device_index."
                )
                self._aec_loopback_queue = None
                return

        extra_settings = None
        WasapiSettings = getattr(backend, "WasapiSettings", None)
        if loopback_backend == "wasapi" and WasapiSettings is not None:
            try:
                extra_settings = WasapiSettings(loopback=True)
            except Exception:
                extra_settings = None
        elif loopback_backend == "wasapi" and WasapiSettings is None:
            log.warning(
                "audio.processing: WASAPI loopback requested but sounddevice "
                "does not expose WasapiSettings; loopback capture is disabled"
            )
            self._aec_loopback_queue = None
            return

        def _callback(indata, frames, _time, status) -> None:
            if not self._running or self._aec is None:
                return
            if status and self._debug:
                log.debug("audio.processing: loopback status %s", status)
            try:
                data = indata.tobytes()  # type: ignore[call-arg]
            except Exception:
                try:
                    data = bytes(indata)
                except Exception:
                    if self._debug:
                        log.debug("audio.processing: unexpected loopback chunk type %r", type(indata))
                    return
            if not data:
                return
            loop.call_soon_threadsafe(_safe_put, data)

        stream_kwargs: dict[str, Any] = dict(
            samplerate=self.input_sample_rate,
            blocksize=blocksize,
            channels=self.channels,
            dtype="int16",
            callback=_callback,
        )
        if device_index is not None:
            try:
                stream_kwargs["device"] = int(device_index)
            except Exception:
                stream_kwargs["device"] = device_index
        if extra_settings is not None:
            stream_kwargs["extra_settings"] = extra_settings

        try:
            stream = backend.InputStream(**stream_kwargs)
        except Exception:
            if self._debug:
                log.exception(
                    "audio.processing: failed to open %s loopback stream",
                    loopback_backend,
                )
            else:
                log.warning(
                    "audio.processing: failed to open %s loopback stream",
                    loopback_backend,
                    exc_info=False,
                )
            self._aec_loopback_queue = None
            return

        try:
            start_method = getattr(stream, "start", None)
            if callable(start_method):
                start_method()
        except Exception:
            try:
                stream.close()
            except Exception:
                pass
            if self._debug:
                log.exception(
                    "audio.processing: failed to start %s loopback stream",
                    loopback_backend,
                )
            else:
                log.warning(
                    "audio.processing: failed to start %s loopback stream",
                    loopback_backend,
                    exc_info=False,
                )
            self._aec_loopback_queue = None
            return

        self._aec_loopback_stream = stream

        self._aec_loopback_task = asyncio.create_task(
            _drain(), name="AudioProcessingLoopback"
        )
        device_desc = self._describe_sounddevice_device(backend, device_index)
        log.info(
            "audio.processing: AEC loopback capture started "
            "(backend=%s, device=%s, sample_rate=%d, blocksize=%d)",
            loopback_backend,
            device_desc,
            self.input_sample_rate,
            blocksize,
        )

    async def _stop_aec_loopback_capture(self) -> None:
        task = self._aec_loopback_task
        self._aec_loopback_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        queue = self._aec_loopback_queue
        self._aec_loopback_queue = None
        if queue is not None:
            try:
                queue.put_nowait(None)
            except Exception:
                pass

        stream = self._aec_loopback_stream
        self._aec_loopback_stream = None
        if stream is not None:
            for method_name in ("stop", "close"):
                method = getattr(stream, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        if self._debug:
                            log.exception(
                                "audio.processing: loopback stream %s() failed", method_name
                            )

    def _resolve_loopback_backend(self, requested: object) -> str:
        try:
            value = str(requested or "auto").strip().lower()
        except Exception:
            value = "auto"
        if value not in {"auto", "wasapi", "pipewire", "pulse", "sounddevice_monitor"}:
            value = "auto"
        if value != "auto":
            return value

        system = platform.system().strip().lower()
        if system.startswith("win"):
            return "wasapi"
        if system == "linux":
            return "sounddevice_monitor"
        return "sounddevice_monitor"

    def _should_prefer_pulse_loopback(self, loopback_backend: str) -> bool:
        system = platform.system().strip().lower()
        if system != "linux":
            return False
        if loopback_backend == "wasapi":
            return False
        explicit_source = self._aec_loopback_params.get("source_name")
        if explicit_source not in ("", None):
            return True
        explicit_device_index = self._aec_loopback_params.get("device_index")
        if explicit_device_index is not None and loopback_backend == "sounddevice_monitor":
            return False
        return loopback_backend in {"pulse", "pipewire", "sounddevice_monitor"}

    def _create_pulse_loopback_stream(
        self,
        *,
        source_name: str,
        blocksize: int,
        on_chunk: Any,
    ) -> Any:
        return _ParecLoopbackStream(
            source_name=source_name,
            sample_rate=self.input_sample_rate,
            channels=self.channels,
            blocksize=blocksize,
            on_chunk=on_chunk,
            debug=self._debug,
        )

    def _resolve_pulse_loopback_source_name(self) -> str | None:
        explicit_source = self._aec_loopback_params.get("source_name")
        if explicit_source not in ("", None):
            return self._resolve_pulse_source_alias(str(explicit_source))

        name_contains = self._aec_loopback_params.get("device_name_contains")
        if name_contains not in ("", None):
            return self._find_pulse_monitor_source(name_contains=str(name_contains))

        return self._resolve_pulse_source_alias("@DEFAULT_MONITOR@")

    def _resolve_pulse_source_alias(self, source_name: str) -> str | None:
        source = str(source_name or "").strip()
        if not source:
            return None
        if source == "@DEFAULT_SOURCE@":
            return self._run_pactl("get-default-source")
        if source != "@DEFAULT_MONITOR@":
            return source

        default_sink = self._run_pactl("get-default-sink")
        sources = self._query_pulse_sources()
        names = {source["name"] for source in sources}
        if default_sink:
            candidate = f"{default_sink}.monitor"
            if not names or candidate in names:
                return candidate
        monitors = [source["name"] for source in sources if source["is_monitor"]]
        if monitors:
            return monitors[0]
        return None

    def _find_pulse_monitor_source(self, *, name_contains: str) -> str | None:
        sources = self._query_pulse_sources()
        if not sources:
            return None
        default_sink = self._run_pactl("get-default-sink")
        default_monitor = f"{default_sink}.monitor" if default_sink else None
        needle = str(name_contains or "").strip().lower()
        matches: list[tuple[int, str]] = []
        for source in sources:
            if not source["is_monitor"]:
                continue
            name = source["name"]
            haystack = f"{name} {source['driver']} {source['state']}".lower()
            if needle and needle not in haystack:
                continue
            score = 0
            if name == default_monitor:
                score += 100
            if source["state"] == "RUNNING":
                score += 20
            if needle:
                score += 20
            if "monitor" in haystack:
                score += 10
            matches.append((score, name))
        if not matches:
            return None
        matches.sort(key=lambda item: (-item[0], item[1]))
        return matches[0][1]

    def _query_pulse_sources(self) -> list[dict[str, str | bool]]:
        output = self._run_pactl("list", "sources", "short")
        if not output:
            return []
        sources: list[dict[str, str | bool]] = []
        for line in output.splitlines():
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            _index, name, driver, sample_spec, state = parts[:5]
            sources.append(
                {
                    "name": name,
                    "driver": driver,
                    "sample_spec": sample_spec,
                    "state": state,
                    "is_monitor": ".monitor" in name.lower(),
                }
            )
        return sources

    def _run_pactl(self, *args: str) -> str | None:
        if shutil.which("pactl") is None:
            return None
        try:
            proc = subprocess.run(
                ["pactl", *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        text = proc.stdout.strip()
        return text or None

    def _describe_sounddevice_device(self, backend: Any, device_index: object) -> str:
        if device_index is None:
            return "default"
        try:
            index = int(device_index)
        except Exception:
            return str(device_index)
        try:
            devices = list(backend.query_devices())
        except Exception:
            return str(index)
        if index < 0 or index >= len(devices):
            return str(index)
        info = devices[index]
        if not isinstance(info, dict):
            return str(index)
        name = str(info.get("name", "") or "").strip()
        if not name:
            return str(index)
        return f"{index}:{name}"

    def _note_loopback_queue_full(self, queue: "asyncio.Queue[bytes]") -> None:
        if not self._debug:
            return
        self._aec_loopback_dropped_frames += 1
        now = time.monotonic()
        last = self._aec_loopback_last_drop_log_ts
        if last <= 0.0:
            self._aec_loopback_last_drop_log_ts = now
            return
        if now - last < 2.0:
            return
        dropped = self._aec_loopback_dropped_frames
        self._aec_loopback_dropped_frames = 0
        self._aec_loopback_last_drop_log_ts = now
        log.debug(
            "audio.processing: dropping loopback frames "
            "(queue full, dropped=%d, queued=%d/%d)",
            dropped,
            queue.qsize(),
            queue.maxsize,
        )

    def _find_loopback_monitor_device(
        self,
        backend: Any,
        *,
        name_contains: object = None,
        backend_hint: str = "sounddevice_monitor",
    ) -> int | None:
        try:
            devices = list(backend.query_devices())
        except Exception:
            if self._debug:
                log.exception("audio.processing: failed to query sound devices")
            return None

        try:
            hostapis = list(backend.query_hostapis())
        except Exception:
            hostapis = []

        needle = ""
        if name_contains not in ("", None):
            try:
                needle = str(name_contains).strip().lower()
            except Exception:
                needle = ""

        matches: list[tuple[int, int]] = []
        for index, info in enumerate(devices):
            if not isinstance(info, dict):
                continue
            try:
                max_input_channels = int(info.get("max_input_channels", 0) or 0)
            except Exception:
                max_input_channels = 0
            if max_input_channels <= 0:
                continue

            name = str(info.get("name", "") or "")
            hostapi_name = self._hostapi_name(hostapis, info.get("hostapi"))
            haystack = f"{name} {hostapi_name}".lower()
            if needle and needle not in haystack:
                continue

            score = self._loopback_monitor_score(haystack, backend_hint)
            if needle:
                score += 20
            if score <= 0:
                continue
            matches.append((score, index))

        if not matches:
            return None
        matches.sort(key=lambda item: (-item[0], item[1]))
        return matches[0][1]

    @staticmethod
    def _hostapi_name(hostapis: list[Any], hostapi_index: object) -> str:
        try:
            index = int(hostapi_index)
        except Exception:
            return ""
        if index < 0 or index >= len(hostapis):
            return ""
        api = hostapis[index]
        if not isinstance(api, dict):
            return ""
        return str(api.get("name", "") or "")

    @staticmethod
    def _loopback_monitor_score(haystack: str, backend_hint: str) -> int:
        text = haystack.lower()
        score = 0
        if "monitor" in text:
            score += 100
        if "sink" in text:
            score += 30
        if "output" in text:
            score += 20
        if "pipewire" in text:
            score += 10
        if "pulse" in text:
            score += 10
        if backend_hint == "pipewire" and "pipewire" in text:
            score += 30
        if backend_hint == "pulse" and "pulse" in text:
            score += 30
        return score

    def _load_sounddevice_backend(self) -> Any | None:
        if self._sounddevice_backend is not None:
            return self._sounddevice_backend
        if self._sounddevice_backend_error:
            return None
        try:
            import sounddevice  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            self._sounddevice_backend_error = True
            if self._debug:
                log.exception("audio.processing: sounddevice is unavailable for loopback")
            else:
                log.warning(
                    "audio.processing: sounddevice is unavailable for loopback (%s)",
                    exc,
                    exc_info=False,
                )
            return None
        self._sounddevice_backend = sounddevice
        return sounddevice

    async def _worker(self) -> None:
        try:
            while True:
                item = await self._input_queue.get()
                if item is None:
                    if self._debug:
                        log.debug("audio.processing: worker received stop signal")
                    break
                frame = self._process_frame(item)
                segment_duration = (
                    frame.voice_active_duration
                    if (frame.voice_detected or self._segment_ended_in_current_chunk)
                    else 0.0
                )
                await self._bus.publish(
                    cfg.events.audio.processed_frame,
                    data=frame.data,
                    sample_rate=frame.sample_rate,
                    channels=frame.channels,
                    voice_activity=frame.voice_detected,
                    timestamp=frame.timestamp,
                    vad_probability=frame.vad_probability,
                    vad_speech_frames=frame.vad_speech_frames,
                    vad_total_frames=frame.vad_total_frames,
                    webrtc_probability=frame.webrtc_probability,
                    silero_probability=frame.silero_probability,
                    silero_invocations=frame.silero_invocations,
                    voice_active_duration=frame.voice_active_duration,
                    segment_duration=segment_duration,
                )
                vad_cfg = getattr(self.config, "vad", None)
                if vad_cfg and getattr(vad_cfg, "publish_voice_activity", False):
                    await self._maybe_publish_vad(frame)
                await self._output_queue.put(frame)
        except Exception:
            log.exception("audio.processing: worker crashed")
        finally:
            self._running = False
            await self._output_queue.put(None)

    async def _maybe_publish_vad(self, frame: ProcessedAudioFrame) -> None:
        vad_cfg = getattr(self.config, "vad", None)
        if not vad_cfg:
            return
        state = frame.voice_detected
        if state or self._segment_ended_in_current_chunk:
            duration = frame.voice_active_duration
        else:
            duration = 0.0
        payload = {
            "active": state,
            "timestamp": frame.timestamp,
            "vad_probability": frame.vad_probability,
            "vad_speech_frames": frame.vad_speech_frames,
            "vad_total_frames": frame.vad_total_frames,
            "webrtc_probability": frame.webrtc_probability,
            "silero_probability": frame.silero_probability,
            "silero_invocations": frame.silero_invocations,
            "voice_active_duration": duration,
            "segment_duration": duration,
        }
        hangover_default_ms = getattr(
            vad_cfg, "hangover_ms", DEFAULT_ACTIVE_REPUBLISH_INTERVAL_MS
        )
        try:
            hangover_default_ms = float(hangover_default_ms)
        except (TypeError, ValueError):
            hangover_default_ms = DEFAULT_ACTIVE_REPUBLISH_INTERVAL_MS
        if hangover_default_ms <= 0.0:
            hangover_default_ms = DEFAULT_ACTIVE_REPUBLISH_INTERVAL_MS

        raw_interval_ms = getattr(
            vad_cfg,
            "active_republish_interval_ms",
            hangover_default_ms,
        )
        if raw_interval_ms in (None, ""):
            republish_interval_ms = hangover_default_ms
        else:
            try:
                republish_interval_ms = float(raw_interval_ms)
            except (TypeError, ValueError):
                republish_interval_ms = hangover_default_ms
        if republish_interval_ms <= 0.0:
            republish_interval_ms = hangover_default_ms
        if republish_interval_ms <= 0.0:
            republish_interval_ms = DEFAULT_ACTIVE_REPUBLISH_INTERVAL_MS
        republish_interval_s = republish_interval_ms / 1000.0

        publish_event = False
        if self._last_voice_state is None or state != self._last_voice_state:
            self._last_voice_state = state
            self._last_voice_state_change_ts = frame.timestamp
            publish_event = True
        elif state:
            if frame.timestamp - self._last_active_republish_ts >= republish_interval_s:
                publish_event = True

        if publish_event:
            if state:
                self._last_active_republish_ts = frame.timestamp
            await self._bus.publish(cfg.events.audio.voice_activity, **payload)

    def _process_frame(self, data: bytes) -> ProcessedAudioFrame:
        process_start_ns = perf.now_ns()
        timestamp = time.time()
        pcm = np.frombuffer(data, dtype=np.int16).astype(np.int16, copy=True)
        pcm = self._apply_aec(pcm)
        channels = max(1, int(self.channels))
        highpass_cfg = getattr(self.config, "highpass", None)
        highpass_enabled = bool(
            self.config.enabled and getattr(highpass_cfg, "enabled", True)
        )
        pcm = self._dc_filter.process(pcm, enabled=highpass_enabled)
        pcm = self._apply_noise_suppression(pcm)
        if self.output_sample_rate > 0 and channels > 0:
            frame_samples = pcm.size // channels
            chunk_duration_ms = (frame_samples / float(self.output_sample_rate)) * 1000.0
        else:
            chunk_duration_ms = 0.0
        voice, vad_metrics = self._detect_voice(pcm, chunk_duration_ms)
        if voice:
            active_duration_ms = self._current_voice_active_duration_ms()
        elif self._segment_ended_in_current_chunk:
            active_duration_ms = self._last_segment_padded_duration_ms
        else:
            active_duration_ms = 0.0
        active_duration = active_duration_ms / 1000.0
        agc_cfg = getattr(self.config, "agc", None)
        agc_settings = AGCSettings(
            enabled=bool(self.config.enabled and getattr(agc_cfg, "enabled", False)),
            target_level_dbfs=float(getattr(agc_cfg, "target_level_dbfs", -20.0)),
            max_gain_db=float(getattr(agc_cfg, "max_gain_db", 20.0)),
            attack_ms=float(getattr(agc_cfg, "attack_ms", 10.0)),
            release_ms=float(getattr(agc_cfg, "release_ms", 200.0)),
        )
        processed_pcm = self._agc.process(
            pcm,
            sample_rate=self.output_sample_rate,
            channels=self.channels,
            settings=agc_settings,
        )
        processed_data = processed_pcm.tobytes()
        #if self._debug:
        #    log.debug(
        #        (
        #            "audio.processing: frame processed (voice=%s, vad=%.3f, "
        #            "webrtc=%.3f, silero=%.3f, invocations=%d, duration_ms=%.2f)"
        #        ),
        #        voice,
        #        vad_metrics["vad_probability"],
        #        vad_metrics["webrtc_probability"],
        #        vad_metrics["silero_probability"],
        #        vad_metrics["silero_invocations"],
        #        chunk_duration_ms,
        #    )
        frame = ProcessedAudioFrame(
            data=processed_data,
            sample_rate=self.output_sample_rate,
            channels=self.channels,
            voice_detected=voice,
            timestamp=timestamp,
            vad_probability=vad_metrics["vad_probability"],
            vad_speech_frames=vad_metrics["vad_speech_frames"],
            vad_total_frames=vad_metrics["vad_total_frames"],
            voice_active_duration=active_duration,
            webrtc_probability=vad_metrics["webrtc_probability"],
            silero_probability=vad_metrics["silero_probability"],
            silero_invocations=int(vad_metrics["silero_invocations"]),
        )
        perf.emit(
            "input",
            "processor_frame",
            duration_ms=perf.elapsed_ms(process_start_ns),
            voice=voice,
            vad_probability=vad_metrics["vad_probability"],
            bytes=len(processed_data),
        )
        return frame

    @staticmethod
    def _compute_rms(pcm: np.ndarray) -> float:
        if pcm.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))

    def _apply_aec(self, pcm: np.ndarray) -> np.ndarray:
        data = np.ascontiguousarray(pcm, dtype=np.int16)
        if data.size == 0:
            return data

        canceller = self._aec
        if canceller is not None:
            try:
                cleaned = canceller.process(data)
            except Exception:
                if self._debug:
                    log.exception("audio.processing: AEC failure; disabling module")
                else:
                    log.warning(
                        "audio.processing: AEC failure; disabling module", exc_info=False
                    )
                self._aec = None
            else:
                cleaned_arr = np.ascontiguousarray(cleaned, dtype=np.int16)
                if self._resampler is None:
                    if cleaned_arr.size == 0:
                        return data
                    if cleaned_arr.size == data.size:
                        data = cleaned_arr
                    elif cleaned_arr.size < data.size:
                        tail = np.ascontiguousarray(data[cleaned_arr.size :], dtype=np.int16)
                        data = np.concatenate([cleaned_arr, tail])
                    else:
                        data = np.ascontiguousarray(cleaned_arr[: data.size], dtype=np.int16)
                else:
                    data = cleaned_arr

        resampler = self._resampler
        if resampler is not None and data.size > 0:
            return resampler.process(data)
        return data

    def _apply_noise_suppression(self, pcm: np.ndarray) -> np.ndarray:
        suppressor = self._noise_suppressor
        if suppressor is None or pcm.size == 0:
            return pcm
        try:
            return suppressor.process(pcm)
        except Exception:
            if self._debug:
                log.exception("audio.processing: noise suppression failure")
            else:
                log.warning("audio.processing: noise suppression failure", exc_info=False)
            self._noise_suppressor = None
            return pcm

    def _detect_voice(
        self, pcm: np.ndarray, chunk_duration_ms: float
    ) -> tuple[bool, dict[str, float | int]]:
        vad_start_ns = perf.now_ns()
        was_active = self._voice_active or self._pending_segment_end
        vad_cfg = getattr(self.config, "vad", None)

        metrics: dict[str, float | int] = {
            "vad_probability": 0.0,
            "vad_speech_frames": 0,
            "vad_total_frames": 0,
            "webrtc_probability": 0.0,
            "silero_probability": float(self._silero_last_probability),
            "silero_invocations": 0,
        }
        self._segment_ended_in_current_chunk = False
        if not (self.config.enabled and getattr(vad_cfg, "enabled", True)):
            if chunk_duration_ms > 0.0:
                self._update_vad_state(False, chunk_duration_ms)
            return self._voice_active, metrics

        # --- Energy gate ---
        # Compute RMS and dBFS.
        rms = self._compute_rms(pcm)
        db = self._rms_to_db(rms)
        if db < float(getattr(vad_cfg, "energy_threshold_db", -90.0)):
            if chunk_duration_ms > 0.0:
                self._update_vad_state(False, chunk_duration_ms)
            return self._voice_active, metrics

        if pcm.size == 0:
            if chunk_duration_ms > 0.0:
                self._update_vad_state(False, chunk_duration_ms)
            return self._voice_active, metrics

        if self._vad_downmix and self.channels > 1:
            try:
                mono = pcm.reshape(-1, self.channels)[:, 0]
            except ValueError:
                mono = pcm[:: self.channels]
        else:
            mono = pcm

        mono = np.ascontiguousarray(mono, dtype=np.int16)

        frame_bytes = self._vad_frame_bytes
        if frame_bytes <= 0:
            if chunk_duration_ms > 0.0:
                self._update_vad_state(False, chunk_duration_ms)
            return self._voice_active, metrics

        frames = self._split_for_vad(mono.tobytes(), frame_bytes)
        total_frames = len(frames)
        metrics["vad_total_frames"] = total_frames
        if total_frames == 0:
            if chunk_duration_ms > 0.0:
                self._update_vad_state(False, chunk_duration_ms)
            return self._voice_active, metrics

        sample_rate = int(self.output_sample_rate)
        frame_duration_ms = float(self._vad_frame_duration_ms)
        if frame_duration_ms <= 0.0 and total_frames > 0:
            frame_duration_ms = chunk_duration_ms / total_frames if chunk_duration_ms > 0.0 else 0.0

        pipeline_mode = self._vad_pipeline
        vad = self._vad if self._allow_webrtc else None
        webrtc_results: list[bool] = []
        webrtc_speech_frames = 0
        if vad is not None:
            for frame in frames:
                result = bool(vad.is_speech(frame, sample_rate))
                if result:
                    webrtc_speech_frames += 1
                webrtc_results.append(result)

        webrtc_probability = (
            webrtc_speech_frames / total_frames if total_frames and vad is not None else 0.0
        )
        metrics["webrtc_probability"] = webrtc_probability

        silero_helper = self._silero_vad if self._allow_silero else None
        use_silero = False
        low_band_cfg = getattr(vad_cfg, "webrtc_escalation_low_threshold", 0.0)
        try:
            low_band = float(low_band_cfg)
        except (TypeError, ValueError):
            low_band = 0.0
        low_band = max(0.0, min(1.0, low_band))

        high_band_cfg = getattr(vad_cfg, "webrtc_escalation_high_threshold", 1.0)
        try:
            high_band = float(high_band_cfg)
        except (TypeError, ValueError):
            high_band = 1.0
        high_band = max(low_band, min(1.0, max(0.0, high_band)))

        if pipeline_mode == "silero":
            use_silero = silero_helper is not None
        elif pipeline_mode == "hybrid":
            if vad is None and silero_helper is not None:
                use_silero = True
            elif vad is not None:
                if webrtc_probability < low_band:
                    use_silero = False
                elif webrtc_probability > high_band:
                    use_silero = False
                else:
                    use_silero = silero_helper is not None
        else:
            use_silero = False

        silero_probability = float(self._silero_last_probability)
        silero_invocations = 0

        if use_silero and silero_helper is not None and total_frames > 0:
            joined_frames = b"".join(frames)
            if joined_frames:
                self._silero_pending_audio.extend(joined_frames)
                frame_window_ms = frame_duration_ms if frame_duration_ms > 0.0 else 0.0
                if frame_window_ms <= 0.0:
                    frame_window_ms = chunk_duration_ms
                    self._silero_pending_ms += frame_window_ms
                else:
                    self._silero_pending_ms += frame_window_ms * total_frames

            cadence_cfg = getattr(vad_cfg, "silero_cadence_ms", None)
            min_duration_cfg = getattr(vad_cfg, "silero_min_activation_duration_ms", 0.0)
            try:
                cadence_ms = float(cadence_cfg) if cadence_cfg is not None else 0.0
            except (TypeError, ValueError):
                cadence_ms = 0.0
            if cadence_ms <= 0.0:
                fallback_cadence_ms = float(self._vad_frame_duration_ms)
                if fallback_cadence_ms <= 0.0:
                    fallback_cadence_ms = frame_duration_ms
                cadence_ms = max(fallback_cadence_ms, 0.0)

            try:
                min_activation_ms = float(min_duration_cfg)
            except (TypeError, ValueError):
                min_activation_ms = float(self._vad_frame_duration_ms)
            if min_activation_ms < 0.0:
                min_activation_ms = 0.0

            required_ms = max(min_activation_ms, cadence_ms)
            if required_ms <= 0.0:
                required_ms = frame_duration_ms if frame_duration_ms > 0.0 else 0.0

            if required_ms <= 0.0 or self._silero_pending_ms >= required_ms:
                if self._silero_pending_audio:
                    pcm_batch = np.frombuffer(
                        bytes(self._silero_pending_audio), dtype=np.int16
                    )
                else:
                    pcm_batch = np.zeros(0, dtype=np.int16)
                try:
                    silero_probability = float(silero_helper.predict(pcm_batch, sample_rate))
                    self._silero_last_probability = silero_probability
                    silero_invocations = 1
                except Exception as exc:
                    if self._debug:
                        log.exception(
                            "audio.processing: Silero predict failed; disabling Silero VAD "
                            "for this processor and falling back to WebRTC when available"
                        )
                    else:
                        log.warning(
                            "audio.processing: Silero predict failed; disabling Silero VAD "
                            "for this processor and falling back to WebRTC when available (%s)",
                            exc,
                            exc_info=False,
                        )
                    self._silero_vad = None
                    self._allow_silero = False
                    silero_helper = None
                    use_silero = False
                    self._silero_last_probability = 0.0
                    silero_probability = 0.0
                finally:                
                    self._silero_pending_audio.clear()
                    self._silero_pending_ms = 0.0
            else:
                silero_probability = float(self._silero_last_probability)
        else:
            if self._silero_pending_audio:
                self._silero_pending_audio.clear()
            self._silero_pending_ms = 0.0
            if not use_silero:
                self._silero_last_probability = 0.0

        metrics["silero_probability"] = silero_probability
        metrics["silero_invocations"] = silero_invocations

        threshold = float(getattr(vad_cfg, "probability_threshold", 0.0))
        final_probability = webrtc_probability
        frame_results: list[bool] = webrtc_results.copy() if webrtc_results else []

        if use_silero and silero_helper is not None:
            final_probability = silero_probability
            decision = final_probability >= threshold and total_frames > 0
            frame_results = [decision] * total_frames
        elif vad is not None:
            if webrtc_probability < low_band:
                frame_results = [False] * total_frames
            else:
                frame_results = webrtc_results
        else:
            frame_results = [False] * total_frames

        if use_silero and silero_helper is not None and final_probability < threshold:
            frame_results = [False] * total_frames

        speech_frames = sum(1 for flag in frame_results if flag)
        metrics["vad_speech_frames"] = speech_frames
        metrics["vad_probability"] = final_probability

        if frame_duration_ms > 0.0:
            for result in frame_results:
                self._update_vad_state(result, frame_duration_ms)

        if self._voice_active:
            voice_detected = True
        elif self._pending_segment_end:
            voice_detected = self._is_hangover_active()
        else:
            self._is_hangover_active()
            voice_detected = False

        #if self._debug:
        #    log.debug(
        #        (
        #            "audio.processing: VAD decision voice=%s (final=%.3f, threshold=%.3f, "
        #            "webrtc=%.3f, silero=%.3f, speech_frames=%d/%d)"
        #        ),
        #        voice_detected,
        #        final_probability,
        #        threshold,
        #        webrtc_probability,
        #        silero_probability,
        #        speech_frames,
        #        total_frames,
        #    )

        is_active = self._voice_active or self._pending_segment_end
        if is_active and not was_active:
            perf.emit(
                "input",
                "vad_start",
                vad_probability=metrics["vad_probability"],
                duration_ms=perf.elapsed_ms(vad_start_ns),
            )
        elif was_active and not is_active:
            perf.emit(
                "input",
                "vad_end",
                vad_probability=metrics["vad_probability"],
                duration_ms=perf.elapsed_ms(vad_start_ns),
            )

        return voice_detected, metrics

    def _update_vad_state(self, is_speech: bool, duration_ms: float) -> None:
        if duration_ms <= 0.0:
            return

        vad_cfg = getattr(self.config, "vad", None)
        pad_ms = max(0.0, float(getattr(vad_cfg, "speech_pad_ms", 0.0)))
        min_speech_ms = max(0.0, float(getattr(vad_cfg, "min_speech_duration_ms", 0.0)))
        min_silence_ms = max(0.0, float(getattr(vad_cfg, "min_silence_duration_ms", 0.0)))
        hangover_ms = max(0.0, float(getattr(vad_cfg, "hangover_ms", 0.0)))

        frame_start_ms = self._stream_time_ms
        self._stream_time_ms += duration_ms
        frame_end_ms = self._stream_time_ms

        if is_speech:
            if self._segment_start_ms is None:
                self._segment_start_ms = frame_start_ms
            self._speech_run_ms += duration_ms
            self._silence_run_ms = 0.0
            self._last_speech_time_ms = frame_end_ms
            self._last_positive_vad_ms = frame_end_ms
            if hangover_ms > 0.0:
                self._hangover_expires_ms = frame_end_ms + hangover_ms
            else:
                self._hangover_expires_ms = None
            if not self._voice_active and self._speech_run_ms >= min_speech_ms:
                speech_start_ms = (
                    self._segment_start_ms if self._segment_start_ms is not None else frame_start_ms
                )
                self._segment_padded_start_ms = max(0.0, speech_start_ms - pad_ms)
                self._voice_active = True
                self._last_segment_padded_duration_ms = 0.0
                self._pending_segment_end = False
                if self._debug:
                    log.debug(
                        "audio.processing: VAD segment started (start_ms=%.2f, pad_ms=%.2f)",
                        speech_start_ms,
                        pad_ms,
                    )
        else:
            self._silence_run_ms += duration_ms
            self._speech_run_ms = 0.0
            if self._voice_active and self._silence_run_ms >= min_silence_ms:
                speech_start_ms = (
                    self._segment_start_ms
                    if self._segment_start_ms is not None
                    else (
                        self._last_speech_time_ms
                        if self._last_speech_time_ms is not None
                        else frame_end_ms - self._silence_run_ms
                    )
                )
                padded_start_ms = (
                    self._segment_padded_start_ms
                    if self._segment_padded_start_ms is not None
                    else max(0.0, speech_start_ms - pad_ms)
                )
                last_speech_ms = (
                    self._last_speech_time_ms
                    if self._last_speech_time_ms is not None
                    else frame_end_ms - self._silence_run_ms
                )
                padded_end_ms = last_speech_ms + pad_ms
                self._last_segment_padded_duration_ms = max(
                    0.0, padded_end_ms - padded_start_ms
                )
                raw_segment_duration_ms = max(0.0, last_speech_ms - speech_start_ms)
                self._voice_active = False
                self._last_positive_vad_ms = last_speech_ms
                if hangover_ms > 0.0:
                    self._hangover_expires_ms = last_speech_ms + hangover_ms
                    self._pending_segment_end = True
                    if self._debug:
                        log.debug(
                            "audio.processing: VAD hangover active until %.2f ms",
                            self._hangover_expires_ms,
                        )
                else:
                    self._pending_segment_end = False
                    self._hangover_expires_ms = None
                    self._last_positive_vad_ms = None
                    self._segment_ended_in_current_chunk = True
                    self._finalize_segment_state()
                    if self._debug:
                        log.debug(
                            "audio.processing: VAD segment ended (duration_ms=%.2f, padded_duration_ms=%.2f)",
                            raw_segment_duration_ms,
                            self._last_segment_padded_duration_ms,
                        )
            elif not self._voice_active:
                if self._silence_run_ms >= min_silence_ms:
                    self._segment_start_ms = None
                    self._segment_padded_start_ms = None
                if hangover_ms <= 0.0:
                    self._hangover_expires_ms = None

    def _finalize_segment_state(self) -> None:
        self._segment_start_ms = None
        self._segment_padded_start_ms = None
        self._last_speech_time_ms = None

    def _is_hangover_active(self) -> bool:
        if not self._pending_segment_end:
            expire_ms = self._hangover_expires_ms
            if expire_ms is not None and self._stream_time_ms > expire_ms:
                self._hangover_expires_ms = None
                if self._last_positive_vad_ms is not None:
                    self._last_positive_vad_ms = None
            return False

        expire_ms = self._hangover_expires_ms
        if expire_ms is None:
            self._pending_segment_end = False
            self._segment_ended_in_current_chunk = True
            self._finalize_segment_state()
            self._last_positive_vad_ms = None
            return False

        current_time_ms = self._stream_time_ms
        if current_time_ms <= expire_ms:
            return True

        self._hangover_expires_ms = None
        self._pending_segment_end = False
        self._segment_ended_in_current_chunk = True
        self._finalize_segment_state()
        if self._last_positive_vad_ms is not None and current_time_ms > expire_ms:
            self._last_positive_vad_ms = None
        return False

    def _current_voice_active_duration_ms(self) -> float:
        if not self._voice_active:
            if self._pending_segment_end:
                return self._last_segment_padded_duration_ms
            return 0.0

        vad_cfg = getattr(self.config, "vad", None)
        pad_ms = max(0.0, float(getattr(vad_cfg, "speech_pad_ms", 0.0)))
        speech_start_ms = self._segment_start_ms
        if speech_start_ms is None:
            return 0.0

        padded_start_ms = (
            self._segment_padded_start_ms
            if self._segment_padded_start_ms is not None
            else max(0.0, speech_start_ms - pad_ms)
        )

        last_speech_ms = (
            self._last_speech_time_ms
            if self._last_speech_time_ms is not None
            else self._stream_time_ms
        )
        current_end_ms = max(self._stream_time_ms, last_speech_ms)
        duration_ms = max(0.0, current_end_ms - padded_start_ms)
        return duration_ms

    def _split_for_vad(self, data: bytes, frame_bytes: int) -> list[bytes]:
        if frame_bytes <= 0:
            return []

        tail = self._vad_tail
        if data:
            tail.extend(data)

        total = len(tail)
        if total < frame_bytes:
            return []

        frames: list[bytes] = []
        offset = 0
        while offset + frame_bytes <= total:
            frames.append(bytes(tail[offset : offset + frame_bytes]))
            offset += frame_bytes

        if offset:
            del tail[:offset]

        return frames

    @staticmethod
    def _rms_to_db(rms: float) -> float:
        if rms <= 0.0:
            return -120.0
        return 20.0 * math.log10(rms / 32768.0)


WindowsAudioProcessor = AudioProcessor

__all__ = ["ProcessedAudioFrame", "AudioProcessor", "WindowsAudioProcessor"]
