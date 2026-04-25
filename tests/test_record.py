#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live audio recording test with processing and WAV output.
- Reads PCM from microphone via MicrophoneStream (PyAudio callback API).
- Passes audio through WindowsAudioProcessor (RMS/VAD calculation; denoise is handled by driver).
- Writes result as 16-bit PCM WAV.
- (New option) --voice-only: write only VAD-active speech segments to WAV.

Prints VAD (voice activity) events and brief statistics.

Environment requirements:
- Project modules must be available: core.config.cfg, microphone.MicrophoneStream, processing.WindowsAudioProcessor.
=======
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import logging
import math
import os
import sys
import time
import wave
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

# --- Project imports ---
try:
    from core.config import cfg  # project configuration
except Exception as exc:
    print("Failed to import core.config.cfg. Run from project root.", file=sys.stderr)
    raise

try:
    from audio.processing import WindowsAudioProcessor

except Exception as exc:
    print(
        "Failed to import processing.WindowsAudioProcessor.",
        file=sys.stderr,
    )
    raise

try:
    from audio.microphone import MicrophoneStream  # async PCM stream from microphone
except ImportError:
    MicrophoneStream = None  # acceptable if raw audio mode is not needed
    print(
        "Failed to import microphone.MicrophoneStream. "
        "Install project dependencies; recording is unavailable otherwise.",
        file=sys.stderr,
    )
from core.config import cfg, load as load_config

log = logging.getLogger("test_record")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record audio to WAV through WindowsAudioProcessor. Supports A/B output and plots."
    )
    p.add_argument("--out", "-o", type=str, default="recording.wav", help="Path to base output WAV file (processed)")
    p.add_argument("--duration", "-d", type=float, default=10.0, help="Recording duration in seconds")
    p.add_argument("--rate", type=int, default=None, help="Sample rate, Hz (default from cfg)")
    p.add_argument("--channels", type=int, default=None, help="Channel count (default from cfg)")
    p.add_argument("--chunk", type=int, default=None, help="Audio frame/buffer size (default from cfg)")
    p.add_argument("--raw", action="store_true", help="Write raw audio without processing")
    p.add_argument("--ab", action="store_true", help="Save RAW and PROCESSED simultaneously (creates *_raw.wav and *_proc.wav)")
    p.add_argument("--voice-only", action="store_true", help="Write only segments where VAD=VOICE (if VAD is off, writes everything)")
    p.add_argument("--no-vad-log", action="store_true", help="Disable VAD event logging")
    p.add_argument("--queue", type=int, default=128, help="Max queue size (0 = unlimited)")
    p.add_argument("--log", type=str, default="INFO", help="Log level: DEBUG/INFO/WARN/ERROR")
    p.add_argument("--metrics", type=str, default=None, help="Path to metrics CSV (time, RMS, dB, VAD)")
    p.add_argument("--plot", action="store_true", help="Save PNG plots for level and VAD")
    p.add_argument("--plot-prefix", type=str, default=None, help="PNG filename prefix (default derived from --out)")
    return p.parse_args()


def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclass
class MetricRow:
    t: float
    rms_raw: float | None
    db_raw: float | None
    rms_proc: float | None
    db_proc: float | None
    vad: int | None  # 1=voice, 0=silence, None=unknown


def compute_rms_db(data: bytes) -> tuple[float, float]:
    pcm = np.frombuffer(data, dtype=np.int16)
    if pcm.size == 0:
        return 0.0, -120.0
    rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
    if rms <= 0.0:
        return rms, -120.0
    energy_db = 20.0 * math.log10(rms / 32768.0)
    return rms, energy_db


def ensure_wav_path(path: str) -> str:
    return path if path.lower().endswith(".wav") else f"{path}.wav"


async def record_raw_to_wav(
    out_path: str,
    *,
    duration: float,
    sample_rate: Optional[int],
    chunk_size: Optional[int],
    channels: Optional[int],
    queue_maxsize: int,
) -> None:
    """Record raw audio without processing into WAV."""
    if MicrophoneStream is None:
        raise RuntimeError("MicrophoneStream is unavailable. Install project dependencies.")
    sr = int(sample_rate or cfg.audio.input.input_sample_rate)
    ch = int(channels or cfg.audio.input.channels)
    cs = int(chunk_size or cfg.audio.input.chunk_size)

    frames = []
    t_end = time.time() + max(0.1, duration)

    async with MicrophoneStream(
        sample_rate=sr,
        chunk_size=cs,
        channels=ch,
        device_index=cfg.audio.input.device_index,
        queue_maxsize=queue_maxsize,
    ) as mic:
        async for chunk in mic:
            frames.append(chunk)
            if time.time() >= t_end:
                break

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"".join(frames))

    log.info("RAW saved to %s (%.2f sec, %d frames)", out_path, duration, len(frames))


async def record_processed_to_wav(
    out_path: str,
    *,
    duration: float,
    sample_rate: Optional[int],
    chunk_size: Optional[int],
    channels: Optional[int],
    queue_maxsize: int,
    vad_log: bool,
    ab_mode: bool,
    metrics_csv: Optional[str],
    plot: bool,
    plot_prefix: Optional[str],
    voice_only: bool,
) -> None:

    """Record audio processed by WindowsAudioProcessor into WAV."""
    if MicrophoneStream is None:
        raise RuntimeError(
            "MicrophoneStream is unavailable. Install project dependencies for audio recording."
        )

    sr = int(sample_rate or cfg.audio.input.input_sample_rate)
    ch = int(channels or cfg.audio.input.channels)
    cs = int(chunk_size or cfg.audio.input.chunk_size)

    # File names.
    base_out = ensure_wav_path(out_path)
    far_out = f"{os.path.splitext(base_out)[0]}_far.wav"
    far_bytes_written = 0
    far_frames_written = 0
    wf_far: wave.Wave_write | None = None
    far_sample_rate: Optional[int] = None
    far_channels: Optional[int] = None
    far_capture_enabled = False

    def close_far_handle() -> None:
        nonlocal wf_far
        if wf_far is not None:
            wf_far.close()
            wf_far = None

    far_close = close_far_handle

    if ab_mode:
        proc_out = base_out.replace(".wav", "_proc.wav")
        raw_out = base_out.replace(".wav", "_raw.wav")
    else:
        proc_out = base_out
        raw_out = None

    wf_proc = wave.open(proc_out, "wb")
    wf_proc.setnchannels(ch)
    wf_proc.setsampwidth(2)

    wf_raw = None
    if raw_out:
        wf_raw = wave.open(raw_out, "wb")
        wf_raw.setnchannels(ch)
        wf_raw.setsampwidth(2)
        wf_raw.setframerate(sr)

    wrote_frames = 0
    t0 = time.time()
    t_end = t0 + max(0.1, duration)
    last_vad = None
    metrics: List[MetricRow] = []

    async with contextlib.AsyncExitStack() as stack:
        mic = await stack.enter_async_context(
            MicrophoneStream(
                sample_rate=sr,
                chunk_size=cs,
                channels=ch,
                device_index=cfg.audio.input.device_index,
                queue_maxsize=queue_maxsize,
            )
        )
        proc = await stack.enter_async_context(
            WindowsAudioProcessor(
                sample_rate=sr,
                channels=ch,
                queue_maxsize=queue_maxsize,
                config=cfg.audio.processing,
            )
        )

        processed_rate_value = getattr(proc, "sample_rate", sr)
        try:
            processed_rate = int(processed_rate_value or sr)
        except (TypeError, ValueError):
            processed_rate = sr
        if processed_rate <= 0:
            processed_rate = sr
        wf_proc.setframerate(processed_rate)

        aec = getattr(proc, "_aec", None)
        if aec is not None and hasattr(aec, "submit_farend"):
            far_capture_enabled = True
            far_sample_rate_value = getattr(aec, "sample_rate", sr)
            try:
                far_sample_rate = int(far_sample_rate_value)
            except (TypeError, ValueError):
                far_sample_rate = sr
            if far_sample_rate <= 0:
                far_sample_rate = sr

            far_channels_value = getattr(aec, "channels", ch)
            try:
                far_channels = int(far_channels_value)
            except (TypeError, ValueError):
                far_channels = ch if ch else 1
            if far_channels <= 0:
                far_channels = ch if ch else 1
            if far_channels <= 0:
                far_channels = 1

            original_submit_farend = aec.submit_farend

            def restore_submit_farend() -> None:
                aec.submit_farend = original_submit_farend

            stack.callback(restore_submit_farend)
            stack.callback(close_far_handle)

            def farend_wrapper(pcm: np.ndarray) -> None:
                nonlocal wf_far, far_bytes_written, far_frames_written
                if wf_far is None:
                    try:
                        wf_far = wave.open(far_out, "wb")
                        wf_far.setnchannels(int(far_channels))
                        wf_far.setsampwidth(2)
                        wf_far.setframerate(int(far_sample_rate))
                    except Exception:
                        log.exception("Failed to open WAV for far-end recording: %s", far_out)
                        wf_far = None
                if wf_far is not None:
                    pcm_bytes = np.asarray(pcm, dtype=np.int16).tobytes()
                    if pcm_bytes:
                        wf_far.writeframes(pcm_bytes)
                        far_bytes_written += len(pcm_bytes)
                        channels_for_frames = int(far_channels) if far_channels else 1
                        frame_bytes = 2 * max(1, channels_for_frames)
                        far_frames_written += len(pcm_bytes) // frame_bytes
                original_submit_farend(pcm)

            aec.submit_farend = farend_wrapper  # type: ignore[assignment]

        # Enable voice filtering only if VAD is actually enabled.
        voice_filter_active = bool(
            voice_only and getattr(proc, "config", None) is not None
            and getattr(proc.config, "enabled", True)
            and getattr(getattr(proc.config, "vad", None), "enabled", False)
        )
        if voice_only and not voice_filter_active:
            log.warning("--voice-only requested, but VAD is disabled in config; writing full stream without filtering")

        stop_requested = False
        skip_remaining = False

        async def producer() -> None:
            nonlocal stop_requested
            async for mic_bytes in mic:
                if stop_requested:
                    break
                # A/B mode: write RAW immediately (without voice filtering for fair comparison).
                if wf_raw is not None and mic_bytes:
                    wf_raw.writeframes(mic_bytes)
                await proc.submit(mic_bytes)
                if time.time() >= t_end:
                    stop_requested = True
                    break
            await mic.stop()
            await proc.stop()

        async def consumer() -> None:
            nonlocal wrote_frames, last_vad, stop_requested, skip_remaining
            async for frame in proc:
                if skip_remaining:
                    continue

                # Filter: write voice frames only when enabled.
                if voice_filter_active and not frame.voice_detected:
                    # Metrics and logs below are still collected.
                    pass
                else:
                    wf_proc.writeframes(frame.data)
                    wrote_frames += 1

                now = frame.timestamp
                rms_proc, db_proc = compute_rms_db(frame.data)
                # VAD logging.
                if vad_log:
                    if last_vad is None or frame.voice_detected != last_vad:
                        last_vad = frame.voice_detected
                        state = "VOICE" if frame.voice_detected else "SILENCE"
                        log.info(
                            "VAD: %-7s  energy %.1f dB  rms=%.3f",
                            state,
                            db_proc,
                            rms_proc,
                        )
                # Metrics.
                if ab_mode or metrics_csv or plot:
                    metrics.append(
                        MetricRow(
                            t=now - t0,
                            rms_raw=None,
                            db_raw=None,
                            rms_proc=float(rms_proc),
                            db_proc=float(db_proc),
                            vad=int(bool(frame.voice_detected)),
                        )
                    )
                if not stop_requested and now >= t_end:
                    stop_requested = True
                if stop_requested:
                    skip_remaining = True

        await asyncio.gather(producer(), consumer())

        far_close()
        if (
            far_capture_enabled
            and wf_far is None
            and far_bytes_written > 0
            and far_sample_rate
            and far_channels
        ):
            far_duration = far_bytes_written / (2 * far_sample_rate * far_channels)
            log.info(
                "Far-end saved to %s (%.2f sec, %d frames)",
                far_out,
                far_duration,
                far_frames_written,
            )
        else:
            log.warning(
                "Far-end WAV %s skipped: AEC is disabled or no far-end data was received",
                far_out,
            )

    # Close files.
    wf_proc.close()
    if wf_raw is not None:
        wf_raw.close()

    dur = time.time() - t0
    log.info(
        "Processed saved to %s (requested=%.2f sec, actual=%.2f sec, %d frames)",
        proc_out,
        duration,
        dur,
        wrote_frames,
    )
    if wf_raw is not None:
        log.info("RAW saved to %s", raw_out)

    # Save metrics CSV if requested.
    if metrics and metrics_csv:
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_sec", "rms_raw", "db_raw", "rms_proc", "db_proc", "vad"])
            for m in metrics:
                w.writerow([f"{m.t:.3f}", safe(m.rms_raw), safe(m.db_raw), safe(m.rms_proc), safe(m.db_proc), m.vad])
        log.info("Metrics saved to %s", metrics_csv)

    # Build plots if requested.
    if plot and metrics:
        try:
            import matplotlib.pyplot as plt
            prefix = plot_prefix or proc_out.rsplit(".", 1)[0]
            # Processed level plot (dB).
            t_vals = [m.t for m in metrics]
            db_proc_vals = [m.db_proc if m.db_proc is not None else float("nan") for m in metrics]

            plt.figure()
            plt.plot(t_vals, db_proc_vals)
            plt.xlabel("Time, s")
            plt.ylabel("Level (dBFS) - processed")
            plt.title("Processed level")
            level_png = f"{prefix}_level.png"
            plt.savefig(level_png, dpi=120, bbox_inches="tight")
            plt.close()

            # Processed VAD (0/1) plot.
            vad_vals = [m.vad if m.vad is not None else 0 for m in metrics]

            plt.figure()
            plt.step(t_vals, vad_vals, where="post")
            plt.xlabel("Time, s")
            plt.ylabel("VAD (0/1) — processed")
            plt.title("Processed VAD")
            vad_png = f"{prefix}_vad.png"
            plt.ylim(-0.1, 1.1)
            plt.savefig(vad_png, dpi=120, bbox_inches="tight")
            plt.close()

            log.info("Plots saved: %s, %s", level_png, vad_png)
        except Exception:
            log.exception("Failed to build plots. Is matplotlib available?")


def safe(x):
    return "" if x is None else f"{x:.6f}"


async def main() -> None:
    load_config()
    args = parse_args()
    setup_logging(args.log)

    out_path = args.out
    log.info(
        "Parameters: duration=%.2f, rate=%s, channels=%s, chunk=%s, raw=%s, ab=%s, voice_only=%s",
        args.duration, args.rate, args.channels, args.chunk, args.raw, args.ab, args.voice_only,
    )

    try:
        if args.raw:
            await record_raw_to_wav(
                ensure_wav_path(out_path),
                duration=args.duration,
                sample_rate=args.rate,
                chunk_size=args.chunk,
                channels=args.channels,
                queue_maxsize=args.queue,
            )
        else:
            await record_processed_to_wav(
                out_path,
                duration=args.duration,
                sample_rate=args.rate,
                chunk_size=args.chunk,
                channels=args.channels,
                queue_maxsize=args.queue,
                vad_log=not args.no_vad_log,
                ab_mode=args.ab,
                metrics_csv=args.metrics,
                plot=args.plot,
                plot_prefix=args.plot_prefix,
                voice_only=args.voice_only,
            )
        log.info("Done.")
    except KeyboardInterrupt:
        log.warning("Stopped by user (Ctrl+C).")
    except Exception:
        log.exception("Recording failed")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            import asyncio
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
        except Exception:
            pass
    asyncio.run(main())
