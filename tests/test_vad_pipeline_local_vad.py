#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test: MicrophoneStream -> local WebRTC VAD (inside script) -> live indicator.

Why: if WindowsAudioProcessor does not publish VAD events, this test confirms
audio input is fine and WebRTC VAD works with correctly framed input.

Dependencies:
    pip install webrtcvad numpy

Run:
    python test_vad_pipeline_local_vad.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from collections import deque

import numpy as np
import webrtcvad

from audio.microphone import MicrophoneStream  # Reuse project class.


def render_line(db: float, active: bool, stalled: bool = False):
    bar_len = 40
    norm = float(np.clip((db + 60.0) / 60.0, 0.0, 1.0))  # -60..0 dBFS -> 0..1
    filled = int(bar_len * norm)
    bar = "█" * filled + "·" * (bar_len - filled)
    state = "VOICE" if active else ("silence..." if not stalled else "-")
    mark = "🎙️" if active else "…"
    sys.stdout.write(f"\r{mark} [{bar}] {db:6.1f} dBFS  {state}   ")
    sys.stdout.flush()


def pcm_rms_dbfs(chunk: bytes, channels: int = 1) -> float:
    if not chunk:
        return -60.0
    pcm = np.frombuffer(chunk, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, 0]
    rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) if pcm.size else 0.0
    if rms <= 1e-6:
        return -60.0
    return 20.0 * np.log10(rms / 32768.0)


def to_mono16le(pcm: bytes, channels: int) -> bytes:
    """Downmix to mono (take left channel) for int16 little-endian."""
    if channels == 1:
        return pcm
    arr = np.frombuffer(pcm, dtype=np.int16).reshape(-1, channels)[:, 0].astype(np.int16)
    return arr.tobytes()


def frame_generator(chunk: bytes, frame_bytes: int, carry: bytearray) -> list[bytes]:
    """Split stream into fixed-size frames while carrying the tail."""
    carry.extend(chunk)
    frames = []
    total = len(carry)
    take = total - (total % frame_bytes)
    if take > 0:
        frames = [bytes(carry[i:i+frame_bytes]) for i in range(0, take, frame_bytes)]
        del carry[:take]
    return frames


async def main():
    # VAD parameters: 16 kHz, 30 ms, mode 2.
    sr = 16000
    frame_ms = 30
    vad_mode = 2
    channels_in_stream = 1  # If MicrophoneStream outputs stereo, set to 2.

    vad = webrtcvad.Vad(vad_mode)
    bytes_per_sample = 2
    frame_bytes = int(sr * frame_ms / 1000) * bytes_per_sample  # For mono int16.
    carry = bytearray()

    last_db = -60.0
    last_active = False
    last_update = 0.0
    stop = asyncio.Event()

    # Render indicator at 20 FPS independently.
    async def renderer():
        try:
            while not stop.is_set():
                stalled = (time.time() - last_update) > 1.0
                render_line(last_db, last_active, stalled=stalled)
                await asyncio.sleep(0.05)
        finally:
            print("\nDone.")

    async with MicrophoneStream(sample_rate=sr, channels=channels_in_stream) as mic:
        # IMPORTANT: MicrophoneStream must output int16 little-endian, sr == 16kHz.
        async def feeder_vad():
            nonlocal last_db, last_active, last_update
            while True:
                chunk = await mic.__anext__()  # Equivalent to "async for", but explicit control is needed.
                if chunk is None:
                    break

                # Level.
                db = pcm_rms_dbfs(chunk, channels=channels_in_stream)
                last_db = 0.8 * last_db + 0.2 * db

                # Downmix to mono for VAD.
                mono = to_mono16le(chunk, channels_in_stream)

                # Generate full 30 ms frames.
                for frame in frame_generator(mono, frame_bytes, carry):
                    is_speech = vad.is_speech(frame, sr)
                    last_active = bool(is_speech)
                    last_update = time.time()

        await asyncio.gather(feeder_vad(), renderer())

    stop.set()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDone.")
