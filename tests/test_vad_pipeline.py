# test_vad_pipeline.py
import asyncio
import sys
import numpy as np
import time

from audio.microphone import MicrophoneStream
from audio.processing.windows_processing import WindowsAudioProcessor
from core.bus import bus

from core.config import cfg, load as load_config


def render_line(
    db: float,
    active: bool,
    probability: float,
    segment_duration: float,
    stalled: bool = False,
):
    bar_len = 40
    norm = float(np.clip((db + 60.0) / 60.0, 0.0, 1.0))  # -60..0 dBFS -> 0..1
    filled = int(bar_len * norm)
    bar = "█" * filled + "·" * (bar_len - filled)
    state = "VOICE" if active else ("silence..." if not stalled else "-----")
    mark = "🎙️" if active else "…"
    sys.stdout.write(
        f"\r{mark} [{bar}] {db:6.1f} dBFS  {state}   "
        f"Pvad={probability:0.2f}  segment={segment_duration:0.2f}s"
    )
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

async def main():
    load_config()  # Ensure cfg.audio.processing.vad.mode from config/config.json is applied.

    # 1) Start EventBus router BEFORE subscriptions.
    bus.start()  # Starts router task (asyncio.create_task)  :contentReference[oaicite:5]{index=5}

    last_db = -60.0
    last_active = False
    last_update = 0.0
    last_prob = 0.0
    last_segment = 0.0
    stop = asyncio.Event()

    async def on_vad(ev):
        nonlocal last_active, last_update, last_db, last_prob, last_segment
        payload = ev.payload  # Event(topic, payload)  :contentReference[oaicite:6]{index=6}
        last_active = bool(payload.get("active", False))
        last_prob = float(payload.get("vad_probability", last_prob))
        last_segment = float(payload.get("voice_active_duration", last_segment))
        last_update = time.time()

    # Subscribe BEFORE starting the pipeline.
    bus.subscribe(cfg.events.audio.voice_activity, on_vad)  # VAD is published by WindowsAudioProcessor  :contentReference[oaicite:7]{index=7}

    async def renderer():
        try:
            while not stop.is_set():
                stalled = (time.time() - last_update) > 1.0
                render_line(
                    last_db,
                    last_active,
                    last_prob,
                    last_segment,
                    stalled=stalled,
                )
                await asyncio.sleep(0.05)
        finally:
            print("\nDone.")

    async with MicrophoneStream() as mic, WindowsAudioProcessor(config=cfg.audio.processing) as proc:
        async def feeder():
            async for chunk in mic:
                # Also compute level from raw data so indicator is always responsive.
                nonlocal last_db
                db = pcm_rms_dbfs(chunk, channels=1)
                last_db = 0.8 * last_db + 0.2 * db
                # Send chunk to processor (it splits into VAD frames and publishes events).
                await proc.submit(chunk)

        async def drain():
            async for _ in proc:  # Drain output so queue does not fill up.
                pass

        await asyncio.gather(feeder(), drain(), renderer())

    stop.set()
    # 2) Gracefully stop the bus.
    await bus.stop()  # Sends sentinel and stops router  :contentReference[oaicite:8]{index=8}

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDone.")
