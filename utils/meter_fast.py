import sys
import time
import platform
import argparse
from pathlib import Path

import numpy as np

# --- Parameters ---
SR = 16000            # sample rate (Hz)
CH = 1                # mono
BLOCK = 256           # block size (smaller block -> lower latency)
REF_FULL_SCALE = 32768.0
BAR_LEN = 50
SAVE_SECONDS = 3.0    # seconds to save on 'S'
RING_SECONDS = 10.0   # ring buffer depth for retrospective save

# Exponential RMS smoothing (lower alpha = smoother).
EMA_ALPHA = 0.3

def dbfs_from_signal_int16(xi: np.ndarray):
    x = xi.astype(np.float32) / REF_FULL_SCALE
    peak = np.max(np.abs(x)) + 1e-12
    rms = np.sqrt(np.mean(x * x)) + 1e-12
    return 20*np.log10(rms), 20*np.log10(peak)

def make_bar(dbfs_rms, min_db=-90, max_db=0):
    val = (dbfs_rms - min_db) / (max_db - min_db)
    val = float(np.clip(val, 0.0, 1.0))
    filled = int(val * BAR_LEN)
    return "█"*filled + "·"*(BAR_LEN - filled)

def list_input_devices():
    import sounddevice as sd

    devs = sd.query_devices()
    rows = []
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            rows.append((i, d["name"], d.get("hostapi", None), d["max_input_channels"]))
    return rows

def select_input_device():
    import sounddevice as sd

    print("Available input devices:")
    rows = list_input_devices()
    for i, name, hostapi, ch in rows:
        api_name = sd.query_hostapis(hostapi)["name"] if hostapi is not None else "?"
        print(f"  [{i:>2}] {name}  |  API: {api_name}  |  in={ch}")

    inp = input("Enter microphone index (Enter for default): ").strip()
    if inp == "":
        print("-> Using default device.")
        return None
    try:
        idx = int(inp)
        sd.query_devices(idx)  # Validate that device exists.
        return idx
    except Exception as e:
        print(f"Invalid index: {inp} ({e}). Using default device.")
        return None


def resolve_sample_rate(sd, device_idx, requested_rate=None) -> int:
    if requested_rate:
        return int(requested_rate)
    try:
        if device_idx is None:
            info = sd.query_devices(None, "input")
        else:
            info = sd.query_devices(device_idx, "input")
        rate = int(round(float(info.get("default_samplerate", SR))))
    except Exception:
        rate = SR
    return rate if rate > 0 else SR


def self_test() -> int:
    sample = np.array([0, 1000, -1000, 500, -500], dtype=np.int16)
    rms_db, peak_db = dbfs_from_signal_int16(sample)
    bar = make_bar(rms_db)
    print(f"self-test ok: rms={rms_db:.2f} peak={peak_db:.2f} bar_len={len(bar)}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Low-latency microphone level meter.")
    parser.add_argument("--self-test", action="store_true", help="Run math smoke test without audio devices.")
    parser.add_argument("--device", type=int, default=None, help="Input device index; skips prompt when set.")
    parser.add_argument("--duration", type=float, default=None, help="Run for N seconds, then exit.")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Input sample rate. Defaults to the selected device default rate.",
    )
    parser.add_argument("--block-size", type=int, default=BLOCK, help="Read block size in frames.")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()

    import sounddevice as sd
    import soundfile as sf

    # Device selection.
    device_idx = args.device if args.device is not None else select_input_device()
    sample_rate = resolve_sample_rate(sd, device_idx, requested_rate=args.sample_rate)
    block_size = max(1, int(args.block_size))

    # Prepare ring buffer with last RING_SECONDS.
    max_samples = int(RING_SECONDS * sample_rate)
    ring = np.zeros((max_samples, CH), dtype=np.int16)
    ring_pos = 0

    # Hotkeys (Windows: msvcrt).
    is_windows = platform.system().lower().startswith("win")
    if is_windows:
        import msvcrt
        print("\n[hotkeys]  S - save ~{:.1f}s WAV  |  Q - quit\n".format(SAVE_SECONDS))
    else:
        print("\n(Note: instant S/Q hotkeys are configured for Windows. On other OSes press Ctrl+C to quit.)\n")

    # Quick usage notes.
    print("Real-time Audio Meter - quick AEC/AGC check")
    print("- Toggle microphone enhancements in device properties and watch live indicators.")
    print("- Crest = Peak - RMS (in dB). AGC usually reduces and stabilizes crest.")
    if device_idx is not None:
        print(f"- Device: index {device_idx}")
    print(f"- Sample rate: {sample_rate} Hz")
    print(f"- Block size: {block_size}\n")

    # Smoothing state.
    ema_rms = None
    last_print = 0.0

    # Create input stream.
    stream_args = dict(
        channels=CH,
        samplerate=sample_rate,
        dtype='int16',
        blocksize=block_size,
        latency='low',
        device=device_idx
    )

    # No queue needed here; read blocks directly via .read().
    deadline = time.time() + args.duration if args.duration and args.duration > 0 else None

    with sd.InputStream(**stream_args) as stream:
        while True:
            if deadline is not None and time.time() >= deadline:
                print("\n[duration] Exit.")
                break
            # Non-blocking key handling on Windows.
            if is_windows and msvcrt.kbhit():
                key = msvcrt.getch()
                if not key:
                    pass
                else:
                    try:
                        ch = key.decode("utf-8").lower()
                    except Exception:
                        ch = ""
                    if ch == 'q':
                        print("\n[Q] Exit.")
                        break
                    elif ch == 's':
                        # Save last SAVE_SECONDS from ring buffer.
                        take = int(min(SAVE_SECONDS, RING_SECONDS) * sample_rate)
                        start = (ring_pos - take) % max_samples
                        if start + take <= max_samples:
                            cut = ring[start:start+take].copy()
                        else:
                            first = max_samples - start
                            cut = np.vstack([ring[start:], ring[:take-first]])
                        ts = time.strftime("%Y%m%d-%H%M%S")
                        fname = Path(f"sample_{ts}.wav")
                        sf.write(str(fname), cut, sample_rate, subtype='PCM_16')
                        print(f"\n[+] Saved {fname.name} ({SAVE_SECONDS:.1f}s) - compare before/after.")

            # Read next block (without queues/extra threads).
            in_data, _ = stream.read(block_size)  # ndarray int16, shape=(block_size, CH)

            # Update ring buffer.
            n = in_data.shape[0]
            end = ring_pos + n
            if end <= max_samples:
                ring[ring_pos:end] = in_data
            else:
                first = max_samples - ring_pos
                ring[ring_pos:] = in_data[:first]
                ring[:end - max_samples] = in_data[first:]
            ring_pos = (ring_pos + n) % max_samples

            # Compute metrics.
            rms_db, peak_db = dbfs_from_signal_int16(in_data[:, 0])
            if ema_rms is None:
                ema_rms = rms_db
            else:
                ema_rms = (1 - EMA_ALPHA) * ema_rms + EMA_ALPHA * rms_db
            crest = peak_db - ema_rms

            # Render ~10 times per second.
            now = time.time()
            if now - last_print >= 0.1:
                last_print = now
                bar = make_bar(ema_rms)
                line = f"\rRMS {ema_rms:6.1f} dBFS  |  Peak {peak_db:6.1f} dBFS  |  Crest {crest:4.1f} dB  | {bar}"
                print(line, end='', flush=True)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Exit.")
