import argparse
from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from livekit.rtc.apm import AudioProcessingModule
from livekit.rtc.audio_frame import AudioFrame

TARGET_SR = 16000
CHANNELS = 1
FRAME_MS = 10
SAMPLES_PER_CH = TARGET_SR * FRAME_MS // 1000  # 160

def ensure_mono(x: np.ndarray) -> np.ndarray:
    """Downmix to mono: average channels for stereo/multichannel input."""
    if x.ndim == 1:
        return x
    # Average across the channel axis.
    return np.mean(x, axis=1)

def to_int16(x: np.ndarray) -> np.ndarray:
    """Convert float32/float64 to int16 with safe clipping."""
    if x.dtype == np.int16:
        return x
    # Safety: clamp to [-1, 1] before scaling.
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz using polyphase FIR (resample_poly)."""
    if sr == TARGET_SR:
        return x
    # Choose integer up/down factors.
    g = gcd(sr, TARGET_SR)
    up = TARGET_SR // g
    down = sr // g
    # resample_poly works along axis 0.
    y = resample_poly(x, up, down, axis=0)
    # Leave final int16 quantization to conversion step.
    return y

def read_wav_mono_16k_int16(path: str) -> np.ndarray:
    """Read WAV with auto downmix/resample and return int16 mono."""
    data, sr = sf.read(path, always_2d=False)  # default dtype is float
    data = ensure_mono(data)
    data = resample_to_16k(data, sr)
    data = to_int16(data)
    return data

def chunk10ms(x: np.ndarray) -> list[np.ndarray]:
    """Split int16 stream into 10 ms frames; drop tail < 10 ms."""
    n_full = (len(x) // SAMPLES_PER_CH) * SAMPLES_PER_CH
    x = x[:n_full]
    return [x[i:i+SAMPLES_PER_CH] for i in range(0, n_full, SAMPLES_PER_CH)]

def frame_from_int16(frame_i16: np.ndarray) -> AudioFrame:
    """Pack a 10 ms int16 frame into LiveKit AudioFrame (mono)."""
    return AudioFrame(frame_i16.tobytes(order="C"), TARGET_SR, CHANNELS, len(frame_i16))

def run_aec(far_end_wav: str, near_end_wav: str, out_wav: str, delay_ms: int = 80):
    # 1) Read and convert both streams to mono int16 @ 16 kHz.
    far = read_wav_mono_16k_int16(far_end_wav)
    mic = read_wav_mono_16k_int16(near_end_wav)
    # Trim to common length.
    n = min(len(far), len(mic))
    far = far[:n]; mic = mic[:n]

    # 2) Split into 10 ms frames.
    far_frames = chunk10ms(far)
    mic_frames = chunk10ms(mic)
    m = min(len(far_frames), len(mic_frames))
    far_frames = far_frames[:m]
    mic_frames = mic_frames[:m]

    # 3) Initialize APM (AEC3 inside).
    apm = AudioProcessingModule(
        echo_cancellation=True,
        noise_suppression=True,
        high_pass_filter=True,
        auto_gain_control=False,
    )
    apm.set_stream_delay_ms(delay_ms)  # Important for AEC.

    out = np.empty(0, dtype=np.int16)

    # 4) Main loop: 10 ms reverse, then 10 ms near-end.
    for fr_rev, fr_near in zip(far_frames, mic_frames):
        rev_frame = frame_from_int16(fr_rev)
        apm.process_reverse_stream(rev_frame)

        near_frame = frame_from_int16(fr_near)
        apm.process_stream(near_frame)  # Mutates near_frame.data in-place.

        cleaned = np.frombuffer(near_frame.data, dtype=np.int16)
        out = np.concatenate([out, cleaned.copy()])

    # 5) Save result.
    sf.write(out_wav, out, TARGET_SR, subtype="PCM_16")
    print(f"Done: {out_wav}")


def self_test() -> int:
    x = np.array([[0.5, -0.5], [0.25, 0.75]], dtype=np.float32)
    mono = ensure_mono(x)
    converted = to_int16(mono)
    frames = chunk10ms(np.zeros(SAMPLES_PER_CH * 2, dtype=np.int16))
    print(
        "self-test ok: "
        f"mono_shape={mono.shape} int16_dtype={converted.dtype} frames={len(frames)}"
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run LiveKit AEC over far/near WAV files.")
    parser.add_argument("--far", default="aec_out_ref.wav", help="Far-end/reference WAV path.")
    parser.add_argument("--near", default="aec_out_mic.wav", help="Near-end/microphone WAV path.")
    parser.add_argument("--out", default="clean.wav", help="Output cleaned WAV path.")
    parser.add_argument("--delay-ms", type=int, default=80, help="AEC stream delay in milliseconds.")
    parser.add_argument("--self-test", action="store_true", help="Run pure helper smoke test without WAV files.")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()

    far = Path(args.far)
    near = Path(args.near)
    missing = [str(path) for path in (far, near) if not path.exists()]
    if missing:
        print("Missing input WAV file(s): " + ", ".join(missing))
        return 1
    run_aec(str(far), str(near), args.out, delay_ms=args.delay_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
