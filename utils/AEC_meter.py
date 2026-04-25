import sys
import time
import queue
import threading
import argparse
import shutil
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# === Settings ===
SR = 16000           # sample rate
CH = 1               # mono
BLOCK = 1024         # block size (frame)
REF_FULL_SCALE = 32768.0  # for int16 -> dBFS
BAR_LEN = 50         # ASCII bar length
SAVE_SECONDS = 3.0   # duration of saved fragment on 'S'

# Queue for audio blocks.
q_audio = queue.Queue()

def dbfs_from_signal(xi: np.ndarray):
    # xi: int16 [-32768..32767]
    x = xi.astype(np.float32) / REF_FULL_SCALE
    peak = np.max(np.abs(x)) + 1e-12
    rms = np.sqrt(np.mean(x * x)) + 1e-12
    return 20*np.log10(rms), 20*np.log10(peak)

def make_bar(dbfs_rms, min_db=-90, max_db=0):
    val = np.clip((dbfs_rms - min_db) / (max_db - min_db), 0, 1)
    filled = int(val * BAR_LEN)
    return "█"*filled + "·"*(BAR_LEN - filled)

def audio_callback(indata, frames, time_info, status):
    if status:
        # Overflow/underflow: print to STDERR but do not crash.
        print(f"[audio status] {status}", file=sys.stderr)
    q_audio.put(indata.copy())


def _queue_callback(target_queue, label):
    def _callback(indata, frames, time_info, status):
        if status:
            print(f"[{label} status] {status}", file=sys.stderr)
        try:
            target_queue.put_nowait(indata.copy())
        except queue.Full:
            pass

    return _callback


class PulseRawCapture:
    """Read one PulseAudio/PipeWire source as raw mono int16 via parec."""

    def __init__(
        self,
        *,
        source: str,
        sample_rate: int,
        block_size: int,
        label: str,
        queue_size: int = 128,
    ) -> None:
        self.source = source
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.label = label
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=queue_size)
        self._proc: subprocess.Popen[bytes] | None = None
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._stderr_tail: list[str] = []

    def __enter__(self) -> "PulseRawCapture":
        if shutil.which("parec") is None:
            raise RuntimeError("parec is not installed; install pulseaudio-utils or pipewire-pulse tools")
        cmd = [
            "parec",
            "--raw",
            "--format=s16le",
            f"--rate={self.sample_rate}",
            "--channels=1",
            f"--device={self.source}",
            "--latency-msec=20",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader.start()
        self._stderr_reader.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._stop.set()
        proc = self._proc
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

    def get(self, timeout: float = 0.2) -> np.ndarray:
        return self.queue.get(timeout=timeout)

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def error_tail(self) -> str:
        return "\n".join(self._stderr_tail[-6:]).strip()

    def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        chunk_bytes = max(1, self.block_size) * 2
        while not self._stop.is_set():
            data = self._proc.stdout.read(chunk_bytes)
            if not data:
                break
            if len(data) % 2:
                data = data[:-1]
            if not data:
                continue
            block = np.frombuffer(data, dtype=np.int16).copy()
            try:
                self.queue.put_nowait(block)
            except queue.Full:
                pass

    def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for raw_line in iter(self._proc.stderr.readline, b""):
            if self._stop.is_set():
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                self._stderr_tail.append(line)
                if len(self._stderr_tail) > 20:
                    self._stderr_tail = self._stderr_tail[-20:]

def saver_thread(stop_evt, sample_rate, ringbuf_seconds=10):
    import soundfile as sf

    """
    Background recording into ring buffer so hotkey can save last SAVE_SECONDS.
    """
    max_samples = int(ringbuf_seconds * sample_rate)
    ring = np.zeros((max_samples, CH), dtype=np.int16)
    idx = 0
    while not stop_evt.is_set():
        try:
            block = q_audio.get(timeout=0.1)
        except queue.Empty:
            continue
        n = block.shape[0]
        if n > max_samples:
            block = block[-max_samples:]
            n = block.shape[0]
        end = idx + n
        if end <= max_samples:
            ring[idx:end] = block
        else:
            first = max_samples - idx
            ring[idx:] = block[:first]
            ring[:end - max_samples] = block[first:]
        idx = (idx + n) % max_samples
        # Save last SAVE_SECONDS on request.
        if save_request.is_set():
            save_request.clear()
            take = int(min(SAVE_SECONDS, ringbuf_seconds) * sample_rate)
            # Cut latest "take" from ring tail.
            start = (idx - take) % max_samples
            if start + take <= max_samples:
                cut = ring[start:start+take].copy()
            else:
                first = max_samples - start
                cut = np.vstack([ring[start:], ring[:take-first]])
            # Output file.
            ts = time.strftime("%Y%m%d-%H%M%S")
            fname = Path(f"sample_{ts}.wav")
            sf.write(str(fname), cut, sample_rate, subtype='PCM_16')
            print(f"\n[+] Saved {fname.name} ({SAVE_SECONDS:.1f}s) - compare before/after enabling enhancements.")

# Global control signals.
stop_saver = threading.Event()
save_request = threading.Event()

def keyboard_listener():
    print("\n[hotkeys]  S - save {0:.1f}s WAV  |  Q - quit\n".format(SAVE_SECONDS))
    while True:
        ch = sys.stdin.read(1)
        if not ch:
            break
        ch = ch.strip().lower()
        if ch == 's':
            save_request.set()
        elif ch == 'q':
            stop_saver.set()
            break


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


def _pactl_get(*args: str) -> str | None:
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
    value = proc.stdout.strip()
    return value or None


def _pactl_source_names() -> set[str]:
    output = _pactl_get("list", "sources", "short")
    if not output:
        return set()
    names: set[str] = set()
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            names.add(parts[1])
    return names


def resolve_pulse_source_name(source: str) -> str:
    """Resolve friendly Pulse/PipeWire aliases to concrete source names."""
    source = (source or "").strip()
    if source == "@DEFAULT_SOURCE@":
        return _pactl_get("get-default-source") or source
    if source == "@DEFAULT_MONITOR@":
        default_sink = _pactl_get("get-default-sink")
        if default_sink:
            candidate = f"{default_sink}.monitor"
            names = _pactl_source_names()
            if not names or candidate in names:
                return candidate
        names = sorted(name for name in _pactl_source_names() if ".monitor" in name.lower())
        return names[0] if names else source
    return source


def _resample_i16(pcm: np.ndarray, input_rate: int, output_rate: int) -> np.ndarray:
    if input_rate == output_rate or pcm.size == 0:
        return np.ascontiguousarray(pcm, dtype=np.int16)
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(int(input_rate), int(output_rate))
    up = int(output_rate) // g
    down = int(input_rate) // g
    y = resample_poly(np.asarray(pcm, dtype=np.float32), up, down)
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def self_test() -> int:
    sample = np.array([0, 1000, -1000, 500, -500], dtype=np.int16)
    rms_db, peak_db = dbfs_from_signal(sample)
    bar = make_bar(rms_db)
    print(f"self-test ok: rms={rms_db:.2f} peak={peak_db:.2f} bar_len={len(bar)}")
    return 0


def _mono_i16(block: np.ndarray) -> np.ndarray:
    arr = np.asarray(block, dtype=np.int16)
    if arr.ndim == 1:
        return np.ascontiguousarray(arr, dtype=np.int16)
    if arr.shape[1] == 1:
        return np.ascontiguousarray(arr[:, 0], dtype=np.int16)
    mixed = arr.astype(np.int32).mean(axis=1)
    return np.clip(np.round(mixed), -32768, 32767).astype(np.int16)


def _drain_farend(far_queue, canceller, *, far_rate: int, aec_rate: int) -> tuple[int, float | None]:
    drained = 0
    last_rms = None
    while True:
        try:
            far_block = far_queue.get_nowait()
        except queue.Empty:
            break
        far = _mono_i16(far_block)
        last_rms, _peak = dbfs_from_signal(far)
        far_for_aec = _resample_i16(far, far_rate, aec_rate)
        canceller.submit_farend(far_for_aec)
        drained += 1
    return drained, last_rms


def _create_aec_canceller(args, sample_rate: int):
    from audio.processing.echo_cancellation import (
        AcousticEchoCancellationSettings,
        AcousticEchoCanceller,
    )

    settings = AcousticEchoCancellationSettings(
        enabled=True,
        frame_duration_ms=args.aec_frame_ms,
        stream_delay_ms=args.aec_delay_ms,
        noise_suppression=args.aec_noise_suppression,
        high_pass_filter=args.aec_high_pass_filter,
        auto_gain_control=args.aec_auto_gain,
    )
    return AcousticEchoCanceller(
        sample_rate=sample_rate,
        channels=CH,
        settings=settings,
    )


def _run_live_aec_pulse(args, sample_rate: int, block_size: int) -> int:
    if block_size <= 0:
        block_size = max(1, int(sample_rate * max(1, int(args.aec_frame_ms)) / 1000))

    mic_source_requested = args.mic_source or "@DEFAULT_SOURCE@"
    loopback_source_requested = args.loopback_source or "@DEFAULT_MONITOR@"
    mic_source = resolve_pulse_source_name(mic_source_requested)
    loopback_source = resolve_pulse_source_name(loopback_source_requested)
    canceller = _create_aec_canceller(args, sample_rate)

    print("Pulse/PipeWire AEC mode")
    print(
        f"- Mic source: {mic_source_requested}"
        + (f" -> {mic_source}" if mic_source != mic_source_requested else "")
    )
    print(
        f"- Loopback source: {loopback_source_requested}"
        + (
            f" -> {loopback_source}"
            if loopback_source != loopback_source_requested
            else ""
        )
    )
    print(f"- Sample rate: {sample_rate} Hz")
    print(f"- Block size: {block_size}")
    print(f"- AEC frame: {args.aec_frame_ms} ms")
    print(f"- AEC delay hint: {args.aec_delay_ms} ms\n")

    far_hist: list[float] = []
    raw_hist: list[float] = []
    clean_hist: list[float] = []
    last_print = 0.0
    deadline = time.time() + args.duration if args.duration and args.duration > 0 else None

    try:
        with PulseRawCapture(
            source=loopback_source,
            sample_rate=sample_rate,
            block_size=block_size,
            label="loopback",
        ) as far_capture, PulseRawCapture(
            source=mic_source,
            sample_rate=sample_rate,
            block_size=block_size,
            label="mic",
        ) as mic_capture:
            while True:
                if deadline is not None and time.time() >= deadline:
                    break
                _far_count, far_rms = _drain_farend(
                    far_capture.queue,
                    canceller,
                    far_rate=sample_rate,
                    aec_rate=sample_rate,
                )
                if far_rms is not None:
                    far_hist.append(float(far_rms))
                    if len(far_hist) > 20:
                        far_hist.pop(0)

                try:
                    mic_block = mic_capture.get(timeout=0.2)
                except queue.Empty:
                    if not mic_capture.running():
                        detail = mic_capture.error_tail()
                        raise RuntimeError(
                            f"mic source capture stopped: {detail or mic_source}"
                        )
                    if not far_capture.running():
                        detail = far_capture.error_tail()
                        raise RuntimeError(
                            f"loopback source capture stopped: {detail or loopback_source}"
                        )
                    continue

                _far_count, far_rms = _drain_farend(
                    far_capture.queue,
                    canceller,
                    far_rate=sample_rate,
                    aec_rate=sample_rate,
                )
                if far_rms is not None:
                    far_hist.append(float(far_rms))
                    if len(far_hist) > 20:
                        far_hist.pop(0)

                raw = _mono_i16(mic_block)
                clean = canceller.process(raw)
                if clean.size == 0:
                    clean = raw

                raw_rms, raw_peak = dbfs_from_signal(raw)
                clean_rms, clean_peak = dbfs_from_signal(clean)
                raw_hist.append(raw_rms)
                clean_hist.append(clean_rms)
                if len(raw_hist) > 20:
                    raw_hist.pop(0)
                if len(clean_hist) > 20:
                    clean_hist.pop(0)

                raw_med = float(np.median(raw_hist))
                clean_med = float(np.median(clean_hist))
                far_med = float(np.median(far_hist)) if far_hist else -120.0
                reduction = raw_med - clean_med

                now = time.time()
                if now - last_print >= 0.1:
                    last_print = now
                    bar = make_bar(clean_med)
                    line = (
                        f"\rFar {far_med:6.1f} dBFS | Raw {raw_med:6.1f} dBFS | Clean {clean_med:6.1f} dBFS "
                        f"| Echo red {reduction:5.1f} dB | "
                        f"RawPk {raw_peak:6.1f} | ClnPk {clean_peak:6.1f} | {bar}"
                    )
                    print(line, end="", flush=True)
    except Exception as exc:
        print(f"\nFailed to run Pulse/PipeWire AEC meter: {type(exc).__name__}: {exc}")
        return 1

    print("\nExit. Bye.")
    return 0


def _run_live_aec(args, sd, sample_rate: int, block_size: int) -> int:
    if args.loopback_device is None:
        print("AEC mode requires --loopback-device with a PipeWire/Pulse monitor input.")
        print("On Linux with PipeWire/Pulse, try: utils/AEC_meter.py --aec --pulse")
        return 2

    canceller = _create_aec_canceller(args, sample_rate)

    loopback_rate = resolve_sample_rate(
        sd,
        args.loopback_device,
        requested_rate=args.loopback_sample_rate,
    )
    loopback_block_size = (
        max(1, int(args.loopback_block_size))
        if args.loopback_block_size
        else max(1, int(loopback_rate * max(1, int(args.aec_frame_ms)) / 1000))
    )

    mic_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
    far_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)

    mic_kwargs = dict(
        channels=CH,
        samplerate=sample_rate,
        dtype="int16",
        blocksize=block_size,
        callback=_queue_callback(mic_queue, "mic"),
    )
    if args.device is not None:
        mic_kwargs["device"] = args.device

    far_kwargs = dict(
        channels=CH,
        samplerate=loopback_rate,
        dtype="int16",
        blocksize=loopback_block_size,
        callback=_queue_callback(far_queue, "loopback"),
        device=args.loopback_device,
    )

    print("Live AEC mode")
    print(f"- Mic device: {'default' if args.device is None else args.device}")
    print(f"- Loopback device: {args.loopback_device}")
    print(f"- Mic sample rate: {sample_rate} Hz")
    print(f"- Loopback sample rate: {loopback_rate} Hz")
    print(f"- Mic block size: {block_size}")
    print(f"- Loopback block size: {loopback_block_size}")
    print(f"- AEC frame: {args.aec_frame_ms} ms")
    print(f"- AEC delay hint: {args.aec_delay_ms} ms\n")

    far_hist: list[float] = []
    raw_hist: list[float] = []
    clean_hist: list[float] = []
    last_print = 0.0
    deadline = time.time() + args.duration if args.duration and args.duration > 0 else None

    try:
        with sd.InputStream(**far_kwargs), sd.InputStream(**mic_kwargs):
            while True:
                if deadline is not None and time.time() >= deadline:
                    break
                _far_count, far_rms = _drain_farend(
                    far_queue,
                    canceller,
                    far_rate=loopback_rate,
                    aec_rate=sample_rate,
                )
                if far_rms is not None:
                    far_hist.append(float(far_rms))
                    if len(far_hist) > 20:
                        far_hist.pop(0)
                try:
                    mic_block = mic_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                _far_count, far_rms = _drain_farend(
                    far_queue,
                    canceller,
                    far_rate=loopback_rate,
                    aec_rate=sample_rate,
                )
                if far_rms is not None:
                    far_hist.append(float(far_rms))
                    if len(far_hist) > 20:
                        far_hist.pop(0)
                raw = _mono_i16(mic_block)
                clean = canceller.process(raw)
                if clean.size == 0:
                    clean = raw

                raw_rms, raw_peak = dbfs_from_signal(raw)
                clean_rms, clean_peak = dbfs_from_signal(clean)
                raw_hist.append(raw_rms)
                clean_hist.append(clean_rms)
                if len(raw_hist) > 20:
                    raw_hist.pop(0)
                if len(clean_hist) > 20:
                    clean_hist.pop(0)

                raw_med = float(np.median(raw_hist))
                clean_med = float(np.median(clean_hist))
                far_med = float(np.median(far_hist)) if far_hist else -120.0
                reduction = raw_med - clean_med

                now = time.time()
                if now - last_print >= 0.1:
                    last_print = now
                    bar = make_bar(clean_med)
                    line = (
                        f"\rFar {far_med:6.1f} dBFS | Raw {raw_med:6.1f} dBFS | Clean {clean_med:6.1f} dBFS "
                        f"| Echo red {reduction:5.1f} dB | "
                        f"RawPk {raw_peak:6.1f} | ClnPk {clean_peak:6.1f} | {bar}"
                    )
                    print(line, end="", flush=True)
    except Exception as exc:
        print(f"\nFailed to run live AEC meter: {type(exc).__name__}: {exc}")
        return 1

    print("\nExit. Bye.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Microphone level meter for AEC/AGC validation.")
    parser.add_argument("--self-test", action="store_true", help="Run math smoke test without audio devices.")
    parser.add_argument("--device", type=int, default=None, help="Input device index.")
    parser.add_argument("--duration", type=float, default=None, help="Run for N seconds, then exit.")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Input sample rate. Defaults to the selected device default rate.",
    )
    parser.add_argument("--block-size", type=int, default=None, help="Read block size in frames.")
    parser.add_argument("--aec", action="store_true", help="Run LiveKit AEC and print raw vs cleaned levels.")
    parser.add_argument(
        "--pulse",
        action="store_true",
        help="In AEC mode, capture Pulse/PipeWire sources via parec instead of sounddevice device indexes.",
    )
    parser.add_argument(
        "--mic-source",
        type=str,
        default=None,
        help="Pulse/PipeWire mic source name for --aec --pulse. Defaults to @DEFAULT_SOURCE@.",
    )
    parser.add_argument(
        "--loopback-source",
        type=str,
        default=None,
        help="Pulse/PipeWire monitor source name for --aec --pulse. Defaults to @DEFAULT_MONITOR@.",
    )
    parser.add_argument("--loopback-device", type=int, default=None, help="PipeWire/Pulse monitor input device for AEC far-end reference.")
    parser.add_argument(
        "--loopback-sample-rate",
        type=int,
        default=None,
        help="Loopback input sample rate. Defaults to the loopback device default rate.",
    )
    parser.add_argument(
        "--loopback-block-size",
        type=int,
        default=None,
        help="Loopback read block size in frames.",
    )
    parser.add_argument("--aec-delay-ms", type=int, default=80, help="AEC stream delay hint in milliseconds.")
    parser.add_argument("--aec-frame-ms", type=int, default=10, help="AEC frame duration in milliseconds.")
    parser.add_argument("--aec-noise-suppression", action="store_true", help="Enable LiveKit APM noise suppression in AEC mode.")
    parser.add_argument("--aec-high-pass-filter", action="store_true", help="Enable LiveKit APM high-pass filter in AEC mode.")
    parser.add_argument("--aec-auto-gain", action="store_true", help="Enable LiveKit APM auto gain in AEC mode.")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()

    use_pulse_aec = bool(args.aec and (args.pulse or args.mic_source or args.loopback_source))
    if use_pulse_aec:
        sample_rate = int(args.sample_rate or 48000)
        if args.block_size is None:
            block_size = max(1, int(sample_rate * max(1, int(args.aec_frame_ms)) / 1000))
        else:
            block_size = max(1, int(args.block_size))
        return _run_live_aec_pulse(args, sample_rate, block_size)

    import sounddevice as sd
    sample_rate = resolve_sample_rate(sd, args.device, requested_rate=args.sample_rate)
    if args.block_size is None and args.aec:
        block_size = max(1, int(sample_rate * max(1, int(args.aec_frame_ms)) / 1000))
    else:
        block_size = max(1, int(args.block_size or BLOCK))

    if args.aec:
        return _run_live_aec(args, sd, sample_rate, block_size)

    # Usage instructions.
    print("Real-time Audio Meter (AEC/AGC validation)")
    print("- Play music/speech on speakers and watch RMS. Toggle microphone enhancements in device properties.")
    print("- During pauses track noise floor (RMS). With AGC speech peaks become smoother; with AEC speaker bleed should drop.")
    print("- If you have multiple devices, choose one via sd.default.device = (input_id, None)\n")
    print(f"- Device: {'default' if args.device is None else args.device}")
    print(f"- Sample rate: {sample_rate} Hz")
    print(f"- Block size: {block_size}\n")

    # Device list hint.
    try:
        devs = sd.query_devices()
        ins = [f"[{i}] {d['name']} (in={d['max_input_channels']})" for i, d in enumerate(devs) if d['max_input_channels']>0]
        print("Available inputs:\n" + "\n".join(ins) + "\n")
    except Exception:
        pass

    # Saver thread.
    t_saver = threading.Thread(target=saver_thread, args=(stop_saver, sample_rate), daemon=True)
    t_saver.start()

    # Keyboard thread (works in console).
    t_keys = threading.Thread(target=keyboard_listener, daemon=True)
    t_keys.start()

    # Smooth median filter for RMS.
    rms_hist = []

    stream_kwargs = dict(
        channels=CH,
        samplerate=sample_rate,
        dtype="int16",
        blocksize=block_size,
        callback=audio_callback,
    )
    if args.device is not None:
        stream_kwargs["device"] = args.device

    deadline = time.time() + args.duration if args.duration and args.duration > 0 else None

    with sd.InputStream(**stream_kwargs):
        last_print = 0.0
        while not stop_saver.is_set():
            if deadline is not None and time.time() >= deadline:
                stop_saver.set()
                break
            try:
                block = q_audio.get(timeout=0.2)
            except queue.Empty:
                continue

            rms_db, peak_db = dbfs_from_signal(block[:, 0])
            rms_hist.append(rms_db)
            if len(rms_hist) > 20:  # ~20 blocks = sliding window ~ (BLOCK/SR)*20 sec
                rms_hist.pop(0)
            rms_med = float(np.median(rms_hist))

            # Print ~10 times per second.
            now = time.time()
            if now - last_print >= 0.1:
                last_print = now
                bar = make_bar(rms_med)
                # Crest factor (peak - RMS), indicator of compression/AGC.
                crest = peak_db - rms_med
                line = f"\rRMS {rms_med:6.1f} dBFS  |  Peak {peak_db:6.1f} dBFS  |  Crest {crest:4.1f} dB  | {bar}"
                print(line, end='', flush=True)

    print("\nExit. Bye.")
    t_keys.join(timeout=0.2)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        stop_saver.set()
        print("\n[Ctrl+C] Stopping...")
