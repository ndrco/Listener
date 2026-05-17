from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import shutil
import sys
import tempfile
import unicodedata
import wave
from pathlib import Path
from typing import Protocol

import numpy as np

from audio.ducking import PulseAudioDucker

try:  # pragma: no cover - availability is environment-dependent
    import sounddevice as sd
except Exception:  # pragma: no cover - handled by playback fallback
    sd = None  # type: ignore[assignment]

from .config import PiperConfig, PlaybackConfig
from .emoji import extract_emoji_for_speech


class SpeechError(RuntimeError):
    pass


log = logging.getLogger(__name__)


class SpeechEngine(Protocol):
    async def speak(self, text: str) -> None:
        ...


class PiperSpeechEngine:
    def __init__(
        self,
        piper: PiperConfig,
        playback: PlaybackConfig,
        *,
        prefetch: bool = False,
        manage_ducking: bool = True,
    ) -> None:
        self.piper = piper
        self.playback = playback
        self.prefetch = bool(prefetch)
        self.manage_ducking = bool(manage_ducking)
        self._worker: _PiperWorkerClient | None = None
        self._worker_failed = False

    async def speak(self, text: str) -> None:
        parsed = extract_emoji_for_speech(text)
        if parsed.tokens:
            log.debug("PiperSpeechEngine: stripped %d emoji(s) before synthesis", len(parsed.tokens))
        units = split_speech_units(parsed.speech_text)
        if not units:
            return
        if self.prefetch:
            await self._speak_prefetch(units)
            return
        with tempfile.TemporaryDirectory(prefix="speaker-") as tmp:
            ducker = PulseAudioDucker(self.playback.ducking)
            ducked = False
            try:
                for index, unit in enumerate(units):
                    output = Path(tmp) / f"speech-{index}.wav"
                    synth_start_ns = _perf_now()
                    await self._synthesize(unit, output)
                    synth_done_ns = _perf_now()
                    synth_ms = _perf_elapsed(synth_start_ns, synth_done_ns)
                    _perf_emit(
                        "speaker",
                        "synth_done",
                        segment_index=index,
                        synth_ms=synth_ms,
                    )
                    if self.manage_ducking and not ducked:
                        await ducker.duck()
                        ducked = True
                    play_start_ns = _perf_now()
                    _perf_emit("speaker", "playback_start", segment_index=index)
                    await self._play(output)
                    _perf_emit(
                        "speaker",
                        "playback_done",
                        segment_index=index,
                        playback_total_ms=_perf_elapsed(play_start_ns),
                    )
                    _perf_emit(
                        "summary",
                        "tts_segment",
                        segment_index=index,
                        text_to_synth_start_ms=0.0,
                        synth_ms=synth_ms,
                        playback_start_delay_ms=_perf_elapsed(synth_done_ns, play_start_ns),
                        playback_total_ms=_perf_elapsed(play_start_ns),
                    )
            finally:
                if self.manage_ducking and ducked:
                    await ducker.restore()

    async def _speak_prefetch(self, units: list[str]) -> None:
        with tempfile.TemporaryDirectory(prefix="speaker-") as tmp:
            tasks: dict[int, asyncio.Task[tuple[Path, int, int, float | None]]] = {}

            async def _synth(index: int, unit: str) -> tuple[Path, int, int, float | None]:
                output = Path(tmp) / f"speech-{index}.wav"
                synth_start_ns = _perf_now()
                await self._synthesize(unit, output)
                synth_done_ns = _perf_now()
                synth_ms = _perf_elapsed(synth_start_ns, synth_done_ns)
                _perf_emit(
                    "speaker",
                    "synth_done",
                    segment_index=index,
                    synth_ms=synth_ms,
                )
                return output, synth_start_ns, synth_done_ns, synth_ms

            tasks[0] = asyncio.create_task(_synth(0, units[0]), name="Speaker.synth.0")
            try:
                for index in range(len(units)):
                    next_index = index + 1
                    if next_index < len(units) and next_index not in tasks:
                        tasks[next_index] = asyncio.create_task(
                            _synth(next_index, units[next_index]),
                            name=f"Speaker.synth.{next_index}",
                        )
                    output, synth_start_ns, synth_done_ns, synth_ms = await tasks.pop(index)
                    play_start_ns = _perf_now()
                    _perf_emit("speaker", "playback_start", segment_index=index)
                    await self._play(output)
                    playback_total_ms = _perf_elapsed(play_start_ns)
                    _perf_emit(
                        "speaker",
                        "playback_done",
                        segment_index=index,
                        playback_total_ms=playback_total_ms,
                    )
                    _perf_emit(
                        "summary",
                        "tts_segment",
                        segment_index=index,
                        text_to_synth_start_ms=0.0,
                        synth_ms=synth_ms,
                        playback_start_delay_ms=_perf_elapsed(synth_done_ns, play_start_ns),
                        playback_total_ms=playback_total_ms,
                    )
            finally:
                for task in tasks.values():
                    task.cancel()
                for task in tasks.values():
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    async def _synthesize(self, text: str, output: Path) -> None:
        if self.prefetch and not self.piper.extra_args and not self._worker_failed:
            try:
                await self._synthesize_with_worker(text, output)
                return
            except Exception as exc:  # noqa: BLE001 - persistent path is optional
                self._worker_failed = True
                log.warning("PiperSpeechEngine: persistent worker failed, using subprocess: %s", exc)
                worker = self._worker
                self._worker = None
                if worker is not None:
                    await worker.close()

        args = build_piper_args(self.piper, output)
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await _communicate_or_kill(
                proc,
                input_data=text.encode("utf-8"),
                timeout_s=self.piper.timeout_s,
                timeout_message=f"Piper timed out after {self.piper.timeout_s:.1f}s",
            )
        except asyncio.TimeoutError as exc:
            raise SpeechError(f"Piper timed out after {self.piper.timeout_s:.1f}s") from exc
        if proc.returncode != 0:
            raise SpeechError(_format_subprocess_error("Piper", proc.returncode, stderr))
        if not output.exists() or output.stat().st_size == 0:
            raise SpeechError("Piper did not produce a WAV file")

    async def _synthesize_with_worker(self, text: str, output: Path) -> None:
        worker = self._worker
        if worker is None:
            python = resolve_worker_python_command(self.piper.command)
            if python is None:
                raise SpeechError("Piper worker requires a Python command or venv Piper entrypoint")
            worker = _PiperWorkerClient(
                python=python,
                model=self.piper.model,
                volume=self.piper.volume,
                sentence_silence=self.piper.sentence_silence,
                timeout_s=self.piper.timeout_s,
            )
            self._worker = worker
        await worker.synthesize(text, output)
        if not output.exists() or output.stat().st_size == 0:
            raise SpeechError("Piper worker did not produce a WAV file")

    async def _play(self, output: Path) -> None:
        backend = str(getattr(self.playback, "backend", "auto") or "auto").strip().casefold()
        if backend == "auto" and sys.platform.startswith("linux") and _playback_command_available(self.playback):
            await self._play_subprocess(output)
            return
        if backend in {"auto", "sounddevice"}:
            try:
                await self._play_sounddevice(output)
                return
            except Exception as exc:  # noqa: BLE001 - auto falls back to paplay
                if backend == "sounddevice":
                    raise SpeechError(f"sounddevice playback failed: {exc}") from exc
                log.debug("PiperSpeechEngine: sounddevice playback unavailable, using paplay: %s", exc)

        await self._play_subprocess(output)

    async def _play_sounddevice(self, output: Path) -> None:
        if sd is None:
            raise SpeechError("sounddevice is not installed")
        try:
            await asyncio.wait_for(
                asyncio.to_thread(_play_wav_sounddevice, output),
                timeout=self.playback.timeout_s,
            )
        except asyncio.TimeoutError as exc:
            with contextlib.suppress(Exception):
                sd.stop()
            raise SpeechError(f"Player timed out after {self.playback.timeout_s:.1f}s") from exc

    async def _play_subprocess(self, output: Path) -> None:
        proc = await asyncio.create_subprocess_exec(
            *build_playback_args(self.playback, output),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await _communicate_or_kill(
                proc,
                input_data=None,
                timeout_s=self.playback.timeout_s,
                timeout_message=f"Player timed out after {self.playback.timeout_s:.1f}s",
            )
        except asyncio.TimeoutError as exc:
            raise SpeechError(f"Player timed out after {self.playback.timeout_s:.1f}s") from exc
        if proc.returncode != 0:
            raise SpeechError(_format_subprocess_error("Player", proc.returncode, stderr))


def split_speech_units(text: str, *, include_incomplete: bool = True) -> list[str]:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return []

    units: list[str] = []
    start = 0
    index = 0
    while index < len(value):
        if _is_sentence_terminal(value, index):
            end = index + 1
            while end < len(value) and value[end] in ".!?…\"')]}»”’":
                end += 1
            unit = value[start:end].strip()
            if _has_speakable_content(unit):
                units.append(unit)
            while end < len(value) and value[end].isspace():
                end += 1
            start = end
            index = end
            continue
        index += 1

    tail = value[start:].strip()
    if include_incomplete and _has_speakable_content(tail):
        units.append(tail)
    return units


def _is_sentence_terminal(value: str, index: int) -> bool:
    char = value[index]
    if char in "!?…":
        return True
    if char != ".":
        return False
    previous_char = value[index - 1] if index > 0 else ""
    next_char = value[index + 1] if index + 1 < len(value) else ""
    if previous_char.isalnum() and next_char.isalnum():
        return False
    return True


def split_complete_speech_units(text: str) -> list[str]:
    return split_speech_units(text, include_incomplete=False)


def build_piper_args(config: PiperConfig, output: Path) -> list[str]:
    return [
        *resolve_piper_command(config.command),
        "--model",
        config.model,
        "--output-file",
        str(output),
        "--sentence-silence",
        str(config.sentence_silence),
        "--volume",
        str(config.volume),
        *config.extra_args,
    ]


def build_playback_args(config: PlaybackConfig, output: Path) -> list[str]:
    return [
        config.command,
        "--client-name",
        config.client_name,
        "--stream-name",
        config.stream_name,
        "--volume=65536",
        "--property=application.id=speaker",
        "--property=state.restore-props=false",
        "--property=state.restore-target=false",
        f"--property=module-stream-restore.id={build_stream_restore_id(output)}",
        str(output),
    ]


def build_stream_restore_id(output: Path) -> str:
    digest = hashlib.sha1(str(output).encode("utf-8")).hexdigest()[:16]
    return f"speaker-tts-{digest}"


def _play_wav_sounddevice(output: Path) -> None:
    if sd is None:
        raise SpeechError("sounddevice is not installed")
    with wave.open(str(output), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise SpeechError(f"Only 16-bit PCM WAV playback is supported, got {sample_width * 8}-bit")
    data = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        data = data.reshape(-1, channels)
    sd.play(data, samplerate=sample_rate, blocking=True)


def _has_speakable_content(value: str) -> bool:
    return any(char.isalnum() or unicodedata.category(char).startswith("S") for char in value)


def _playback_command_available(config: PlaybackConfig) -> bool:
    command = str(getattr(config, "command", "") or "").strip()
    if not command:
        return False
    path = Path(command)
    return path.exists() or shutil.which(command) is not None


def resolve_piper_command(command: str) -> list[str]:
    value = str(command or "").strip()
    if not value:
        raise SpeechError("Piper command is not configured")

    path = Path(value)
    name = path.name.casefold()
    if name in {"piper", "piper.exe"}:
        python = _find_venv_python(path)
        if python is not None:
            return [str(python), "-m", "piper"]
    if name.startswith("python"):
        return [value, "-m", "piper"]
    return [value]


def resolve_worker_python_command(command: str) -> list[str] | None:
    value = str(command or "").strip()
    if not value:
        return None
    path = Path(value)
    name = path.name.casefold()
    if name in {"piper", "piper.exe"}:
        python = _find_venv_python(path)
        return [str(python)] if python is not None else None
    if name.startswith("python"):
        return [value]
    return None


class _PiperWorkerClient:
    def __init__(
        self,
        *,
        python: list[str],
        model: str,
        volume: float,
        sentence_silence: float,
        timeout_s: float,
    ) -> None:
        self.python = list(python)
        self.model = str(model)
        self.volume = float(volume)
        self.sentence_silence = float(sentence_silence)
        self.timeout_s = float(timeout_s)
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def synthesize(self, text: str, output: Path) -> None:
        async with self._lock:
            await self._ensure_started()
            proc = self._proc
            if proc is None or proc.stdin is None or proc.stdout is None:
                raise SpeechError("Piper worker is not running")
            request = json.dumps(
                {"text": text, "output": str(output)},
                ensure_ascii=False,
            )
            proc.stdin.write((request + "\n").encode("utf-8"))
            await proc.stdin.drain()
            raw = await asyncio.wait_for(proc.stdout.readline(), timeout=self.timeout_s)
            if not raw:
                stderr = await self._read_stderr_tail()
                raise SpeechError(f"Piper worker stopped unexpectedly{stderr}")
            response = json.loads(raw.decode("utf-8"))
            if not isinstance(response, dict) or response.get("ok") is not True:
                raise SpeechError(str(response.get("error") if isinstance(response, dict) else response))

    async def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.stdin is not None:
            with contextlib.suppress(Exception):
                proc.stdin.close()
                await proc.stdin.wait_closed()
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()

    def __del__(self) -> None:
        proc = self._proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()

    async def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        worker_path = Path(__file__).with_name("piper_worker.py")
        self._proc = await asyncio.create_subprocess_exec(
            *self.python,
            str(worker_path),
            "--model",
            self.model,
            "--volume",
            str(self.volume),
            "--sentence-silence",
            str(self.sentence_silence),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        proc = self._proc
        assert proc.stdout is not None
        raw = await asyncio.wait_for(proc.stdout.readline(), timeout=self.timeout_s)
        if not raw:
            stderr = await self._read_stderr_tail()
            raise SpeechError(f"Piper worker failed to start{stderr}")
        response = json.loads(raw.decode("utf-8"))
        if not isinstance(response, dict) or response.get("ready") is not True:
            raise SpeechError(f"Piper worker returned invalid startup response: {response!r}")

    async def _read_stderr_tail(self) -> str:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return ""
        with contextlib.suppress(Exception):
            data = await asyncio.wait_for(proc.stderr.read(1000), timeout=0.1)
            if data:
                return f": {data.decode('utf-8', errors='replace').strip()}"
        return ""


async def _communicate_or_kill(
    proc,
    *,
    input_data: bytes | None,
    timeout_s: float,
    timeout_message: str,
) -> tuple[bytes, bytes]:
    try:
        return await asyncio.wait_for(proc.communicate(input_data), timeout=timeout_s)
    except asyncio.TimeoutError:
        await _kill_process(proc)
        raise asyncio.TimeoutError(timeout_message)
    except asyncio.CancelledError:
        await _kill_process(proc)
        raise


async def _kill_process(proc) -> None:
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    with contextlib.suppress(Exception):
        await proc.wait()


def _find_venv_python(entrypoint: Path) -> Path | None:
    bin_dir = entrypoint.parent
    if bin_dir.name not in {"bin", "Scripts"}:
        return None

    candidates = ("python3", "python") if bin_dir.name == "bin" else ("python.exe", "python")
    for candidate in candidates:
        python = bin_dir / candidate
        if python.exists():
            return python
    return None


def _format_subprocess_error(name: str, returncode: int | None, stderr: bytes) -> str:
    details = stderr.decode("utf-8", errors="replace").strip()
    if details:
        details = details[-500:]
        return f"{name} failed ({returncode}): {details}"
    return f"{name} failed ({returncode})"


def _perf_now() -> int:
    try:
        from core import perf

        return perf.now_ns()
    except Exception:
        return 0


def _perf_elapsed(start_ns: int, end_ns: int | None = None) -> float | None:
    try:
        from core import perf

        return perf.elapsed_ms(start_ns, end_ns)
    except Exception:
        return None


def _perf_emit(namespace: str, stage: str, **fields) -> None:
    try:
        from core import perf

        perf.emit(namespace, stage, **fields)
    except Exception:
        return
