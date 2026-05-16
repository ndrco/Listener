from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Protocol

from audio.ducking import (
    PulseAudioDucker,
    SinkInputVolume,
    build_ducking_steps,
    parse_sink_input_volumes,
)

from .config import PiperConfig, PlaybackConfig
from .emoji import extract_emoji_for_speech


class SpeechError(RuntimeError):
    pass


log = logging.getLogger(__name__)


class SpeechEngine(Protocol):
    async def speak(self, text: str) -> None:
        ...


class PiperSpeechEngine:
    def __init__(self, piper: PiperConfig, playback: PlaybackConfig) -> None:
        self.piper = piper
        self.playback = playback

    async def speak(self, text: str) -> None:
        parsed = extract_emoji_for_speech(text)
        if parsed.tokens:
            log.debug("PiperSpeechEngine: stripped %d emoji(s) before synthesis", len(parsed.tokens))
        units = split_speech_units(parsed.speech_text)
        if not units:
            return
        with tempfile.TemporaryDirectory(prefix="speaker-") as tmp:
            ducker = PulseAudioDucker(self.playback.ducking)
            ducked = False
            try:
                for index, unit in enumerate(units):
                    output = Path(tmp) / f"speech-{index}.wav"
                    await self._synthesize(unit, output)
                    if not ducked:
                        await ducker.duck()
                        ducked = True
                    await self._play(output)
            finally:
                if ducked:
                    await ducker.restore()

    async def _synthesize(self, text: str, output: Path) -> None:
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

    async def _play(self, output: Path) -> None:
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
        if value[index] in ".!?…":
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
        "--property=application.id=speaker",
        "--property=media.role=a11y",
        str(output),
    ]


def _has_speakable_content(value: str) -> bool:
    return any(char.isalnum() or unicodedata.category(char).startswith("S") for char in value)


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
