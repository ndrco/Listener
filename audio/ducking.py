"""PulseAudio/PipeWire volume ducking helpers."""

from __future__ import annotations

import atexit
import asyncio
import json
import logging
import math
import subprocess
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

_ACTIVE_DUCKERS: set["PulseAudioDucker"] = set()
_DUCKING_CLEANUP_INSTALLED = False
_DUCKING_STEP_MS = 20


class DuckingError(RuntimeError):
    """Raised when system audio ducking cannot be applied."""


@dataclass(frozen=True, slots=True)
class SinkInputVolume:
    sink_input_id: int
    volumes: list[int]
    application_id: str = ""
    application_name: str = ""
    media_name: str = ""
    media_role: str = ""


class PulseAudioDucker:
    """Temporarily scales PulseAudio/PipeWire sink input volumes."""

    def __init__(self, config: Any, *, exclude_speaker: bool = True) -> None:
        self.config = config
        self.exclude_speaker = exclude_speaker
        self._snapshot: list[SinkInputVolume] = []

    async def duck(self) -> None:
        if not bool(getattr(self.config, "enabled", False)):
            return
        volume_scale = float(getattr(self.config, "volume_scale", 1.0) or 1.0)
        if volume_scale >= 1.0:
            return
        _install_ducking_cleanup()
        try:
            self._snapshot = await _list_sink_inputs(exclude_speaker=self.exclude_speaker)
            if not self._snapshot:
                return
            _ACTIVE_DUCKERS.add(self)
            await self._apply_steps(
                build_ducking_steps(
                    self._snapshot,
                    volume_scale,
                    int(getattr(self.config, "fade_in_ms", 0) or 0),
                ),
                int(getattr(self.config, "fade_in_ms", 0) or 0),
            )
        except DuckingError as exc:
            await _restore_snapshot_best_effort(self._snapshot)
            _ACTIVE_DUCKERS.discard(self)
            self._snapshot = []
            log.debug("audio.ducking: unavailable: %s", exc)

    async def restore(self) -> None:
        if not bool(getattr(self.config, "enabled", False)) or not self._snapshot:
            return
        _ACTIVE_DUCKERS.discard(self)
        snapshot, self._snapshot = self._snapshot, []
        try:
            await self._apply_steps(
                build_ducking_steps(
                    snapshot,
                    float(getattr(self.config, "volume_scale", 1.0) or 1.0),
                    int(getattr(self.config, "fade_out_ms", 0) or 0),
                    restore=True,
                ),
                int(getattr(self.config, "fade_out_ms", 0) or 0),
            )
        except DuckingError:
            await _restore_snapshot_best_effort(snapshot)

    async def _apply_steps(self, steps: list[list[SinkInputVolume]], duration_ms: int) -> None:
        if not steps:
            return
        delay_s = 0.0
        if duration_ms > 0 and len(steps) > 1:
            delay_s = duration_ms / (len(steps) - 1) / 1000.0
        for index, snapshot in enumerate(steps):
            for item in snapshot:
                await _run_pactl(
                    "set-sink-input-volume",
                    str(item.sink_input_id),
                    *[str(value) for value in item.volumes],
                )
            if delay_s > 0 and index + 1 < len(steps):
                await asyncio.sleep(delay_s)


async def _list_sink_inputs(*, exclude_speaker: bool = True) -> list[SinkInputVolume]:
    proc = await asyncio.create_subprocess_exec(
        "pactl",
        "--format",
        "json",
        "list",
        "sink-inputs",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise DuckingError(_format_subprocess_error("pactl", proc.returncode, stderr))
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise DuckingError("pactl returned invalid JSON") from exc
    if not isinstance(payload, list):
        raise DuckingError("pactl returned unexpected sink input payload")
    return parse_sink_input_volumes(payload, exclude_speaker=exclude_speaker)


async def _run_pactl(*args: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "pactl",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise DuckingError(_format_subprocess_error("pactl", proc.returncode, stderr))


def build_ducking_steps(
    snapshot: list[SinkInputVolume],
    volume_scale: float,
    duration_ms: int,
    *,
    restore: bool = False,
    step_ms: int = _DUCKING_STEP_MS,
) -> list[list[SinkInputVolume]]:
    if not snapshot:
        return []

    target_scale = min(1.0, max(0.0, float(volume_scale)))
    start_scale = target_scale if restore else 1.0
    end_scale = 1.0 if restore else target_scale
    if math.isclose(start_scale, end_scale):
        return [[_scale_sink_input(item, end_scale) for item in snapshot]]

    if duration_ms <= 0:
        return [[_scale_sink_input(item, end_scale) for item in snapshot]]

    interval_count = max(1, math.ceil(duration_ms / max(1, step_ms)))
    denominator = interval_count + 1
    steps: list[list[SinkInputVolume]] = []
    last_signature: tuple[tuple[int, tuple[int, ...]], ...] | None = None
    for step_index in range(1, denominator + 1):
        progress = step_index / denominator
        scale = start_scale + (end_scale - start_scale) * progress
        current = [_scale_sink_input(item, scale) for item in snapshot]
        signature = tuple((item.sink_input_id, tuple(item.volumes)) for item in current)
        if signature != last_signature:
            steps.append(current)
            last_signature = signature
    return steps


def parse_sink_input_volumes(
    payload: list[object],
    *,
    exclude_speaker: bool = True,
) -> list[SinkInputVolume]:
    snapshot: list[SinkInputVolume] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        sink_input_id = item.get("index")
        if not isinstance(sink_input_id, int):
            continue
        raw_volume = item.get("volume")
        if not isinstance(raw_volume, dict):
            continue
        properties = item.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        if exclude_speaker and _is_speaker_sink_input(properties):
            continue
        channels: list[int] = []
        for channel in raw_volume.values():
            if not isinstance(channel, dict):
                continue
            value = channel.get("value")
            if isinstance(value, int) and value >= 0:
                channels.append(value)
        if channels:
            snapshot.append(
                SinkInputVolume(
                    sink_input_id=sink_input_id,
                    volumes=channels,
                    application_id=_property_text(properties.get("application.id")),
                    application_name=_property_text(properties.get("application.name")),
                    media_name=_property_text(properties.get("media.name")),
                    media_role=_property_text(properties.get("media.role")),
                )
            )
    return snapshot


async def _restore_snapshot_best_effort(snapshot: list[SinkInputVolume]) -> None:
    for item in snapshot:
        try:
            await _run_pactl(
                "set-sink-input-volume",
                str(item.sink_input_id),
                *[str(value) for value in item.volumes],
            )
        except DuckingError:
            continue


def _install_ducking_cleanup() -> None:
    global _DUCKING_CLEANUP_INSTALLED
    if _DUCKING_CLEANUP_INSTALLED:
        return
    atexit.register(_restore_active_duckers_sync)
    _DUCKING_CLEANUP_INSTALLED = True


def _restore_active_duckers_sync() -> None:
    while _ACTIVE_DUCKERS:
        ducker = _ACTIVE_DUCKERS.pop()
        snapshot, ducker._snapshot = ducker._snapshot, []
        for item in snapshot:
            _run_pactl_sync(
                "set-sink-input-volume",
                str(item.sink_input_id),
                *[str(value) for value in item.volumes],
            )


def _run_pactl_sync(*args: str) -> None:
    result = subprocess.run(
        ["pactl", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        log.debug("audio.ducking: synchronous restore failed for args: %s", args)


def _scale_sink_input(item: SinkInputVolume, scale: float) -> SinkInputVolume:
    return SinkInputVolume(
        sink_input_id=item.sink_input_id,
        volumes=[max(0, int(round(value * scale))) for value in item.volumes],
        application_id=item.application_id,
        application_name=item.application_name,
        media_name=item.media_name,
        media_role=item.media_role,
    )


def _is_speaker_sink_input(properties: dict[object, object]) -> bool:
    application_id = _property_text(properties.get("application.id")).casefold()
    application_name = _property_text(properties.get("application.name")).casefold()
    media_name = _property_text(properties.get("media.name")).casefold()
    return application_id == "speaker" or (
        application_name == "speaker" and media_name == "speaker tts"
    )


def _property_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _format_subprocess_error(name: str, returncode: int | None, stderr: bytes) -> str:
    details = stderr.decode("utf-8", errors="replace").strip()
    if details:
        details = details[-500:]
        return f"{name} failed ({returncode}): {details}"
    return f"{name} failed ({returncode})"
