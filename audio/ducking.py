"""PulseAudio/PipeWire volume ducking helpers."""

from __future__ import annotations

import atexit
import asyncio
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

_ACTIVE_DUCKERS: set["PulseAudioDucker"] = set()
_DUCKING_ACTIVE_SCALES: dict[int, list[float]] = {}
_DUCKING_CLEANUP_INSTALLED = False
_DUCKING_LOCK: asyncio.Lock | None = None
_DUCKING_LOCK_LOOP: asyncio.AbstractEventLoop | None = None
_DUCKING_ORIGINALS: dict[int, SinkInputVolume] = {}
_DUCKING_STATE_PATH = Path(
    os.environ.get(
        "LISTENER_DUCKING_STATE_PATH",
        str(Path(__file__).resolve().parents[1] / "state" / "ducking_state.json"),
    )
)
_DUCKING_STEP_MS = 20


class DuckingError(RuntimeError):
    """Raised when system audio ducking cannot be applied."""


@dataclass(frozen=True, slots=True)
class SinkInputVolume:
    sink_input_id: int
    volumes: list[int]
    channel_names: list[str] = field(default_factory=list)
    application_id: str = ""
    application_name: str = ""
    media_name: str = ""
    media_role: str = ""
    node_name: str = ""


class PulseAudioDucker:
    """Temporarily scales PulseAudio/PipeWire sink input volumes."""

    def __init__(self, config: Any, *, exclude_speaker: bool = True) -> None:
        self.config = config
        self.exclude_speaker = exclude_speaker
        self._snapshot: list[SinkInputVolume] = []
        self._volume_scale = 1.0

    async def duck(self) -> None:
        if not bool(getattr(self.config, "enabled", False)):
            return
        volume_scale = float(getattr(self.config, "volume_scale", 1.0) or 1.0)
        if volume_scale >= 1.0:
            return
        self._volume_scale = min(1.0, max(0.0, volume_scale))
        _install_ducking_cleanup()
        async with _get_ducking_lock():
            try:
                self._snapshot = await _list_sink_inputs(exclude_speaker=self.exclude_speaker)
                if not self._snapshot:
                    return
                original = _original_targets_for_snapshot(self._snapshot)
                _register_ducker_snapshot(self, self._snapshot)
                _remember_persisted_originals(original)
                target = _effective_targets_for_snapshot(self._snapshot)
                _ACTIVE_DUCKERS.add(self)
                log.debug(
                    "audio.ducking: ducking %d sink input(s) scale=%.3f "
                    "exclude_speaker=%s ids=%s",
                    len(self._snapshot),
                    self._volume_scale,
                    self.exclude_speaker,
                    [item.sink_input_id for item in self._snapshot],
                )
                await self._apply_steps(
                    build_volume_transition_steps(
                        self._snapshot,
                        target,
                        int(getattr(self.config, "fade_in_ms", 0) or 0),
                    ),
                    int(getattr(self.config, "fade_in_ms", 0) or 0),
                )
                await _restore_route_settings_best_effort(original)
            except DuckingError as exc:
                original = _original_targets_for_snapshot(self._snapshot)
                target = _release_ducker_snapshot(self)
                await _restore_snapshot_best_effort(target)
                await _restore_route_settings_best_effort(original)
                _clear_persisted_originals(original)
                _ACTIVE_DUCKERS.discard(self)
                self._snapshot = []
                log.debug("audio.ducking: unavailable: %s", exc)

    async def restore(self) -> None:
        if not bool(getattr(self.config, "enabled", False)) or not self._snapshot:
            return
        async with _get_ducking_lock():
            _ACTIVE_DUCKERS.discard(self)
            touched = self._snapshot
            original = _original_targets_for_snapshot(touched)
            targets = _release_ducker_snapshot(self)
            self._snapshot = []
            current = await _current_snapshot_for(touched)
            if not current:
                log.debug(
                    "audio.ducking: restore skipped because sink input(s) disappeared ids=%s",
                    [item.sink_input_id for item in touched],
                )
                await _restore_route_settings_best_effort(original)
                _forget_inactive_originals(item.sink_input_id for item in touched)
                return
            log.debug(
                "audio.ducking: restoring %d sink input(s) ids=%s",
                len(current),
                [item.sink_input_id for item in current],
            )
            try:
                await self._apply_steps(
                    build_volume_transition_steps(
                        current,
                        targets,
                        int(getattr(self.config, "fade_out_ms", 0) or 0),
                    ),
                    int(getattr(self.config, "fade_out_ms", 0) or 0),
                )
            except DuckingError:
                await _restore_snapshot_best_effort(targets)
            finally:
                await _restore_route_settings_best_effort(original)
                _forget_inactive_originals(item.sink_input_id for item in touched)

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


async def restore_all_ducking() -> dict[str, Any]:
    """Best-effort recovery for any stuck global ducking state."""

    async with _get_ducking_lock():
        originals = list(_DUCKING_ORIGINALS.values())
        persisted_originals = _load_persisted_originals()
        active_duckers = list(_ACTIVE_DUCKERS)
        for ducker in active_duckers:
            ducker._snapshot = []
        _ACTIVE_DUCKERS.clear()
        _DUCKING_ACTIVE_SCALES.clear()
        _DUCKING_ORIGINALS.clear()

        listener_state = await normalize_listener_output_volume_state()
        listener_ids = list(listener_state.get("stream_ids") or [])
        listener_route_keys = list(listener_state.get("route_keys") or [])
        restore_targets = _merge_originals_by_route_key([*originals, *persisted_originals])
        if not restore_targets:
            return {
                "restored": bool(listener_ids),
                "active_duckers": len(active_duckers),
                "restored_sink_input_ids": [],
                "missing_sink_input_ids": [],
                "listener_stream_ids": listener_ids,
                "listener_route_keys": listener_route_keys,
                "persisted_baselines": len(persisted_originals),
            }

        current = await _list_sink_inputs(exclude_speaker=False)
        restored_ids, missing_keys = await _restore_targets_against_current_streams(
            restore_targets,
            current,
        )
        await _restore_route_settings_best_effort(restore_targets)
        _clear_persisted_originals(restore_targets)
        log.info(
            "audio.ducking: forced restore active_duckers=%d persisted=%d "
            "restored_ids=%s missing_keys=%s",
            len(active_duckers),
            len(persisted_originals),
            restored_ids,
            missing_keys,
        )
        return {
            "restored": bool(restored_ids or missing_keys or active_duckers or listener_ids),
            "active_duckers": len(active_duckers),
            "restored_sink_input_ids": restored_ids,
            "missing_sink_input_ids": [],
            "missing_route_keys": missing_keys,
            "listener_stream_ids": listener_ids,
            "listener_route_keys": listener_route_keys,
            "persisted_baselines": len(persisted_originals),
        }


def _get_ducking_lock() -> asyncio.Lock:
    global _DUCKING_LOCK, _DUCKING_LOCK_LOOP
    loop = asyncio.get_running_loop()
    if _DUCKING_LOCK is None or _DUCKING_LOCK_LOOP is not loop:
        _DUCKING_LOCK = asyncio.Lock()
        _DUCKING_LOCK_LOOP = loop
    return _DUCKING_LOCK


async def _current_snapshot_for(touched: list[SinkInputVolume]) -> list[SinkInputVolume]:
    wanted = {item.sink_input_id for item in touched}
    if not wanted:
        return []
    current = await _list_sink_inputs(exclude_speaker=False)
    return [item for item in current if item.sink_input_id in wanted]


def _register_ducker_snapshot(
    ducker: PulseAudioDucker,
    snapshot: list[SinkInputVolume],
) -> None:
    for item in snapshot:
        sink_input_id = item.sink_input_id
        if sink_input_id not in _DUCKING_ORIGINALS:
            _DUCKING_ORIGINALS[sink_input_id] = item
        _DUCKING_ACTIVE_SCALES.setdefault(sink_input_id, []).append(ducker._volume_scale)


def _original_targets_for_snapshot(snapshot: list[SinkInputVolume]) -> list[SinkInputVolume]:
    return [_DUCKING_ORIGINALS.get(item.sink_input_id, item) for item in snapshot]


def _release_ducker_snapshot(ducker: PulseAudioDucker) -> list[SinkInputVolume]:
    targets: list[SinkInputVolume] = []
    touched_ids: list[int] = []
    for item in ducker._snapshot:
        sink_input_id = item.sink_input_id
        touched_ids.append(sink_input_id)
        scales = _DUCKING_ACTIVE_SCALES.get(sink_input_id)
        if scales:
            _remove_one_scale(scales, ducker._volume_scale)
            if not scales:
                _DUCKING_ACTIVE_SCALES.pop(sink_input_id, None)
        original = _DUCKING_ORIGINALS.get(sink_input_id, item)
        targets.append(_effective_target_for_original(original))
    _forget_inactive_originals(touched_ids)
    return targets


def _remove_one_scale(scales: list[float], target: float) -> None:
    for index, value in enumerate(scales):
        if math.isclose(value, target):
            scales.pop(index)
            return
    scales.pop()


def _effective_targets_for_snapshot(snapshot: list[SinkInputVolume]) -> list[SinkInputVolume]:
    targets: list[SinkInputVolume] = []
    for item in snapshot:
        original = _DUCKING_ORIGINALS.get(item.sink_input_id, item)
        targets.append(_effective_target_for_original(original))
    return targets


def _effective_target_for_original(original: SinkInputVolume) -> SinkInputVolume:
    scales = _DUCKING_ACTIVE_SCALES.get(original.sink_input_id) or []
    if not scales:
        return original
    return _scale_sink_input(original, min(scales))


def _forget_inactive_originals(sink_input_ids: Iterable[int]) -> None:
    inactive_originals: list[SinkInputVolume] = []
    for sink_input_id in sink_input_ids:
        if not _DUCKING_ACTIVE_SCALES.get(sink_input_id):
            original = _DUCKING_ORIGINALS.get(sink_input_id)
            if original is not None:
                inactive_originals.append(original)
            _DUCKING_ACTIVE_SCALES.pop(sink_input_id, None)
            _DUCKING_ORIGINALS.pop(sink_input_id, None)
    inactive_keys = {_wireplumber_route_settings_key(item) for item in inactive_originals}
    inactive_keys.discard("")
    if not inactive_keys:
        return
    active_keys = {
        _wireplumber_route_settings_key(item)
        for sink_input_id, item in _DUCKING_ORIGINALS.items()
        if _DUCKING_ACTIVE_SCALES.get(sink_input_id)
    }
    _clear_persisted_original_keys(inactive_keys - active_keys)


async def _list_sink_inputs(*, exclude_speaker: bool = True) -> list[SinkInputVolume]:
    payload = await _list_raw_sink_inputs()
    return parse_sink_input_volumes(payload, exclude_speaker=exclude_speaker)


async def normalize_active_listener_output_streams() -> list[int]:
    """Compatibility wrapper returning only active Listener playback stream ids."""

    state = await normalize_listener_output_volume_state()
    return list(state.get("stream_ids") or [])


async def normalize_listener_output_volume_state() -> dict[str, Any]:
    """Normalize active Listener playback and persisted route volume state."""

    return await asyncio.to_thread(normalize_listener_output_volume_state_sync)


def normalize_active_listener_output_streams_sync(
    *,
    retries: int = 5,
    delay_s: float = 0.05,
) -> list[int]:
    state = normalize_listener_output_volume_state_sync(retries=retries, delay_s=delay_s)
    return list(state.get("stream_ids") or [])


def normalize_listener_output_volume_state_sync(
    *,
    retries: int = 5,
    delay_s: float = 0.05,
) -> dict[str, list[int] | list[str]]:
    route_keys = _normalize_listener_output_route_settings_sync()
    stream_ids = _normalize_active_listener_output_streams_sync(
        retries=retries,
        delay_s=delay_s,
    )
    return {
        "route_keys": route_keys,
        "stream_ids": stream_ids,
    }


def _normalize_active_listener_output_streams_sync(
    *,
    retries: int = 5,
    delay_s: float = 0.05,
) -> list[int]:
    for attempt in range(max(1, int(retries))):
        try:
            payload = _list_raw_sink_inputs_sync()
        except DuckingError:
            return []
        restored: list[int] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            sink_input_id = item.get("index")
            raw_volume = item.get("volume")
            properties = item.get("properties")
            if not isinstance(sink_input_id, int) or not isinstance(raw_volume, dict):
                continue
            if not isinstance(properties, dict) or not _is_speaker_sink_input(properties):
                continue
            channel_names = [str(name) for name, value in raw_volume.items() if isinstance(value, dict)]
            if not channel_names:
                channel_names = ["mono"]
            target = SinkInputVolume(
                sink_input_id=sink_input_id,
                volumes=[65536] * len(channel_names),
                channel_names=channel_names,
                application_id=_property_text(properties.get("application.id")),
                application_name=_property_text(properties.get("application.name")),
                media_name=_property_text(properties.get("media.name")),
                media_role=_property_text(properties.get("media.role")),
                node_name=_property_text(properties.get("node.name")),
            )
            _run_pactl_sync(
                "set-sink-input-volume",
                str(sink_input_id),
                *[str(value) for value in target.volumes],
            )
            _restore_route_settings_sync(target)
            restored.append(sink_input_id)
        if restored or attempt + 1 >= max(1, int(retries)):
            if restored:
                log.info(
                    "audio.ducking: normalized listener output stream(s) ids=%s",
                    restored,
                )
            return restored
        time.sleep(max(0.0, float(delay_s)))
    return []


def _normalize_listener_output_route_settings_sync() -> list[str]:
    restored: list[str] = []
    for target in _listener_output_route_targets():
        key = _wireplumber_route_settings_key(target)
        _restore_route_settings_sync(target)
        if key:
            restored.append(key)
    return restored


def _listener_output_route_targets() -> list[SinkInputVolume]:
    targets = [
        SinkInputVolume(
            sink_input_id=0,
            volumes=[65536],
            channel_names=["mono"],
            application_id="speaker",
        )
    ]
    for application_name in sorted(_listener_python_application_names()):
        targets.append(
            SinkInputVolume(
                sink_input_id=0,
                volumes=[65536],
                channel_names=["mono"],
                application_name=application_name,
            )
        )
    return targets


async def _list_raw_sink_inputs() -> list[object]:
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
    return _decode_sink_inputs_json(stdout)


def _list_raw_sink_inputs_sync() -> list[object]:
    result = subprocess.run(
        ["pactl", "--format", "json", "list", "sink-inputs"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise DuckingError(_format_subprocess_error("pactl", result.returncode, result.stderr))
    return _decode_sink_inputs_json(result.stdout)


def _decode_sink_inputs_json(stdout: bytes) -> list[object]:
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise DuckingError("pactl returned invalid JSON") from exc
    if not isinstance(payload, list):
        raise DuckingError("pactl returned unexpected sink input payload")
    return payload


async def _restore_targets_against_current_streams(
    targets: list[SinkInputVolume],
    current: list[SinkInputVolume],
) -> tuple[list[int], list[str]]:
    restore_snapshot: list[SinkInputVolume] = []
    missing_keys: list[str] = []
    current_by_key: dict[str, list[SinkInputVolume]] = {}
    current_by_id = {item.sink_input_id: item for item in current}
    for item in current:
        key = _wireplumber_route_settings_key(item)
        if key:
            current_by_key.setdefault(key, []).append(item)

    for target in targets:
        key = _wireplumber_route_settings_key(target)
        matches = current_by_key.get(key, []) if key else []
        if not matches and target.sink_input_id in current_by_id:
            matches = [current_by_id[target.sink_input_id]]
        if not matches:
            missing_keys.append(key or str(target.sink_input_id))
            continue
        for match in matches:
            restore_snapshot.append(
                SinkInputVolume(
                    sink_input_id=match.sink_input_id,
                    volumes=list(target.volumes),
                    channel_names=match.channel_names or target.channel_names,
                    application_id=target.application_id,
                    application_name=target.application_name,
                    media_name=target.media_name,
                    media_role=target.media_role,
                    node_name=target.node_name,
                )
            )

    if restore_snapshot:
        await _restore_snapshot_best_effort(restore_snapshot)
    return [item.sink_input_id for item in restore_snapshot], missing_keys


def _merge_originals_by_route_key(items: list[SinkInputVolume]) -> list[SinkInputVolume]:
    merged: dict[str, SinkInputVolume] = {}
    for item in items:
        key = _wireplumber_route_settings_key(item) or str(item.sink_input_id)
        merged.setdefault(key, item)
    return list(merged.values())


def _remember_persisted_originals(snapshot: list[SinkInputVolume]) -> None:
    baselines = _load_persisted_baseline_map()
    changed = False
    for item in snapshot:
        key = _wireplumber_route_settings_key(item)
        if not key or key in baselines:
            continue
        baselines[key] = _sink_input_to_state(item)
        changed = True
    if changed:
        _save_persisted_baseline_map(baselines)


def _load_persisted_originals() -> list[SinkInputVolume]:
    return [
        item
        for item in (
            _sink_input_from_state(value)
            for value in _load_persisted_baseline_map().values()
        )
        if item is not None
    ]


def _clear_persisted_originals(snapshot: list[SinkInputVolume]) -> None:
    keys = {_wireplumber_route_settings_key(item) for item in snapshot}
    keys.discard("")
    _clear_persisted_original_keys(keys)


def _clear_persisted_original_keys(keys: set[str]) -> None:
    if not keys:
        return
    baselines = _load_persisted_baseline_map()
    changed = False
    for key in keys:
        if key in baselines:
            baselines.pop(key, None)
            changed = True
    if changed:
        _save_persisted_baseline_map(baselines)


def _load_persisted_baseline_map() -> dict[str, dict[str, Any]]:
    path = _DUCKING_STATE_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        log.debug("audio.ducking: unable to read persisted ducking baselines", exc_info=True)
        return {}
    if not isinstance(payload, dict):
        return {}
    raw_baselines = payload.get("baselines")
    if not isinstance(raw_baselines, dict):
        return {}
    return {
        str(key): value
        for key, value in raw_baselines.items()
        if isinstance(value, dict)
    }


def _save_persisted_baseline_map(baselines: dict[str, dict[str, Any]]) -> None:
    path = _DUCKING_STATE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not baselines:
            path.unlink(missing_ok=True)
            return
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "updated_at": time.time(),
                    "baselines": baselines,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError:
        log.debug("audio.ducking: unable to write persisted ducking baselines", exc_info=True)


def _sink_input_to_state(item: SinkInputVolume) -> dict[str, Any]:
    return {
        "volumes": list(item.volumes),
        "channel_names": list(item.channel_names),
        "application_id": item.application_id,
        "application_name": item.application_name,
        "media_name": item.media_name,
        "media_role": item.media_role,
        "node_name": item.node_name,
        "updated_at": time.time(),
    }


def _sink_input_from_state(value: dict[str, Any]) -> SinkInputVolume | None:
    raw_volumes = value.get("volumes")
    if not isinstance(raw_volumes, list):
        return None
    volumes = [item for item in raw_volumes if isinstance(item, int) and item >= 0]
    if not volumes:
        return None
    raw_channels = value.get("channel_names")
    channel_names = [
        str(item)
        for item in raw_channels
        if isinstance(item, str)
    ] if isinstance(raw_channels, list) else []
    return SinkInputVolume(
        sink_input_id=0,
        volumes=volumes,
        channel_names=channel_names[: len(volumes)],
        application_id=str(value.get("application_id") or ""),
        application_name=str(value.get("application_name") or ""),
        media_name=str(value.get("media_name") or ""),
        media_role=str(value.get("media_role") or ""),
        node_name=str(value.get("node_name") or ""),
    )


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


async def _run_pw_metadata(*args: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "pw-metadata",
        "-n",
        "route-settings",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise DuckingError(_format_subprocess_error("pw-metadata", proc.returncode, stderr))


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


def build_volume_transition_steps(
    start_snapshot: list[SinkInputVolume],
    target_snapshot: list[SinkInputVolume],
    duration_ms: int,
    *,
    step_ms: int = _DUCKING_STEP_MS,
) -> list[list[SinkInputVolume]]:
    if not start_snapshot or not target_snapshot:
        return []

    targets = {item.sink_input_id: item for item in target_snapshot}
    pairs = [
        (start, targets[start.sink_input_id])
        for start in start_snapshot
        if start.sink_input_id in targets
    ]
    if not pairs:
        return []
    if duration_ms <= 0:
        return [[target for _, target in pairs]]

    interval_count = max(1, math.ceil(duration_ms / max(1, step_ms)))
    denominator = interval_count + 1
    steps: list[list[SinkInputVolume]] = []
    last_signature: tuple[tuple[int, tuple[int, ...]], ...] | None = None
    for step_index in range(1, denominator + 1):
        progress = step_index / denominator
        current = [
            _interpolate_sink_input(start, target, progress)
            for start, target in pairs
        ]
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
        channel_names: list[str] = []
        for channel_name, channel in raw_volume.items():
            if not isinstance(channel, dict):
                continue
            value = channel.get("value")
            if isinstance(value, int) and value >= 0:
                channels.append(value)
                channel_names.append(str(channel_name))
        if channels:
            snapshot.append(
                SinkInputVolume(
                    sink_input_id=sink_input_id,
                    volumes=channels,
                    channel_names=channel_names[: len(channels)],
                    application_id=_property_text(properties.get("application.id")),
                    application_name=_property_text(properties.get("application.name")),
                    media_name=_property_text(properties.get("media.name")),
                    media_role=_property_text(properties.get("media.role")),
                    node_name=_property_text(properties.get("node.name")),
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


async def _restore_route_settings_best_effort(snapshot: list[SinkInputVolume]) -> None:
    for item in snapshot:
        key = _wireplumber_route_settings_key(item)
        if not key:
            continue
        payload = _wireplumber_route_settings_payload(item)
        try:
            await _run_pw_metadata("0", key, payload, "Spa:String:JSON")
        except DuckingError as exc:
            log.debug("audio.ducking: route-settings restore failed for %s: %s", key, exc)


def _install_ducking_cleanup() -> None:
    global _DUCKING_CLEANUP_INSTALLED
    if _DUCKING_CLEANUP_INSTALLED:
        return
    atexit.register(_restore_active_duckers_sync)
    _DUCKING_CLEANUP_INSTALLED = True


def _restore_active_duckers_sync() -> None:
    for ducker in list(_ACTIVE_DUCKERS):
        ducker._snapshot = []
    _ACTIVE_DUCKERS.clear()
    originals = list(_DUCKING_ORIGINALS.values())
    _DUCKING_ACTIVE_SCALES.clear()
    _DUCKING_ORIGINALS.clear()
    for item in originals:
        _run_pactl_sync(
            "set-sink-input-volume",
            str(item.sink_input_id),
            *[str(value) for value in item.volumes],
        )
        _restore_route_settings_sync(item)


def _run_pactl_sync(*args: str) -> None:
    result = subprocess.run(
        ["pactl", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        log.debug("audio.ducking: synchronous restore failed for args: %s", args)


def _run_pw_metadata_sync(*args: str) -> None:
    result = subprocess.run(
        ["pw-metadata", "-n", "route-settings", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        log.debug("audio.ducking: synchronous route-settings restore failed for args: %s", args)


def _scale_sink_input(item: SinkInputVolume, scale: float) -> SinkInputVolume:
    return SinkInputVolume(
        sink_input_id=item.sink_input_id,
        volumes=[max(0, int(round(value * scale))) for value in item.volumes],
        channel_names=item.channel_names,
        application_id=item.application_id,
        application_name=item.application_name,
        media_name=item.media_name,
        media_role=item.media_role,
        node_name=item.node_name,
    )


def _interpolate_sink_input(
    start: SinkInputVolume,
    target: SinkInputVolume,
    progress: float,
) -> SinkInputVolume:
    volumes: list[int] = []
    for index, target_value in enumerate(target.volumes):
        start_value = start.volumes[index] if index < len(start.volumes) else target_value
        value = start_value + (target_value - start_value) * progress
        volumes.append(max(0, int(round(value))))
    return SinkInputVolume(
        sink_input_id=target.sink_input_id,
        volumes=volumes,
        channel_names=target.channel_names,
        application_id=target.application_id,
        application_name=target.application_name,
        media_name=target.media_name,
        media_role=target.media_role,
        node_name=target.node_name,
    )


def _restore_route_settings_sync(item: SinkInputVolume) -> None:
    key = _wireplumber_route_settings_key(item)
    if not key:
        return
    _run_pw_metadata_sync("0", key, _wireplumber_route_settings_payload(item), "Spa:String:JSON")


def _wireplumber_route_settings_key(item: SinkInputVolume) -> str:
    property_name = ""
    property_value = ""
    for candidate_name, candidate_value in (
        ("media.role", item.media_role),
        ("application.id", item.application_id),
        ("application.name", item.application_name),
        ("media.name", item.media_name),
        ("node.name", item.node_name),
    ):
        if candidate_value:
            property_name = candidate_name
            property_value = candidate_value
            break
    if not property_name:
        return ""
    return f"restore.stream.Output/Audio.{property_name}:{property_value}"


def _wireplumber_route_settings_payload(item: SinkInputVolume) -> str:
    volume_values = [round(value / 65536.0, 6) for value in item.volumes]
    payload: dict[str, object] = {
        "volumes": volume_values,
        "volume": round(sum(volume_values) / len(volume_values), 6) if volume_values else 1.0,
        "mute": False,
    }
    channels = [_wireplumber_channel_name(name) for name in item.channel_names]
    channels = [name for name in channels if name]
    if channels and len(channels) == len(volume_values):
        payload["channels"] = channels
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _wireplumber_channel_name(name: str) -> str:
    return {
        "front-left": "FL",
        "front-right": "FR",
        "front-center": "FC",
        "rear-left": "RL",
        "rear-right": "RR",
        "lfe": "LFE",
        "mono": "MONO",
    }.get(str(name or "").strip().casefold(), "")


def _is_speaker_sink_input(properties: dict[object, object]) -> bool:
    application_id = _property_text(properties.get("application.id")).casefold()
    application_name = _property_text(properties.get("application.name")).casefold()
    media_name = _property_text(properties.get("media.name")).casefold()
    process_id = _property_text(properties.get("application.process.id"))
    if application_id == "speaker" or (
        application_name == "speaker" and media_name == "speaker tts"
    ):
        return True
    if process_id and process_id == str(os.getpid()):
        return True
    if media_name == "alsa playback" and application_name in _listener_python_application_names():
        return True
    return False


def _listener_python_application_names() -> set[str]:
    executable_names = {
        os.path.basename(sys.executable).strip().casefold(),
        os.path.basename(os.path.realpath(sys.executable)).strip().casefold(),
    }
    values = set()
    for executable_name in executable_names:
        if not executable_name:
            continue
        values.add(f"pipewire alsa [{executable_name}]")
        if executable_name.startswith("python") and "." in executable_name:
            values.add(f"pipewire alsa [{executable_name.split('.', 1)[0]}]")
    return values


def _property_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _format_subprocess_error(name: str, returncode: int | None, stderr: bytes) -> str:
    details = stderr.decode("utf-8", errors="replace").strip()
    if details:
        details = details[-500:]
        return f"{name} failed ({returncode}): {details}"
    return f"{name} failed ({returncode})"
