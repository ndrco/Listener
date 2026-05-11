"""Shared helpers for calling OpenClaw Gateway RPC methods."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
import time
from typing import Any

from core.config import cfg

log = logging.getLogger(__name__)

_LAST_CHAT_RUN_BY_SESSION: dict[str, dict[str, Any]] = {}


def build_openclaw_base_command(raw: Any) -> list[str]:
    if isinstance(raw, list):
        tokens = [str(part).strip() for part in raw if str(part).strip()]
    else:
        text = str(raw or "").strip()
        if not text:
            text = "openclaw"
        try:
            tokens = shlex.split(text, posix=False)
        except Exception:
            tokens = [text]
        tokens = [part.strip() for part in tokens if part and part.strip()]

    if not tokens:
        tokens = ["openclaw"]

    first = tokens[0].lower()
    contains_openclaw = any(_token_is_openclaw_binary(token) for token in tokens)
    if first in {"wsl", "wsl.exe"} and not contains_openclaw:
        tokens.append("openclaw")
    return tokens


def _token_is_openclaw_binary(token: str) -> bool:
    cleaned = str(token or "").strip().strip("\"'").strip()
    if not cleaned or cleaned.startswith("-"):
        return False
    name = os.path.basename(cleaned).lower()
    return name in {"openclaw", "openclaw.exe"}


def remember_openclaw_chat_run(session_key: str, run_id: str) -> None:
    normalized_session_key = str(session_key or "").strip() or "main"
    normalized_run_id = str(run_id or "").strip()
    if not normalized_run_id:
        return
    _LAST_CHAT_RUN_BY_SESSION[normalized_session_key] = {
        "run_id": normalized_run_id,
        "remembered_at": time.time(),
    }


def get_openclaw_chat_run(session_key: str) -> dict[str, Any] | None:
    normalized_session_key = str(session_key or "").strip() or "main"
    run_info = _LAST_CHAT_RUN_BY_SESSION.get(normalized_session_key)
    if not isinstance(run_info, dict):
        return None
    return dict(run_info)


def clear_openclaw_chat_run(session_key: str, run_id: str | None = None) -> None:
    normalized_session_key = str(session_key or "").strip() or "main"
    if run_id is None:
        _LAST_CHAT_RUN_BY_SESSION.pop(normalized_session_key, None)
        return
    existing = _LAST_CHAT_RUN_BY_SESSION.get(normalized_session_key)
    if not isinstance(existing, dict):
        return
    if str(existing.get("run_id") or "") == str(run_id or ""):
        _LAST_CHAT_RUN_BY_SESSION.pop(normalized_session_key, None)


async def _gateway_call(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    args = [
        *build_openclaw_base_command(getattr(cfg.openclaw, "command", "openclaw")),
        "gateway",
        "call",
        method,
        "--json",
        "--params",
        json.dumps(payload, ensure_ascii=False),
    ]

    gateway_url = getattr(cfg.openclaw, "gateway_url", None)
    if gateway_url:
        args.extend(["--url", str(gateway_url)])

    gateway_token = getattr(cfg.openclaw, "gateway_token", None)
    if gateway_token:
        args.extend(["--token", str(gateway_token)])

    timeout_s = float(getattr(cfg.openclaw, "call_timeout_s", 12.0) or 12.0)
    if timeout_s <= 0:
        timeout_s = 12.0

    def _run_cmd() -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            args,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )

    try:
        proc = await asyncio.to_thread(_run_cmd)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"OpenClaw {method} timeout after {timeout_s:.1f}s") from exc

    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="ignore").strip()
        out = (proc.stdout or b"").decode("utf-8", errors="ignore").strip()
        details = err or out or "unknown error"
        raise RuntimeError(f"OpenClaw {method} failed ({proc.returncode}): {details}")

    try:
        payload_data = json.loads((proc.stdout or b"{}").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OpenClaw {method} returned invalid JSON") from exc
    if not isinstance(payload_data, dict):
        raise RuntimeError(f"OpenClaw {method} returned unexpected payload")
    return payload_data


def _extract_running_session_keys(payload: dict[str, Any]) -> list[str]:
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        return []
    running: list[str] = []
    seen: set[str] = set()
    for item in sessions:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "").strip().lower()
        key = str(item.get("key") or "").strip()
        if status != "running" or not key or key in seen:
            continue
        seen.add(key)
        running.append(key)
    return running


def _choose_session_fallback_keys(requested_session_key: str, running_session_keys: list[str]) -> list[str]:
    requested = str(requested_session_key or "").strip() or "main"
    candidates: list[str] = []
    seen: set[str] = {requested}
    for key in running_session_keys:
        if key == requested:
            continue
        if key.endswith(f":{requested}") and key not in seen:
            seen.add(key)
            candidates.append(key)
    if len(candidates) == 1:
        return candidates
    if len(running_session_keys) == 1:
        only_key = running_session_keys[0]
        if only_key not in seen:
            return [only_key]
    return candidates


async def _abort_via_sessions_abort(session_key: str) -> dict[str, Any]:
    normalized_session_key = str(session_key or "").strip() or "main"
    sessions_abort_payload = await _gateway_call(
        "sessions.abort",
        {"key": normalized_session_key},
    )
    aborted_run_id = str(sessions_abort_payload.get("abortedRunId") or "").strip()
    if aborted_run_id:
        clear_openclaw_chat_run(normalized_session_key, aborted_run_id)
    return {
        **sessions_abort_payload,
        "aborted": bool(aborted_run_id),
        "runIds": [aborted_run_id] if aborted_run_id else [],
        "method": "sessions.abort",
        "resolvedSessionKey": normalized_session_key,
    }


async def _steer_stop_session(session_key: str) -> dict[str, Any]:
    normalized_session_key = str(session_key or "").strip() or "main"
    steer_payload = await _steer_session(normalized_session_key, "/stop")
    interrupted_active = bool(steer_payload.get("interruptedActiveRun"))
    aborted_run_ids = [
        str(run_id or "").strip()
        for run_id in (steer_payload.get("runIds") or [])
        if str(run_id or "").strip()
    ]
    return {
        **steer_payload,
        "aborted": bool(steer_payload.get("aborted")) or interrupted_active or bool(aborted_run_ids),
        "runIds": aborted_run_ids,
        "interruptedActiveRun": interrupted_active,
        "method": "sessions.steer",
        "resolvedSessionKey": normalized_session_key,
    }


async def _steer_session(session_key: str, message: str) -> dict[str, Any]:
    normalized_session_key = str(session_key or "").strip() or "main"
    return await _gateway_call(
        "sessions.steer",
        {
            "key": normalized_session_key,
            "message": message,
        },
    )


def _build_session_candidates(
    requested_session_key: str,
    running_session_keys: list[str],
    fallback_session_keys: list[str],
) -> list[str]:
    normalized_session_key = str(requested_session_key or "").strip() or "main"
    candidates: list[str] = []
    seen_candidates: set[str] = set()
    if normalized_session_key in running_session_keys:
        seen_candidates.add(normalized_session_key)
        candidates.append(normalized_session_key)
    for candidate in fallback_session_keys:
        normalized_candidate = str(candidate or "").strip()
        if not normalized_candidate or normalized_candidate in seen_candidates:
            continue
        seen_candidates.add(normalized_candidate)
        candidates.append(normalized_candidate)
    if len(running_session_keys) == 1:
        only_key = str(running_session_keys[0] or "").strip()
        if only_key and only_key not in seen_candidates:
            seen_candidates.add(only_key)
            candidates.append(only_key)
    return candidates


async def steer_openclaw_chat_session(session_key: str, message: str) -> dict[str, Any]:
    normalized_session_key = str(session_key or "").strip() or "main"
    normalized_message = " ".join(str(message or "").split())
    if not normalized_message:
        return {
            "ok": True,
            "steered": False,
            "reason": "empty",
            "method": "sessions.steer",
            "requestedSessionKey": normalized_session_key,
            "resolvedSessionKey": normalized_session_key,
            "runningSessionKeys": [],
            "fallbackSessionKeys": [],
            "fallbackUsed": False,
        }

    sessions_list_payload = await _gateway_call("sessions.list", {})
    running_session_keys = _extract_running_session_keys(sessions_list_payload)
    fallback_session_keys = _choose_session_fallback_keys(
        normalized_session_key,
        running_session_keys,
    )
    steer_candidates = _build_session_candidates(
        normalized_session_key,
        running_session_keys,
        fallback_session_keys,
    )

    for steer_session_key in steer_candidates:
        steer_payload = await _steer_session(steer_session_key, normalized_message)
        steered = bool(steer_payload.get("ok", True))
        return {
            **steer_payload,
            "ok": bool(steer_payload.get("ok", True)),
            "steered": steered,
            "method": "sessions.steer",
            "requestedSessionKey": normalized_session_key,
            "resolvedSessionKey": steer_session_key,
            "runningSessionKeys": running_session_keys,
            "fallbackSessionKeys": list(fallback_session_keys),
            "fallbackUsed": steer_session_key != normalized_session_key,
        }

    return {
        "ok": True,
        "steered": False,
        "reason": "no_running_session",
        "method": "sessions.steer",
        "requestedSessionKey": normalized_session_key,
        "resolvedSessionKey": normalized_session_key,
        "runningSessionKeys": running_session_keys,
        "fallbackSessionKeys": list(fallback_session_keys),
        "fallbackUsed": False,
    }


async def abort_openclaw_chat_session(session_key: str) -> dict[str, Any]:
    normalized_session_key = str(session_key or "").strip() or "main"
    remembered_run = get_openclaw_chat_run(normalized_session_key)
    run_id = str((remembered_run or {}).get("run_id") or "").strip()
    aborted_run_ids: list[str] = []
    aborted_run_id_set: set[str] = set()
    fallback_used = False
    steer_used = False
    interrupted_active = False
    resolved_session_key = normalized_session_key
    running_session_keys: list[str] = []
    fallback_session_keys: list[str] = []
    abort_method = "sessions.abort"

    # Prefer a narrow stop first, then clear the OpenClaw session queue,
    # and only then fall back to plain session abort.

    def _record_success(
        result: dict[str, Any],
        *,
        resolved_key: str,
        method_name: str,
        used_fallback: bool = False,
        used_steer: bool = False,
    ) -> None:
        nonlocal abort_method, fallback_used, interrupted_active, resolved_session_key, steer_used
        abort_method = method_name
        fallback_used = fallback_used or used_fallback
        steer_used = steer_used or used_steer
        interrupted_active = interrupted_active or bool(result.get("interruptedActiveRun"))
        resolved_session_key = resolved_key
        for raw_run_id in result.get("runIds") or []:
            normalized_run_id = str(raw_run_id or "").strip()
            if not normalized_run_id or normalized_run_id in aborted_run_id_set:
                continue
            aborted_run_id_set.add(normalized_run_id)
            aborted_run_ids.append(normalized_run_id)
            clear_openclaw_chat_run(normalized_session_key, normalized_run_id)
        clear_openclaw_chat_run(normalized_session_key)
        if resolved_key != normalized_session_key:
            clear_openclaw_chat_run(resolved_key)

    if run_id:
        chat_abort_payload = await _gateway_call(
            "chat.abort",
            {"sessionKey": normalized_session_key, "runId": run_id},
        )
        chat_abort_payload.setdefault("method", "chat.abort")
        if bool(chat_abort_payload.get("aborted")):
            _record_success(
                chat_abort_payload,
                resolved_key=normalized_session_key,
                method_name="chat.abort",
            )
        else:
            log.debug(
                "OpenClaw chat.abort did not match active run for session=%s run_id=%s; "
                "falling back to session-level stop",
                normalized_session_key,
                run_id,
            )

    try:
        sessions_list_payload = await _gateway_call("sessions.list", {})
    except Exception:
        log.exception(
            "OpenClaw sessions.list probe failed for requested session=%s",
            normalized_session_key,
        )
        sessions_list_payload = {}

    running_session_keys = _extract_running_session_keys(sessions_list_payload)
    fallback_session_keys = _choose_session_fallback_keys(
        normalized_session_key,
        running_session_keys,
    )

    steer_candidates = _build_session_candidates(
        normalized_session_key,
        running_session_keys,
        fallback_session_keys,
    )

    for steer_session_key in steer_candidates:
        try:
            steer_result = await _steer_stop_session(steer_session_key)
        except Exception:
            log.exception(
                "OpenClaw sessions.steer stop probe failed for requested session=%s target=%s",
                normalized_session_key,
                steer_session_key,
            )
            continue
        if bool(steer_result.get("aborted")):
            _record_success(
                steer_result,
                resolved_key=str(steer_result.get("resolvedSessionKey") or steer_session_key),
                method_name="sessions.steer",
                used_fallback=steer_session_key != normalized_session_key,
                used_steer=True,
            )
            break

    should_try_abort_backup = not steer_used and (bool(steer_candidates) or not aborted_run_ids)
    if should_try_abort_backup:
        abort_candidates: list[str] = []
        seen_abort_candidates: set[str] = set()
        for candidate in [*steer_candidates, normalized_session_key]:
            normalized_candidate = str(candidate or "").strip()
            if not normalized_candidate or normalized_candidate in seen_abort_candidates:
                continue
            seen_abort_candidates.add(normalized_candidate)
            abort_candidates.append(normalized_candidate)

        for abort_session_key in abort_candidates:
            abort_result = await _abort_via_sessions_abort(abort_session_key)
            if bool(abort_result.get("aborted")):
                _record_success(
                    abort_result,
                    resolved_key=str(abort_result.get("resolvedSessionKey") or abort_session_key),
                    method_name="sessions.abort",
                    used_fallback=abort_session_key != normalized_session_key,
                )
                break

    return {
        "ok": True,
        "aborted": bool(aborted_run_ids) or interrupted_active,
        "runIds": list(aborted_run_ids),
        "method": abort_method,
        "requestedSessionKey": normalized_session_key,
        "attempt": 1,
        "runningSessionKeys": running_session_keys,
        "fallbackSessionKeys": list(fallback_session_keys),
        "fallbackUsed": fallback_used,
        "steerUsed": steer_used,
        "interruptedActiveRun": interrupted_active,
        "resolvedSessionKey": resolved_session_key,
    }


__all__ = [
    "abort_openclaw_chat_session",
    "build_openclaw_base_command",
    "clear_openclaw_chat_run",
    "get_openclaw_chat_run",
    "remember_openclaw_chat_run",
    "steer_openclaw_chat_session",
]
