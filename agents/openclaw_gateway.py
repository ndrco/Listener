"""Shared helpers for calling OpenClaw Gateway RPC methods."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
from typing import Any

from core.config import cfg


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


async def abort_openclaw_chat_session(session_key: str) -> dict[str, Any]:
    payload = {"sessionKey": session_key}
    args = [
        *build_openclaw_base_command(getattr(cfg.openclaw, "command", "openclaw")),
        "gateway",
        "call",
        "chat.abort",
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
        raise TimeoutError(f"OpenClaw chat.abort timeout after {timeout_s:.1f}s") from exc

    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="ignore").strip()
        out = (proc.stdout or b"").decode("utf-8", errors="ignore").strip()
        details = err or out or "unknown error"
        raise RuntimeError(f"OpenClaw chat.abort failed ({proc.returncode}): {details}")

    try:
        payload_data = json.loads((proc.stdout or b"{}").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("OpenClaw chat.abort returned invalid JSON") from exc
    if not isinstance(payload_data, dict):
        raise RuntimeError("OpenClaw chat.abort returned unexpected payload")
    return payload_data


__all__ = ["abort_openclaw_chat_session", "build_openclaw_base_command"]
