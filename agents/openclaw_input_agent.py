"""Forward accepted voice phrases into OpenClaw chat via CLI gateway calls."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import subprocess
import uuid
from typing import Any

from core.bus import Event, EventBus, bus as default_bus
from core.config import cfg

log = logging.getLogger(__name__)


class OpenClawInputAgent:
    """Consumes speech phrases and sends them to OpenClaw `chat.send`."""

    def __init__(self, *, bus: EventBus | None = None) -> None:
        self._bus = bus or default_bus
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._paused = False
        self._topic = ""
        self._command_missing_logged = False

    async def start(self) -> None:
        if self._running:
            return
        if not getattr(cfg.openclaw, "enabled", False):
            log.info("OpenClawInputAgent: disabled")
            return

        topic = str(getattr(cfg.openclaw, "source_topic", "") or "").strip()
        if not topic:
            topic = cfg.events.llm.input_text

        self._topic = topic
        self._queue = asyncio.Queue()
        self._running = True
        self._paused = False
        self._bus.subscribe(topic, self._on_phrase)
        self._task = asyncio.create_task(self._worker(), name="OpenClawInputAgent.worker")
        log.info("OpenClawInputAgent: started (topic=%s)", topic)

    async def pause(self) -> None:
        if not self._running or self._paused:
            return
        self._paused = True
        await self._drain_queue()
        log.info("OpenClawInputAgent: paused")

    async def resume(self) -> None:
        if not self._running or not self._paused:
            return
        self._paused = False
        log.info("OpenClawInputAgent: resumed")

    async def close(self) -> None:
        if not self._running:
            return
        self._running = False
        self._paused = False

        if self._topic:
            self._bus.unsubscribe(self._topic, self._on_phrase)
        self._topic = ""

        await self._queue.put(None)
        task = self._task
        self._task = None
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        await self._drain_queue()
        log.info("OpenClawInputAgent: stopped")

    async def _on_phrase(self, event: Event) -> None:
        if not self._running or self._paused:
            return

        payload = dict(event.payload or {})
        message = self._build_message(payload)
        if not message:
            return

        params: dict[str, Any] = {
            "message": message,
            "idempotencyKey": uuid.uuid4().hex,
        }
        session_key = str(getattr(cfg.openclaw, "session_key", "") or "").strip()
        if session_key:
            params["sessionKey"] = session_key

        await self._queue.put(params)

    async def _worker(self) -> None:
        while self._running:
            item = await self._queue.get()
            if item is None:
                break
            if self._paused:
                continue
            try:
                await self._send_chat_send(item)
            except Exception:
                log.exception("OpenClawInputAgent: failed to send chat message")

    async def _send_chat_send(self, params: dict[str, Any]) -> None:
        base_cmd = self._build_base_command(getattr(cfg.openclaw, "command", "openclaw"))

        args = [
            *base_cmd,
            "gateway",
            "call",
            "chat.send",
            "--params",
            json.dumps(params, ensure_ascii=False),
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
        except FileNotFoundError:
            if not self._command_missing_logged:
                log.error(
                    "OpenClawInputAgent: command not found: %s. "
                    "Set openclaw.command in config.",
                    base_cmd[0] if base_cmd else "openclaw",
                )
                self._command_missing_logged = True
            return
        except subprocess.TimeoutExpired:
            log.warning("OpenClawInputAgent: chat.send timeout after %.1fs", timeout_s)
            return

        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", errors="ignore").strip()
            out = (proc.stdout or b"").decode("utf-8", errors="ignore").strip()
            details = err or out or "unknown error"
            log.warning("OpenClawInputAgent: chat.send failed (%s): %s", proc.returncode, details)
            return

        if cfg.debug:
            out = (proc.stdout or b"").decode("utf-8", errors="ignore").strip()
            if out:
                log.info("OpenClawInputAgent: chat.send ok: %s", out)

    @staticmethod
    def _build_base_command(raw: Any) -> list[str]:
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
        contains_openclaw = any(
            OpenClawInputAgent._token_is_openclaw_binary(token) for token in tokens
        )
        if first in {"wsl", "wsl.exe"} and not contains_openclaw:
            tokens.append("openclaw")
        return tokens

    @staticmethod
    def _token_is_openclaw_binary(token: str) -> bool:
        cleaned = str(token or "").strip().strip("\"'").strip()
        if not cleaned:
            return False
        if cleaned.startswith("-"):
            return False
        name = os.path.basename(cleaned).lower()
        return name in {"openclaw", "openclaw.exe"}

    def _build_message(self, payload: dict[str, Any]) -> str:
        raw_text = payload.get("text")
        text = " ".join(str(raw_text or "").split())
        return text

    async def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break


__all__ = ["OpenClawInputAgent"]
