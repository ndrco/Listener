"""Local HTTP control API for Listener runtime operations."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
from typing import Any
from urllib.parse import urlsplit

from agents.speech_gate_agent import SpeechGateAgent
from core.config import cfg

log = logging.getLogger(__name__)


class ControlAgent:
    """Small stdlib-only HTTP API bound to the local Listener process."""

    def __init__(
        self,
        *,
        speech_gate: SpeechGateAgent | None = None,
        speaker: Any | None = None,
    ) -> None:
        self._speech_gate = speech_gate
        self._speaker = speaker
        self._server: asyncio.AbstractServer | None = None
        self._host = ""
        self._port = 0
        self._token: str | None = None
        self._max_ttl_seconds = 86400.0

    async def start(self) -> None:
        if self._server is not None:
            return
        control_cfg = getattr(cfg, "control", object())
        if not bool(getattr(control_cfg, "enabled", True)):
            log.info("ControlAgent: disabled")
            return

        self._host = str(getattr(control_cfg, "host", "127.0.0.1") or "127.0.0.1").strip()
        try:
            self._port = int(getattr(control_cfg, "port", 18790))
        except (TypeError, ValueError):
            self._port = 18790
        token = getattr(control_cfg, "token", None)
        self._token = str(token).strip() if token not in (None, "") else None
        self._max_ttl_seconds = float(
            getattr(control_cfg, "max_ttl_seconds", 86400.0) or 86400.0
        )

        if not self._is_loopback_host(self._host) and not self._token:
            raise RuntimeError(
                "ControlAgent refuses non-loopback host without control.token"
            )

        self._server = await asyncio.start_server(
            self._handle_client,
            host=self._host,
            port=self._port,
        )
        sockets = self._server.sockets or []
        if sockets:
            sock_host, sock_port = sockets[0].getsockname()[:2]
            self._host = str(sock_host)
            self._port = int(sock_port)
        log.info("ControlAgent: started http://%s:%s", self._host, self._port)

    async def close(self) -> None:
        server = self._server
        self._server = None
        if not server:
            return
        server.close()
        await server.wait_closed()
        log.info("ControlAgent: stopped")

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            try:
                request = await self._read_request(reader)
            except (
                asyncio.TimeoutError,
                asyncio.IncompleteReadError,
                asyncio.LimitOverrunError,
                ConnectionError,
                OSError,
            ):
                return
            if request is None:
                await self._send_json(writer, 400, {"ok": False, "error": "bad_request"})
                return
            method, path, headers, body = request
            if not self._authorised(headers):
                await self._send_json(writer, 401, {"ok": False, "error": "unauthorized"})
                return

            if method == "GET" and path == "/":
                await self._handle_root(writer)
            elif method == "GET" and path == "/health":
                await self._send_json(writer, 200, {"ok": True, "service": "listener-control"})
            elif method == "GET" and path == "/speech-gate/status":
                await self._handle_status(writer)
            elif method == "POST" and path == "/speech-gate/mode":
                await self._handle_set_mode(writer, body)
            elif method == "GET" and path == "/speaker/status":
                await self._handle_speaker_status(writer)
            elif method == "POST" and path == "/speaker/enabled":
                await self._handle_speaker_enabled(writer, body)
            else:
                await self._send_json(writer, 404, {"ok": False, "error": "not_found"})
        except Exception:
            log.exception("ControlAgent: request failed")
            with contextlib.suppress(ConnectionError, BrokenPipeError, OSError):
                await self._send_json(writer, 500, {"ok": False, "error": "internal_error"})
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, BrokenPipeError, OSError):
                await writer.wait_closed()

    async def _handle_root(self, writer: asyncio.StreamWriter) -> None:
        await self._send_json(
            writer,
            200,
            {
                "ok": True,
                "service": "listener-control",
                "endpoints": {
                    "health": "GET /health",
                    "speech_gate_status": "GET /speech-gate/status",
                    "speech_gate_mode": "POST /speech-gate/mode",
                    "speaker_status": "GET /speaker/status",
                    "speaker_enabled": "POST /speaker/enabled",
                },
            },
        )

    async def _handle_status(self, writer: asyncio.StreamWriter) -> None:
        if self._speech_gate is None:
            await self._send_json(
                writer,
                503,
                {"ok": False, "error": "speech_gate_unavailable"},
            )
            return
        await self._send_json(writer, 200, {"ok": True, "speech_gate": self._speech_gate.get_status()})

    async def _handle_set_mode(self, writer: asyncio.StreamWriter, body: bytes) -> None:
        if self._speech_gate is None:
            await self._send_json(
                writer,
                503,
                {"ok": False, "error": "speech_gate_unavailable"},
            )
            return
        try:
            payload = json.loads(body.decode("utf-8") if body else "{}")
        except json.JSONDecodeError:
            await self._send_json(writer, 400, {"ok": False, "error": "invalid_json"})
            return
        if not isinstance(payload, dict):
            await self._send_json(writer, 400, {"ok": False, "error": "invalid_payload"})
            return

        ttl_seconds = payload.get("ttl_seconds")
        try:
            if ttl_seconds not in (None, ""):
                ttl_seconds = float(ttl_seconds)
                if ttl_seconds <= 0:
                    ttl_seconds = None
                elif ttl_seconds > self._max_ttl_seconds:
                    await self._send_json(
                        writer,
                        400,
                        {
                            "ok": False,
                            "error": "ttl_too_large",
                            "max_ttl_seconds": self._max_ttl_seconds,
                        },
                    )
                    return
        except (TypeError, ValueError):
            await self._send_json(writer, 400, {"ok": False, "error": "invalid_ttl"})
            return

        try:
            status = await self._speech_gate.set_mode(
                payload.get("mode"),
                ttl_seconds=ttl_seconds,
                source=str(payload.get("source") or "api"),
                reason=str(payload.get("reason") or ""),
            )
        except ValueError as exc:
            await self._send_json(writer, 400, {"ok": False, "error": str(exc)})
            return

        await self._send_json(writer, 200, {"ok": True, "speech_gate": status})

    async def _handle_speaker_status(self, writer: asyncio.StreamWriter) -> None:
        if self._speaker is None:
            await self._send_json(
                writer,
                503,
                {"ok": False, "error": "speaker_unavailable"},
            )
            return
        await self._send_json(writer, 200, {"ok": True, "speaker": self._speaker.get_status()})

    async def _handle_speaker_enabled(
        self,
        writer: asyncio.StreamWriter,
        body: bytes,
    ) -> None:
        if self._speaker is None:
            await self._send_json(
                writer,
                503,
                {"ok": False, "error": "speaker_unavailable"},
            )
            return
        try:
            payload = json.loads(body.decode("utf-8") if body else "{}")
        except json.JSONDecodeError:
            await self._send_json(writer, 400, {"ok": False, "error": "invalid_json"})
            return
        if not isinstance(payload, dict):
            await self._send_json(writer, 400, {"ok": False, "error": "invalid_payload"})
            return
        enabled = self._parse_bool(payload.get("enabled"))
        if enabled is None:
            await self._send_json(writer, 400, {"ok": False, "error": "invalid_enabled"})
            return
        status = await self._speaker.set_enabled(
            enabled,
            source=str(payload.get("source") or "api"),
            reason=str(payload.get("reason") or ""),
        )
        await self._send_json(writer, 200, {"ok": True, "speaker": status})

    async def _read_request(
        self, reader: asyncio.StreamReader
    ) -> tuple[str, str, dict[str, str], bytes] | None:
        header_bytes = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5.0)
        if len(header_bytes) > 65536:
            return None
        try:
            header_text = header_bytes.decode("iso-8859-1")
        except UnicodeDecodeError:
            return None
        lines = header_text.split("\r\n")
        request_line = lines[0]
        try:
            method, raw_target, _version = request_line.split(" ", 2)
        except ValueError:
            return None
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if not line or ":" not in line:
                continue
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()

        content_length = 0
        if "content-length" in headers:
            try:
                content_length = int(headers["content-length"])
            except ValueError:
                return None
        if content_length < 0 or content_length > 65536:
            return None
        body = await reader.readexactly(content_length) if content_length else b""
        return method.upper(), urlsplit(raw_target).path, headers, body

    def _authorised(self, headers: dict[str, str]) -> bool:
        if not self._token:
            return True
        auth = headers.get("authorization", "")
        if auth == f"Bearer {self._token}":
            return True
        return headers.get("x-listener-control-token") == self._token

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        text = str(host or "").strip().lower()
        if text == "localhost":
            return True
        try:
            return ipaddress.ip_address(text).is_loopback
        except ValueError:
            return False

    @staticmethod
    def _parse_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on", "enabled"}:
                return True
            if lowered in {"0", "false", "no", "off", "disabled"}:
                return False
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        return None

    @staticmethod
    async def _send_json(
        writer: asyncio.StreamWriter,
        status: int,
        payload: dict[str, Any],
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        reason = {
            200: "OK",
            400: "Bad Request",
            401: "Unauthorized",
            404: "Not Found",
            413: "Payload Too Large",
            500: "Internal Server Error",
            503: "Service Unavailable",
        }.get(status, "OK")
        writer.write(
            (
                f"HTTP/1.1 {status} {reason}\r\n"
                "Content-Type: application/json; charset=utf-8\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n"
                "\r\n"
            ).encode("ascii")
            + body
        )
        await writer.drain()

__all__ = ["ControlAgent"]
