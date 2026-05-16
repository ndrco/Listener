from __future__ import annotations

import asyncio
import json
import locale
import logging
import platform
import uuid
from typing import Any, AsyncIterator

try:  # pragma: no cover - import availability depends on installed extras
    import websockets
    from websockets.legacy.client import WebSocketClientProtocol
except Exception:  # pragma: no cover - handled at connect time
    websockets = None  # type: ignore[assignment]
    WebSocketClientProtocol = Any  # type: ignore[misc, assignment]

from . import __version__
from .config import GatewayConfig

log = logging.getLogger(__name__)
PROTOCOL_VERSION = 3


class GatewayError(RuntimeError):
    pass


class GatewayClient:
    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self._ws: WebSocketClientProtocol | None = None
        self._reader: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._events: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def connect(self) -> None:
        if websockets is None:
            raise GatewayError("websockets is not installed; install Listener requirements")
        self._ws = await websockets.connect(
            self.config.url,
            open_timeout=self.config.connect_timeout_s,
            max_size=26 * 1024 * 1024,
            user_agent_header=f"speaker/{__version__}",
        )
        await self._consume_optional_challenge()
        await self._send_connect()
        self._reader = asyncio.create_task(self._reader_loop())

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
            try:
                await self._reader
            except asyncio.CancelledError:
                pass
            self._reader = None
        if self._ws:
            await self._ws.close()
            self._ws = None
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()

    async def request(self, method: str, params: dict[str, Any] | None = None, timeout_s: float = 10.0) -> dict:
        if self._ws is None:
            raise GatewayError("Gateway is not connected")
        request_id = f"speaker-{uuid.uuid4().hex}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future
        frame = {
            "type": "req",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        await self._ws.send(json.dumps(frame, ensure_ascii=False))
        try:
            return await asyncio.wait_for(future, timeout=timeout_s)
        finally:
            self._pending.pop(request_id, None)

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def _consume_optional_challenge(self) -> None:
        assert self._ws is not None
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            return
        frame = _decode_frame(raw)
        if frame.get("type") == "event" and frame.get("event") == "connect.challenge":
            return
        await self._events.put(frame)

    async def _send_connect(self) -> None:
        assert self._ws is not None
        request_id = f"speaker-connect-{uuid.uuid4().hex}"
        auth: dict[str, str] = {}
        if self.config.token:
            auth["token"] = self.config.token
        frame = {
            "type": "req",
            "id": request_id,
            "method": "connect",
            "params": {
                "minProtocol": PROTOCOL_VERSION,
                "maxProtocol": PROTOCOL_VERSION,
                "client": {
                    "id": "gateway-client",
                    "displayName": "Speaker",
                    "version": __version__,
                    "platform": platform.system().lower() or "linux",
                    "mode": "backend",
                },
                "role": "operator",
                "scopes": ["operator.read"],
                "caps": [],
                "commands": [],
                "permissions": {},
                "auth": auth,
                "locale": locale.getlocale()[0] or "en-US",
                "userAgent": f"speaker/{__version__}",
            },
        }
        await self._ws.send(json.dumps(frame, ensure_ascii=False))

        while True:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=self.config.connect_timeout_s)
            response = _decode_frame(raw)
            if response.get("type") != "res" or response.get("id") != request_id:
                await self._events.put(response)
                continue
            if response.get("ok") is True:
                return
            raise GatewayError(_format_error(response.get("error")))

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            try:
                async for raw in self._ws:
                    frame = _decode_frame(raw)
                    if frame.get("type") == "res":
                        request_id = str(frame.get("id") or "")
                        future = self._pending.get(request_id)
                        if future and not future.done():
                            if frame.get("ok") is True:
                                payload = frame.get("payload")
                                future.set_result(payload if isinstance(payload, dict) else {})
                            else:
                                future.set_exception(GatewayError(_format_error(frame.get("error"))))
                        continue
                    if frame.get("type") == "event":
                        await self._events.put(frame)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                for future in self._pending.values():
                    if not future.done():
                        future.set_exception(GatewayError(str(exc)))
                log.debug("Gateway reader stopped: %s", exc)
        finally:
            await self._events.put(None)


def _decode_frame(raw: str | bytes) -> dict[str, Any]:
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        frame = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GatewayError("Gateway sent invalid JSON") from exc
    if not isinstance(frame, dict):
        raise GatewayError("Gateway sent non-object frame")
    return frame


def _format_error(error: Any) -> str:
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or "Gateway request failed"
        return str(message)
    if error:
        return str(error)
    return "Gateway request failed"
