import asyncio
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from agents.control_agent import ControlAgent  # noqa: E402
from core.config import cfg  # noqa: E402
from llm.speech_gate import SpeechGateMode  # noqa: E402
from utils.listenerctl import request_json  # noqa: E402


class FakeSpeechGate:
    def __init__(self) -> None:
        self.mode = "normal"
        self.calls: list[dict] = []

    def get_status(self) -> dict:
        return {
            "running": True,
            "paused": False,
            "mode": self.mode,
            "temporary": False,
        }

    async def set_mode(
        self,
        mode,
        *,
        ttl_seconds=None,
        source="api",
        reason="",
    ) -> dict:
        parsed = SpeechGateMode.parse(mode)
        if parsed == SpeechGateMode.STANDBY and ttl_seconds is None:
            raise ValueError("standby mode requires ttl_seconds")
        self.mode = parsed.value
        self.calls.append(
            {
                "mode": parsed.value,
                "ttl_seconds": ttl_seconds,
                "source": source,
                "reason": reason,
            }
        )
        return self.get_status()


def _save_control_cfg() -> tuple:
    return (
        cfg.control.enabled,
        cfg.control.host,
        cfg.control.port,
        cfg.control.token,
        cfg.control.max_ttl_seconds,
    )


def _restore_control_cfg(saved: tuple) -> None:
    (
        cfg.control.enabled,
        cfg.control.host,
        cfg.control.port,
        cfg.control.token,
        cfg.control.max_ttl_seconds,
    ) = saved


def test_control_agent_status_and_set_mode_api():
    async def _runner() -> None:
        saved = _save_control_cfg()
        cfg.control.enabled = True
        cfg.control.host = "127.0.0.1"
        cfg.control.port = 0
        cfg.control.token = None
        cfg.control.max_ttl_seconds = 100.0
        speech_gate = FakeSpeechGate()
        agent = ControlAgent(speech_gate=speech_gate)  # type: ignore[arg-type]
        try:
            await agent.start()
            status_code, data = await asyncio.to_thread(
                request_json,
                agent.base_url,
                "/",
            )
            assert status_code == 200
            assert data["ok"] is True
            assert data["endpoints"]["speech_gate_status"] == "GET /speech-gate/status"

            status_code, data = await asyncio.to_thread(
                request_json,
                agent.base_url,
                "/speech-gate/status",
            )
            assert status_code == 200
            assert data["speech_gate"]["mode"] == "normal"

            status_code, data = await asyncio.to_thread(
                request_json,
                agent.base_url,
                "/speech-gate/mode",
                method="POST",
                payload={
                    "mode": "chatty",
                    "ttl_seconds": 42,
                    "source": "test",
                    "reason": "conversation",
                },
            )
            assert status_code == 200
            assert data["speech_gate"]["mode"] == "chatty"
            assert speech_gate.calls[-1] == {
                "mode": "chatty",
                "ttl_seconds": 42.0,
                "source": "test",
                "reason": "conversation",
            }

            status_code, data = await asyncio.to_thread(
                request_json,
                agent.base_url,
                "/speech-gate/mode",
                method="POST",
                payload={"mode": "standby"},
            )
            assert status_code == 400
            assert data["ok"] is False
        finally:
            await agent.close()
            _restore_control_cfg(saved)

    asyncio.run(_runner())


def test_control_agent_rejects_non_loopback_without_token():
    async def _runner() -> None:
        saved = _save_control_cfg()
        cfg.control.enabled = True
        cfg.control.host = "0.0.0.0"
        cfg.control.port = 0
        cfg.control.token = None
        agent = ControlAgent(speech_gate=FakeSpeechGate())  # type: ignore[arg-type]
        try:
            with pytest.raises(RuntimeError):
                await agent.start()
        finally:
            await agent.close()
            _restore_control_cfg(saved)

    asyncio.run(_runner())


def test_control_agent_requires_token_when_configured():
    async def _runner() -> None:
        saved = _save_control_cfg()
        cfg.control.enabled = True
        cfg.control.host = "127.0.0.1"
        cfg.control.port = 0
        cfg.control.token = "secret"
        cfg.control.max_ttl_seconds = 100.0
        agent = ControlAgent(speech_gate=FakeSpeechGate())  # type: ignore[arg-type]
        try:
            await agent.start()
            status_code, data = await asyncio.to_thread(
                request_json,
                agent.base_url,
                "/speech-gate/status",
            )
            assert status_code == 401
            assert data["error"] == "unauthorized"

            status_code, data = await asyncio.to_thread(
                request_json,
                agent.base_url,
                "/speech-gate/status",
                token="secret",
            )
            assert status_code == 200
            assert data["ok"] is True
        finally:
            await agent.close()
            _restore_control_cfg(saved)

    asyncio.run(_runner())
