import asyncio
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from agents.openclaw_gateway import (  # noqa: E402
    abort_openclaw_chat_session,
    clear_openclaw_chat_run,
    get_openclaw_chat_run,
    remember_openclaw_chat_run,
)
from agents.openclaw_input_agent import OpenClawInputAgent  # noqa: E402
from core.config import cfg  # noqa: E402


def test_abort_openclaw_chat_session_uses_remembered_run_id_first(monkeypatch):
    async def _runner() -> None:
        calls: list[tuple[str, dict]] = []

        async def _fake_gateway_call(method: str, payload: dict):
            calls.append((method, dict(payload)))
            if method == "chat.abort":
                return {"ok": True, "aborted": True, "runIds": ["run-1"]}
            if method == "sessions.list":
                return {"sessions": []}
            raise AssertionError(f"unexpected call {method} {payload}")

        remember_openclaw_chat_run("main", "run-1")
        monkeypatch.setattr("agents.openclaw_gateway._gateway_call", _fake_gateway_call)
        try:
            result = await abort_openclaw_chat_session("main")
        finally:
            clear_openclaw_chat_run("main")

        assert calls == [
            ("chat.abort", {"sessionKey": "main", "runId": "run-1"}),
            ("sessions.list", {}),
        ]
        assert result["aborted"] is True
        assert result["runIds"] == ["run-1"]
        assert result["method"] == "chat.abort"
        assert result["resolvedSessionKey"] == "main"
        assert result["fallbackUsed"] is False
        assert result["steerUsed"] is False
        assert get_openclaw_chat_run("main") is None

    asyncio.run(_runner())


def test_abort_openclaw_chat_session_clears_queue_after_run_id_abort(monkeypatch):
    async def _runner() -> None:
        calls: list[tuple[str, dict]] = []

        async def _fake_gateway_call(method: str, payload: dict):
            calls.append((method, dict(payload)))
            if method == "chat.abort":
                return {"ok": True, "aborted": True, "runIds": ["run-5"]}
            if method == "sessions.list":
                return {"sessions": [{"key": "agent:main:main", "status": "running"}]}
            if method == "sessions.steer":
                return {
                    "ok": True,
                    "aborted": False,
                    "runIds": [],
                    "interruptedActiveRun": True,
                }
            raise AssertionError(f"unexpected call {method} {payload}")

        remember_openclaw_chat_run("main", "run-5")
        monkeypatch.setattr("agents.openclaw_gateway._gateway_call", _fake_gateway_call)
        try:
            result = await abort_openclaw_chat_session("main")
        finally:
            clear_openclaw_chat_run("main")

        assert calls == [
            ("chat.abort", {"sessionKey": "main", "runId": "run-5"}),
            ("sessions.list", {}),
            ("sessions.steer", {"key": "agent:main:main", "message": "/stop"}),
        ]
        assert result["aborted"] is True
        assert result["runIds"] == ["run-5"]
        assert result["method"] == "sessions.steer"
        assert result["resolvedSessionKey"] == "agent:main:main"
        assert result["fallbackUsed"] is True
        assert result["steerUsed"] is True
        assert result["interruptedActiveRun"] is True
        assert get_openclaw_chat_run("main") is None

    asyncio.run(_runner())


def test_abort_openclaw_chat_session_uses_abort_backup_if_queue_stop_does_not_land(monkeypatch):
    async def _runner() -> None:
        calls: list[tuple[str, dict]] = []

        async def _fake_gateway_call(method: str, payload: dict):
            calls.append((method, dict(payload)))
            if method == "chat.abort":
                return {"ok": True, "aborted": True, "runIds": ["run-6"]}
            if method == "sessions.list":
                return {"sessions": [{"key": "agent:main:main", "status": "running"}]}
            if method == "sessions.steer":
                return {
                    "ok": True,
                    "aborted": False,
                    "runIds": [],
                    "interruptedActiveRun": False,
                }
            if method == "sessions.abort":
                return {"ok": True, "abortedRunId": "run-7", "status": "aborted"}
            raise AssertionError(f"unexpected call {method} {payload}")

        remember_openclaw_chat_run("main", "run-6")
        monkeypatch.setattr("agents.openclaw_gateway._gateway_call", _fake_gateway_call)
        try:
            result = await abort_openclaw_chat_session("main")
        finally:
            clear_openclaw_chat_run("main")

        assert calls == [
            ("chat.abort", {"sessionKey": "main", "runId": "run-6"}),
            ("sessions.list", {}),
            ("sessions.steer", {"key": "agent:main:main", "message": "/stop"}),
            ("sessions.abort", {"key": "agent:main:main"}),
        ]
        assert result["aborted"] is True
        assert result["runIds"] == ["run-6", "run-7"]
        assert result["method"] == "sessions.abort"
        assert result["resolvedSessionKey"] == "agent:main:main"
        assert result["fallbackUsed"] is True
        assert result["steerUsed"] is False
        assert result["interruptedActiveRun"] is False

    asyncio.run(_runner())


def test_abort_openclaw_chat_session_falls_back_to_sessions_abort(monkeypatch):
    async def _runner() -> None:
        calls: list[tuple[str, dict]] = []

        async def _fake_gateway_call(method: str, payload: dict):
            calls.append((method, dict(payload)))
            if method == "chat.abort":
                return {"ok": True, "aborted": False, "runIds": []}
            if method == "sessions.list":
                return {"sessions": []}
            if method == "sessions.abort":
                return {"ok": True, "abortedRunId": "run-2", "status": "aborted"}
            raise AssertionError(f"unexpected call {method} {payload}")

        remember_openclaw_chat_run("main", "run-1")
        monkeypatch.setattr("agents.openclaw_gateway._gateway_call", _fake_gateway_call)
        try:
            result = await abort_openclaw_chat_session("main")
        finally:
            clear_openclaw_chat_run("main")

        assert calls == [
            ("chat.abort", {"sessionKey": "main", "runId": "run-1"}),
            ("sessions.list", {}),
            ("sessions.abort", {"key": "main"}),
        ]
        assert result["aborted"] is True
        assert result["runIds"] == ["run-2"]
        assert result["method"] == "sessions.abort"
        assert result["resolvedSessionKey"] == "main"
        assert result["fallbackUsed"] is False
        assert result["steerUsed"] is False
        assert get_openclaw_chat_run("main") is None

    asyncio.run(_runner())


def test_abort_openclaw_chat_session_uses_sessions_abort_without_run_id(monkeypatch):
    async def _runner() -> None:
        calls: list[tuple[str, dict]] = []

        async def _fake_gateway_call(method: str, payload: dict):
            calls.append((method, dict(payload)))
            if method == "sessions.list":
                return {"sessions": []}
            if method == "sessions.abort":
                return {"ok": True, "abortedRunId": None, "status": "no-active-run"}
            raise AssertionError(f"unexpected call {method} {payload}")

        clear_openclaw_chat_run("main")
        monkeypatch.setattr("agents.openclaw_gateway._gateway_call", _fake_gateway_call)
        result = await abort_openclaw_chat_session("main")

        assert calls == [
            ("sessions.list", {}),
            ("sessions.abort", {"key": "main"}),
        ]
        assert result["aborted"] is False
        assert result["runIds"] == []
        assert result["method"] == "sessions.abort"
        assert result["resolvedSessionKey"] == "main"
        assert result["runningSessionKeys"] == []
        assert result["fallbackSessionKeys"] == []
        assert result["attempt"] == 1

    asyncio.run(_runner())


def test_abort_openclaw_chat_session_uses_sessions_steer_for_running_session(
    monkeypatch,
):
    async def _runner() -> None:
        calls: list[tuple[str, dict]] = []

        async def _fake_gateway_call(method: str, payload: dict):
            calls.append((method, dict(payload)))
            if method == "sessions.list":
                return {
                    "sessions": [
                        {"key": "agent:main:main", "status": "running"},
                        {"key": "agent:main:telegram:direct:268979884", "status": "done"},
                    ]
                }
            if method == "sessions.steer" and payload == {
                "key": "agent:main:main",
                "message": "/stop",
            }:
                return {
                    "ok": True,
                    "aborted": False,
                    "runIds": [],
                    "interruptedActiveRun": True,
                }
            raise AssertionError(f"unexpected call {method} {payload}")

        clear_openclaw_chat_run("main")
        monkeypatch.setattr("agents.openclaw_gateway._gateway_call", _fake_gateway_call)
        result = await abort_openclaw_chat_session("main")

        assert calls == [
            ("sessions.list", {}),
            ("sessions.steer", {"key": "agent:main:main", "message": "/stop"}),
        ]
        assert result["aborted"] is True
        assert result["runIds"] == []
        assert result["method"] == "sessions.steer"
        assert result["requestedSessionKey"] == "main"
        assert result["resolvedSessionKey"] == "agent:main:main"
        assert result["fallbackUsed"] is True
        assert result["steerUsed"] is True
        assert result["interruptedActiveRun"] is True
        assert result["runningSessionKeys"] == ["agent:main:main"]
        assert result["fallbackSessionKeys"] == ["agent:main:main"]

    asyncio.run(_runner())


def test_abort_openclaw_chat_session_uses_sessions_abort_as_backup_after_steer_probe(
    monkeypatch,
):
    async def _runner() -> None:
        calls: list[tuple[str, dict]] = []

        async def _fake_gateway_call(method: str, payload: dict):
            calls.append((method, dict(payload)))
            if method == "sessions.list":
                return {"sessions": [{"key": "agent:main:main", "status": "running"}]}
            if method == "sessions.steer":
                return {
                    "ok": True,
                    "aborted": False,
                    "runIds": [],
                    "interruptedActiveRun": False,
                }
            if method == "sessions.abort" and payload == {"key": "agent:main:main"}:
                return {"ok": True, "abortedRunId": "run-4", "status": "aborted"}
            if method == "sessions.abort" and payload == {"key": "main"}:
                return {"ok": True, "abortedRunId": None, "status": "no-active-run"}
            raise AssertionError(f"unexpected call {method} {payload}")

        clear_openclaw_chat_run("main")
        monkeypatch.setattr("agents.openclaw_gateway._gateway_call", _fake_gateway_call)
        result = await abort_openclaw_chat_session("main")

        assert calls == [
            ("sessions.list", {}),
            ("sessions.steer", {"key": "agent:main:main", "message": "/stop"}),
            ("sessions.abort", {"key": "agent:main:main"}),
        ]
        assert result["aborted"] is True
        assert result["runIds"] == ["run-4"]
        assert result["method"] == "sessions.abort"
        assert result["resolvedSessionKey"] == "agent:main:main"
        assert result["fallbackUsed"] is True
        assert result["steerUsed"] is False
        assert result["attempt"] == 1

    asyncio.run(_runner())


def test_openclaw_input_agent_remembers_run_id(monkeypatch):
    async def _runner() -> None:
        async def _fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        def _fake_run(*args, **kwargs):
            del args, kwargs
            return subprocess.CompletedProcess(
                ["openclaw", "gateway", "call", "chat.send"],
                0,
                stdout=b'{"runId":"run-voice-1","status":"started"}',
                stderr=b"",
            )

        old_enabled = cfg.openclaw.enabled
        old_command = cfg.openclaw.command
        old_session_key = cfg.openclaw.session_key
        old_debug = cfg.debug
        cfg.openclaw.enabled = True
        cfg.openclaw.command = "openclaw"
        cfg.openclaw.session_key = "voice-main"
        cfg.debug = False
        monkeypatch.setattr("agents.openclaw_input_agent.asyncio.to_thread", _fake_to_thread)
        monkeypatch.setattr("agents.openclaw_input_agent.subprocess.run", _fake_run)
        clear_openclaw_chat_run("voice-main")
        try:
            agent = OpenClawInputAgent()
            await agent._send_chat_send(  # pylint: disable=protected-access
                {
                    "message": "hello",
                    "idempotencyKey": "abc",
                    "sessionKey": "voice-main",
                }
            )
        finally:
            cfg.openclaw.enabled = old_enabled
            cfg.openclaw.command = old_command
            cfg.openclaw.session_key = old_session_key
            cfg.debug = old_debug

        run_info = get_openclaw_chat_run("voice-main")
        assert isinstance(run_info, dict)
        assert run_info["run_id"] == "run-voice-1"
        assert isinstance(run_info["remembered_at"], float)
        clear_openclaw_chat_run("voice-main")

    asyncio.run(_runner())


def test_openclaw_input_agent_clear_pending_messages():
    async def _runner() -> None:
        agent = OpenClawInputAgent()
        await agent._queue.put({"message": "one"})  # pylint: disable=protected-access
        await agent._queue.put({"message": "two"})  # pylint: disable=protected-access

        dropped = await agent.clear_pending_messages()

        assert dropped == 2
        assert agent._queue.empty()  # pylint: disable=protected-access

    asyncio.run(_runner())
