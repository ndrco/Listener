from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as main_module  # noqa: E402
from core.bus import Event  # noqa: E402
from core.config import cfg  # noqa: E402


class FakeBus:
    def __init__(self) -> None:
        self.subscriptions: dict[str, list] = {}
        self.stopped = False

    def start(self) -> None:
        return None

    def subscribe(self, pattern: str, handler) -> None:
        self.subscriptions.setdefault(pattern, []).append(handler)

    async def publish(self, topic: str, **payload) -> None:
        for handler in self.subscriptions.get(topic, []):
            await handler(Event(topic=topic, payload=payload))

    async def stop(self) -> None:
        self.stopped = True


class FakeIndicators:
    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


class FakeAudioAgent:
    fail_start = False

    async def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("audio boom")

    async def close(self) -> None:
        return None


class FakeSpeakerAgent:
    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def interrupt(self, *, reason: str) -> int:
        return 0


class FakeOpenClawInputAgent:
    def __init__(self, *, on_barge_in_interrupt) -> None:
        self.on_barge_in_interrupt = on_barge_in_interrupt

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def clear_pending_messages(self) -> int:
        return 0


class FakeSpeechGateAgent:
    def __init__(
        self,
        *,
        on_local_stop,
        on_local_speaker_on=None,
        on_local_speaker_off=None,
    ) -> None:
        self.on_local_stop = on_local_stop
        self.on_local_speaker_on = on_local_speaker_on
        self.on_local_speaker_off = on_local_speaker_off

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


class FakeControlAgent:
    instances: list["FakeControlAgent"] = []

    def __init__(self, *, speech_gate, speaker, status_provider, shutdown_handler) -> None:
        self.status_provider = status_provider
        self.shutdown_handler = shutdown_handler
        self.ready_status = None
        self.closed = False
        self.instances.append(self)

    async def start(self) -> None:
        self.ready_status = self.status_provider()
        await self.shutdown_handler("test")

    async def close(self) -> None:
        self.closed = True


def _patch_runtime(monkeypatch) -> FakeBus:
    fake_bus = FakeBus()
    FakeAudioAgent.fail_start = False
    FakeControlAgent.instances = []
    monkeypatch.setattr(main_module, "load", lambda: None)
    monkeypatch.setattr(main_module, "configure_logging", lambda *, debug, info: None)
    async def _fake_restore_all_ducking():
        return {"restored": False}
    monkeypatch.setattr(main_module, "restore_all_ducking", _fake_restore_all_ducking)
    monkeypatch.setattr(main_module, "bus", fake_bus)
    monkeypatch.setattr(main_module, "indicators", FakeIndicators())
    monkeypatch.setattr(main_module, "AudioAgent", FakeAudioAgent)
    monkeypatch.setattr(main_module, "SpeakerAgent", FakeSpeakerAgent)
    monkeypatch.setattr(main_module, "OpenClawInputAgent", FakeOpenClawInputAgent)
    monkeypatch.setattr(main_module, "SpeechGateAgent", FakeSpeechGateAgent)
    monkeypatch.setattr(main_module, "ControlAgent", FakeControlAgent)
    return fake_bus


def test_main_best_effort_startup_does_not_raise_when_critical_component_fails(monkeypatch):
    async def _runner() -> None:
        old_strict = cfg.service.strict_startup
        old_speaker_enabled = cfg.speaker.enabled
        fake_bus = _patch_runtime(monkeypatch)
        FakeAudioAgent.fail_start = True
        cfg.service.strict_startup = False
        cfg.speaker.enabled = False
        try:
            await main_module.main()
        finally:
            cfg.service.strict_startup = old_strict
            cfg.speaker.enabled = old_speaker_enabled

        assert fake_bus.stopped is True
        assert FakeControlAgent.instances[0].closed is True
        assert FakeControlAgent.instances[0].ready_status["ready"] is False

    asyncio.run(_runner())


def test_main_strict_startup_raises_when_critical_component_fails(monkeypatch):
    async def _runner() -> None:
        old_strict = cfg.service.strict_startup
        old_speaker_enabled = cfg.speaker.enabled
        fake_bus = _patch_runtime(monkeypatch)
        FakeAudioAgent.fail_start = True
        cfg.service.strict_startup = True
        cfg.speaker.enabled = False
        try:
            with pytest.raises(RuntimeError, match="strict startup failed"):
                await main_module.main()
        finally:
            cfg.service.strict_startup = old_strict
            cfg.speaker.enabled = old_speaker_enabled

        assert fake_bus.stopped is True
        assert FakeControlAgent.instances[0].closed is True

    asyncio.run(_runner())
