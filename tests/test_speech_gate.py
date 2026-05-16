import asyncio
import json
import logging
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from core.bus import Event  # noqa: E402
from core.config import cfg  # noqa: E402
import agents.speech_gate_agent as speech_gate_agent_module  # noqa: E402
import llm.speech_gate as speech_gate_module  # noqa: E402
from llm.speech_gate import SpeechDirectionGate, SpeechGateMode  # noqa: E402
from agents.speech_gate_agent import SpeechGateAgent  # noqa: E402


def _build_gate() -> SpeechDirectionGate:
    gate = SpeechDirectionGate.from_config()
    gate.enabled = True
    gate.mode = SpeechGateMode.NORMAL
    gate.rules_threshold = 0.7
    gate.final_threshold = 0.5
    gate.attention_window = 10.0
    gate.attention_extension = 5.0
    gate._ml_threshold = 0.7
    gate._attention_until = 0.0
    gate.patterns = {
        "assistant_names": {"kissa"},
        "command_verbs": {"расскажи"},
        "politeness_markers": {"пожалуйста"},
        "question_markers": {"как"},
        "modal_markers": {"можешь"},
        "continuation_patterns": {"и еще"},
    }
    return gate


def test_speech_gate_allows_assistant_name_rule():
    gate = _build_gate()
    decision = gate.should_allow("Привет, kissa")

    assert decision.allowed is True
    assert decision.reason == "rules"
    assert decision.rule_score == pytest.approx(1.0)
    assert decision.final_score >= gate.final_threshold


def test_speech_gate_attention_window_allows_followup():
    gate = _build_gate()

    first = gate.should_allow("Kissa, расскажи новости")
    second = gate.should_allow("подробности")

    assert first.allowed is True
    assert second.allowed is True
    assert second.reason == "attention"


def test_speech_gate_ml_branch_used_when_rules_not_enough():
    gate = _build_gate()
    gate.patterns["assistant_names"] = set()

    class FakeClassifier:
        @staticmethod
        def predict_directed_prob(text: str) -> float:
            assert isinstance(text, str)
            return 0.9

    gate._load_classifier = lambda: FakeClassifier()  # type: ignore[method-assign]

    decision = gate.should_allow("Расскажи про погоду")

    assert decision.allowed is True
    assert decision.reason == "ml"
    assert decision.rule_score == pytest.approx(0.3)
    assert decision.ml_score == pytest.approx(0.9)
    assert decision.final_score == pytest.approx(0.66, abs=1e-6)


def test_speech_gate_classifier_falls_back_to_cpu_on_cuda_oom_during_load(
    monkeypatch, tmp_path, caplog
):
    model_dir = tmp_path / "gate-model"
    model_dir.mkdir()
    calls: list[str] = []

    class FakeClassifier:
        def __init__(self, model_path: str, device: str = "cpu", *, max_length: int = 64) -> None:
            del model_path, max_length
            calls.append(device)
            self.device = device
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")

    gate = _build_gate()
    gate.root_path = tmp_path
    gate._classifier = None
    gate._classifier_disabled = False

    old_path = cfg.speech_gate.model.path
    old_device = cfg.speech_gate.model.device
    old_max_length = cfg.speech_gate.model.max_length
    cfg.speech_gate.model.path = str(model_dir)
    cfg.speech_gate.model.device = "cuda"
    cfg.speech_gate.model.max_length = 64
    monkeypatch.setattr(speech_gate_module, "DirectedIntentClassifier", FakeClassifier)
    monkeypatch.setattr(speech_gate_module.torch.cuda, "is_available", lambda: True)
    caplog.set_level(logging.WARNING, logger=speech_gate_module.log.name)
    try:
        classifier = gate._load_classifier()  # pylint: disable=protected-access
    finally:
        cfg.speech_gate.model.path = old_path
        cfg.speech_gate.model.device = old_device
        cfg.speech_gate.model.max_length = old_max_length

    assert calls == ["cuda", "cpu"]
    assert classifier.device == "cpu"
    assert gate._classifier_device == "cpu"  # pylint: disable=protected-access
    assert any(
        "classifier CUDA initialisation ran out of memory; retrying on cpu"
        in record.getMessage()
        for record in caplog.records
    )


def test_speech_gate_classifier_falls_back_to_cpu_on_cuda_oom_during_inference(
    monkeypatch, tmp_path, caplog
):
    model_dir = tmp_path / "gate-model"
    model_dir.mkdir()
    calls: list[str] = []

    class FakeClassifier:
        def __init__(self, model_path: str, device: str = "cpu", *, max_length: int = 64) -> None:
            del model_path, max_length
            calls.append(device)
            self.device = device

        def predict_directed_prob(self, text: str) -> float:
            assert isinstance(text, str)
            if self.device == "cuda":
                raise RuntimeError("CUDA out of memory")
            return 0.9

    gate = _build_gate()
    gate.patterns["assistant_names"] = set()
    gate.root_path = tmp_path
    gate._classifier = None
    gate._classifier_disabled = False

    old_path = cfg.speech_gate.model.path
    old_device = cfg.speech_gate.model.device
    old_max_length = cfg.speech_gate.model.max_length
    cfg.speech_gate.model.path = str(model_dir)
    cfg.speech_gate.model.device = "cuda"
    cfg.speech_gate.model.max_length = 64
    monkeypatch.setattr(speech_gate_module, "DirectedIntentClassifier", FakeClassifier)
    monkeypatch.setattr(speech_gate_module.torch.cuda, "is_available", lambda: True)
    caplog.set_level(logging.WARNING, logger=speech_gate_module.log.name)
    try:
        decision = gate.should_allow("Расскажи про погоду")
    finally:
        cfg.speech_gate.model.path = old_path
        cfg.speech_gate.model.device = old_device
        cfg.speech_gate.model.max_length = old_max_length

    assert calls == ["cuda", "cpu"]
    assert decision.allowed is True
    assert decision.reason == "ml"
    assert decision.ml_score == pytest.approx(0.9)
    assert gate._classifier_device == "cpu"  # pylint: disable=protected-access
    assert any(
        "classifier CUDA inference ran out of memory; retrying on cpu"
        in record.getMessage()
        for record in caplog.records
    )
    assert not any(
        "classifier unavailable, using rules-only fallback" in record.getMessage()
        for record in caplog.records
    )


def test_speech_gate_fallback_drops_weak_phrase_without_model():
    gate = _build_gate()
    gate.patterns["assistant_names"] = set()
    gate.patterns["command_verbs"] = set()
    gate.patterns["question_markers"] = set()
    gate.patterns["modal_markers"] = set()
    gate.patterns["politeness_markers"] = set()

    def _raise_no_model():
        raise RuntimeError("no model")

    gate._load_classifier = _raise_no_model  # type: ignore[method-assign]
    decision = gate.should_allow("Просто мысль вслух")

    assert decision.allowed is False
    assert decision.reason == "low_score"
    assert decision.final_score == pytest.approx(0.0)


def test_speech_gate_modes_keep_expected_behavior():
    gate = _build_gate()

    gate.set_mode(SpeechGateMode.CHATTY)
    assert gate.should_allow("просто фраза").allowed is True

    gate.set_mode(SpeechGateMode.MUTE)
    assert gate.should_allow("просто фраза").reason == "mute"
    assert gate.should_allow("Kissa, привет").allowed is True
    assert gate.should_allow("включи").reason == "mute"

    gate.set_mode(SpeechGateMode.STANDBY)
    decision = gate.should_allow("Kissa, привет")
    assert decision.allowed is False
    assert decision.reason == "standby"


def test_speech_gate_mute_requires_leading_assistant_name():
    gate = _build_gate()
    gate.patterns["assistant_names"] = {"марина"}
    gate.set_mode(SpeechGateMode.MUTE)

    assert gate.should_allow("Марина, что там?").allowed is True
    assert gate.should_allow("Я думаю марина потом не будет").allowed is False
    assert gate.should_allow("Да полмарина не будет. Ну, что там?").allowed is False


def test_speech_gate_mode_override_uses_segment_time_mode():
    gate = _build_gate()
    gate.set_mode(SpeechGateMode.NORMAL)

    decision = gate.should_allow("не профильная фраза", mode=SpeechGateMode.CHATTY)

    assert decision.allowed is True
    assert decision.reason == "attention"


def test_speech_gate_loads_assistant_name_from_openclaw_identity(tmp_path):
    identity_path = tmp_path / ".openclaw" / "workspace" / "IDENTITY.md"
    identity_path.parent.mkdir(parents=True)
    identity_path.write_text(
        "# Identity\n\nName: Marina\nИмя: Марина\n",
        encoding="utf-8",
    )

    gate = _build_gate()
    gate.patterns = {}
    gate.root_path = tmp_path

    class DummyCfg:
        patterns_file = None
        identity_file = ".openclaw/workspace/IDENTITY.md"
        assistant_names: list[str] = []
        command_verbs: list[str] = []
        politeness_markers: list[str] = []
        question_markers: list[str] = []
        modal_markers: list[str] = []
        continuation_patterns: list[str] = []

    gate.patterns = gate._load_patterns(DummyCfg())  # pylint: disable=protected-access

    assert gate.patterns["assistant_names"] == {"marina", "марина"}
    decision = gate.should_allow("Марина, как дела?")
    assert decision.allowed is True
    assert decision.reason == "rules"
    assert decision.rule_score == pytest.approx(1.0)


def test_speech_gate_auto_discovers_openclaw_workspace_from_config(tmp_path, monkeypatch):
    home = tmp_path / "home"
    openclaw_config = home / ".openclaw" / "openclaw.json"
    workspace = tmp_path / "openclaw-workspace"
    identity_path = workspace / "IDENTITY.md"
    openclaw_config.parent.mkdir(parents=True)
    workspace.mkdir(parents=True)
    openclaw_config.write_text(
        json.dumps({"agents": {"defaults": {"workspace": str(workspace)}}}),
        encoding="utf-8",
    )
    identity_path.write_text(
        "# IDENTITY.md\n\n- **Имя:** Марина\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("OPENCLAW_IDENTITY_FILE", raising=False)
    monkeypatch.delenv("OPENCLAW_WORKSPACE", raising=False)
    monkeypatch.delenv("OPENCLAW_STATE_DIR", raising=False)
    monkeypatch.delenv("OPENCLAW_CONFIG_PATH", raising=False)

    gate = _build_gate()
    gate.patterns = {}
    gate.root_path = tmp_path

    class DummyCfg:
        patterns_file = None
        identity_file = None
        assistant_names: list[str] = []
        command_verbs: list[str] = []
        politeness_markers: list[str] = []
        question_markers: list[str] = []
        modal_markers: list[str] = []
        continuation_patterns: list[str] = []

    gate.patterns = gate._load_patterns(DummyCfg())  # pylint: disable=protected-access

    assert gate.patterns["assistant_names"] == {"марина"}
    assert gate.should_allow("Марина, как дела?").allowed is True


def test_speech_gate_ignores_assistant_names_from_patterns_file(tmp_path):
    patterns_path = tmp_path / "patterns.json"
    patterns_path.write_text(
        json.dumps(
            {
                "assistant_names": ["legacy"],
                "command_verbs": ["расскажи"],
            }
        ),
        encoding="utf-8",
    )

    gate = _build_gate()
    gate.patterns = {}
    gate.root_path = tmp_path

    class DummyCfg:
        patterns_file = str(patterns_path)
        identity_file = str(tmp_path / "missing_identity.md")
        assistant_names: list[str] = []
        command_verbs: list[str] = []
        politeness_markers: list[str] = []
        question_markers: list[str] = []
        modal_markers: list[str] = []
        continuation_patterns: list[str] = []

    gate.patterns = gate._load_patterns(DummyCfg())  # pylint: disable=protected-access

    assert "assistant_names" not in gate.patterns
    assert gate.patterns["command_verbs"] == {"расскажи"}
    assert gate.should_allow("Привет, legacy").allowed is False


def test_speech_gate_detects_local_commands_without_swallowing_regular_prompt():
    gate = _build_gate()

    mute = gate.detect_local_command("Kissa, пожалуйста, помолчи")
    assert mute is not None
    assert mute.action == "mute"
    assert mute.assistant_name == "kissa"
    assert mute.phrase == "помолчи"

    standby = gate.detect_local_command("Kissa, не слушай")
    assert standby is not None
    assert standby.action == "standby"

    stop = gate.detect_local_command("Kissa, стоп")
    assert stop is not None
    assert stop.action == "stop_generation"

    assert gate.find_leading_assistant_name("Kissa, помолчи") == "kissa"
    assert gate.find_leading_assistant_name("Пожалуйста, Kissa, помолчи") is None
    assert gate.detect_local_command("Пожалуйста, Kissa, помолчи") is None
    assert gate.detect_local_command("Включи Kissa, говори") is None
    assert gate.detect_local_command("Kissa, ответь какая погода") is None
    assert gate.detect_local_command("Kissa, слушай, какая сегодня погода") is None
    assert gate.detect_barge_in_command("Kissa, ответь какая погода") is None
    assert gate.detect_barge_in_command("Kissa, нет, я имел в виду Амстердам").phrase == "нет"  # type: ignore[union-attr]
    assert gate.detect_barge_in_command("Kissa, точнее Амстердам").phrase == "точнее"  # type: ignore[union-attr]
    assert gate.detect_barge_in_command("Пожалуйста, Kissa, нет") is None


def test_speech_gate_agent_loads_identity_once_on_start(monkeypatch):
    async def _runner() -> None:
        class DummyBus:
            def __init__(self) -> None:
                self.subscriptions: dict[str, object] = {}

            def subscribe(self, topic: str, handler):
                self.subscriptions[topic] = handler

            def unsubscribe(self, topic: str, handler):
                current = self.subscriptions.get(topic)
                if current is handler:
                    del self.subscriptions[topic]

            async def publish(self, topic: str, **payload):
                del topic, payload

        real_factory = speech_gate_agent_module.SpeechDirectionGate.from_config
        calls = 0

        def _wrapped_factory():
            nonlocal calls
            calls += 1
            return real_factory()

        old_names = list(cfg.speech_gate.assistant_names)
        old_patterns_file = cfg.speech_gate.patterns_file
        old_identity_file = cfg.speech_gate.identity_file
        cfg.speech_gate.assistant_names = ["kissa"]
        cfg.speech_gate.patterns_file = None
        cfg.speech_gate.identity_file = str(ROOT / "missing_identity.md")
        monkeypatch.setattr(
            speech_gate_agent_module.SpeechDirectionGate,
            "from_config",
            staticmethod(_wrapped_factory),
        )
        try:
            agent = SpeechGateAgent(bus=DummyBus())  # type: ignore[arg-type]
            await agent.start()
            await agent.close()
        finally:
            cfg.speech_gate.assistant_names = old_names
            cfg.speech_gate.patterns_file = old_patterns_file
            cfg.speech_gate.identity_file = old_identity_file

        assert calls == 1

    asyncio.run(_runner())


def test_speech_gate_agent_publishes_filtered_topic():
    async def _runner() -> None:
        class DummyBus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict]] = []
                self.subscriptions: dict[str, object] = {}

            def subscribe(self, topic: str, handler):
                self.subscriptions[topic] = handler

            def unsubscribe(self, topic: str, handler):
                current = self.subscriptions.get(topic)
                if current is handler:
                    del self.subscriptions[topic]

            async def publish(self, topic: str, **payload):
                self.events.append((topic, payload))

        old_names = list(cfg.speech_gate.assistant_names)
        old_patterns_file = cfg.speech_gate.patterns_file
        old_identity_file = cfg.speech_gate.identity_file
        cfg.speech_gate.assistant_names = ["kissa"]
        cfg.speech_gate.patterns_file = None
        cfg.speech_gate.identity_file = str(ROOT / "missing_identity.md")
        try:
            bus = DummyBus()
            agent = SpeechGateAgent(bus=bus)  # type: ignore[arg-type]
            await agent.start()
            try:
                await agent._on_input_text(  # pylint: disable=protected-access
                    Event(topic="llm/input_text", payload={"text": "Привет, kissa"})
                )
            finally:
                await agent.close()
        finally:
            cfg.speech_gate.assistant_names = old_names
            cfg.speech_gate.patterns_file = old_patterns_file
            cfg.speech_gate.identity_file = old_identity_file

        assert any(topic == "llm/accepted_phrase" for topic, _ in bus.events)
        published_payload = [payload for topic, payload in bus.events if topic == "llm/accepted_phrase"][0]
        assert published_payload["text"] == "Привет, kissa"
        assert "speech_gate_reason" in published_payload

    asyncio.run(_runner())


def test_speech_gate_agent_handles_local_voice_commands(monkeypatch):
    async def _runner() -> None:
        class DummyBus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict]] = []
                self.subscriptions: dict[str, object] = {}

            def subscribe(self, topic: str, handler):
                self.subscriptions[topic] = handler

            def unsubscribe(self, topic: str, handler):
                current = self.subscriptions.get(topic)
                if current is handler:
                    del self.subscriptions[topic]

            async def publish(self, topic: str, **payload):
                self.events.append((topic, payload))

        indicator_calls: list[str] = []

        async def _fake_emit(kind: str) -> bool:
            indicator_calls.append(kind)
            return True

        old_names = list(cfg.speech_gate.assistant_names)
        old_patterns_file = cfg.speech_gate.patterns_file
        old_identity_file = cfg.speech_gate.identity_file
        cfg.speech_gate.assistant_names = ["kissa"]
        cfg.speech_gate.patterns_file = None
        cfg.speech_gate.identity_file = str(ROOT / "missing_identity.md")
        monkeypatch.setattr(speech_gate_agent_module, "emit_indicator", _fake_emit)
        try:
            bus = DummyBus()
            agent = SpeechGateAgent(bus=bus)  # type: ignore[arg-type]
            await agent.start()
            try:
                await agent._on_input_text(  # pylint: disable=protected-access
                    Event(topic="llm/input_text", payload={"text": "Kissa, помолчи"})
                )
                assert agent.get_status()["mode"] == "mute"
                assert bus.events == []

                await agent._on_input_text(  # pylint: disable=protected-access
                    Event(topic="llm/input_text", payload={"text": "включи"})
                )
                assert agent.get_status()["mode"] == "mute"
                assert bus.events == []

                await agent._on_input_text(  # pylint: disable=protected-access
                    Event(topic="llm/input_text", payload={"text": "включи Kissa, говори"})
                )
                assert agent.get_status()["mode"] == "mute"

                await agent._on_input_text(  # pylint: disable=protected-access
                    Event(topic="llm/input_text", payload={"text": "Kissa, говори"})
                )
                assert agent.get_status()["mode"] == "normal"
                assert len(bus.events) <= 1

                await agent._on_input_text(  # pylint: disable=protected-access
                    Event(topic="llm/input_text", payload={"text": "Kissa, ответь какая погода"})
                )
                assert bus.events[-1][0] == "llm/accepted_phrase"
                assert bus.events[-1][1]["text"] == "Kissa, ответь какая погода"
                assert bus.events[-1][1]["speech_gate_leading_assistant_name"] == "kissa"
                assert "speech_gate_barge_in" not in bus.events[-1][1]

                await agent._on_input_text(  # pylint: disable=protected-access
                    Event(
                        topic="llm/input_text",
                        payload={"text": "Kissa, нет, я имел в виду Амстердам"},
                    )
                )
                assert bus.events[-1][1]["text"] == "Kissa, нет, я имел в виду Амстердам"
                assert bus.events[-1][1]["speech_gate_barge_in"] is True
                assert bus.events[-1][1]["speech_gate_barge_in_phrase"] == "нет"

                await agent._on_input_text(  # pylint: disable=protected-access
                    Event(topic="llm/input_text", payload={"text": "Kissa, выключись"})
                )
                status = agent.get_status()
                assert status["mode"] == "standby"
                assert status["temporary"] is False

                await agent._on_input_text(  # pylint: disable=protected-access
                    Event(topic="llm/input_text", payload={"text": "Kissa, говори"})
                )
                assert agent.get_status()["mode"] == "normal"
                assert bus.events[-1][1]["text"] == "Kissa, нет, я имел в виду Амстердам"
                assert indicator_calls.count("local_handled") == 4
            finally:
                await agent.close()
        finally:
            cfg.speech_gate.assistant_names = old_names
            cfg.speech_gate.patterns_file = old_patterns_file
            cfg.speech_gate.identity_file = old_identity_file

    asyncio.run(_runner())


def test_speech_gate_agent_emits_rejected_indicator(monkeypatch):
    async def _runner() -> None:
        class DummyBus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict]] = []
                self.subscriptions: dict[str, object] = {}

            def subscribe(self, topic: str, handler):
                self.subscriptions[topic] = handler

            def unsubscribe(self, topic: str, handler):
                current = self.subscriptions.get(topic)
                if current is handler:
                    del self.subscriptions[topic]

            async def publish(self, topic: str, **payload):
                self.events.append((topic, payload))

        indicator_calls: list[str] = []

        async def _fake_emit(kind: str) -> bool:
            indicator_calls.append(kind)
            return True

        old_names = list(cfg.speech_gate.assistant_names)
        old_patterns_file = cfg.speech_gate.patterns_file
        old_identity_file = cfg.speech_gate.identity_file
        cfg.speech_gate.assistant_names = ["kissa"]
        cfg.speech_gate.patterns_file = None
        cfg.speech_gate.identity_file = str(ROOT / "missing_identity.md")
        monkeypatch.setattr(speech_gate_agent_module, "emit_indicator", _fake_emit)
        try:
            bus = DummyBus()
            agent = SpeechGateAgent(bus=bus)  # type: ignore[arg-type]
            await agent.start()
            try:
                await agent._on_input_text(  # pylint: disable=protected-access
                    Event(topic="llm/input_text", payload={"text": "просто мысль вслух"})
                )
            finally:
                await agent.close()
        finally:
            cfg.speech_gate.assistant_names = old_names
            cfg.speech_gate.patterns_file = old_patterns_file
            cfg.speech_gate.identity_file = old_identity_file

        assert bus.events == []
        assert indicator_calls == ["rejected"]

    asyncio.run(_runner())


def test_speech_gate_agent_local_stop_command_calls_openclaw_abort(monkeypatch):
    async def _runner() -> None:
        class DummyBus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict]] = []
                self.subscriptions: dict[str, object] = {}

            def subscribe(self, topic: str, handler):
                self.subscriptions[topic] = handler

            def unsubscribe(self, topic: str, handler):
                current = self.subscriptions.get(topic)
                if current is handler:
                    del self.subscriptions[topic]

            async def publish(self, topic: str, **payload):
                self.events.append((topic, payload))

        calls: list[str] = []
        indicator_calls: list[str] = []

        async def _fake_abort(session_key: str):
            calls.append(session_key)
            return {"ok": True, "aborted": True, "runIds": ["run-1"]}

        async def _fake_clear_pending() -> int:
            calls.append("clear")
            return 2

        async def _fake_emit(kind: str) -> bool:
            indicator_calls.append(kind)
            return True

        old_names = list(cfg.speech_gate.assistant_names)
        old_patterns_file = cfg.speech_gate.patterns_file
        old_identity_file = cfg.speech_gate.identity_file
        old_session_key = cfg.openclaw.session_key
        cfg.speech_gate.assistant_names = ["kissa"]
        cfg.speech_gate.patterns_file = None
        cfg.speech_gate.identity_file = str(ROOT / "missing_identity.md")
        cfg.openclaw.session_key = "voice-main"
        monkeypatch.setattr(speech_gate_agent_module, "abort_openclaw_chat_session", _fake_abort)
        monkeypatch.setattr(speech_gate_agent_module, "emit_indicator", _fake_emit)
        try:
            bus = DummyBus()
            agent = SpeechGateAgent(  # type: ignore[arg-type]
                bus=bus,
                on_local_stop=_fake_clear_pending,
            )
            await agent.start()
            try:
                await agent._on_input_text(  # pylint: disable=protected-access
                    Event(topic="llm/input_text", payload={"text": "Kissa, стоп"})
                )
            finally:
                await agent.close()
        finally:
            cfg.speech_gate.assistant_names = old_names
            cfg.speech_gate.patterns_file = old_patterns_file
            cfg.speech_gate.identity_file = old_identity_file
            cfg.openclaw.session_key = old_session_key

        assert calls == ["clear", "voice-main"]
        assert bus.events == []
        assert indicator_calls == ["interrupted"]

    asyncio.run(_runner())


def test_speech_gate_agent_runtime_mode_control():
    async def _runner() -> None:
        class DummyBus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict]] = []
                self.subscriptions: dict[str, object] = {}

            def subscribe(self, topic: str, handler):
                self.subscriptions[topic] = handler

            def unsubscribe(self, topic: str, handler):
                current = self.subscriptions.get(topic)
                if current is handler:
                    del self.subscriptions[topic]

            async def publish(self, topic: str, **payload):
                self.events.append((topic, payload))

        old_names = list(cfg.speech_gate.assistant_names)
        old_patterns_file = cfg.speech_gate.patterns_file
        old_identity_file = cfg.speech_gate.identity_file
        cfg.speech_gate.assistant_names = ["kissa"]
        cfg.speech_gate.patterns_file = None
        cfg.speech_gate.identity_file = str(ROOT / "missing_identity.md")
        try:
            bus = DummyBus()
            agent = SpeechGateAgent(bus=bus)  # type: ignore[arg-type]
            await agent.start()
            try:
                await agent._on_input_text(  # pylint: disable=protected-access
                    Event(topic="llm/input_text", payload={"text": "Привет, kissa"})
                )
                await agent.set_mode("mute", source="test", reason="clear attention")
                await agent._on_input_text(  # pylint: disable=protected-access
                    Event(topic="llm/input_text", payload={"text": "подробности"})
                )
                assert len(bus.events) == 1
                assert agent.get_status()["mode"] == "mute"

                with pytest.raises(ValueError):
                    await agent.set_mode("standby")

                await agent.set_mode("chatty", ttl_seconds=0.05, source="test")
                assert agent.get_status()["mode"] == "chatty"
                await asyncio.sleep(0.08)
                status = agent.get_status()
                assert status["mode"] == "mute"
                assert status["temporary"] is False

                await agent.set_mode("normal")
                assert agent.get_status()["mode"] == "normal"
            finally:
                await agent.close()
        finally:
            cfg.speech_gate.assistant_names = old_names
            cfg.speech_gate.patterns_file = old_patterns_file
            cfg.speech_gate.identity_file = old_identity_file

    asyncio.run(_runner())


def test_speech_gate_agent_restores_temporary_mode_after_restart(tmp_path):
    async def _runner() -> None:
        class DummyBus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict]] = []
                self.subscriptions: dict[str, object] = {}

            def subscribe(self, topic: str, handler):
                self.subscriptions[topic] = handler

            def unsubscribe(self, topic: str, handler):
                current = self.subscriptions.get(topic)
                if current is handler:
                    del self.subscriptions[topic]

            async def publish(self, topic: str, **payload):
                self.events.append((topic, payload))

        old_names = list(cfg.speech_gate.assistant_names)
        old_patterns_file = cfg.speech_gate.patterns_file
        old_identity_file = cfg.speech_gate.identity_file
        old_state_path = cfg.control.state_path
        cfg.speech_gate.assistant_names = ["kissa"]
        cfg.speech_gate.patterns_file = None
        cfg.speech_gate.identity_file = str(ROOT / "missing_identity.md")
        cfg.control.state_path = str(tmp_path / "runtime_state.json")
        try:
            first = SpeechGateAgent(bus=DummyBus())  # type: ignore[arg-type]
            await first.start()
            try:
                await first.set_mode("chatty", ttl_seconds=0.2, source="test", reason="persist me")
                await asyncio.sleep(0.05)
            finally:
                await first.close()

            second = SpeechGateAgent(bus=DummyBus())  # type: ignore[arg-type]
            await second.start()
            try:
                status = second.get_status()
                assert status["mode"] == "chatty"
                assert status["temporary"] is True
                assert status["restore_mode"] == "normal"
                assert status["source"] == "test"
                assert status["reason"] == "persist me"
                assert 0.0 < float(status["expires_in_seconds"]) <= 0.2
                await asyncio.sleep(0.2)
                assert second.get_status()["mode"] == "normal"
                assert second.get_status()["temporary"] is False
            finally:
                await second.close()
        finally:
            cfg.speech_gate.assistant_names = old_names
            cfg.speech_gate.patterns_file = old_patterns_file
            cfg.speech_gate.identity_file = old_identity_file
            cfg.control.state_path = old_state_path

    asyncio.run(_runner())


def test_speech_gate_agent_restores_permanent_mode_after_restart(tmp_path):
    async def _runner() -> None:
        class DummyBus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict]] = []
                self.subscriptions: dict[str, object] = {}

            def subscribe(self, topic: str, handler):
                self.subscriptions[topic] = handler

            def unsubscribe(self, topic: str, handler):
                current = self.subscriptions.get(topic)
                if current is handler:
                    del self.subscriptions[topic]

            async def publish(self, topic: str, **payload):
                self.events.append((topic, payload))

        old_names = list(cfg.speech_gate.assistant_names)
        old_patterns_file = cfg.speech_gate.patterns_file
        old_identity_file = cfg.speech_gate.identity_file
        old_state_path = cfg.control.state_path
        cfg.speech_gate.assistant_names = ["kissa"]
        cfg.speech_gate.patterns_file = None
        cfg.speech_gate.identity_file = str(ROOT / "missing_identity.md")
        cfg.control.state_path = str(tmp_path / "runtime_state.json")
        try:
            first = SpeechGateAgent(bus=DummyBus())  # type: ignore[arg-type]
            await first.start()
            try:
                await first.set_mode("standby", ttl_seconds=5.0, source="test")
                await first.set_mode("mute", source="test", reason="keep it")
            finally:
                await first.close()

            second = SpeechGateAgent(bus=DummyBus())  # type: ignore[arg-type]
            await second.start()
            try:
                status = second.get_status()
                assert status["mode"] == "mute"
                assert status["temporary"] is False
                assert status["source"] == "test"
                assert status["reason"] == "keep it"
            finally:
                await second.close()
        finally:
            cfg.speech_gate.assistant_names = old_names
            cfg.speech_gate.patterns_file = old_patterns_file
            cfg.speech_gate.identity_file = old_identity_file
            cfg.control.state_path = old_state_path

    asyncio.run(_runner())


def test_speech_gate_agent_uses_mode_active_at_segment_start():
    async def _runner() -> None:
        class DummyBus:
            def __init__(self) -> None:
                self.events: list[tuple[str, dict]] = []
                self.subscriptions: dict[str, object] = {}

            def subscribe(self, topic: str, handler):
                self.subscriptions[topic] = handler

            def unsubscribe(self, topic: str, handler):
                current = self.subscriptions.get(topic)
                if current is handler:
                    del self.subscriptions[topic]

            async def publish(self, topic: str, **payload):
                self.events.append((topic, payload))

        old_names = list(cfg.speech_gate.assistant_names)
        old_patterns_file = cfg.speech_gate.patterns_file
        old_identity_file = cfg.speech_gate.identity_file
        cfg.speech_gate.assistant_names = ["kissa"]
        cfg.speech_gate.patterns_file = None
        cfg.speech_gate.identity_file = str(ROOT / "missing_identity.md")
        try:
            bus = DummyBus()
            agent = SpeechGateAgent(bus=bus)  # type: ignore[arg-type]
            await agent.start()
            try:
                await agent.set_mode("chatty", ttl_seconds=0.05, source="test")
                segment_start = agent.get_status()["changed_at"] + 0.01
                await asyncio.sleep(0.08)
                assert agent.get_status()["mode"] == "normal"

                await agent._on_input_text(  # pylint: disable=protected-access
                    Event(
                        topic="llm/input_text",
                        payload={
                            "text": "не профильная фраза",
                            "start_timestamp": segment_start,
                        },
                    )
                )
            finally:
                await agent.close()
        finally:
            cfg.speech_gate.assistant_names = old_names
            cfg.speech_gate.patterns_file = old_patterns_file
            cfg.speech_gate.identity_file = old_identity_file

        assert len(bus.events) == 1
        payload = bus.events[0][1]
        assert payload["speech_gate_reason"] == "attention"

    asyncio.run(_runner())
