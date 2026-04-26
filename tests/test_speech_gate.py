import asyncio
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from core.bus import Event  # noqa: E402
from core.config import cfg  # noqa: E402
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

        assert any(topic == "llm/speaker_phrase" for topic, _ in bus.events)
        published_payload = [payload for topic, payload in bus.events if topic == "llm/speaker_phrase"][0]
        assert published_payload["text"] == "Привет, kissa"
        assert "speech_gate_reason" in published_payload

    asyncio.run(_runner())
