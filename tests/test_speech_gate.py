import asyncio
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from core.bus import Event  # noqa: E402
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

        bus = DummyBus()
        agent = SpeechGateAgent(bus=bus)  # type: ignore[arg-type]
        await agent.start()
        try:
            await agent._on_input_text(  # pylint: disable=protected-access
                Event(topic="llm/input_text", payload={"text": "Привет, kissa"})
            )
        finally:
            await agent.close()

        assert any(topic == "llm/speaker_phrase" for topic, _ in bus.events)
        published_payload = [payload for topic, payload in bus.events if topic == "llm/speaker_phrase"][0]
        assert published_payload["text"] == "Привет, kissa"
        assert "speech_gate_reason" in published_payload

    asyncio.run(_runner())
