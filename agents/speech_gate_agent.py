"""Filter STT phrases with a rules+ML speech gate before forwarding to LLM."""

from __future__ import annotations

import logging
from typing import Any

from core.bus import Event, EventBus, bus as default_bus
from core.config import cfg
from llm.speech_gate import SpeechDirectionGate

log = logging.getLogger(__name__)


class SpeechGateAgent:
    """Consumes ``llm/input_text`` and publishes filtered ``llm/speaker_phrase``."""

    def __init__(self, *, bus: EventBus | None = None) -> None:
        self._bus = bus or default_bus
        self._running = False
        self._paused = False
        self._input_topic = ""
        self._output_topic = ""
        self._gate = SpeechDirectionGate.from_config()

    async def start(self) -> None:
        if self._running:
            return

        self._input_topic = str(getattr(cfg.events.llm, "input_text", "") or "").strip()
        self._output_topic = str(
            getattr(cfg.events.llm, "speaker_phrase", self._input_topic) or ""
        ).strip()

        if not self._input_topic or not self._output_topic:
            log.warning("SpeechGateAgent: topics are not configured; disabled")
            return
        if self._input_topic == self._output_topic:
            log.warning(
                "SpeechGateAgent: input and output topics are identical (%s); disabled to avoid loops",
                self._input_topic,
            )
            return

        self._gate = SpeechDirectionGate.from_config()
        self._bus.subscribe(self._input_topic, self._on_input_text)
        self._running = True
        self._paused = False
        log.info(
            "SpeechGateAgent: started (input=%s, output=%s, enabled=%s)",
            self._input_topic,
            self._output_topic,
            getattr(cfg.speech_gate, "enable", True),
        )

    async def pause(self) -> None:
        if not self._running or self._paused:
            return
        self._paused = True
        log.info("SpeechGateAgent: paused")

    async def resume(self) -> None:
        if not self._running or not self._paused:
            return
        self._paused = False
        log.info("SpeechGateAgent: resumed")

    async def close(self) -> None:
        if not self._running:
            return
        self._running = False
        self._paused = False
        if self._input_topic:
            self._bus.unsubscribe(self._input_topic, self._on_input_text)
        self._input_topic = ""
        self._output_topic = ""
        log.info("SpeechGateAgent: stopped")

    async def _on_input_text(self, event: Event) -> None:
        if not self._running or self._paused:
            return

        payload: dict[str, Any] = dict(event.payload or {})
        text = " ".join(str(payload.get("text") or "").split())
        if not text:
            return

        # Keep this synchronous: in some Linux runtimes torch-initialized processes
        # can deadlock when creating worker threads via asyncio.to_thread().
        decision = self._gate.should_allow(text, payload=payload)
        if not decision.allowed:
            if cfg.debug:
                log.info(
                    "speech_gate: drop phrase reason=%s rules=%.2f ml=%.2f final=%.2f continuation=%s text=%s",
                    decision.reason,
                    decision.rule_score,
                    decision.ml_score,
                    decision.final_score,
                    decision.continuation,
                    text,
                )
            return

        if cfg.debug:
            log.info(
                "speech_gate: allow phrase reason=%s rules=%.2f ml=%.2f final=%.2f continuation=%s text=%s",
                decision.reason,
                decision.rule_score,
                decision.ml_score,
                decision.final_score,
                decision.continuation,
                text,
            )

        publish_payload = dict(payload)
        publish_payload["text"] = text
        publish_payload["speech_gate_reason"] = decision.reason
        publish_payload["speech_gate_rule_score"] = decision.rule_score
        publish_payload["speech_gate_ml_score"] = decision.ml_score
        publish_payload["speech_gate_final_score"] = decision.final_score
        publish_payload["speech_gate_continuation"] = decision.continuation

        await self._bus.publish(self._output_topic, **publish_payload)


__all__ = ["SpeechGateAgent"]
