from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .config import GatewayConfig, StreamingConfig
from .messages import extract_assistant_message_text
from .tts import split_complete_speech_units, split_speech_units


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    identifier: str
    text: str
    run_id: str
    final: bool = False


@dataclass(frozen=True, slots=True)
class SpeechRouteResult:
    segments: list[SpeechSegment]
    needs_history: bool = False


@dataclass(slots=True)
class _RunSpeechState:
    last_text: str = ""
    emitted_text: str = ""
    seq: int = 0
    fallback_final_only: bool = False
    unsafe_after_partial: bool = False


@dataclass(slots=True)
class ChatSpeechRouter:
    gateway: GatewayConfig
    streaming: StreamingConfig
    _runs: dict[str, _RunSpeechState] = field(default_factory=dict)

    def route(self, event: dict[str, Any]) -> SpeechRouteResult:
        if event.get("type") != "event" or event.get("event") != "chat":
            return SpeechRouteResult([])
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return SpeechRouteResult([])
        if not self.gateway.matches_session(str(payload.get("sessionKey") or "")):
            return SpeechRouteResult([])

        state = str(payload.get("state") or "")
        run_id = str(payload.get("runId") or "unknown")
        if state == "delta":
            return SpeechRouteResult(self._route_delta(run_id, payload))
        if state == "final":
            return self._route_final(run_id, payload)
        if state in {"aborted", "error"}:
            self._runs.pop(run_id, None)
        return SpeechRouteResult([])

    def route_final_text(self, run_id: str, text: str) -> SpeechRouteResult:
        run_key = str(run_id or "unknown")
        state = self._runs.setdefault(run_key, _RunSpeechState())
        state.last_text = str(text or "")
        return self._flush_final(run_key, state)

    def discard(self, run_id: str) -> None:
        self._runs.pop(str(run_id or "unknown"), None)

    def _route_delta(self, run_id: str, payload: dict[str, Any]) -> list[SpeechSegment]:
        extracted = _extract_payload_message(payload, run_id)
        if extracted is None:
            return []

        state = self._runs.setdefault(run_id, _RunSpeechState())
        if state.fallback_final_only:
            state.last_text = extracted.text
            return []
        if state.emitted_text and not extracted.text.startswith(state.emitted_text):
            state.fallback_final_only = True
            state.unsafe_after_partial = True
            state.last_text = extracted.text
            return []

        state.last_text = extracted.text
        return self._emit_available(run_id, state, final=False)

    def _route_final(self, run_id: str, payload: dict[str, Any]) -> SpeechRouteResult:
        state = self._runs.setdefault(run_id, _RunSpeechState())
        extracted = _extract_payload_message(payload, run_id)
        if extracted is not None:
            state.last_text = extracted.text
            return self._flush_final(run_id, state)
        if not state.last_text:
            return SpeechRouteResult([], needs_history=True)

        segments = self._emit_available(run_id, state, final=bool(self.streaming.flush_on_final))
        return SpeechRouteResult(segments, needs_history=True)

    def _flush_final(self, run_id: str, state: _RunSpeechState) -> SpeechRouteResult:
        if state.fallback_final_only:
            if not state.emitted_text:
                segments = self._emit_available(run_id, state, final=True)
                self._runs.pop(run_id, None)
                return SpeechRouteResult(segments)
            if state.last_text.startswith(state.emitted_text):
                segments = self._emit_available(run_id, state, final=True)
            else:
                segments = []
            self._runs.pop(run_id, None)
            return SpeechRouteResult(segments)

        segments = self._emit_available(run_id, state, final=bool(self.streaming.flush_on_final))
        self._runs.pop(run_id, None)
        return SpeechRouteResult(segments)

    def _emit_available(self, run_id: str, state: _RunSpeechState, *, final: bool) -> list[SpeechSegment]:
        if state.emitted_text and not state.last_text.startswith(state.emitted_text):
            state.fallback_final_only = True
            return []

        new_text = state.last_text[len(state.emitted_text) :].strip()
        if not new_text:
            return []

        units = (
            split_speech_units(new_text, include_incomplete=True)
            if final
            else split_complete_speech_units(new_text)
        )
        if not final:
            forced = _forced_incomplete_chunk(new_text, units, self.streaming)
            if forced:
                units.append(forced)
        if not units:
            return []

        segments: list[SpeechSegment] = []
        for unit in units:
            state.seq += 1
            state.emitted_text = _append_text(state.emitted_text, unit)
            segments.append(
                SpeechSegment(
                    identifier=_segment_identifier(run_id, state.seq, unit),
                    text=unit,
                    run_id=run_id,
                    final=final,
                )
            )
        return segments


def _extract_payload_message(payload: dict[str, Any], run_id: str):
    return extract_assistant_message_text(
        payload.get("message"),
        fallback_identifier=f"run:{run_id}",
    )


def _forced_incomplete_chunk(text: str, complete_units: list[str], config: StreamingConfig) -> str | None:
    if complete_units:
        emitted = ""
        for unit in complete_units:
            emitted = _append_text(emitted, unit)
        if text.startswith(emitted):
            text = text[len(emitted) :].strip()
    if len(text) < config.max_chars:
        return None
    cut = text.rfind(" ", 0, config.max_chars + 1)
    if cut < max(1, config.min_chars):
        cut = config.max_chars
    chunk = text[:cut].strip()
    if len(chunk) < config.min_chars:
        return None
    return chunk


def _append_text(base: str, addition: str) -> str:
    addition = addition.strip()
    if not addition:
        return base
    if not base:
        return addition
    return f"{base} {addition}"


def _segment_identifier(run_id: str, seq: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"stream:{run_id}:{seq}:{digest}"
