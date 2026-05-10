"""Filter STT phrases with a rules+ML speech gate before forwarding to LLM."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agents.openclaw_gateway import abort_openclaw_chat_session
from core.bus import Event, EventBus, bus as default_bus
from core.config import cfg
from llm.speech_gate import LocalCommandMatch, SpeechDirectionGate, SpeechGateMode

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _ModeInterval:
    mode: SpeechGateMode
    start_at: float
    end_at: float | None = None
    restore_mode: SpeechGateMode | None = None


class SpeechGateAgent:
    """Consumes ``llm/input_text`` and publishes phrases accepted by SpeechGate."""

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        on_local_stop: Callable[[], Awaitable[int] | int | None] | None = None,
    ) -> None:
        self._bus = bus or default_bus
        self._on_local_stop = on_local_stop
        self._running = False
        self._paused = False
        self._input_topic = ""
        self._output_topic = ""
        self._gate = SpeechDirectionGate.from_config()
        self._mode_ttl_task: asyncio.Task[None] | None = None
        self._mode_version = 0
        self._mode_restore: SpeechGateMode | None = None
        self._mode_expires_at: float | None = None
        self._mode_changed_at: float | None = None
        self._mode_source = "config"
        self._mode_reason = ""
        self._mode_history: list[_ModeInterval] = []

    async def start(self) -> None:
        if self._running:
            return

        self._input_topic = str(getattr(cfg.events.llm, "input_text", "") or "").strip()
        self._output_topic = str(
            getattr(cfg.events.llm, "accepted_phrase", self._input_topic) or ""
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

        self._clear_mode_state(source="config", reason="")
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
        await self._cancel_mode_timer()
        if self._input_topic:
            self._bus.unsubscribe(self._input_topic, self._on_input_text)
        self._input_topic = ""
        self._output_topic = ""
        log.info("SpeechGateAgent: stopped")

    async def set_mode(
        self,
        mode: str | SpeechGateMode,
        *,
        ttl_seconds: float | None = None,
        source: str = "api",
        reason: str = "",
    ) -> dict[str, Any]:
        target_mode = SpeechGateMode.parse(mode)
        ttl = self._normalise_ttl(ttl_seconds)
        return await self._apply_mode_change(
            target_mode,
            ttl_seconds=ttl,
            source=source,
            reason=reason,
        )

    async def _apply_mode_change(
        self,
        mode: SpeechGateMode,
        *,
        ttl_seconds: float | None,
        source: str,
        reason: str,
        allow_standby_without_ttl: bool = False,
    ) -> dict[str, Any]:
        target_mode = mode if isinstance(mode, SpeechGateMode) else SpeechGateMode.parse(mode)
        ttl = self._normalise_ttl(ttl_seconds)
        if (
            target_mode == SpeechGateMode.STANDBY
            and ttl is None
            and not allow_standby_without_ttl
        ):
            raise ValueError("standby mode requires ttl_seconds")

        previous_mode = self._gate.mode
        await self._cancel_mode_timer()
        self._mode_version += 1
        self._gate.set_mode(target_mode)
        self._gate.clear_attention()

        now = time.time()
        self._mode_changed_at = now
        self._mode_source = str(source or "api")
        self._mode_reason = str(reason or "")

        if target_mode == SpeechGateMode.NORMAL:
            self._mode_restore = None
            self._mode_expires_at = None
        elif ttl is None:
            self._mode_restore = None
            self._mode_expires_at = None
        else:
            self._mode_restore = (
                previous_mode
                if previous_mode != target_mode
                else self._mode_restore or SpeechGateMode.NORMAL
            )
            self._mode_expires_at = now + ttl
            self._mode_ttl_task = asyncio.create_task(
                self._restore_mode_after(ttl, self._mode_restore, self._mode_version),
                name="SpeechGateAgent.mode_ttl",
            )
        self._append_mode_interval(
            target_mode,
            start_at=now,
            end_at=self._mode_expires_at,
            restore_mode=self._mode_restore,
        )

        log.info(
            "SpeechGateAgent: mode changed %s -> %s source=%s ttl=%s reason=%s",
            previous_mode.value,
            target_mode.value,
            self._mode_source,
            f"{ttl:.1f}s" if ttl is not None else "none",
            self._mode_reason,
        )
        return self.get_status()

    async def _handle_local_command(self, match: LocalCommandMatch, text: str) -> None:
        reason = f"voice command via {match.assistant_name}: {match.phrase}"
        source = "speech_gate_local"

        if match.action == "mute":
            status = await self._apply_mode_change(
                SpeechGateMode.MUTE,
                ttl_seconds=None,
                source=source,
                reason=reason,
            )
            log.info(
                "speech_gate: local command -> mode=%s assistant=%s phrase=%s text=%s",
                status["mode"],
                match.assistant_name,
                match.phrase,
                text,
            )
            return

        if match.action == "normal":
            status = await self._apply_mode_change(
                SpeechGateMode.NORMAL,
                ttl_seconds=None,
                source=source,
                reason=reason,
            )
            log.info(
                "speech_gate: local command -> mode=%s assistant=%s phrase=%s text=%s",
                status["mode"],
                match.assistant_name,
                match.phrase,
                text,
            )
            return

        if match.action == "standby":
            status = await self._apply_mode_change(
                SpeechGateMode.STANDBY,
                ttl_seconds=None,
                source=source,
                reason=reason,
                allow_standby_without_ttl=True,
            )
            log.info(
                "speech_gate: local command -> mode=%s assistant=%s phrase=%s text=%s",
                status["mode"],
                match.assistant_name,
                match.phrase,
                text,
            )
            return

        if match.action == "stop_generation":
            session_key = str(getattr(cfg.openclaw, "session_key", "") or "main").strip() or "main"
            dropped_pending = 0
            if self._on_local_stop is not None:
                try:
                    maybe_result = self._on_local_stop()
                    if asyncio.iscoroutine(maybe_result):
                        maybe_result = await maybe_result
                    dropped_pending = int(maybe_result or 0)
                except Exception:
                    log.exception(
                        "speech_gate: failed to clear pending OpenClaw input before local stop"
                    )
            try:
                result = await abort_openclaw_chat_session(session_key)
            except FileNotFoundError:
                log.warning(
                    "speech_gate: local stop command ignored because OpenClaw command is not available"
                )
            except Exception:
                log.exception(
                    "speech_gate: failed to abort OpenClaw generation via local stop command"
                )
            else:
                aborted = bool(result.get("aborted"))
                method = str(result.get("method") or "unknown")
                resolved_session_key = str(result.get("resolvedSessionKey") or session_key)
                fallback_used = bool(result.get("fallbackUsed"))
                steer_used = bool(result.get("steerUsed"))
                interrupted_active = bool(result.get("interruptedActiveRun"))
                attempt = int(result.get("attempt") or 1)
                running_session_keys = result.get("runningSessionKeys") or []
                log_method = log.info if aborted else log.warning
                log_method(
                    "speech_gate: local stop command sent to OpenClaw "
                    "(assistant=%s phrase=%s session=%s method=%s aborted=%s runs=%s "
                    "interrupted_active=%s dropped_pending=%s attempts=%s resolved_session=%s "
                    "fallback_used=%s steer_used=%s running_sessions=%s)",
                    match.assistant_name,
                    match.phrase,
                    session_key,
                    method,
                    aborted,
                    len(result.get("runIds") or []),
                    interrupted_active,
                    dropped_pending,
                    attempt,
                    resolved_session_key,
                    fallback_used,
                    steer_used,
                    running_session_keys,
                )
            return

        log.debug(
            "speech_gate: ignored unsupported local command action=%s assistant=%s text=%s",
            match.action,
            match.assistant_name,
            text,
        )

    def get_status(self) -> dict[str, Any]:
        now = time.time()
        expires_in = None
        if self._mode_expires_at is not None:
            expires_in = max(0.0, self._mode_expires_at - now)
        return {
            "running": self._running,
            "paused": self._paused,
            "mode": self._gate.mode.value,
            "configured_mode": SpeechGateMode.from_value(
                getattr(cfg.speech_gate, "mode", SpeechGateMode.NORMAL.value)
            ).value,
            "temporary": self._mode_expires_at is not None,
            "changed_at": self._mode_changed_at,
            "expires_at": self._mode_expires_at,
            "expires_in_seconds": expires_in,
            "restore_mode": self._mode_restore.value if self._mode_restore else None,
            "source": self._mode_source,
            "reason": self._mode_reason,
            "input_topic": self._input_topic,
            "output_topic": self._output_topic,
        }

    @staticmethod
    def _normalise_ttl(ttl_seconds: float | int | str | None) -> float | None:
        if ttl_seconds in (None, ""):
            return None
        ttl = float(ttl_seconds)
        if ttl <= 0:
            return None
        return ttl

    def _clear_mode_state(self, *, source: str, reason: str) -> None:
        self._mode_version += 1
        self._mode_restore = None
        self._mode_expires_at = None
        self._mode_changed_at = time.time()
        self._mode_source = source
        self._mode_reason = reason
        self._mode_history = [
            _ModeInterval(mode=self._gate.mode, start_at=self._mode_changed_at)
        ]

    def _append_mode_interval(
        self,
        mode: SpeechGateMode,
        *,
        start_at: float,
        end_at: float | None = None,
        restore_mode: SpeechGateMode | None = None,
    ) -> None:
        if self._mode_history:
            last = self._mode_history[-1]
            if last.end_at is None or last.end_at > start_at:
                last.end_at = start_at
        self._mode_history.append(
            _ModeInterval(
                mode=mode,
                start_at=start_at,
                end_at=end_at,
                restore_mode=restore_mode,
            )
        )
        # Keep enough history for delayed STT finalisation without growing forever.
        cutoff = start_at - 3600.0
        self._mode_history = [
            interval
            for interval in self._mode_history
            if interval.end_at is None or interval.end_at >= cutoff
        ]

    def _effective_mode_for_payload(self, payload: dict[str, Any]) -> SpeechGateMode:
        timestamp = self._extract_segment_start_timestamp(payload)
        if timestamp is None:
            return self._gate.mode
        for interval in reversed(self._mode_history):
            if timestamp < interval.start_at:
                continue
            if interval.end_at is None or timestamp <= interval.end_at:
                return interval.mode
            if interval.restore_mode is not None:
                return interval.restore_mode
            return self._gate.mode
        return self._gate.mode

    @staticmethod
    def _extract_segment_start_timestamp(payload: dict[str, Any]) -> float | None:
        for key in ("start_timestamp", "segment_start_timestamp", "speech_start_timestamp"):
            value = payload.get(key)
            if value in (None, ""):
                continue
            try:
                timestamp = float(value)
            except (TypeError, ValueError):
                continue
            if timestamp > 0:
                return timestamp
        return None

    async def _cancel_mode_timer(self) -> None:
        task = self._mode_ttl_task
        self._mode_ttl_task = None
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _restore_mode_after(
        self, ttl_seconds: float, restore_mode: SpeechGateMode, version: int
    ) -> None:
        try:
            await asyncio.sleep(ttl_seconds)
        except asyncio.CancelledError:
            raise
        if version != self._mode_version:
            return

        previous_mode = self._gate.mode
        self._gate.set_mode(restore_mode)
        self._gate.clear_attention()
        self._mode_ttl_task = None
        self._mode_restore = None
        self._mode_expires_at = None
        self._mode_changed_at = time.time()
        self._mode_source = "ttl"
        self._mode_reason = f"expired temporary {previous_mode.value} mode"
        self._append_mode_interval(restore_mode, start_at=self._mode_changed_at)
        log.info(
            "SpeechGateAgent: mode TTL expired %s -> %s",
            previous_mode.value,
            restore_mode.value,
        )

    async def _on_input_text(self, event: Event) -> None:
        if not self._running or self._paused:
            return

        payload: dict[str, Any] = dict(event.payload or {})
        text = " ".join(str(payload.get("text") or "").split())
        if not text:
            return

        local_command = self._gate.detect_local_command(text)
        if local_command is not None:
            await self._handle_local_command(local_command, text)
            return

        # Keep this synchronous: in some Linux runtimes torch-initialized processes
        # can deadlock when creating worker threads via asyncio.to_thread().
        effective_mode = self._effective_mode_for_payload(payload)
        decision = self._gate.should_allow(text, payload=payload, mode=effective_mode)
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
