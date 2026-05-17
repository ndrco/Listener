import asyncio
import contextlib
import logging
import signal
import sys
from typing import Optional

from agents.audio_agent import AudioAgent
from agents.control_agent import ControlAgent
from agents.openclaw_input_agent import OpenClawInputAgent
from agents.speaker_agent import SpeakerAgent
from agents.speech_gate_agent import SpeechGateAgent
from core.bus import bus
from core.config import cfg, load
from core.logging_utils import configure_logging
from core.sound_indicators import indicators


class RuntimeStatus:
    """Small status snapshot for service readiness probes."""

    def __init__(self) -> None:
        self._components: dict[str, dict] = {}

    def set(
        self,
        name: str,
        state: str,
        *,
        critical: bool = False,
        error: str | None = None,
    ) -> None:
        self._components[name] = {
            "state": state,
            "ok": state in {"started", "disabled"},
            "critical": bool(critical),
            "error": error,
        }

    def as_dict(self) -> dict:
        components = {name: dict(value) for name, value in self._components.items()}
        ready = all(
            bool(component.get("ok"))
            for component in components.values()
            if component.get("critical")
        )
        last_error = None
        for component in reversed(list(components.values())):
            if component.get("error"):
                last_error = component["error"]
                break
        return {
            "ok": True,
            "service": "listener",
            "ready": ready,
            "components": components,
            "last_error": last_error,
        }


async def main() -> None:
    load()
    configure_logging(debug=cfg.debug, info=cfg.info)
    if cfg.debug:
        logging.info("Listener voice runtime starting in DEBUG mode")
    elif cfg.info:
        logging.info("Listener voice runtime starting")

    status = RuntimeStatus()
    stop = asyncio.Event()

    async def _request_shutdown(reason: str = "api") -> None:
        await bus.publish(
            cfg.events.app.stop,
            source="control",
            reason=str(reason or "api"),
        )

    async def _on_app_stop(ev) -> None:
        logging.info(
            "main: got app/stop from %s (%s)",
            ev.payload.get("source"),
            ev.payload.get("reason"),
        )
        stop.set()

    bus.start()
    bus.subscribe(cfg.events.app.stop, _on_app_stop)

    try:
        await indicators.start()
        status.set("indicators", "started")
    except Exception:
        status.set("indicators", "failed", error="failed to start sound indicators")
        logging.exception("main: failed to start sound indicators")

    audio: Optional[AudioAgent] = None
    speech_gate: Optional[SpeechGateAgent] = None
    control: Optional[ControlAgent] = None
    openclaw_input: Optional[OpenClawInputAgent] = None
    speaker: Optional[SpeakerAgent] = None

    if getattr(cfg, "audio", None):
        audio = AudioAgent()
        try:
            await audio.start()
            status.set("audio", "started", critical=True)
        except Exception:
            status.set("audio", "failed", critical=True, error="failed to start AudioAgent")
            logging.exception("main: failed to start AudioAgent")
            audio = None
    else:
        status.set("audio", "disabled", critical=True)

    speaker = SpeakerAgent()
    try:
        await speaker.start()
        status.set("speaker", "started", critical=bool(cfg.speaker.enabled))
    except Exception:
        status.set(
            "speaker",
            "failed",
            critical=bool(cfg.speaker.enabled),
            error="failed to start SpeakerAgent",
        )
        logging.exception("main: failed to start SpeakerAgent")
        speaker = None

    async def _interrupt_speaker_for_barge_in() -> int:
        if speaker is None:
            return 0
        return await speaker.interrupt(reason="barge_in")

    openclaw_input = OpenClawInputAgent(
        on_barge_in_interrupt=_interrupt_speaker_for_barge_in,
    )

    async def _handle_local_stop() -> int:
        dropped = 0
        if openclaw_input is not None:
            dropped += int(await openclaw_input.clear_pending_messages() or 0)
        if speaker is not None:
            dropped += int(await speaker.interrupt(reason="local_stop") or 0)
        return dropped

    speech_gate = SpeechGateAgent(
        on_local_stop=_handle_local_stop,
    )
    try:
        await speech_gate.start()
        status.set("speech_gate", "started", critical=True)
    except Exception:
        status.set(
            "speech_gate",
            "failed",
            critical=True,
            error="failed to start SpeechGateAgent",
        )
        logging.exception("main: failed to start SpeechGateAgent")
        speech_gate = None

    if speech_gate is not None:
        control = ControlAgent(
            speech_gate=speech_gate,
            speaker=speaker,
            status_provider=status.as_dict,
            shutdown_handler=_request_shutdown,
        )
        try:
            await control.start()
            status.set("control", "started")
        except Exception:
            status.set("control", "failed", error="failed to start ControlAgent")
            logging.exception("main: failed to start ControlAgent")
            control = None
    else:
        status.set("control", "disabled")

    try:
        await openclaw_input.start()
        status.set("openclaw_input", "started", critical=True)
    except Exception:
        status.set(
            "openclaw_input",
            "failed",
            critical=True,
            error="failed to start OpenClawInputAgent",
        )
        logging.exception("main: failed to start OpenClawInputAgent")
        openclaw_input = None

    startup_error: RuntimeError | None = None
    if bool(getattr(cfg.service, "strict_startup", False)) and not status.as_dict()["ready"]:
        startup_error = RuntimeError(f"strict startup failed: {status.as_dict()['last_error']}")
        logging.error("main: %s", startup_error)
        stop.set()

    key_task: Optional[asyncio.Task] = None
    if sys.platform.startswith("win"):
        try:
            import msvcrt  # type: ignore

            async def _console_quit_watcher() -> None:
                logging.info("main: key watcher started")
                while not stop.is_set():
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b"\x1b", b"q", b"Q"):
                            logging.info("main: quit key pressed")
                            stop.set()
                            break
                    await asyncio.sleep(0.02)

            key_task = asyncio.create_task(
                _console_quit_watcher(), name="console_quit_watcher"
            )
        except Exception:
            key_task = None

    loop = asyncio.get_running_loop()
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig:
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:
                pass

    try:
        logging.info("main: waiting for stop...")
        await stop.wait()
    finally:
        logging.info("main: stopping...")

        if key_task:
            try:
                await asyncio.wait_for(key_task, timeout=0.3)
            except asyncio.TimeoutError:
                key_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await key_task
            except asyncio.CancelledError:
                pass

        try:
            if openclaw_input:
                await openclaw_input.close()
        except Exception:
            pass

        try:
            if speaker:
                await speaker.close()
        except Exception:
            pass

        try:
            if control:
                await control.close()
        except Exception:
            pass

        try:
            if speech_gate:
                await speech_gate.close()
        except Exception:
            pass

        try:
            if audio:
                await audio.close()
        except Exception:
            pass

        try:
            await bus.stop()
        except Exception:
            pass

        try:
            await indicators.close()
        except Exception:
            pass

        if cfg.debug or cfg.info:
            logging.info("Listener voice runtime shutdown complete")
        if startup_error is not None:
            raise startup_error


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
        except Exception:
            pass
    asyncio.run(main())
