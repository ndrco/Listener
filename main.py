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


async def main() -> None:
    load()
    configure_logging(debug=cfg.debug, info=cfg.info)
    if cfg.debug:
        logging.info("Listener voice runtime starting in DEBUG mode")
    elif cfg.info:
        logging.info("Listener voice runtime starting")

    bus.start()

    try:
        await indicators.start()
    except Exception:
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
        except Exception:
            logging.exception("main: failed to start AudioAgent")
            audio = None

    speaker = SpeakerAgent()
    try:
        await speaker.start()
    except Exception:
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
    except Exception:
        logging.exception("main: failed to start SpeechGateAgent")
        speech_gate = None

    if speech_gate is not None:
        control = ControlAgent(speech_gate=speech_gate, speaker=speaker)
        try:
            await control.start()
        except Exception:
            logging.exception("main: failed to start ControlAgent")
            control = None

    try:
        await openclaw_input.start()
    except Exception:
        logging.exception("main: failed to start OpenClawInputAgent")
        openclaw_input = None

    stop = asyncio.Event()

    async def _on_app_stop(ev) -> None:
        logging.info(
            "main: got app/stop from %s (%s)",
            ev.payload.get("source"),
            ev.payload.get("reason"),
        )
        stop.set()

    bus.subscribe(cfg.events.app.stop, _on_app_stop)

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


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
        except Exception:
            pass
    asyncio.run(main())
