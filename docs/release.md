# Release Checklist

Use this checklist before publishing the first GitHub release.

## Repository Hygiene

- `git status --short` contains only intentional release changes.
- No model weights, `.venv`, caches, local `.openclaw`, or machine-specific paths are tracked.
- `README.md`, `INSTALL.md`, `docs/audio.md`, `docs/stt.md`, and
  `docs/openclaw.md` describe the current behavior.
- `LICENSE` is present.

## Verification

```bash
. .venv/bin/activate
python -m py_compile main.py agents/control_agent.py agents/openclaw_input_agent.py \
  agents/speaker_agent.py agents/speech_gate_agent.py audio/ducking.py \
  llm/speech_gate.py speaker/*.py utils/listenerctl.py
python -m pytest -q
```

Manual smoke:

```bash
.venv/bin/python main.py
curl -s http://127.0.0.1:18790/ | jq
.venv/bin/python utils/listenerctl.py speech-gate set-mode chatty --ttl 10
.venv/bin/python utils/listenerctl.py speech-gate status
.venv/bin/python utils/listenerctl.py speaker status
.venv/bin/python utils/listenerctl.py speaker off
.venv/bin/python utils/listenerctl.py speaker on
```

## Suggested First Tag

```bash
git tag -a v0.1.0 -m "Listener v0.1.0"
git push origin main --tags
```

Suggested release title:

```text
Listener v0.1.0 - Linux voice runtime for OpenClaw
```

Suggested release notes:

- Linux-first microphone, VAD, STT and AEC loopback pipeline.
- Whisper STT integration with async executor isolation.
- SpeechGate directed-speech filtering with OpenClaw identity discovery.
- Runtime SpeechGate control API and `listenerctl`.
- OpenClaw workspace skill for voice-mode control.
- Integrated Speaker playback for OpenClaw replies, with ducking and runtime control.
