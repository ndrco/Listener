# Release Workflow

Use this checklist before publishing a Listener release.

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

## Next Tag

```bash
git tag -a v0.2.6 -m "Listener v0.2.6"
git push origin main --tags
```

Suggested release title:

```text
Listener v0.2.6 - Speaker voice controls and ducking recovery
```

Suggested release notes:

- Added local voice commands to disable and re-enable spoken replies with
  phrases like `"<Имя> выключи озвучку"` and `"<Имя> включи озвучку"`.
- Added an OpenClaw `listener-speaker-off` skill/tool for turning Listener
  speech output off from the workspace.
- Fixed ducking recovery so persisted baseline volumes survive stream
  recreation and short sink-input disappearance during or after speech.
- Taught forced ducking recovery to match streams by route key, not only by the
  old sink-input id, which helps Chrome/PipeWire sessions recover their volume.
- Expanded docs and regression coverage for speaker voice controls and ducking
  restoration.
- Bumped runtime version to `0.2.6`.
