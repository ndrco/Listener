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
git tag -a v0.2.7 -m "Listener v0.2.7"
git push origin main --tags
```

Suggested release title:

```text
Listener v0.2.7 - OpenClaw gateway v4 and emoji display while speaker is off
```

Suggested release notes:

- Updated Listener's OpenClaw websocket clients to negotiate Gateway protocol
  v4 for both input-side forwarding and SpeakerAgent history/event reads.
- Kept emoji-display working even when spoken replies are disabled, so emoji can
  still appear on the external display while local TTS stays off.
- Added a loopback-only fallback for reading the local OpenClaw gateway token,
  which fixes `device identity required` failures for Listener's websocket path.
- Added an explicit PATH entry for `/home/re/.local/bin` in the Listener
  systemd template so CLI fallback can still find `openclaw` when needed.
- Expanded regression coverage for Gateway v4 compatibility and emoji-display
  behavior with `speaker.enabled=false`.
- Bumped runtime version to `0.2.7`.
