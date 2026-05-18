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
git tag -a v0.2.5 -m "Listener v0.2.5"
git push origin main --tags
```

Suggested release title:

```text
Listener v0.2.5 - Emoji display simplification and import resilience
```

Suggested release notes:

- Simplified emoji display delivery: Listener now sends only the latest emoji
  token from a segment and defaults `speaker.emoji_display.send` to `last`.
- Kept backward compatibility for existing configs by accepting legacy `send=all`
  and mapping it to `last`.
- Made `audio` and `llm` package exports lazy so control, Speaker, and status
  code paths can import cleanly before optional ML dependencies are loaded.
- Changed `torch`-backed SpeechGate and Silero helpers to fail only when those
  features are instantiated, rather than during unrelated module imports.
- Expanded README and test coverage for the new emoji-display behavior.
- Bumped runtime version to `0.2.5`.
