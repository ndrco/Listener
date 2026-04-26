---
name: listener-control
description: Control the local Listener voice runtime SpeechGate mode via listenerctl. Use when the user asks OpenClaw to change listening behavior, quiet mode, conversation mode, standby, or normal voice routing.
---

# Listener Control

Use this skill when the user wants to change how Listener accepts voice input.

## Commands

Assume `LISTENER_HOME` points to the Listener repo. If it is not set, check the
workspace `TOOLS.md` for the local Listener path.

```bash
$LISTENER_HOME/.venv/bin/python $LISTENER_HOME/utils/listenerctl.py speech-gate status
```

```bash
$LISTENER_HOME/.venv/bin/python $LISTENER_HOME/utils/listenerctl.py speech-gate set-mode normal
```

```bash
$LISTENER_HOME/.venv/bin/python $LISTENER_HOME/utils/listenerctl.py speech-gate set-mode mute --reason "quiet mode"
```

```bash
$LISTENER_HOME/.venv/bin/python $LISTENER_HOME/utils/listenerctl.py speech-gate set-mode chatty --ttl 600 --reason "conversation mode"
```

```bash
$LISTENER_HOME/.venv/bin/python $LISTENER_HOME/utils/listenerctl.py speech-gate set-mode standby --ttl 300 --reason "standby requested"
```

## Intent Mapping

- Conversation mode, listen to everything, no wake name required, active listening on -> `chatty --ttl 600` unless the user gives a duration.
- Quiet mode, name-only mode, stop listening to background speech, active listening off -> `mute`.
- Do not listen, stop listening completely, standby mode, go to standby -> `standby --ttl 300` unless the user gives a duration.
- Normal mode, come back, listen normally, leave active listening mode -> `normal`.
- If the user asks about listening activity/status, run `speech-gate status` first and report the current mode.

After changing mode, run `speech-gate status` and summarize the resulting mode
briefly. Do not use `standby` without `--ttl`.
