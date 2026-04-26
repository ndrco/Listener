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

- "режим разговора", "слушай все", "можно без обращения по имени" -> `chatty --ttl 600` unless the user gives a duration.
- "тихий режим", "реагируй только на имя" -> `mute`.
- "не слушай", "уйди в standby", "режим ожидания" -> `standby --ttl 300` unless the user gives a duration.
- "обычный режим", "вернись", "слушай нормально" -> `normal`.

After changing mode, run `speech-gate status` and summarize the resulting mode
briefly. Do not use `standby` without `--ttl`.
