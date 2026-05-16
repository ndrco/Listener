---
name: listener-control
description: Control the local Listener voice runtime SpeechGate mode via listenerctl. Use when the user asks OpenClaw to change listening behavior, quiet mode, conversation mode, standby, normal voice routing, or current listening status.
---

# Listener Control

Use this skill when the user wants to change or inspect how Listener accepts voice input.

## Preferred Command

Use the bundled helper script `scripts/listener-control` (resolve relative to this SKILL.md).
It discovers:

- `LISTENER_HOME` from env, OpenClaw workspace `TOOLS.md`, the source skill path, or common local paths.
- control URL from `LISTENER_CONTROL_URL`, `TOOLS.md` (`Control URL:`), or Listener `config/config.json`.
- control token from `LISTENER_CONTROL_TOKEN`, `TOOLS.md`, or Listener `config/config.json`.

Examples:

```bash
scripts/listener-control status
```

```bash
scripts/listener-control speaker status
```

```bash
scripts/listener-control normal --reason "normal listening"
```

```bash
scripts/listener-control mute --reason "quiet mode"
```

```bash
scripts/listener-control chatty --ttl 600 --reason "conversation mode"
```

```bash
scripts/listener-control standby --reason "standby requested"
```

```bash
scripts/listener-control speaker off --reason "disable spoken replies"
```

```bash
scripts/listener-control speaker on --reason "enable spoken replies"
```

The helper delegates to `listenerctl.py`. Direct commands also work when `LISTENER_HOME` is known, e.g. `$LISTENER_HOME/.venv/bin/python $LISTENER_HOME/utils/listenerctl.py status`.

## Intent Mapping

- Conversation mode, listen to everything, no wake name required, active listening on -> `chatty --ttl 600` unless the user gives a duration.
- Quiet mode, name-only mode, stop listening to background speech, active listening off -> `mute`.
- Do not listen, stop listening completely, standby mode, go deaf -> `standby`. If the user gives a duration you can use `--ttl`, for example `--ttl 600`.
- Normal mode, come back, listen normally, leave active listening mode -> `normal`.
- Disable spoken replies, turn voice output off, stop reading answers aloud, do not speak answers -> `speaker off`.
- Enable spoken replies, turn voice output on, read answers aloud again -> `speaker on`.
- If the user asks about spoken reply / voice output status, run `speaker status`.
- If the user asks about listening activity/status, run `status` first and report the current mode.

After changing mode, run `status` and summarize the resulting mode briefly. The CLI output includes mode, permanent/temporary state, expiry time, and restore mode. Do not use `chatty` without `--ttl`.
After changing Speaker state, run `speaker status` and summarize whether spoken replies are on or off.
