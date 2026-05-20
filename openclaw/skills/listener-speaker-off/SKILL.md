---
name: listener-speaker-off
description: Disable Listener spoken replies. Use when the user asks OpenClaw to stop reading answers aloud, turn voice output off, or disable local TTS replies.
---

# Listener Speaker Off

Use this skill when the user specifically wants Listener to stop speaking
assistant replies aloud.

## Preferred Command

Use the bundled helper script `scripts/listener-speaker-off` (resolve relative
to this SKILL.md).

Example:

```bash
scripts/listener-speaker-off
```

This helper delegates to the existing `listener-control` skill, turns Speaker
off, then prints `speaker status` so you can confirm the result.

## Intent Mapping

- Turn spoken replies off
- Disable voice output
- Stop reading answers aloud
- Do not speak replies
- Mute Listener TTS

After running the helper, briefly report whether spoken replies are now off.
