---
name: listener-tts-file
description: Create a local WAV file from text through Listener's already-running VoxCPM2 or CosyVoice3 model. Use when the user asks OpenClaw to synthesize, narrate, or save text as a spoken audio file.
---

# Listener TTS File

Use the bundled helper resolved from this skill's installed directory. Never
run bare `scripts/listener-tts-file` from the workspace root. The usual command
path is:

```bash
TTS_FILE_HELPER="${OPENCLAW_WORKSPACE:-$HOME/.openclaw/workspace}/skills/listener-tts-file/scripts/listener-tts-file"
```

If this `SKILL.md` was read from another workspace path, use the `scripts`
directory next to that exact file instead. The helper delegates to Listener's
local control API and never starts a model process of its own.

Render text and wait for the WAV file:

```bash
"$TTS_FILE_HELPER" render --text "🙂 Добро пожаловать!" --filename welcome --wait --json
```

For long text, write or reuse a UTF-8 text file and pass it without changing its
contents:

```bash
"$TTS_FILE_HELPER" render --text-file /path/to/text.txt --filename narration --wait --json
```

The completed job contains `output_path`. Return that path to the user or use it
for the next requested local-file operation. Do not invoke CosyVoice3/VoxCPM2
Python environments directly and do not launch another TTS worker.

## Style

Prefer one leading allowlisted emoji in the text. Listener removes it from the
spoken text and converts it to a safe model instruction:

- `🙂` / `😊` warm
- `😄` / `🎉` cheerful
- `😌` calm
- `🤔` thoughtful
- `😔` / `😢` sad and empathetic
- `😠` firm
- `😮` surprised
- `😏` playful
- `⚠️` urgent
- `😂` amused

An explicit allowlisted style is also available, for example `--style calm`.
Never pass arbitrary model instructions. Available styles are `neutral`, `warm`,
`cheerful`, `calm`, `thoughtful`, `sad`, `firm`, `surprised`, `playful`,
`urgent`, and `amused`.

## Job Operations

```bash
"$TTS_FILE_HELPER" list --json
"$TTS_FILE_HELPER" status JOB_ID --json
"$TTS_FILE_HELPER" cancel JOB_ID --json
```

If rendering is unavailable, report that Listener must be running with
`speaker.tts_mode=persistent` and `speaker.tts.backend=cosyvoice3` or `voxcpm2`.
