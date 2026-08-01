# OpenClaw Integration

[Русская версия](openclaw_RUS.md)

Listener integrates with OpenClaw in several directions:

1. Accepted voice phrases are sent to OpenClaw through `openclaw gateway call chat.send`.
2. OpenClaw can control Listener's SpeechGate mode through the bundled
   `listener-control` workspace skill and `utils/listenerctl.py`.
3. Listener can voice OpenClaw replies locally through the integrated Speaker
   agent and lets OpenClaw toggle spoken replies on or off.
4. OpenClaw can ask the running neural TTS worker to save text as a local WAV
   file through the bundled `listener-tts-file` skill.

The reply path is:

```text
OpenClaw Gateway chat events -> SpeakerAgent -> selected TTS -> local audio playback
OpenClaw skill -> Listener control API -> existing neural worker -> WAV file
```

## Sending Voice Phrases to OpenClaw

Enable OpenClaw in `config/config.json`:

```json
{
  "openclaw": {
    "enabled": true,
    "command": "openclaw",
    "source_topic": "llm/accepted_phrase",
    "session_key": "main",
    "gateway_url": null,
    "gateway_token": null
  }
}
```

On Windows with OpenClaw inside WSL, use
`config/config.windows.example.json` as the starting point.

## Assistant Name / Identity

The speech gate does not keep assistant names in
`config/speech_gate_patterns.json`. Instead, Listener auto-discovers OpenClaw's
workspace identity file:

- `OPENCLAW_IDENTITY_FILE`
- `OPENCLAW_WORKSPACE/IDENTITY.md`
- `OPENCLAW_STATE_DIR/workspace/IDENTITY.md`
- `OPENCLAW_CONFIG_PATH`
- `~/.openclaw/openclaw.json`
- `~/.openclaw-dev/openclaw.json`
- `~/.openclaw-*/openclaw.json`

The identity file should contain one of:

```markdown
Name: Marina
Имя: Марина
```

If auto-discovery is not correct, set:

```json
{
  "speech_gate": {
    "identity_file": "/path/to/openclaw/workspace/IDENTITY.md"
  }
}
```

## Runtime Control API

Listener starts a local HTTP control API when `control.enabled=true`:

```text
GET  /
GET  /health
GET  /speech-gate/status
POST /speech-gate/mode
POST /speech-gate/reset
GET  /speaker/status
POST /speaker/enabled
POST /tts/files
GET  /tts/files
GET  /tts/files/{job_id}
POST /tts/files/{job_id}/cancel
```

Example:

```bash
curl -s http://127.0.0.1:18790/speech-gate/status | jq
```

Switch modes:

```bash
curl -s -X POST http://127.0.0.1:18790/speech-gate/mode \
  -H 'Content-Type: application/json' \
  -d '{"mode":"chatty","ttl_seconds":600,"source":"curl"}' | jq
curl -s -X POST http://127.0.0.1:18790/speech-gate/reset \
  -H 'Content-Type: application/json' \
  -d '{"source":"curl","reason":"recover voice"}' | jq
curl -s -X POST http://127.0.0.1:18790/speaker/enabled \
  -H 'Content-Type: application/json' \
  -d '{"enabled":false,"source":"curl","reason":"quiet"}' | jq
curl -s -X POST http://127.0.0.1:18790/tts/files \
  -H 'Content-Type: application/json' \
  -d '{"text":"😔 Иногда тишина всё объясняет.","filename":"thought"}' | jq
```

`POST /speech-gate/reset` is a recovery endpoint for the rare case where a
stuck barge-in or interrupted run leaves Listener's own voice cues ducked. It
forces `speech_gate` back to `normal`, re-enables `speaker`, interrupts active
reply playback, and restores tracked sink-input volumes.

Supported modes:

- `normal` - regular directed-speech filtering.
- `mute` - only direct assistant-name calls pass.
- `chatty` - all non-empty phrases pass.
- `standby` - all phrases are blocked; TTL is required.

`chatty` and other temporary modes are evaluated by STT segment start time, so a
phrase that started inside a TTL window still uses that mode even if Whisper
finishes after the TTL expires.

## Speaker Runtime

For spoken replies to work, Listener needs:

- OpenClaw Gateway reachable at `speaker.gateway.url` and matching `speaker.gateway.session_key`;
- `websockets` installed in Listener `.venv` via `requirements.txt`;
- `piper` available through `speaker.piper.command`;
- a valid voice model at `speaker.piper.model`;
- a working playback command such as `/usr/bin/paplay`.

Optional emoji display support is configured under `speaker.emoji_display`.
Listener removes emoji from the text sent to Piper, then forwards extracted
symbols to an external HTTP service such as the sibling `emoji-display` project.
The hardware/COM connection intentionally stays outside Listener.

In streaming mode, Listener speaks complete sentence-like chunks as they arrive
from OpenClaw and performs a short final `chat.history` check to recover any
tail that was visible in the UI but absent from the gateway deltas. With
`speaker.tts_mode="persistent"` the selected worker stays warm. Piper WAV
playback uses `speaker.playback.backend`; neural PCM uses the separate
`speaker.playback.streaming_backend`. Linux `auto` prefers an isolated `pacat`
process and reuses it for all sentence segments in the same OpenClaw run.

Speaker ducking is per OpenClaw run. For neural TTS it starts only after the
one-time PCM prebuffer is ready and is restored after the isolated player drains
the last queued segment.
If an interrupt or barge-in leaves audio quiet, call
`listenerctl.py speech_gate_reset` or `systemctl --user reload listener.service`
to reset SpeechGate/Speaker and restore remembered PipeWire/PulseAudio volumes.

The default Piper command uses Listener's own `.venv`; its default model is
`models/ru_RU-irina-medium.onnx`. Place the matching JSON sidecar next to it, or
set `speaker.enabled=false` until the local Piper setup is ready.

Useful runtime checks:

```bash
.venv/bin/python utils/listenerctl.py speaker status
curl -s http://127.0.0.1:18790/speaker/status | jq
```

Key status fields:

- `speaker=on|off` - whether spoken replies are enabled;
- `agent=running|stopped` - whether `SpeakerAgent` is alive inside Listener;
- `gateway=connected|disconnected` - whether Listener is subscribed to OpenClaw Gateway;
- `queue` and `current` - queued or active speech segments;
- `last_interrupt` - last stop/barge-in reason;
- `error` - last gateway, Piper, or playback failure.

## Local Voice Commands

Listener can also intercept a few assistant-name voice commands locally before
the phrase is forwarded to OpenClaw:

- `Имя, помолчи` -> switches SpeechGate to `mute`
- `Имя, говори` -> switches SpeechGate to `normal`
- `Имя, отключись` -> switches SpeechGate to `standby`
- `Имя, включи озвучку` or `Имя, верни озвучку` -> enables spoken replies
- `Имя, отключи озвучку` or `Имя, выключи озвучку` -> disables spoken replies
- `Имя, стоп` -> calls OpenClaw `chat.abort` for the configured `openclaw.session_key`

These local commands are intentionally swallowed by Listener and are not sent as
regular `chat.send` input. OpenClaw's own control skill is still useful for
typed commands, richer mode changes such as temporary `chatty`, and manual
inspection through `listenerctl`.

When integrated Speaker is enabled, `Имя, стоп` also interrupts current TTS
playback and clears queued speech. Explicit barge-in phrases forwarded through
`sessions.steer` interrupt Speaker playback before the steer request waits for
OpenClaw.

## Speaker Troubleshooting

Start Listener with DEBUG logging when you need to trace lost or interrupted
spoken replies:

```bash
.venv/bin/python main.py 2>&1 | tee /tmp/listener-speaker.log
```

Look for the Speaker chain:

```bash
rg "SpeakerAgent: (connected|final event needs history check|history check produced|queued speech segment|speaking assistant reply|speech failed|interrupted|dropped)" /tmp/listener-speaker.log
rg "EmojiDisplay|extracted .* emoji|emoji-only" /tmp/listener-speaker.log
```

How to read it:

- `history check produced ... final segment(s)` means Listener had to recover a final tail from `chat.history`;
- `queued speech segment` means the text reached Speaker's playback queue;
- `speaking assistant reply` means Piper/playback started;
- `interrupted` means local stop, barge-in, or OpenClaw abort cleared playback;
- `speech failed` points to Piper or playback command failures.

This is the main workflow for bugs where the last sentence is visible in
OpenClaw but not spoken locally.

## Install the OpenClaw Skill

From the Listener repository:

```bash
OPENCLAW_WORKSPACE="$(openclaw config get agents.defaults.workspace)"
mkdir -p "$OPENCLAW_WORKSPACE/skills"
for skill in listener-control listener-speaker-off listener-tts-file; do
  rm -rf "$OPENCLAW_WORKSPACE/skills/$skill"
  cp -R "openclaw/skills/$skill" "$OPENCLAW_WORKSPACE/skills/"
done
```

Add local path notes to OpenClaw `TOOLS.md`:

```bash
OPENCLAW_WORKSPACE="$(openclaw config get agents.defaults.workspace)"
cat >> "$OPENCLAW_WORKSPACE/TOOLS.md" <<EOF

### Listener
- LISTENER_HOME=$(pwd)
- Control URL: http://127.0.0.1:18790
- Use: $OPENCLAW_WORKSPACE/skills/listener-control/scripts/listener-control
EOF
```

Run the command from the Listener repository root so `LISTENER_HOME` is written
as the current project path. The skill helper also falls back to env variables,
Listener `config/config.json`, and common local paths.

Optionally add a short persistent note to OpenClaw `AGENTS.md` so the agent
recognizes that some chat messages may arrive from Listener as voice
transcripts:

```bash
OPENCLAW_WORKSPACE="$(openclaw config get agents.defaults.workspace)"
cat >> "$OPENCLAW_WORKSPACE/AGENTS.md" <<'EOF'

## Listener Voice Input

Some messages may come from Listener as voice transcripts through OpenClaw
chat.send. When the user asks to change listening behavior, use the
listener-control skill: chatty for conversation mode, mute for name-only mode,
standby only with TTL, and normal to return to default filtering. For disabling
spoken replies, use listener-speaker-off or `listener-control speaker off`.
EOF
```

If Listener uses VoxCPM2 or CosyVoice3, also install the response-style
convention:

```bash
OPENCLAW_WORKSPACE="$(openclaw config get agents.defaults.workspace)"
sed -n '1,$p' openclaw/prompts/listener-tts-style.md >> "$OPENCLAW_WORKSPACE/AGENTS.md"
```

The leading position is intentional: Listener queues completed sentences while
OpenClaw is still streaming, so a trailing emoji arrives too late to style that
sentence. Styles are selected by an allowlist and persist only inside the same
OpenClaw `runId`; abort, error, finalization, and barge-in clear run state.

## Manual `listenerctl` Commands

```bash
.venv/bin/python utils/listenerctl.py status
.venv/bin/python utils/listenerctl.py chatty --ttl 600
.venv/bin/python utils/listenerctl.py mute
.venv/bin/python utils/listenerctl.py standby --ttl 300
.venv/bin/python utils/listenerctl.py normal
.venv/bin/python utils/listenerctl.py speaker status
.venv/bin/python utils/listenerctl.py speaker off
.venv/bin/python utils/listenerctl.py speaker on
.venv/bin/python utils/listenerctl.py tts-file render \
  --text '🙂 Добро пожаловать!' --filename welcome --wait
.venv/bin/python utils/listenerctl.py tts-file list
```

`listenerctl` reads:

- `LISTENER_CONTROL_URL`
- `LISTENER_CONTROL_TOKEN`

Its human-readable output includes the current mode, whether it is temporary or
permanent, the expiry time when temporary, and the restore mode.

The OpenClaw skill helper (`openclaw/skills/listener-control/scripts/listener-control`)
also discovers `LISTENER_HOME`, control URL, and control token from env,
OpenClaw `TOOLS.md`, or Listener `config/config.json` before delegating to
`listenerctl.py`.

If the control API is exposed on anything other than loopback, configure a
non-empty `control.token`.

## OpenClaw File-render Skill

The bundled `listener-tts-file` skill creates WAV files through the already
running VoxCPM2 or CosyVoice3 worker. It delegates discovery of Listener home,
control URL and token to the installed `listener-control` helper. It never
activates a model environment or starts a second worker.

Typical skill command:

```bash
openclaw/skills/listener-tts-file/scripts/listener-tts-file render \
  --text '😌 Спокойный текст для записи.' \
  --filename calm-note \
  --wait --json
```

On completion, the returned job contains `output_path`. A leading allowlisted
emoji selects the same fixed safe style instruction as live replies. An
explicit style can be selected with `--style calm`; arbitrary instructions are
not accepted. The output directory, text limit, queue limit and segment size
are configured under `speaker.file_render`. See
[`neural-tts.md`](neural-tts.md#render-wav-files-without-another-tts-process)
for lifecycle, scheduling, disk overhead and failure behavior.

## Speaker Control Skill Mapping

The bundled `listener-control` skill exposes spoken-reply controls:

- "turn spoken replies off", "do not read answers aloud" -> `speaker off`
- "turn spoken replies on", "read answers aloud again" -> `speaker on`
- spoken reply / voice output status -> `speaker status`

For the narrow "disable speech right now" case, the bundled
`listener-speaker-off` skill calls the dedicated helper
`scripts/listener-speaker-off`.

After changing Speaker state, the skill should run `speaker status` and report
whether spoken replies are on or off.
