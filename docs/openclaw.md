# OpenClaw Integration

Listener integrates with OpenClaw in two directions:

1. Accepted voice phrases are sent to OpenClaw through `openclaw gateway call chat.send`.
2. OpenClaw can control Listener's SpeechGate mode through the bundled
   `listener-control` workspace skill and `utils/listenerctl.py`.
3. Listener can voice OpenClaw replies locally through the integrated Speaker
   agent and lets OpenClaw toggle spoken replies on or off.

The reply path is:

```text
OpenClaw Gateway chat events -> SpeakerAgent -> Piper -> local audio playback
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
GET  /speaker/status
POST /speaker/enabled
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
curl -s -X POST http://127.0.0.1:18790/speaker/enabled \
  -H 'Content-Type: application/json' \
  -d '{"enabled":false,"source":"curl","reason":"quiet"}' | jq
```

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

The repository `config/config.json` currently contains a machine-specific example
pointing at a sibling `/home/re/src/Speaker` checkout. On another machine you
should either replace those paths or set `speaker.enabled=false` until your
local Piper setup is ready.

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
rm -rf "$OPENCLAW_WORKSPACE/skills/listener-control"
cp -R openclaw/skills/listener-control "$OPENCLAW_WORKSPACE/skills/"
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
standby only with TTL, and normal to return to default filtering.
EOF
```

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

## Speaker Control Skill Mapping

The bundled `listener-control` skill exposes spoken-reply controls:

- "turn spoken replies off", "do not read answers aloud" -> `speaker off`
- "turn spoken replies on", "read answers aloud again" -> `speaker on`
- spoken reply / voice output status -> `speaker status`

After changing Speaker state, the skill should run `speaker status` and report
whether spoken replies are on or off.
