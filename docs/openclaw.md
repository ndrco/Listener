# OpenClaw Integration

Listener integrates with OpenClaw in two directions:

1. Accepted voice phrases are sent to OpenClaw through `openclaw gateway call chat.send`.
2. OpenClaw can control Listener's SpeechGate mode through the bundled
   `listener-control` workspace skill and `utils/listenerctl.py`.

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
GET  /health
GET  /speech-gate/status
POST /speech-gate/mode
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
```

Supported modes:

- `normal` - regular directed-speech filtering.
- `mute` - only direct assistant-name calls pass.
- `chatty` - all non-empty phrases pass.
- `standby` - all phrases are blocked; TTL is required.

`chatty` and other temporary modes are evaluated by STT segment start time, so a
phrase that started inside a TTL window still uses that mode even if Whisper
finishes after the TTL expires.

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
- Use: \$LISTENER_HOME/.venv/bin/python \$LISTENER_HOME/utils/listenerctl.py
EOF
```

Run the command from the Listener repository root so `LISTENER_HOME` is written
as the current project path.

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
```

`listenerctl` reads:

- `LISTENER_CONTROL_URL`
- `LISTENER_CONTROL_TOKEN`

If the control API is exposed on anything other than loopback, configure a
non-empty `control.token`.
