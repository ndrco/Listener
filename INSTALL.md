# Installation

This guide covers a fresh Linux setup for Listener and the optional OpenClaw
integration. Windows is supported through the sample config in
`config/config.windows.example.json`, but Linux PipeWire/PulseAudio is the
primary tested path.

## 1. System Requirements

- Python 3.12
- PortAudio runtime and headers
- PipeWire or PulseAudio for Linux loopback/AEC
- `pactl` for device/source diagnostics
- Optional: NVIDIA driver with CUDA 12.8-compatible PyTorch for GPU STT
- Optional: OpenClaw CLI in `PATH`

Ubuntu/Debian packages:

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv python3-pip \
  libportaudio2 portaudio19-dev pulseaudio-utils jq
```

## 2. Clone and Create `.venv`

```bash
git clone <repository-url>
cd Listener
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

The default `requirements.txt` is tuned for CUDA 12.8 PyTorch. On a CPU-only
machine you can still run the audio pipeline, but set STT and speech-gate model
devices to `cpu` in `config/config.json`.

## 3. Models

Model weights are intentionally not tracked in git. The default config expects:

- Silero VAD: `models/silero_vad_v6.jit`
- Speech-gate classifier: `models/directed-ruElectra-small-fp16`
- Whisper cache/root: `models/whisper`

Download Silero VAD:

```bash
.venv/bin/python utils/silero_vad_model_downloader.py
```

For Whisper, either:

- place a compatible local snapshot under `models/whisper`;
- set `audio.stt.local_files_only=false` for the first model download;
- or temporarily set `audio.stt.enabled=false` while testing the rest of the app.

For a CPU-only first run, use:

```json
{
  "speech_gate": {
    "model": {
      "device": "cpu"
    }
  },
  "audio": {
    "stt": {
      "device": "cpu",
      "compute_type": "int8"
    }
  }
}
```

## 4. Audio Devices

List all audio devices and PipeWire/PulseAudio sources:

```bash
.venv/bin/python utils/list_devices.py
```

List only monitor/loopback candidates:

```bash
.venv/bin/python utils/list_devices.py --monitors
```

For Linux AEC, `config/config.json` can use Pulse/PipeWire source aliases:

```json
{
  "audio": {
    "processing": {
      "aec": {
        "enabled": true,
        "playback_source": "loopback",
        "loopback_backend": "auto",
        "loopback_source_name": "@DEFAULT_MONITOR@"
      }
    }
  }
}
```

Quick microphone check:

```bash
.venv/bin/python utils/meter_fast.py --duration 10
```

AEC check:

```bash
.venv/bin/python utils/AEC_meter.py --aec --pulse \
  --mic-source @DEFAULT_SOURCE@ \
  --loopback-source @DEFAULT_MONITOR@ \
  --duration 30
```

## 5. Run Listener

```bash
.venv/bin/python main.py
```

In another terminal, check the local control API:

```bash
curl -s http://127.0.0.1:18790/ | jq
curl -s http://127.0.0.1:18790/speech-gate/status | jq
```

Switch SpeechGate to temporary chatty mode:

```bash
.venv/bin/python utils/listenerctl.py chatty --ttl 60
```

Return to normal:

```bash
.venv/bin/python utils/listenerctl.py normal
```

## 6. OpenClaw Integration

Listener sends accepted phrases to OpenClaw through:

```bash
openclaw gateway call chat.send
```

Minimal config:

```json
{
  "openclaw": {
    "enabled": true,
    "command": "openclaw",
    "source_topic": "llm/accepted_phrase",
    "session_key": "main"
  }
}
```

Install the Listener control skill into the active OpenClaw workspace:

```bash
OPENCLAW_WORKSPACE="$(openclaw config get agents.defaults.workspace)"
mkdir -p "$OPENCLAW_WORKSPACE/skills"
rm -rf "$OPENCLAW_WORKSPACE/skills/listener-control"
cp -R openclaw/skills/listener-control "$OPENCLAW_WORKSPACE/skills/"
```

Add local Listener notes to OpenClaw `TOOLS.md`:

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

After that, OpenClaw can use the `listener-control` skill for phrases like:

- "turn on conversation mode" -> `chatty --ttl 600`
- "quiet mode" -> `mute`
- "do not listen for five minutes" -> `standby --ttl 300`
- "return to normal listening" -> `normal`
- "turn active listening on" -> `chatty --ttl 600`
- "turn active listening off" -> `mute`

Listener also reads OpenClaw assistant identity from the OpenClaw workspace
`IDENTITY.md` automatically. Supported keys are `Name:` and `Имя:`.

## 7. Tests

```bash
. .venv/bin/activate
python -m pytest -q
```

Expected result for the current release prep:

```text
52 passed
```
