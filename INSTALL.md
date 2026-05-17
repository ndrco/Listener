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
- Optional: `paplay` playback command for integrated Speaker

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

`requirements-optional.txt` now also installs `piper-tts`, which is used by the
integrated Speaker when spoken replies are enabled.

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

For Speaker, provide a Piper model `.onnx` file and make sure
`speaker.piper.command` and `speaker.piper.model` point to real local paths.
The repository config includes an example wired to a sibling `/home/re/src/Speaker`
checkout; on another machine you should either replace those paths or disable
Speaker for the first run:

```json
{
  "speaker": {
    "enabled": false
  }
}
```

If you want a self-contained Listener setup, use the Listener virtualenv as the
Piper entrypoint:

```json
{
  "speaker": {
    "enabled": true,
    "piper": {
      "command": ".venv/bin/python3",
      "model": "/absolute/path/to/voice-model.onnx"
    }
  }
}
```

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
curl -s http://127.0.0.1:18790/speaker/status | jq
```

Switch SpeechGate to temporary chatty mode:

```bash
.venv/bin/python utils/listenerctl.py chatty --ttl 60
```

Return to normal:

```bash
.venv/bin/python utils/listenerctl.py normal
```

Check Speaker runtime state:

```bash
.venv/bin/python utils/listenerctl.py speaker status
```

## 6. Install As A Service

Run Listener manually once before installing it as a service. This makes audio
device, model, and OpenClaw configuration errors much easier to see directly in
the terminal.

After the manual smoke test works, install a Linux `systemd --user` service for
the current checkout:

```bash
.venv/bin/python utils/install_user_service.py
systemctl --user start listener.service
```

Check liveness and readiness:

```bash
.venv/bin/python utils/listenerctl.py health
.venv/bin/python utils/listenerctl.py ready
```

Follow logs:

```bash
journalctl --user -u listener.service -f
```

Gracefully stop the running service:

```bash
.venv/bin/python utils/listenerctl.py stop --reason manual
```

For the full service workflow, including start-on-login, restart, uninstall,
custom checkout paths, and strict startup mode, see
[docs/service.md](docs/service.md).

## 7. OpenClaw Integration

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

After that, OpenClaw can use the `listener-control` skill for phrases like:

- "turn on conversation mode" -> `chatty --ttl 600`
- "quiet mode" -> `mute`
- "do not listen for five minutes" -> `standby --ttl 300`
- "return to normal listening" -> `normal`
- "turn active listening on" -> `chatty --ttl 600`
- "turn active listening off" -> `mute`

Listener also reads OpenClaw assistant identity from the OpenClaw workspace
`IDENTITY.md` automatically. Supported keys are `Name:` and `Имя:`.

Spoken replies also depend on OpenClaw Gateway chat events. A healthy state is:

- OpenClaw Gateway reachable at `speaker.gateway.url`;
- `listenerctl speaker status` shows `agent=running gateway=connected`;
- no `error=...` field in the status output.

## 8. Tests

```bash
. .venv/bin/activate
python -m pytest -q
```

Expected result: the full test suite passes.
