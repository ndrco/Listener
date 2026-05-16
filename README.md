# Listener

`Listener` is a local voice-input runtime for OpenClaw. It captures microphone
audio, applies audio processing/VAD/AEC, transcribes speech with Whisper, filters
phrases by directed-speech rules, and forwards accepted text to OpenClaw.

The project is Linux-first, with Windows compatibility kept through the
platform-specific sample config in `config/config.windows.example.json`.

## Features

- Microphone capture through `sounddevice`.
- Linux PipeWire/PulseAudio loopback for AEC monitor sources.
- Windows WASAPI loopback compatibility.
- LiveKit AEC, optional NS/HPF/AGC, custom noise suppression and VAD.
- Hybrid VAD pipeline: WebRTC + Silero.
- Whisper STT through `faster-whisper`.
- SpeechGate filtering with assistant name loaded from OpenClaw `IDENTITY.md`.
- Runtime SpeechGate control API and `utils/listenerctl.py`.
- Optional short audio indicators for rejected, forwarded and local control events.
- Integrated local Speaker playback for OpenClaw replies through Piper.
- Bundled OpenClaw workspace skill: `openclaw/skills/listener-control`.

## Pipeline

```text
Microphone -> AudioProcessor -> BufferedSpeechWriter -> WhisperStreamingTranscriber
           -> llm/input_text -> SpeechGateAgent -> llm/accepted_phrase
           -> OpenClawInputAgent -> OpenClaw
```

Reply path when spoken replies are enabled:

```text
OpenClaw Gateway -> SpeakerAgent -> PiperSpeechEngine -> paplay
```

Core modules:

- `agents/` - runtime orchestration.
- `audio/` - microphone, processing, buffering and STT.
- `core/` - config, event bus and logging.
- `llm/` - SpeechGate directed-speech logic.
- `utils/` - diagnostics and control CLI.
- `openclaw/skills/` - OpenClaw workspace skill bundle.

## Quick Start

See [INSTALL.md](INSTALL.md) for the full setup guide.

```bash
git clone <repository-url>
cd Listener
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-optional.txt
python utils/silero_vad_model_downloader.py
python utils/list_devices.py
python main.py
```

In another terminal:

```bash
curl -s http://127.0.0.1:18790/ | jq
.venv/bin/python utils/listenerctl.py status
```

## Configuration

Main runtime settings live in `config/config.json`.

Important sections:

- `control` - local runtime HTTP API for SpeechGate mode control.
- `openclaw` - OpenClaw CLI/gateway forwarding settings.
- `speaker` - integrated OpenClaw reply TTS, Piper playback and TTS ducking.
- `indicators` - short notification tones for SpeechGate/OpenClaw events.
- `speech_gate` - directed-speech rules, identity file and classifier settings.
- `audio.input` - microphone sample rate, channels, chunk size and device.
- `audio.processing` - AEC, VAD, AGC, high-pass and noise suppression.
- `audio.buffer` - speech segment buffering before STT.
- `audio.stt` - Whisper model and decoding settings.
- `events` - internal EventBus topic names.

The repository config currently contains a local-machine `speaker` example with
paths under `/home/re/src/Speaker`. On a fresh clone you should either:

- set `speaker.enabled=false` for the first run; or
- replace `speaker.piper.command`, `speaker.piper.model`, and if needed
  `speaker.gateway.*` with values valid on your machine.

The primary SpeechGate pattern source is `config/speech_gate_patterns.json`.
Inline pattern arrays in `config/config.json` are supported as overrides, but
the default project config keeps them empty to avoid duplicate definitions.

When `indicators.enabled=true`, Listener plays short tones for four cases:

- phrase rejected by SpeechGate;
- phrase forwarded into OpenClaw;
- local voice command handled inside Listener;
- successful interruption/stop sent into OpenClaw.

Each of these can be toggled independently through `indicators.rejected`,
`indicators.forwarded`, `indicators.local_handled`, and
`indicators.interrupted`.

`indicators.ducking` can lower other PulseAudio/PipeWire streams while each
short tone is playing. Unlike Speaker's own TTS ducking, indicator ducking also
ducks any currently playing Speaker stream, so local stop/interruption beeps are
audible.

The default `requirements.txt` is tuned for CUDA 12.8 PyTorch. For CPU-only
machines, set `audio.stt.device="cpu"` and `speech_gate.model.device="cpu"` in
`config/config.json`.

## Models

Model weights are intentionally not tracked in git.

Default expected paths:

- `models/silero_vad_v6.jit`
- `models/directed-ruElectra-small-fp16`
- `models/whisper`
- `config/blacklist.txt`

Download Silero VAD:

```bash
.venv/bin/python utils/silero_vad_model_downloader.py
```

For Whisper, either place a local snapshot under `models/whisper`, temporarily
set `audio.stt.local_files_only=false`, or disable STT while testing the rest of
the pipeline.

For integrated Speaker, Listener also needs a Piper model path and a working
Piper entrypoint. The simplest self-contained setup is:

- install `requirements-optional.txt` into Listener `.venv`;
- set `speaker.piper.command` to `.venv/bin/python3`;
- set `speaker.piper.model` to your local `.onnx` voice model;
- or temporarily set `speaker.enabled=false`.

## Linux Audio Setup

List devices:

```bash
.venv/bin/python utils/list_devices.py
```

List PipeWire/PulseAudio monitor sources for loopback/AEC:

```bash
.venv/bin/python utils/list_devices.py --monitors
```

Recommended Linux AEC defaults:

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

Useful diagnostics:

```bash
.venv/bin/python utils/meter_fast.py --duration 10
.venv/bin/python utils/AEC_meter.py --aec --pulse \
  --mic-source @DEFAULT_SOURCE@ \
  --loopback-source @DEFAULT_MONITOR@ \
  --duration 30
```

More details: [docs/audio.md](docs/audio.md).

## OpenClaw Setup

Enable OpenClaw forwarding:

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

Install the bundled OpenClaw skill:

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

Listener auto-discovers OpenClaw's assistant name from workspace `IDENTITY.md`
using `Name:` or `Имя:`. Full guide: [docs/openclaw.md](docs/openclaw.md).

## Speaker Setup

The integrated Speaker subscribes to OpenClaw Gateway chat events and voices
assistant replies locally. It requires all of the following:

- OpenClaw Gateway reachable at `speaker.gateway.url` (default `ws://127.0.0.1:18789`);
- Python environment with `piper` available through `speaker.piper.command`;
- a valid voice model at `speaker.piper.model`;
- a playback command such as `/usr/bin/paplay`.

Optionally, `speaker.emoji_display` can point to a separate
`emoji-display` daemon. Listener strips emoji from text before Piper and sends
the extracted symbols to that HTTP service; it does not open serial/COM ports.

Quick checks:

```bash
.venv/bin/python utils/listenerctl.py speaker status
curl -s http://127.0.0.1:18790/speaker/status | jq
```

If you do not want spoken replies during initial setup, disable them:

```json
{
  "speaker": {
    "enabled": false
  }
}
```

## Runtime SpeechGate Control

When `main.py` is running, SpeechGate modes can be changed without restarting:

```bash
.venv/bin/python utils/listenerctl.py status
.venv/bin/python utils/listenerctl.py mute --reason "quiet mode"
.venv/bin/python utils/listenerctl.py chatty --ttl 600
.venv/bin/python utils/listenerctl.py standby --ttl 300
.venv/bin/python utils/listenerctl.py normal
.venv/bin/python utils/listenerctl.py speaker status
.venv/bin/python utils/listenerctl.py speaker off
.venv/bin/python utils/listenerctl.py speaker on
```

The status line includes mode, temporary/permanent state, expiry time, and
restore mode.

HTTP examples:

```bash
curl -s http://127.0.0.1:18790/speech-gate/status | jq
curl -s -X POST http://127.0.0.1:18790/speech-gate/mode \
  -H 'Content-Type: application/json' \
  -d '{"mode":"chatty","ttl_seconds":60,"source":"curl"}' | jq
curl -s http://127.0.0.1:18790/speaker/status | jq
```

Modes:

- `normal` - regular directed-speech filtering.
- `mute` - only assistant-name calls pass.
- `chatty` - all non-empty phrases pass.
- `standby` - all phrases are blocked; TTL is required.

Listener can also handle a small set of voice-only mode commands locally before
anything is forwarded to OpenClaw:

- `Имя, помолчи` -> `mute`
- `Имя, говори` -> `normal`
- `Имя, отключись` -> `standby`
- `Имя, стоп` -> OpenClaw `chat.abort` for the configured `session_key`

These local commands are swallowed by `SpeechGateAgent` and are not forwarded as
regular chat input.

When integrated Speaker is enabled, `Имя, стоп` and explicit barge-in phrases
also interrupt local TTS playback and clear queued spoken segments. OpenClaw can
toggle spoken replies through the bundled skill with `speaker on`, `speaker off`
and `speaker status`.

## Speaker Troubleshooting

The most useful first signal is `speaker status`:

- `agent=running gateway=connected` means Listener is subscribed to OpenClaw Gateway;
- `speaker=off` means spoken replies are disabled by config or runtime control;
- `error=...` points to the last gateway, Piper, or playback failure;
- `queue` and `current` show whether speech is waiting or actively playing.

For runtime diagnostics, start Listener in DEBUG mode and inspect Speaker logs:

```bash
.venv/bin/python main.py 2>&1 | tee /tmp/listener-speaker.log
rg "SpeakerAgent: (connected|final event needs history check|history check produced|queued speech segment|speaking assistant reply|speech failed|interrupted|dropped)" /tmp/listener-speaker.log
rg "EmojiDisplay|extracted .* emoji|emoji-only" /tmp/listener-speaker.log
```

This is especially helpful when a final sentence is visible in OpenClaw but was
not spoken: the log chain shows whether the tail was missing from gateway
streaming, dropped from the queue, interrupted, or failed in Piper/playback.

## Tests

```bash
. .venv/bin/activate
python -m pytest -q
```

Expected result: the full test suite passes.

## Documentation

- [INSTALL.md](INSTALL.md) - fresh setup and first run.
- [docs/audio.md](docs/audio.md) - audio processing, VAD and AEC.
- [docs/stt.md](docs/stt.md) - Whisper STT and SpeechGate.
- [docs/openclaw.md](docs/openclaw.md) - OpenClaw forwarding and control skill.
- [docs/release.md](docs/release.md) - release checklist.

## License

MIT. See [LICENSE](LICENSE).
