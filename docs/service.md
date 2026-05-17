# Listener as a user service

Listener should run as a foreground Python process supervised by systemd. Use a
user service on Linux, because Listener needs access to the user's microphone,
PipeWire/PulseAudio session, and local OpenClaw environment.

## Before Installing

Finish the normal setup first:

```bash
git clone <repository-url>
cd Listener
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-optional.txt
.venv/bin/python utils/silero_vad_model_downloader.py
```

Then configure `config/config.json` for this machine:

- `audio.input.device_index` if the default microphone is wrong.
- `audio.processing.aec.loopback_source_name` if AEC should use a specific monitor source.
- `audio.stt.device` and `speech_gate.model.device`, especially on CPU-only machines.
- `speaker.enabled=false` for the first run, or set real `speaker.piper.command` and `speaker.piper.model` paths.
- `openclaw.gateway_url`, `openclaw.gateway_token`, and `openclaw.session_key` if your OpenClaw Gateway is not using the defaults.

Run one manual smoke test before creating the service:

```bash
.venv/bin/python main.py
```

In another terminal:

```bash
.venv/bin/python utils/listenerctl.py health
.venv/bin/python utils/listenerctl.py ready
```

Stop the manual process with `Ctrl+C`, or:

```bash
.venv/bin/python utils/listenerctl.py stop --reason smoke-test
```

## Install

The easiest path is the installer script. It reads the bundled template,
rewrites it for the current checkout path, installs it into
`~/.config/systemd/user`, reloads systemd, and enables the service:

```bash
.venv/bin/python utils/install_user_service.py
```

Start immediately during install:

```bash
.venv/bin/python utils/install_user_service.py --start
```

Preview the generated unit without writing anything:

```bash
.venv/bin/python utils/install_user_service.py --dry-run
```

The raw template lives at `deploy/systemd/listener.service`. It contains the
paths for this checkout and is useful as a readable reference. If you copy it
manually, edit these fields in `~/.config/systemd/user/listener.service`:

- `WorkingDirectory`
- `ExecStart`
- `ExecReload`
- `ExecStop`

## Run

```bash
systemctl --user start listener.service
systemctl --user status listener.service
.venv/bin/python utils/listenerctl.py health
.venv/bin/python utils/listenerctl.py ready
```

`health` only checks that the control API is alive. `ready` checks whether
critical Listener components started successfully.

If `ready` prints `listener=not_ready`, inspect the component list and the last
error:

```bash
.venv/bin/python utils/listenerctl.py ready --json
```

## Logs

```bash
journalctl --user -u listener.service -f
```

Listener writes to stdout/stderr, and journald handles collection and rotation.

## Stop and Restart

```bash
.venv/bin/python utils/listenerctl.py speech_gate_reset --reason manual
systemctl --user reload listener.service
.venv/bin/python utils/listenerctl.py stop --reason manual
systemctl --user restart listener.service
systemctl --user stop listener.service
```

The systemd unit uses `listenerctl stop` for graceful shutdown. The `ExecStop`
command is best-effort, so the unit does not fail if Listener already stopped
itself through `/shutdown`. If the process does not exit within
`TimeoutStopSec`, systemd will terminate it.

`systemctl --user reload listener.service` is wired to Listener's soft recovery
path. It runs `listenerctl.py speech_gate_reset --reason systemd-reload`,
which returns `speech_gate` to `normal`, re-enables `speaker`, interrupts
stuck reply playback, and forces ducking volumes to restore without restarting
the Python process. On PipeWire/WirePlumber systems it also restores persisted
ducking baselines from `state/ducking_state.json` and normalizes the
Speaker/Listener output route settings, so reload is the first recovery command
to try after a bad barge-in or interrupted long OpenClaw response.

## Strict Startup

By default Listener keeps the existing best-effort startup behavior. To make
service startup fail when a critical component cannot start, enable this in
`config/config.json`:

```json
"service": {
  "strict_startup": true
}
```

Critical components are audio input, SpeechGate, OpenClaw input forwarding, and
Speaker when `speaker.enabled=true`.

Runtime mode changes are stored locally in `state/runtime_state.json`. That file
is created by Listener on the installed machine and is intentionally not shipped
in releases.

Recommended rollout:

```bash
.venv/bin/python utils/listenerctl.py ready
systemctl --user restart listener.service
journalctl --user -u listener.service -n 80
```

## Remove

```bash
systemctl --user disable --now listener.service
rm ~/.config/systemd/user/listener.service
systemctl --user daemon-reload
```

## Troubleshooting

- `connection_failed` from `listenerctl`: the service is not running, failed
  during startup, or `control.host`/`control.port` differs from the default.
- `listener=not_ready`: the process is alive, but one or more critical
  components failed. Use `listenerctl ready --json` and `journalctl`.
- No microphone input: run `utils/list_devices.py`, then set
  `audio.input.device_index` or the system default input device.
- No spoken replies: check `speaker.enabled`, Piper paths, and
  `listenerctl speaker status`.
- OpenClaw receives no phrases: check `speech_gate` mode, OpenClaw Gateway
  settings, and `listenerctl ready`.
