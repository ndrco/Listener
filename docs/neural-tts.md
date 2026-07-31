# Neural TTS: VoxCPM2 and CosyVoice3

[Русская версия](neural-tts_RUS.md)

Listener can use VoxCPM2 or Fun-CosyVoice3-0.5B as the primary Speaker
backend. Both integrations keep the model in a persistent child process,
stream raw PCM to Listener over stdio, and fall back to Piper if startup or
generation fails. The same worker can also render WAV files requested through
Listener's control API, without loading a second model process. The checked-in
default remains Piper.

This is an optional installation. Complete the base setup in
[`INSTALL.md`](../INSTALL.md) first.

## Why the environments must be separate

Do not install either neural backend into Listener's `.venv`, and do not put
VoxCPM2 and CosyVoice3 into one shared environment. The tested installation
uses three independent Python runtimes:

| Process | Python | Important tested pins |
| --- | --- | --- |
| Listener | 3.12 | Listener `requirements.txt` |
| VoxCPM2 worker | 3.11 | Torch 2.8/cu128, `voxcpm` 2.0.3, NumPy 2.x |
| CosyVoice3 worker | 3.10 | Torch 2.8/cu128, Transformers 4.51, NumPy 1.26, ONNX Runtime GPU 1.22 |

CosyVoice imports its repository and Matcha-TTS submodule directly. VoxCPM2
uses the installed `voxcpm` package. Their Python, NumPy, Transformers, ONNX
and audio dependency constraints differ; merging them makes upgrades fragile
and can break Listener's STT stack. Listener starts only the worker selected by
`speaker.tts.backend`, so installing both does not consume GPU memory for both
unless two Listener instances are started.

The worker interpreter is configured with an absolute path. It never needs to
be activated by a shell or by `systemd`.

## Hardware and storage overhead

The following measurements are from the current tested host: GeForce RTX 5080
16 GB, driver 580.173.02, CUDA 12.8 PyTorch wheels, reference cloning enabled,
warm-up enabled, VoxCPM2 denoiser disabled, and CosyVoice TensorRT disabled.
They are capacity-planning figures, not model guarantees.

| Item | VoxCPM2 | CosyVoice3 |
| --- | ---: | ---: |
| Isolated environment on disk | about 8.0 GiB | about 8.0 GiB |
| Model snapshot on disk | about 4.7 GiB | about 9.1 GiB |
| Worker-specific text-normalization data | none | about 21 MiB WeText FST |
| Worker model load plus warm-up | about 16.5 s | about 11.4 s in isolated smoke test |
| Output format seen in tests | mono PCM16, 48 kHz | mono PCM16, 24 kHz |

The shared `rutextnorm` dependency lives in Listener's main environment and
adds about 0.2 MiB on the tested host. It does not create another process or
use GPU/model memory. A representative sentence took about 0.23 ms per call
after warm-up on this host, so its CPU and latency cost is negligible compared
with synthesis; the exact time remains hardware- and text-dependent.

With the production VoxCPM2 profile, the persistent worker uses approximately
3.8 GiB resident RAM and 7.0 GiB VRAM. Listener with its CUDA STT and speech
gate uses another approximately 2.4 GiB VRAM on that host. Loading/compilation
temporarily raises host RAM usage; allow at least 16 GiB system RAM for the
combined process set and preferably 32 GiB or more.

CosyVoice3 VRAM varies with its ONNX providers, `fp16`, TensorRT and model
revision, so measure it on the target installation rather than treating the
weight-file size as VRAM usage. On a 16 GB GPU, do not run both neural workers
at once alongside CUDA STT. Normal Listener operation starts only one.

Installing both tested environments and both model snapshots consumes roughly
30 GiB before package/model caches. Caches can require another copy while a
snapshot is downloading.

Useful measurements after startup:

```bash
nvidia-smi
ps -eo pid,rss,cmd | rg 'voxcpm2_worker|cosyvoice3_worker|main.py'
du -sh /opt/listener-tts/*
```

## Common system packages

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 sox libsox-dev git git-lfs
```

The examples below use Conda. Micromamba can be used with equivalent `create
-p` commands. Choose any root directory with at least 40 GiB free; `/opt` is
only an example:

```bash
sudo mkdir -p /opt/listener-tts
sudo chown "$USER":"$USER" /opt/listener-tts
export LISTENER_ROOT=/absolute/path/to/Listener
export TTS_ROOT=/opt/listener-tts
```

The variables are conveniences for installation only. Paths written to JSON
must be absolute; Listener does not expand shell variables in `config.json`.

## Install VoxCPM2

Create a Python 3.11 environment and install the tested Blackwell profile:

```bash
export VOX_ROOT="$TTS_ROOT/VoxCPM2"
export VOX_ENV="$VOX_ROOT/env"
export VOX_MODEL_DIR="$VOX_ROOT/models/VoxCPM2"
mkdir -p "$VOX_ROOT/models" "$VOX_ROOT/reference"
conda create -y -p "$VOX_ENV" python=3.11 pip
conda run -p "$VOX_ENV" python -m pip install --upgrade pip
conda run -p "$VOX_ENV" python -m pip install \
  -r "$LISTENER_ROOT/docs/requirements/voxcpm2-cu128.txt"
```

The checked-in requirements profile is specifically tested on RTX 50-series
hardware. For another GPU, select a PyTorch wheel compatible with that GPU and
driver, but retain Python 3.11 and the separate environment.

Download `openbmb/VoxCPM2` before enabling offline mode:

```bash
export VOX_MODEL_DIR
"$VOX_ENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openbmb/VoxCPM2",
    local_dir=os.environ["VOX_MODEL_DIR"],
)
PY
```

Copy a clean reference recording to `$VOX_ROOT/reference/voice.wav`. Prefer a
mono WAV with one speaker, little noise or reverb, and roughly 5–15 seconds of
speech. A transcript next to it is useful for provenance and future model
changes, but Listener's current VoxCPM2 worker builds its prompt cache directly
from the WAV and does not consume `prompt_text`.

Verify the environment without loading the model:

```bash
"$VOX_ENV/bin/python" - <<'PY'
import torch
from voxcpm import VoxCPM

print(torch.__version__, torch.cuda.is_available())
print(VoxCPM)
PY
```

## Install CosyVoice3

Clone CosyVoice recursively. The Matcha-TTS submodule is mandatory:

```bash
export COSY_ROOT="$TTS_ROOT/CosyVoice3"
export COSY_REPO="$COSY_ROOT/CosyVoice"
export COSY_ENV="$COSY_ROOT/env"
export COSY_MODEL_DIR="$COSY_REPO/pretrained_models/Fun-CosyVoice3-0.5B"
export WETEXT_DIR="$COSY_ROOT/models/wetext"
mkdir -p "$COSY_ROOT/reference" "$COSY_ROOT/models"
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$COSY_REPO"
git -C "$COSY_REPO" submodule update --init --recursive
conda create -y -p "$COSY_ENV" python=3.10 pip
conda run -p "$COSY_ENV" python -m pip install --upgrade pip
conda run -p "$COSY_ENV" python -m pip install \
  -r "$LISTENER_ROOT/docs/requirements/cosyvoice3-cu128.txt"
```

Do not additionally install the repository's upstream `requirements.txt` over
this profile on an RTX 50-series GPU: its older Torch/cu121 and ONNX Runtime
pins replace the Blackwell-compatible packages. On older supported GPUs, the
upstream profile may be appropriate, but it still belongs in this dedicated
Python 3.10 environment.

Download the model and WeText normalizer before setting
`local_files_only=true`:

```bash
export COSY_MODEL_DIR WETEXT_DIR
"$COSY_ENV/bin/python" - <<'PY'
import os
from modelscope import snapshot_download

snapshot_download(
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    local_dir=os.environ["COSY_MODEL_DIR"],
)
snapshot_download(
    "pengzhendong/wetext",
    local_dir=os.environ["WETEXT_DIR"],
)
PY
```

The configured WeText directory must contain all four files below:

```text
en/tn/tagger.fst
en/tn/verbalizer.fst
zh/tn/tagger.fst
zh/tn/verbalizer.fst
```

Copy a clean reference WAV to `$COSY_ROOT/reference/voice.wav`. Listener caches
its speech features once at worker startup. As with the current Vox adapter,
the `prompt_text` config field is reserved but is not consumed by this worker.

Verify imports and CUDA before starting Listener:

```bash
"$COSY_ENV/bin/python" - <<'PY'
import torch
import wetext
from modelscope import snapshot_download

print(torch.__version__, torch.cuda.is_available())
print(wetext.Normalizer, snapshot_download)
PY
test -d "$COSY_REPO/third_party/Matcha-TTS"
test -f "$WETEXT_DIR/en/tn/tagger.fst"
```

## Listener configuration

Neural backends require `speaker.tts_mode="persistent"`. Start from this
shape, replace every `/opt/listener-tts` path with the installation's real
absolute path, and select exactly one `tts.backend`:

```json
{
  "speaker": {
    "enabled": true,
    "tts_mode": "persistent",
    "tts": {
      "backend": "voxcpm2",
      "fallback_backend": "piper",
      "startup_timeout_s": 90,
      "generation_timeout_s": 120,
      "cancel_timeout_s": 1,
      "max_consecutive_errors": 3,
      "normalize_numbers": true,
      "style": {
        "enabled": true,
        "inherit_within_run": true,
        "leading_emoji_only": true,
        "default_style": "neutral"
      }
    },
    "file_render": {
      "enabled": true,
      "output_dir": "state/tts-files",
      "max_text_chars": 5000,
      "max_pending_jobs": 8,
      "max_completed_jobs": 128,
      "segment_chars": 220
    },
    "voxcpm2": {
      "python": "/opt/listener-tts/VoxCPM2/env/bin/python",
      "model_path": "/opt/listener-tts/VoxCPM2/models/VoxCPM2",
      "reference_wav_path": "/opt/listener-tts/VoxCPM2/reference/voice.wav",
      "device": "cuda",
      "optimize": true,
      "load_denoiser": false,
      "local_files_only": true,
      "seed": 42,
      "cfg_value": 2.0,
      "inference_timesteps": 10,
      "warmup": true,
      "compile_threads": 4
    },
    "cosyvoice3": {
      "python": "/opt/listener-tts/CosyVoice3/env/bin/python",
      "repo_path": "/opt/listener-tts/CosyVoice3/CosyVoice",
      "model_path": "/opt/listener-tts/CosyVoice3/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B",
      "prompt_wav_path": "/opt/listener-tts/CosyVoice3/reference/voice.wav",
      "device": "cuda",
      "local_files_only": true,
      "fp16": true,
      "load_trt": false,
      "wetext_path": "/opt/listener-tts/CosyVoice3/models/wetext",
      "warmup": true,
      "speed": 1.0,
      "enable_vocal_events": false
    }
  }
}
```

To use CosyVoice3, change only:

```json
{
  "speaker": {
    "tts": {
      "backend": "cosyvoice3"
    }
  }
}
```

The fallback requires a working Piper configuration. A neural startup failure
opens the fallback circuit for the current Listener process; correct the error
and restart Listener to retry the primary backend.

## Smoke test and service operation

First confirm the effective config without loading a neural model:

```bash
cd "$LISTENER_ROOT"
.venv/bin/python -m speaker.cli --config config/config.json print-config
```

Then synthesize one sentence. This loads and warms the selected worker, plays
the result, and shuts the worker down afterward:

```bash
.venv/bin/python -m speaker.cli --config config/config.json \
  --log-level INFO say '😔 Иногда тишина остаётся единственной собеседницей.'
```

Run Listener normally only after that succeeds:

```bash
.venv/bin/python main.py
.venv/bin/python utils/listenerctl.py speaker status
```

The Speaker status exposes the selected backend, worker PID, startup/generation
errors, first-audio timing, PCM chunks and playback state. With the user
service, use:

```bash
journalctl --user -u listener.service -f
```

## Crash-isolated streaming playback

Neural workers produce mono PCM16 chunks. On Linux, Listener sends those chunks
to a lightweight `pacat` subprocess by default and falls back to `pw-cat` when
needed. PortAudio therefore does not run inside the Listener process for normal
neural playback. Killing or crashing the audio child cannot terminate Listener
or the persistent model worker.

The player is scoped to an OpenClaw `run_id`, not an individual sentence. It
collects the initial prebuffer once, remains open between queued segments, and
drains only after Speaker processes the run-finished marker. Ducking begins
immediately before the child starts and is restored after its stdin and server
buffers have drained. File rendering bypasses playback and is unaffected.

```json
{
  "speaker": {
    "playback": {
      "streaming_backend": "auto",
      "streaming_command": "",
      "prebuffer_ms": 150,
      "latency_ms": 100,
      "queue_ms": 2000,
      "restart_attempts": 1,
      "write_timeout_s": 5
    }
  }
}
```

`auto` prefers `pacat`, then `pw-cat`; on non-Linux systems it may use
`sounddevice`. Explicit values are `pacat`, `pwcat`, and `sounddevice`.
`streaming_command` can override the executable path. `prebuffer_ms` is the
one-time start cushion, `latency_ms` requests the audio-server buffer,
`queue_ms` bounds PCM waiting in Listener, and `restart_attempts` limits new
player processes after a mid-run failure.

Speaker status reports the resolved backend, child PID, queue/prebuffer state,
bytes written, restart count, last exit code, and captured stderr tail. Listener
does not automatically replay a segment after a mid-playback crash because its
beginning may already have been audible; it starts a fresh player for the next
segment instead.

## Render WAV files without another TTS process

When Listener is already running with `tts_mode="persistent"` and the selected
backend is `voxcpm2` or `cosyvoice3`, its control API can queue file-render jobs:

```bash
.venv/bin/python utils/listenerctl.py tts-file render \
  --text '😔 Иногда тишина всё объясняет.' \
  --filename quiet-thought \
  --wait
```

The completed response prints an absolute WAV `path`. Longer input may be read
from a UTF-8 file with `--text-file`. Job controls are:

```bash
.venv/bin/python utils/listenerctl.py tts-file list
.venv/bin/python utils/listenerctl.py tts-file status JOB_ID
.venv/bin/python utils/listenerctl.py tts-file cancel JOB_ID
```

The corresponding authenticated control endpoints are:

```text
POST /tts/files
GET  /tts/files
GET  /tts/files/{job_id}
POST /tts/files/{job_id}/cancel
```

`POST /tts/files` accepts `text`, optional allowlisted `style`, and optional
`filename`. `filename` is only a label: Listener removes path components,
adds a unique suffix, and always writes below `file_render.output_dir`. Output
is first written as `.wav.part` and atomically renamed after the WAV header is
finalized. Partial files are removed on failure or cancellation. Job metadata
is in memory, only the newest `max_completed_jobs` terminal entries are kept,
and all entries are cleared by a Listener restart. Completed WAV files remain
until the user removes or archives them.

This path shares the existing `NeuralWorkerClient`; it does not spawn another
VoxCPM2/CosyVoice3 worker and adds no second copy of the model in RAM or VRAM.
If the worker was still lazy, the first job pays the normal model load/warm-up
cost. Disk use for mono PCM16 is approximately 2.75 MiB/min at CosyVoice3's
24 kHz or 5.49 MiB/min at VoxCPM2's 48 kHz, plus a 44-byte WAV header.

One file job runs at a time. Long text is split into bounded speech segments
and releases the shared generation lock between them, so queued reply playback
can run before the next file segment. An individual in-progress model segment
is not preempted. Playback cancellation is scoped separately from file-job
cancellation, preventing one operation from corrupting the other's persistent
stdio stream. File rendering deliberately uses only the selected neural
backend: it reports a failed job instead of silently saving a Piper fallback
under a neural-backend label.

Install `openclaw/skills/listener-tts-file` together with `listener-control` to
let OpenClaw create these files. The skill calls the control API and explicitly
forbids starting either isolated model environment directly.

## Emoji-to-style instructions

OpenClaw can request speaking style by putting one allowlisted emoji at the
start of a sentence. Listener removes all emoji from spoken text and sends a
fixed, model-neutral instruction for the leading allowlisted emoji. Arbitrary
assistant text is never promoted to an instruction.

| Leading emoji | Style | Instruction intent |
| --- | --- | --- |
| 🙂 😊 ❤️ | `warm` | warm, friendly, gentle |
| 😄 🎉 ✨ | `cheerful` | cheerful, upbeat, lively |
| 😌 | `calm` | calm, soft, unhurried |
| 🤔 🧐 | `thoughtful` | thoughtful, measured |
| 😔 😢 😭 💔 | `sad` | subdued, sad, empathetic |
| 😠 😡 | `firm` | controlled anger, no shouting |
| 😮 😲 🤯 | `surprised` | clear, natural surprise |
| 😏 😼 😉 | `playful` | playful, lightly teasing |
| ⚠️ 🚨 | `urgent` | urgent, clear, slightly faster |
| 😂 🤣 | `amused` | amused, holding back laughter |

VoxCPM2 receives the fixed instruction as a parenthesized prefix to target
text. CosyVoice3 tokenizes it as an instruct prompt. When
`enable_vocal_events=true`, the `amused` style can additionally insert its
supported `[laughter]` event; this is disabled by default because vocal-event
quality depends on the voice and model revision. Piper ignores style metadata.

### Shared Russian text normalization

`tts.normalize_numbers=true` enables Listener's narrow wrapper around
`rutextnorm` before dispatch to any TTS backend. It therefore covers Piper,
VoxCPM2, CosyVoice3, Piper fallback, and neural WAV file rendering consistently.
Only selected numeric or mathematical spans and their directly associated date,
time, percentage, currency, or measurement notation are changed. Text outside
those spans is copied unchanged: Listener does not transliterate Latin words,
model identifiers, or ordinary assistant text. URLs, IP addresses, phone
numbers, absolute paths, inline code, and control tokens are protected.

Software versions use an explicit spoken decimal point, for example
`GPT-5.6-terra` becomes `GPT-пять точка шесть-terra`; the Latin fragments remain
unchanged. Russian decimal measurements such as `3.5 кг` are treated as `3,5 кг`.
Style instructions and `[laughter]` tokens are added outside normalization.

The narrow layer also speaks a single mathematical `=` as `равно` and handles
calendar days with named Russian months (`1 августа` → `первое августа`,
`с 1 августа` → `с первого августа`). It deliberately does not try to repair
noun agreement with a partial dictionary: OpenClaw should produce the correct
source form, such as `2 чашки`. Operators such as `==`, `>=`, and `!=` remain
unchanged, as do expressions inside protected code spans.

CosyVoice3 still calls its tokenizer directly, so the stock English WeText
frontend is not applied afterward. `rutextnorm` is installed in Listener's main
environment through `requirements.txt`; model-specific isolated environments do
not need it. When normalization is enabled, a missing package is an engine
creation error; an unexpected per-request normalization error fails open to the
original text. Disable the feature with `tts.normalize_numbers=false` or
`SPEAKER_TTS_NORMALIZE_NUMBERS=false`. The old CosyVoice-only config key and
environment variable are accepted for migration, but the shared names should be
used for new deployments.

With `leading_emoji_only=true`, a trailing emoji is display-only. With
`inherit_within_run=true`, a leading emoji styles following sentences in the
same OpenClaw `runId`, then the state is discarded at finalization, error,
abort or barge-in. This matters for streaming: a trailing emoji arrives after
the sentence may already have entered the TTS queue.

Install the bundled convention into OpenClaw once:

```bash
OPENCLAW_WORKSPACE="$(openclaw config get agents.defaults.workspace)"
sed -n '1,$p' openclaw/prompts/listener-tts-style.md \
  >> "$OPENCLAW_WORKSPACE/AGENTS.md"
```

## Troubleshooting

- `worker failed to start`: run the environment verification snippet and
  inspect the worker's `stderr_tail` in `speaker status`.
- `CUDA is not available`: verify the NVIDIA driver and the environment's
  Torch wheel with `torch.cuda.is_available()`.
- `local WeText FST directory not found`: download WeText while online and
  point `wetext_path` at the directory containing `en/tn` and `zh/tn`.
- `CosyVoice Matcha-TTS submodule not found`: run `git submodule update --init
  --recursive` inside the CosyVoice repository.
- First startup times out: increase `tts.startup_timeout_s`; compilation and
  cold filesystem caches are slower than subsequent starts.
- Neural synthesis fails repeatedly: Piper remains active after the configured
  error threshold. Check `tts.primary.worker.stderr_tail` in Speaker status.
- Listener's own speech becomes quiet after ducking: run `.venv/bin/python
  utils/listenerctl.py speech_gate_reset --reason "recover ducking"`. Speaker
  streams are excluded from normal ducking, and reset also repairs stale
  PipeWire/WirePlumber route volume.

Keep `local_files_only=true` in production only after model and WeText
downloads are complete. It prevents a service restart from silently accessing
the network or changing the installed snapshot.
