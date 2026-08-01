# Unified Python environment migration

[Русская версия](unified-tts-migration_RUS.md)

This runbook migrates an existing Listener installation from separate VoxCPM2
and CosyVoice3 environments to one Python 3.12 environment. The model workers
remain separate processes; only their interpreter and installed packages are
shared.

## Build without touching the running service

Run these commands from the deployed Listener checkout. Use a versioned sibling
directory so the active `.venv` is not modified in place:

```bash
cd /absolute/path/to/Listener
python3.12 -m venv .venv-unified-20260801
.venv-unified-20260801/bin/python -m pip install --upgrade pip
.venv-unified-20260801/bin/python -m pip install \
  -r requirements-unified-cu128-py312.txt
.venv-unified-20260801/bin/python -m pip check
mkdir -p .deploy-backups/20260801-unified-env
.venv-unified-20260801/bin/python -m pip freeze \
  > .deploy-backups/20260801-unified-env/requirements-unified-20260801.freeze.txt
```

The unified requirements deliberately use the CPU-only TorchCodec wheel. It
decodes reference audio on CPU while PyTorch, Whisper, VoxCPM2, and CosyVoice3
continue to use CUDA.

Copy external model assets into the production checkout before changing the
config. Keep the old directories for rollback:

```bash
mkdir -p models/tts/voxcpm2/model models/tts/cosyvoice3 references/voxcpm2 references/cosyvoice3
rsync -a /old/VoxCPM2/models/VoxCPM2/ models/tts/voxcpm2/model/
rsync -a --exclude='.git/' /old/CosyVoice/ models/tts/cosyvoice3/CosyVoice/
rsync -a /old/wetext/ models/tts/cosyvoice3/wetext/
rsync -a /old/VoxCPM2/Reference/Nata.wav references/voxcpm2/Nata.wav
rsync -a /old/CosyVoice3/Reference/Nata.wav references/cosyvoice3/Nata.wav
```

The standard paths are computed relative to Listener's project root. Remove
the old `model_path`, `reference_wav_path`, `repo_path`, `prompt_wav_path`, and
`wetext_path` fields from the deployed config to use them.

## Pre-switch checks

```bash
.venv-unified-20260801/bin/python -m pytest -q
.venv-unified-20260801/bin/python -m speaker.cli \
  --config config/config.json print-config
.venv-unified-20260801/bin/python - <<'PY'
import torch, torchaudio, torchcodec, transformers, voxcpm
from cosyvoice.cli.cosyvoice import AutoModel

print(torch.__version__, torchaudio.__version__, torch.cuda.is_available())
print(torchcodec.__version__, transformers.__version__, voxcpm.__version__)
print(AutoModel)
PY
```

The CosyVoice repository and Matcha-TTS submodule must already exist at the
paths in `config/config.json`. Run one live sentence with each backend before
the production switch; this verifies model loading, reference WAV decoding,
CUDA execution, PCM streaming, and shutdown.

## Switch production

Make sure the backend `python` fields and obsolete external asset paths are
absent from the deployed config. Then stop the service,
swap the environment directories, and start it again:

```bash
systemctl --user stop listener.service
mv .venv .venv-pre-unified-20260801
ln -s .venv-unified-20260801 .venv
systemctl --user start listener.service
.venv/bin/python utils/listenerctl.py ready --json
systemctl --user status listener.service --no-pager -l
```

Verify the selected worker command in `systemctl --user status`: it must begin
with the deployed Listener `.venv/bin/python`, not an old model environment.
Then synthesize a real OpenClaw reply and inspect the journal for worker startup,
stream completion, audio playback restarts, and fallback events.

## Rollback

Keep the previous `.venv` and explicit backend interpreter paths until the new
environment has passed several days of normal use. Roll back with:

```bash
systemctl --user stop listener.service
mv .venv .venv-unified-link-failed-20260801
mv .venv-pre-unified-20260801 .venv
# Restore explicit voxcpm2.python/cosyvoice3.python paths if they were used.
systemctl --user start listener.service
.venv/bin/python utils/listenerctl.py ready --json
```

Do not delete either old environment as part of the switch. Remove it only
after the rollback window closes and after checking that no service process
uses it.
