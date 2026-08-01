# Миграция на единое окружение Python

[English version](unified-tts-migration.md)

Этот регламент переводит существующую установку Listener с отдельных окружений
VoxCPM2 и CosyVoice3 на одно окружение Python 3.12. Worker-процессы моделей остаются
раздельными; общими становятся только интерпретатор и установленные пакеты.

## Сборка без изменения работающей службы

Выполняйте команды из боевого checkout Listener. Создайте версионированный соседний
каталог, чтобы не изменять активную `.venv` на месте:

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

Единый профиль намеренно использует CPU-only wheel TorchCodec. Он декодирует
референсное аудио на CPU, а PyTorch, Whisper, VoxCPM2 и CosyVoice3 продолжают работать
на CUDA.

До изменения конфига скопируйте внешние ресурсы моделей внутрь боевого checkout.
Старые каталоги оставьте для отката:

```bash
mkdir -p models/tts/voxcpm2/model models/tts/cosyvoice3 references/voxcpm2 references/cosyvoice3
rsync -a /old/VoxCPM2/models/VoxCPM2/ models/tts/voxcpm2/model/
rsync -a --exclude='.git/' /old/CosyVoice/ models/tts/cosyvoice3/CosyVoice/
rsync -a /old/wetext/ models/tts/cosyvoice3/wetext/
rsync -a /old/VoxCPM2/Reference/Nata.wav references/voxcpm2/Nata.wav
rsync -a /old/CosyVoice3/Reference/Nata.wav references/cosyvoice3/Nata.wav
```

Стандартные пути вычисляются относительно корня проекта Listener. Удалите из боевого
конфига старые поля `model_path`, `reference_wav_path`, `repo_path`, `prompt_wav_path`
и `wetext_path`, чтобы использовать новую раскладку.

## Проверки до переключения

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

Репозиторий CosyVoice и подмодуль Matcha-TTS уже должны находиться по путям из
`config/config.json`. До переключения продакшна выполните live-синтез одной фразы каждым
бэкендом. Так проверяются загрузка моделей, чтение референсного WAV, CUDA, передача PCM
и корректное завершение процесса.

## Переключение продакшна

Убедитесь, что поля `python` бэкендов и устаревшие внешние пути отсутствуют в боевом
конфиге. Затем остановите службу, поменяйте
каталоги окружений и снова запустите её:

```bash
systemctl --user stop listener.service
mv .venv .venv-pre-unified-20260801
ln -s .venv-unified-20260801 .venv
systemctl --user start listener.service
.venv/bin/python utils/listenerctl.py ready --json
systemctl --user status listener.service --no-pager -l
```

Проверьте команду выбранного worker-процесса в `systemctl --user status`: она должна
начинаться с боевого `.venv/bin/python` Listener, а не со старого окружения модели.
После этого синтезируйте реальный ответ OpenClaw и проверьте журнал на запуск worker,
завершение потока, перезапуски воспроизведения и события fallback.

## Откат

Храните прежнюю `.venv` и явные пути к интерпретаторам бэкендов до нескольких дней
стабильной работы нового окружения. Команды отката:

```bash
systemctl --user stop listener.service
mv .venv .venv-unified-link-failed-20260801
mv .venv-pre-unified-20260801 .venv
# При необходимости верните явные пути voxcpm2.python/cosyvoice3.python.
systemctl --user start listener.service
.venv/bin/python utils/listenerctl.py ready --json
```

Не удаляйте старые окружения во время переключения. Удалять их следует только после
закрытия окна отката и проверки, что ни один процесс службы их не использует.
