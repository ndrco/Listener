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

Убедитесь, что `voxcpm2.python` и `cosyvoice3.python` отсутствуют в боевом конфиге либо
оба указывают на новый интерпретатор Listener. Затем остановите службу, поменяйте
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
