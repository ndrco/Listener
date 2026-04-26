# Listener

`Listener` это локальный voice-input runtime: он захватывает звук с микрофона, прогоняет его через обработку, режет на речевые сегменты, распознаёт речь через Whisper, фильтрует реплики по направленности и при необходимости отправляет результат в OpenClaw.

Проект построен вокруг шины событий и трёх основных агентов:

- `AudioAgent` запускает микрофон, обработку аудио, буферизацию речи и STT.
- `SpeechGateAgent` решает, действительно ли фраза адресована ассистенту.
- `OpenClawInputAgent` пересылает прошедшие фразы в `openclaw gateway call chat.send`.

Базовая конфигурация лежит в [config/config.json](config/config.json).

## Пайплайн

```text
Microphone -> AudioProcessor -> BufferedSpeechWriter -> WhisperStreamingTranscriber
           -> llm/input_text -> SpeechGateAgent -> llm/speaker_phrase
           -> OpenClawInputAgent -> OpenClaw
```

Внутри `AudioProcessor` сигнал может проходить через:

- AEC
- ресемплинг
- high-pass / DC blocking
- noise suppression
- VAD
- AGC

Подробности по устройству подсистемы есть в [docs/audio.md](docs/audio.md) и [docs/stt.md](docs/stt.md).

## Быстрый старт

Создать локальное окружение и установить зависимости:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

Запустить приложение:

```bash
python main.py
```

Запустить тесты:

```bash
pytest -q
```

## Конфигурация

Основные настройки рантайма находятся в `config/config.json`.

Самые важные секции:

- `audio.input`: частота дискретизации, каналы, размер блока и индекс микрофона.
- `audio.processing`: AEC, VAD, AGC, high-pass и шумоподавление.
- `audio.buffer`: буферизация речевых сегментов перед STT.
- `audio.stt`: модель Whisper и параметры декодирования.
- `speech_gate`: правила и ML-гейт направленности речи.
- `openclaw`: параметры отправки финальных реплик в OpenClaw.
- `events`: топики внутренней шины событий.

Windows-ориентированный пример вынесен в [config/config.windows.example.json](config/config.windows.example.json).

## Модели

Код ожидает наличие путей к моделям в конфиге, но сами веса не должны храниться в исходниках.

По умолчанию используются такие пути:

- `audio.processing.vad.model_path`: `models/silero_vad_v6.jit`
- `speech_gate.model.path`: `models/directed-ruElectra-small-fp16`
- `audio.stt.model`: `avazir/faster-distil-whisper-large-v3-ru`
- `audio.stt.download_root`: `models/whisper`
- `audio.stt.local_files_only`: `true`

Последний флаг особенно важен: при `local_files_only=true` Whisper не будет автоматически скачивать веса. Для чистого клона есть три варианта:

1. Положить модель Whisper в `models/whisper`.
2. Временно переключить `audio.stt.local_files_only` в `false`.
3. Временно отключить STT через `audio.stt.enabled=false`.

Silero VAD можно скачать так:

```bash
python utils/silero_vad_model_downloader.py
```

## Linux

На Linux поддерживаются:

- захват микрофона
- обработка аудио
- VAD
- Whisper STT
- speech gate
- отправка в OpenClaw
- диагностика AEC и loopback

Показать устройства:

```bash
python utils/list_devices.py
```

Показать monitor-источники для loopback/AEC:

```bash
python utils/list_devices.py --monitors
```

`utils/list_devices.py` выводит две разные картины:

- устройства, видимые через `sounddevice`
- реальные PipeWire/Pulse sources, полученные через `pactl`

Это важно, потому что на Linux PipeWire может удерживать raw ALSA-устройство, и тогда микрофон физически существует в системе, но в `sounddevice` не виден как отдельный input.

Для runtime Linux loopback/AEC настраивается через:

- `audio.processing.aec.playback_source = "loopback"`
- `audio.processing.aec.loopback_backend = "auto"` или `pulse` / `pipewire`
- `audio.processing.aec.loopback_device_index`
- `audio.processing.aec.loopback_source_name`
- `audio.processing.aec.loopback_device_name_contains`

Если monitor/source не найден, приложение продолжит работу без loopback-захвата и запишет warning в лог.

Для OpenClaw на Linux обычно достаточно:

```json
{
  "openclaw": {
    "enabled": true,
    "command": "openclaw"
  }
}
```

Если OpenClaw не нужен, установите `openclaw.enabled=false`.

## Диагностика аудио

Полезные утилиты из `utils/`:

- `utils/list_devices.py`: список аудиоустройств и PipeWire/Pulse sources.
- `utils/meter_fast.py`: быстрый live meter для микрофона.
- `utils/AEC_meter.py`: live-проверка AEC и офлайн-режим для AEC по WAV.
- `utils/livekit_test.py`: простой тест LiveKit AEC на WAV-файлах.
- `utils/debug_analysis.py`: анализ артефактов speaker/debug.

Примеры:

```bash
python utils/meter_fast.py --device 0 --duration 10
python utils/AEC_meter.py --aec --pulse --duration 30
python utils/livekit_test.py --self-test
```

Для PipeWire/Pulse в `AEC_meter.py` можно использовать явные source names:

- микрофон через `@DEFAULT_SOURCE@`
- loopback через `@DEFAULT_MONITOR@`

Пример:

```bash
python utils/AEC_meter.py --aec --pulse \
  --mic-source @DEFAULT_SOURCE@ \
  --loopback-source @DEFAULT_MONITOR@ \
  --duration 30
```

## Структура проекта

```text
agents/      оркестрация runtime
audio/       микрофон, обработка, STT, буферизация
core/        конфиг, event bus, логирование
llm/         speech gate
config/      JSON-конфиги и паттерны
docs/        внутренняя документация по audio/STT
tests/       pytest-набор
utils/       диагностические и вспомогательные скрипты
```

## Разработка

Базовый рабочий цикл:

```bash
. .venv/bin/activate
pytest -q
python utils/list_devices.py
python main.py
```

В тестах уже покрыты:

- audio processing
- поведение VAD-пайплайна
- STT helper-логика
- обратная совместимость `WindowsAudioProcessor`
- Linux loopback selection

## Текущие оговорки

- На чистом окружении первый запуск чаще всего упирается в отсутствие локальных моделей или в слишком строгий конфиг.
- На Linux для AEC-диагностики надёжнее работать через Pulse/PipeWire source names, чем только через `sounddevice`.
- Отправка в OpenClaw опциональна и требует, чтобы `openclaw` был доступен в `PATH`.
