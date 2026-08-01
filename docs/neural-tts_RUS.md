# Нейронные TTS: VoxCPM2 и CosyVoice3

[English version](neural-tts.md)

Listener может использовать VoxCPM2 или Fun-CosyVoice3-0.5B в качестве основного бэкенда
Speaker. Обе интеграции сохраняют модель в постоянном дочернем процессе, передают
необработанный PCM в Listener через stdio и возвращаются к Piper в случае сбоя запуска
или генерации. Тот же worker может создавать WAV-файлы по запросу через control API
Listener, не загружая второй процесс модели. В конфигурации репозитория по умолчанию
остаётся Piper.

Это дополнительная установка. Сначала завершите базовую настройку в
[`INSTALL_RUS.md`](../INSTALL_RUS.md).
Для существующей боевой установки используйте поэтапный
[регламент миграции на единое окружение](unified-tts-migration_RUS.md).

## Рекомендуемая схема окружения

Listener, VoxCPM2 и CosyVoice3 могут работать в одном окружении Python 3.12. Для
проверенного хоста с RTX 5080 это рекомендуемый профиль:

| Процесс | Python | Важные проверенные версии |
| --- | --- | --- |
| Listener | 3.12 | Torch 2.11/cu128, Transformers 4.51, NumPy 1.26 |
| Worker VoxCPM2 | тот же интерпретатор | `voxcpm` 2.0.3, CPU TorchCodec 0.15 |
| Worker CosyVoice3 | тот же интерпретатор | ONNX Runtime GPU 1.22, ModelScope 1.22 |

Модели по-прежнему работают в отдельных постоянных дочерних процессах. Такая изоляция
сохраняет отмену, восстановление после сбоя и fallback на Piper, но по умолчанию оба
worker-процесса наследуют `sys.executable` Listener. Listener запускает только выбранный
бэкенд, поэтому общее окружение не загружает обе модели в VRAM.

Ключевые совместимые версии: `modelscope==1.22.0` (требование VoxCPM2),
`transformers==4.51.3` и CPU-only wheel TorchCodec. CPU-wheel декодирует только
референсное аудио; нейронный инференс остаётся на CUDA. Старые раздельные окружения
остаются поддержаны через явные поля `python`.

## Накладные расходы на оборудование и хранилище

Следующие измерения взяты из текущего протестированного хоста: GeForce RTX 5080 16 ГБ,
драйвер 580.173.02, сборки PyTorch для CUDA 12.8, включено клонирование голоса по референсу,
включен прогрев, отключен шумоподавитель VoxCPM2 и отключен CosyVoice TensorRT. Это
показатели планирования мощностей, а не модельные гарантии.

| Параметр | VoxCPM2 | CosyVoice3 |
| --- | ---: | ---: |
| Дополнительное отдельное окружение | не требуется | не требуется |
| Снимок модели на диске | около 4,7 ГиБ | около 9,1 ГиБ |
| Данные нормализации только для worker | нет | около 21 МБ WeText FST |
| Загрузка worker-модели и прогрев | около 16,5 с | около 11,4 с в изолированном smoke-тесте |
| Формат вывода, показанный в тестах | моно PCM16, 48 кГц | моно PCM16, 24 кГц |

Общая зависимость `rutextnorm` находится в основном окружении Listener и на
протестированном хосте занимает около 0,2 МиБ. Она не создаёт дополнительный процесс
и не использует память GPU или модели. После прогрева обработка типичной фразы на этом
хосте заняла в среднем около 0,23 мс, поэтому нагрузка на CPU и задержка ничтожны по
сравнению с синтезом; точное время зависит от оборудования и текста.

В рабочем профиле VoxCPM2 постоянный рабочий процесс использует примерно 3,8 ГиБ
резидентной оперативной памяти и 7,0 ГиБ видеопамяти. Listener с CUDA STT и SpeechGate
использует ещё примерно 2,4 ГиБ видеопамяти на этом хосте. Загрузка и компиляция
временно увеличивает использование оперативной памяти хоста; предоставьте не менее 16
ГиБ системной оперативной памяти для объединенного набора процессов и желательно 32 ГиБ
или более.

VRAM CosyVoice3 зависит от провайдеров ONNX, `fp16`, TensorRT и версии модели, поэтому
измеряйте ее на целевой установке, а не рассматривайте размер файла веса как
использование VRAM. На графическом процессоре емкостью 16 ГБ не запускайте одновременно
оба нейронных worker-процесса вместе с CUDA STT. В штатном режиме Listener запускает
только один из них.

Проверенное единое окружение занимает около 8,6 ГиБ без части опциональных пакетов
Listener и заменяет примерно 23 ГиБ прежних трёх окружений. Снимкам моделей всё ещё
нужно около 14 ГиБ; при загрузке кэш может временно потребовать ещё одну копию.

Полезные измерения после запуска:

```bash
nvidia-smi
ps -eo pid,rss,cmd | rg 'voxcpm2_worker|cosyvoice3_worker|main.py'
du -sh /opt/listener-tts/*
```

## Общие системные пакеты

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 sox libsox-dev git git-lfs
```

В примерах ниже используется Conda. Micromamba можно использовать с эквивалентными
командами `create -p`. Выберите любой корневой каталог, в котором свободно не менее 40
ГиБ; `/opt` — это всего лишь пример:

```bash
sudo mkdir -p /opt/listener-tts
sudo chown "$USER":"$USER" /opt/listener-tts
export LISTENER_ROOT=/absolute/path/to/Listener
export TTS_ROOT=/opt/listener-tts
```

Переменные предназначены только для удобства установки. Пути в JSON могут быть
абсолютными или относительными к корню проекта Listener; переменные оболочки в
`config.json` не раскрываются.

## Установка единого окружения Python 3.12

Для новой установки создайте окружение Listener и установите единый профиль вместо
двух старых файлов зависимостей:

```bash
cd "$LISTENER_ROOT"
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-unified-cu128-py312.txt
.venv/bin/python -m pip check
```

Не заменяйте URL CPU-only TorchCodec стандартным wheel. На проверенном хосте с CUDA
12.8 стандартный TorchCodec 0.15 искал `libnvrtc.so.13`. Закреплённый CPU-wheel
устраняет ошибку загрузчика и не меняет размещение инференса.

Храните ресурсы моделей внутри корня Listener, чтобы production не зависел от
разработческого checkout:

```text
models/tts/voxcpm2/model/
models/tts/cosyvoice3/CosyVoice/
models/tts/cosyvoice3/wetext/
references/voxcpm2/Nata.wav
references/cosyvoice3/Nata.wav
```

Каталог CosyVoice должен содержать подмодуль `third_party/Matcha-TTS` и модель в
`pretrained_models/Fun-CosyVoice3-0.5B`. Это стандартные пути Listener, поэтому поля
путей можно не указывать в `config.json`. Для другой раскладки поддерживаются
относительные и абсолютные override-пути.

## Старые раздельные окружения

Следующие два раздела сохранены для отката и машин, где проверенный профиль Python 3.12
не подходит. При их использовании явно задайте поле `python` соответствующего бэкенда.

### Установка VoxCPM2 в отдельное окружение

Создайте среду Python 3.11 и установите протестированный профиль Blackwell:

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

Зафиксированный профиль зависимостей специально протестирован на оборудовании серии RTX
50. Для другого графического процессора выберите сборку PyTorch, совместимую с этим
графическим процессором и драйвером, но сохраните Python 3.11 и отдельную среду.

Загрузите `openbmb/VoxCPM2` перед включением автономного режима:

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

Скопируйте чистую эталонную запись в `$VOX_ROOT/reference/voice.wav`. Предпочитайте моно
монофонический WAV с одним диктором, небольшим шумом или реверберацией и примерно 5–15
секундами речи. Расшифровка рядом с ним полезна для фиксации происхождения и будущих
изменений модели, но текущий worker VoxCPM2 Listener создаёт свой кэш подсказок
непосредственно из WAV и не использует `prompt_text`.

Проверьте среду, не загружая модель:

```bash
"$VOX_ENV/bin/python" - <<'PY'
import torch
from voxcpm import VoxCPM

print(torch.__version__, torch.cuda.is_available())
print(VoxCPM)
PY
```

### Установка CosyVoice3 в отдельное окружение

Клонируйте CosyVoice рекурсивно. Субмодуль Matcha-TTS является обязательным:

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

Не устанавливайте дополнительно `requirements.txt` из репозитория поверх этого профиля
на графическом процессоре RTX 50-й серии: его старые зафиксированные версии Torch/cu121 и ONNX Runtime
заменяют пакеты, совместимые с Blackwell. На старых поддерживаемых графических
процессорах upstream-профиль может подойти, но его всё равно следует устанавливать в эту
выделенной среде Python 3.10.

Загрузите модель и нормализатор WeText перед установкой `local_files_only=true`:

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

Настроенный каталог WeText должен содержать все четыре файла ниже:

```text
en/tn/tagger.fst
en/tn/verbalizer.fst
zh/tn/tagger.fst
zh/tn/verbalizer.fst
```

Скопируйте чистый эталонный WAV в `$COSY_ROOT/reference/voice.wav`. Listener кэширует
речевые признаки один раз при запуске worker-процесса. Как и в случае с текущим
адаптером Vox, поле конфигурации `prompt_text` зарезервировано, но не используется этим
worker-процессом.

Проверьте импорт и CUDA перед запуском Listener:

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

## Конфигурация Listener

Для нейронных бэкендов требуется `speaker.tts_mode="persistent"`. Возьмите за основу
следующую структуру и выберите ровно один `tts.backend`. Стандартные самодостаточные
пути ресурсов повторять в конфиге не требуется:

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
      "device": "cuda",
      "local_files_only": true,
      "fp16": true,
      "load_trt": false,
      "warmup": true,
      "speed": 1.0,
      "enable_vocal_events": false
    }
  }
}
```

Если поле `python` отсутствует, оба worker-процесса используют тот же интерпретатор,
что и Listener. Для старого раздельного окружения по-прежнему можно указать абсолютный
путь `python`.

Чтобы использовать CosyVoice3, измените только:

```json
{
  "speaker": {
    "tts": {
      "backend": "cosyvoice3"
    }
  }
}
```

Для fallback требуется рабочая конфигурация Piper. Сбой при запуске нейронной модели
переводит текущий процесс Listener на резервный бэкенд; исправьте ошибку и
перезапустите Listener, чтобы повторить попытку основного бэкенда.

## Smoke-тест и работа службы

Сначала подтвердите эффективную конфигурацию без загрузки нейронной модели:

```bash
cd "$LISTENER_ROOT"
.venv/bin/python -m speaker.cli --config config/config.json print-config
```

Затем синтезируйте одно предложение. Команда загрузит и прогреет выбранный worker,
воспроизведёт результат, а затем завершит worker-процесс:

```bash
.venv/bin/python -m speaker.cli --config config/config.json \
  --log-level INFO say '😔 Иногда тишина остаётся единственной собеседницей.'
```

Запустите Listener нормально только после того, как это удастся:

```bash
.venv/bin/python main.py
.venv/bin/python utils/listenerctl.py speaker status
```

Статус Speaker отображает выбранный бэкенд, PID worker-процесса, ошибки запуска и генерации,
время до первого аудиофрагмента, фрагменты PCM и состояние воспроизведения. С
пользовательским сервисом используйте:

```bash
journalctl --user -u listener.service -f
```

## Изолированное потоковое воспроизведение

Нейронные worker-процессы выдают моно PCM16-фрагменты. В Linux Listener по умолчанию
передаёт их лёгкому subprocess `pacat`, а при необходимости использует `pw-cat`.
Поэтому при обычной нейронной озвучке PortAudio не работает внутри процесса Listener:
падение или принудительное завершение аудиопроцесса не может завершить Listener или
постоянный worker модели.

Проигрыватель привязан ко всему `run_id` OpenClaw, а не к отдельному предложению. Он
один раз накапливает начальный prebuffer, остаётся открытым между сегментами очереди и
сливает данные только после маркера завершения ответа. Ducking включается непосредственно
перед запуском subprocess и восстанавливается после закрытия stdin и опустошения буферов
аудиосервера. Файловый рендер не использует playback и не затрагивается.

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

`auto` предпочитает `pacat`, затем `pw-cat`; на других ОС может использоваться
`sounddevice`. Явные значения: `pacat`, `pwcat`, `sounddevice`.
`streaming_command` переопределяет путь к исполняемому файлу. `prebuffer_ms` задаёт
одноразовый стартовый запас, `latency_ms` — запрашиваемый буфер аудиосервера,
`queue_ms` — предел ожидающего PCM внутри Listener, а `restart_attempts` ограничивает
повторные запуски проигрывателя после ошибки в текущем ответе.

Статус Speaker показывает выбранный backend, PID дочернего процесса, состояние очереди
и prebuffer, число записанных байтов, перезапуски, последний exit code и хвост stderr.
Listener не повторяет автоматически сегмент после падения во время playback: его начало
могло уже прозвучать. Новый проигрыватель запускается для следующего сегмента.

## Создание WAV без второго TTS-процесса

Когда Listener уже работает с `tts_mode="persistent"` и выбранным бэкендом
`voxcpm2` или `cosyvoice3`, через control API можно ставить в очередь задачи файловой
озвучки:

```bash
.venv/bin/python utils/listenerctl.py tts-file render \
  --text '😔 Иногда тишина всё объясняет.' \
  --filename quiet-thought \
  --wait
```

После завершения команда печатает абсолютный `path` к WAV. Длинный текст можно прочитать
из UTF-8-файла через `--text-file`. Управление задачами:

```bash
.venv/bin/python utils/listenerctl.py tts-file list
.venv/bin/python utils/listenerctl.py tts-file status JOB_ID
.venv/bin/python utils/listenerctl.py tts-file cancel JOB_ID
```

Соответствующие защищённые общими настройками control API маршруты:

```text
POST /tts/files
GET  /tts/files
GET  /tts/files/{job_id}
POST /tts/files/{job_id}/cancel
```

`POST /tts/files` принимает `text`, необязательный стиль из разрешённого списка и
необязательный `filename`. `filename` служит только меткой: Listener удаляет компоненты
пути, добавляет уникальный суффикс и всегда пишет внутрь `file_render.output_dir`.
Сначала создаётся `.wav.part`, который атомарно переименовывается после завершения
WAV-заголовка. При ошибке или отмене частичный файл удаляется. Метаданные задач хранятся
в памяти, причём сохраняются только последние `max_completed_jobs` завершённых записей;
при перезапуске Listener все записи очищаются. Готовые WAV остаются на диске, пока
пользователь не удалит или не заархивирует их.

Этот путь использует существующий `NeuralWorkerClient`: отдельный worker
VoxCPM2/CosyVoice3 не запускается, второй экземпляр модели в RAM или VRAM не появляется.
Если worker ещё не был загружен, первая задача оплачивает обычное время загрузки и
прогрева. Для моно PCM16 файл занимает примерно 2,75 МиБ/мин при 24 кГц CosyVoice3 или
5,49 МиБ/мин при 48 кГц VoxCPM2 плюс 44 байта WAV-заголовка.

Одновременно выполняется одна файловая задача. Длинный текст делится на ограниченные
речевые сегменты, а общая блокировка генерации освобождается между ними: ожидающая
озвучка ответа может выполниться до следующего файлового сегмента. Уже начатый сегмент
модели не прерывается. Отмена playback и отмена файловой задачи разделены по владельцам,
поэтому одна операция не повреждает постоянный stdio-поток другой. Файловый рендер
намеренно использует только выбранную нейронную модель: при ошибке задача получает
состояние `failed`, а Piper не сохраняется под именем нейронного бэкенда.

Чтобы OpenClaw создавал такие файлы, установите `openclaw/skills/listener-tts-file`
вместе с `listener-control`. Навык обращается к control API и явно запрещает прямой
запуск любой из изолированных сред моделей.

## Инструкции по созданию стиля эмодзи

OpenClaw может запросить стиль речи, поставив один эмодзи из разрешенного списка в
начале предложения. Listener удаляет все emoji из произносимого текста и отправляет
фиксированную, нейтральную для модели инструкцию для ведущих emoji из разрешённого
списка. Произвольный текст помощника никогда не превращается в инструкцию.

| Ведущий emoji | Стиль | Смысл инструкции |
| --- | --- | --- |
| 🙂 😊 ❤️ | `warm` | теплый, дружелюбный, нежный |
| 😄 🎉 ✨ | `cheerful` | веселый, жизнерадостный, живой |
| 😌 | `calm` | спокойный, мягкий, неторопливый |
| 🤔 🧐 | `thoughtful` | вдумчивый, размеренный |
| 😔 😢 😭 💔 | `sad` | приглушенный, грустный, чуткий |
| 😠 😡 | `firm` | контролируемый гнев, отсутствие криков |
| 😮 😲 🤯 | `surprised` | ясное, естественное удивление |
| 😏 😼 😉 | `playful` | игривый, слегка дразнящий |
| ⚠️ 🚨 | `urgent` | срочно, четко, немного быстрее |
| 😂 🤣 | `amused` | позабавился, сдерживая смех |

VoxCPM2 получает фиксированную инструкцию в виде префикса в скобках к целевому тексту.
CosyVoice3 токенизирует её как инструкцию. При `enable_vocal_events=true`
стиль `amused` может дополнительно вставить поддерживаемое событие `[laughter]`; по
умолчанию это отключено, поскольку качество голосовых событий зависит от версии голоса и
модели. Piper игнорирует метаданные стиля.

### Общая нормализация русского текста

`tts.normalize_numbers=true` включает узкую обёртку Listener над `rutextnorm` до
передачи текста любому TTS-бэкенду. Поэтому она одинаково работает для Piper,
VoxCPM2, CosyVoice3, резервного Piper и создания WAV нейронной моделью. Изменяются
только выбранные числовые или математические фрагменты и непосредственно связанные
с ними даты, время, проценты, валюты или единицы измерения. Остальной текст
копируется без изменений: Listener не транслитерирует латинские слова,
идентификаторы моделей и обычный текст ответа. URL, IP-адреса, телефоны, абсолютные
пути, inline-код и управляющие токены защищены от обработки.

В версиях точка произносится явно: например, `GPT-5.6-terra` преобразуется в
`GPT-пять точка шесть-terra`, а латинские части остаются исходными. Русская
десятичная величина `3.5 кг` обрабатывается как `3,5 кг`. Инструкции стиля и токены
`[laughter]` добавляются вне нормализации.

Узкий слой также произносит одиночный математический `=` как «равно» и обрабатывает
календарные дни с русскими названиями месяцев (`1 августа` → `первое августа`,
`с 1 августа` → `с первого августа`). Согласование существительных с помощью
неполного словаря намеренно не выполняется: OpenClaw должен сразу сформировать
правильный исходный текст, например `2 чашки`. Операторы `==`, `>=` и `!=`, а также
выражения внутри защищённого кода остаются без изменений.

CosyVoice3 по-прежнему напрямую вызывает свой токенизатор, поэтому стандартный
английский frontend WeText после этого не применяется. `rutextnorm` устанавливается
в основное окружение Listener через `requirements.txt`; изолированным окружениям
моделей пакет не нужен. Если нормализация включена, отсутствие пакета считается
ошибкой создания TTS-движка; неожиданная ошибка при обработке отдельной реплики
приводит к использованию исходного текста. Отключить функцию можно через
`tts.normalize_numbers=false` или `SPEAKER_TTS_NORMALIZE_NUMBERS=false`. Старые
CosyVoice-специфичные ключ конфигурации и переменная окружения принимаются для
миграции, но в новых установках следует использовать общие имена.

В случае `leading_emoji_only=true` завершающий эмодзи доступен только для отображения.
При `inherit_within_run=true` ведущий emoji задаёт стиль следующим
предложения в том же OpenClaw `runId`, затем состояние отбрасывается при финализации,
ошибке, прерывании или перебивании. Это важно для потоковой передачи: завершающий
emoji появляется после того, как предложение уже могло попасть в очередь TTS.

Установите прилагаемое соглашение в OpenClaw один раз:

```bash
OPENCLAW_WORKSPACE="$(openclaw config get agents.defaults.workspace)"
sed -n '1,$p' openclaw/prompts/listener-tts-style.md \
  >> "$OPENCLAW_WORKSPACE/AGENTS.md"
```

## Поиск неисправностей

- `worker failed to start`: запустите фрагмент проверки среды и проверьте `stderr_tail`
  worker-процесса в `speaker status`.
- `CUDA is not available`: проверьте драйвер NVIDIA и сборку Torch в среде с помощью
  `torch.cuda.is_available()`.
- `local WeText FST directory not found`: загрузите WeText, находясь в сети, и укажите
  `wetext_path` на каталог, содержащий `en/tn` и `zh/tn`.
- `CosyVoice Matcha-TTS submodule not found`: запустите `git submodule update --init
  --recursive` внутри репозитория CosyVoice.
- Время первого запуска истекло: увеличьте `tts.startup_timeout_s`; компиляция и
  холодный кэш файловой системы выполняются медленнее, чем последующие запуски.
- Нейронный синтез неоднократно дает сбой: Piper остается активным после достижения
  настроенного порога ошибки. Проверьте `tts.primary.worker.stderr_tail` в статусе
  Speaker.
- Собственная речь Listener становится тихой после приглушения: запустите
  `.venv/bin/python utils/listenerctl.py speech_gate_reset --reason "recover ducking"`.
  Потоки Speaker исключаются из обычного приглушения, а сброс также восстанавливает
  застрявшую громкость маршрута PipeWire/WirePlumber.

Оставляйте `local_files_only=true` при эксплуатации только после завершения загрузки
модели и WeText. Это предотвращает незаметный доступ к сети или изменение установленного
моментального снимка при перезапуске службы.
