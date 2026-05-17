# Whisper STT движок

Модуль `audio.stt.whisper_engine.WhisperEngine` инкапсулирует загрузку и работу с
моделями [Whisper](https://github.com/openai/whisper) через библиотеку
`faster-whisper`. Он конфигурируется секцией `audio.stt` файла `config/config.json` и
выполняет все вспомогательные задачи:

* инициализирует модель по имени (`tiny`, `base`, `small`, `medium`, `large` или
  путь к локальной директории) и подбирает устройство/тип вычислений;
* приводит входящий поток PCM (`int16`) к целевой частоте дискретизации Whisper
  (`audio.stt.sample_rate`, по умолчанию 16 кГц) с помощью потокового
  ресемплера;
* конвертирует значения в диапазон `[-1, 1]` и передаёт их в
  `WhisperModel.transcribe`, добавляя указанные декодерные опции;
* возвращает список распознанных фрагментов текста (по одному элементу на
  сегмент, который вернула модель).

## Жизненный цикл движка

1. **Инициализация.** Экземпляр создаётся с конфигурацией `WhisperSttCfg`. При
   активированном флаге `enabled` движок сразу загружает веса и готов к
   инференсу.
2. **Подготовка модели.** WhisperEngine создаёт `WhisperModel`, передавая
   параметры `device`, `compute_type`, `cpu_threads`, `num_workers` и
   `download_root`. Отсутствие `faster-whisper` приводит к исключению
   `RuntimeError`.
3. **Ресемплинг.** Метод `transcribe` принимает произвольные блоки PCM с
   указанной частотой дискретизации. Если `sample_rate` отличается от
   `audio.stt.sample_rate`, данные проходят через потоковый ресемплер
   `StreamingResampler`.
4. **Инференс.** После нормализации сигнала выполняется вызов
   `WhisperModel.transcribe`, которому передаются параметры декодера
   (beam-search, температура, фильтр VAD и т. д.). Результирующие строки
   очищаются от ведущих/замыкающих пробелов.

## Метод `transcribe`

```python
WhisperEngine.transcribe(batch_pcm: np.ndarray | bytes | Iterable[int], sample_rate: int) -> list[str]
```

* `batch_pcm` — моно PCM `int16` (байты, numpy-массив или итерируемая
  последовательность). Для многоканального аудио микширование выполняется на
  стороне вызывающего кода.
* `sample_rate` — частота дискретизации поступившего блока.

Метод возвращает список текстовых гипотез. При отключённом модуле (`enabled =
false`) возвращается пустой список.

## Конфигурация `audio.stt`

### Базовые параметры

| Ключ | Назначение | Значение по умолчанию |
|------|------------|-----------------------|
| `enabled` | Включает Whisper STT. | `false` |
| `model` | Имя модели `faster-whisper`, `avazir/faster-distil-whisper-large-v3-ru`, `large-v3`, `large-v3-turbo` либо путь к каталогу с весами. | `"small"` |
| `device` | Устройство инференса (`"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, ...). При `null` библиотека подбирает его автоматически. | `null` |
| `compute_type` | Тип вычислений (`"default"`, `"int8"`, `"int8_float16"`, ...). | `null` |
| `download_root` | Каталог для кэша моделей. | `null` |
| `blacklist_path` | Путь до текстового blacklist-файла для постфильтрации Whisper-фраз. Пути считаются от корня проекта. | `"config/blacklist.txt"` |
| `local_files_only` | Запрещает автоматическое скачивание модели и заставляет `faster-whisper` работать только с локальными файлами. | `false` |
| `cpu_threads` | Количество потоков для CPU-инференса. Игнорируется, если `null`. | `null` |
| `num_workers` | Количество параллельных воркеров `faster-whisper`. | `null` |
| `language` | ISO-код языка (например, `"ru"`, `"en"`). При `null` Whisper пытается определить язык автоматически. | `null` |
| `task` | Тип задания (`"transcribe"` или `"translate"`). | `"transcribe"` |
| `sample_rate` | Целевая частота дискретизации аудио (Гц). | `16000` |
| `partial_topic` / `final_topic` | Темы EventBus для частичных и финальных гипотез (берутся из `cfg.events.audio`). | `"audio/stt/partial"` / `"audio/stt/final"` |
| `min_confidence` | Минимальная уверенность для фиксации фразы. | `0.35` |
| `stability_timeout_s` | Таймаут ожидания новых обновлений. | `1.2` сек |
| `queue_wait_s` | Таймаут ожидания новых сегментов от `BufferedSpeechWriter`. | `0.2` сек |
| `enable_punctuation` | Добавлять завершающий знак пунктуации при публикации финального текста. | `true` |

### Параметры декодера

Следующие ключи передаются напрямую в `WhisperModel.transcribe`, если заданы в
конфигурации:

* `beam_size`, `best_of`, `patience`, `length_penalty` — управление beam search;
* `temperature`, `temperature_increment_on_fallback`,
  `prompt_reset_on_temperature` — температурные параметры;
* `initial_prompt` — префикс для первой гипотезы;
* `condition_on_previous_text` — наследовать ли предыдущий текст в следующих
  окнах;
* `compression_ratio_threshold`, `logprob_threshold`,
  `no_speech_threshold`, `max_initial_timestamp` — эвристики остановки и
  фильтрации;
* `suppress_tokens`, `suppress_blank` — управление подавлением токенов;
* `vad_filter`, `vad_parameters` — активация VAD фильтра внутри
  `faster-whisper`;
* `word_timestamps`, `without_timestamps` — включение таймштампов.

Поля можно сбрасывать в `null`, чтобы использовать значения по умолчанию
`faster-whisper`.

## Потоковый транскрайбер

Модуль `audio.stt.streaming.WhisperStreamingTranscriber` связывает
`BufferedSpeechWriter`, `WhisperEngine` и системную шину событий для непрерывной
транскрипции. Он работает в асинхронной задаче, читает сегменты из очереди
`BufferedSpeechWriter.queue`, передаёт их в Whisper и управляет накоплением
частичных гипотез.

Ключевые обязанности:

1. **Буферизация и состояние.** Транскрайбер отслеживает текущую гипотезу,
   таймштампы обновлений и метаданные сегментов (длительность, уверенность VAD,
   границы отрезков).
2. **Частичные публикации.** После каждого обновления гипотезы публикует событие
   `audio/stt/partial` с полями `text`, `raw_text`, `is_final=false` и
   метаданными сегмента.
3. **Финализация.** По истечении таймаута стабильности или при принудительном
   вызове формирует финальную фразу, применяет постобработку (нормализация
   пробелов, капитализация, опциональная пунктуация) и рассылает событие
   `audio/stt/final`.
4. **Интеграция с LLM.** Финальный текст помещается в асинхронную очередь
   `llm_queue` как строка; дополнительно можно передать callback `on_final`,
   который будет вызван для каждой финальной реплики. В стандартном
   `AudioAgent` этот callback публикует итог в `cfg.events.llm.input_text`.

Внутренний payload callback'а содержит `pcm_data` исходного сегмента, но в
событие `audio/stt/final` это поле не публикуется: перед отправкой в EventBus
оно удаляется.

## Гейтинг направленности речи

Перед отправкой распознанной речи в LLM срабатывает двухэтапный гейт
`llm.speech_gate.SpeechDirectionGate`:

* **Правила + скоринг.** Текст проверяется на имя ассистента из OpenClaw
  identity-файла (строки `Name:` / `Имя:`) и маркеры обращения: командные глаголы,
  вопросительные и модальные слова, вежливые формулы. Маркеры читаются один раз
  при старте из файла `speech_gate.patterns_file` (например,
  `config/speech_gate_patterns.json`), при его отсутствии используются
  inline-списки из конфигурации. Если правило набирает порог
  `speech_gate.rules_threshold` (0.7 по умолчанию), запрос считается
  адресованным и пропускается без ML-проверки.
* **ML-классификатор.** Для сомнительных реплик запускается модель
  `models/directed-ruElectra-small-fp16` (параметры и устройство задаются в
  `speech_gate.model`). Итоговый скор рассчитывается как
  `0.6 * ml + 0.4 * rules`, и при значении ниже `speech_gate.final_threshold`
  (0.5) фраза отбрасывается с диагностикой в логах в режиме DEBUG.

После удачного прохождения гейта включается «attention mode» на несколько
секунд (`speech_gate.attention_window_seconds`), когда последующие реплики
пропускаются без фильтров. Если фраза заканчивается на маркеры продолжения
(`speech_gate.continuation_patterns`, например «и ещё», «а также»), окно
продлевается на `speech_gate.attention_extension_seconds`.

### Режимы работы гейта

Режимы (`standby/mute/normal/chatty`) можно переключать извне. Назначение:

- **normal** — стандартный режим. Срабатывают правила, при необходимости
  подключается ML-классификатор; после успешного обращения включается
  «attention mode».
- **standby** — режим ожидания: гейт блокирует все реплики независимо от
  правил и ML.
- **mute** — «тихий» режим: пропускаются только обращения по имени ассистента,
  причём имя должно быть в начале распознанной фразы; остальные реплики
  блокируются.
- **chatty** — «болтливый» режим: реплики пропускаются без фильтрации, гейт
  фактически всегда открыт.

### Runtime-переключение режимов

Во время работы `main.py` поднимает локальный control API
`http://127.0.0.1:18790`. Пользоваться им проще через CLI:

```bash
.venv/bin/python utils/listenerctl.py speech-gate status
.venv/bin/python utils/listenerctl.py speech-gate set-mode mute --reason "quiet mode"
.venv/bin/python utils/listenerctl.py speech-gate set-mode chatty --ttl 600
.venv/bin/python utils/listenerctl.py speech-gate set-mode standby --ttl 300
.venv/bin/python utils/listenerctl.py speech-gate set-mode normal
.venv/bin/python utils/listenerctl.py speech-gate reset --reason "recover voice"
```

`normal` отменяет временный режим. `mute` и `chatty` могут быть постоянными или
временными. Через HTTP API и `listenerctl` режим `standby` принимается только с
TTL, чтобы не запереть голосовое управление. Любое runtime-переключение
сбрасывает attention-window.

Секция `control` в `config/config.json`:

```json
{
  "control": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 18790,
    "token": null,
    "max_ttl_seconds": 86400
  }
}
```

CLI читает `LISTENER_CONTROL_URL` и `LISTENER_CONTROL_TOKEN`. Если `host` не
loopback, Listener требует непустой `control.token`.

Тот же control API также предоставляет runtime-управление встроенным Speaker:

```bash
.venv/bin/python utils/listenerctl.py speaker status
.venv/bin/python utils/listenerctl.py speaker off
.venv/bin/python utils/listenerctl.py speaker on
```

Команда `speech-gate reset` нужна как recovery-кнопка: она возвращает
`speech_gate` в `normal`, заново включает `speaker` и принудительно
восстанавливает все запомненные ducking-volume'ы, если после barge-in или
прерванной генерации голос/beep Listener остались приглушёнными.

OpenClaw-интеграция v1 реализована как workspace skill:

```bash
mkdir -p "$(openclaw config get agents.defaults.workspace)/skills"
cp -R openclaw/skills/listener-control \
  "$(openclaw config get agents.defaults.workspace)/skills/"
```

В OpenClaw `TOOLS.md` удобно добавить локальную заметку:

```markdown
### Listener
- LISTENER_HOME=<path-to-Listener>
- Control URL: http://127.0.0.1:18790
- Use: $LISTENER_HOME/.venv/bin/python $LISTENER_HOME/utils/listenerctl.py
```

### Настройка `speech_gate`

Ключи секции `speech_gate` в `config/config.json` и их назначение:

| Ключ | Значение | По умолчанию |
|------|----------|--------------|
| `enable` | Включает/выключает гейт. При `false` реплики проходят без проверки правил и ML, режим внимания не используется. | `true` |
| `rules_threshold` | Порог срабатывания правил. Если скор из словарей (имя, глаголы, маркеры) превышает это значение, фраза считается обращением без запуска ML. | `0.7` |
| `final_threshold` | Итоговый порог после смешивания правил и ML (`0.6 * ml + 0.4 * rules`). Реплики ниже порога игнорируются. | `0.5` |
| `attention_window_seconds` | Длительность «attention mode» после успешного обращения, в течение которой гейт пропускает реплики без фильтров. | `8.0` |
| `attention_extension_seconds` | На сколько секунд продлевается окно внимания при обнаружении маркеров продолжения («и ещё», «а также» и т. п.). | `3.0` |
| `patterns_file` | Путь до JSON со списками маркеров (`command_verbs`, `continuation_patterns`, и др.). Пути считаются от корня проекта. `assistant_names` в этом файле игнорируется. | `"config/speech_gate_patterns.json"` |
| `identity_file` | Путь до OpenClaw identity Markdown. `null` или `"auto"` включает автообнаружение через `OPENCLAW_IDENTITY_FILE`, `OPENCLAW_WORKSPACE`, `OPENCLAW_STATE_DIR`, `OPENCLAW_CONFIG_PATH`, `~/.openclaw/openclaw.json` и профильные каталоги `~/.openclaw-*`. Явный относительный путь считается от корня Listener. | `null` |
| `model.path` | Каталог модели `directed-ruElectra-small-fp16` для классификатора направленности речи. | `"models/directed-ruElectra-small-fp16"` |
| `model.device` | Устройство для инференса классификатора (`cpu`, `cuda`, `cuda:0` и т. п.). | `"cpu"` |
| `model.threshold` | Порог вероятности для ответа модели (до смешивания с правилом). | `0.7` |
| `model.max_length` | Максимальная длина токенизации входного текста для классификатора. | `64` |

### Звуковые индикаторы

Listener умеет проигрывать короткие уведомительные сигналы для ключевых
переходов голосового контура. Конфигурация задаётся в секции `indicators`
файла `config/config.json`.

| Ключ | Значение | По умолчанию |
|------|----------|--------------|
| `enabled` | Включает/выключает звуковые индикаторы. | `true` |
| `backend` | `auto`, `sounddevice`, `winsound` или `none`. На Linux обычно используется `sounddevice`, на Windows возможен fallback в `winsound`. | `"auto"` |
| `output_device_index` | Индекс выходного audio-device для сигналов. `null` использует системное устройство по умолчанию. | `null` |
| `sample_rate` | Частота дискретизации синтезированных сигналов. | `24000` |
| `volume` | Громкость сигналов в диапазоне `0.0..1.0`. | `0.18` |
| `queue_maxsize` | Максимум сигналов в очереди playback. При переполнении новые сигналы отбрасываются. | `8` |
| `rejected` | Проигрывать сигнал, когда фраза отвергнута SpeechGate. | `true` |
| `forwarded` | Проигрывать сигнал, когда фраза успешно отправлена в OpenClaw. | `true` |
| `local_handled` | Проигрывать сигнал, когда служебная команда обработана внутри Listener. | `true` |
| `interrupted` | Проигрывать сигнал для успешного barge-in или stop в OpenClaw. | `true` |

По умолчанию есть четыре разных коротких сигнала:

1. фраза отвергнута SpeechGate;
2. фраза прошла SpeechGate и успешно отправлена в OpenClaw;
3. служебная voice-команда (`mute`, `normal`, `standby`) обработана внутри Listener;
4. перебивка или stop-команда успешно дошла до OpenClaw.

Можно выключать типы сигналов по отдельности, например:

```json
{
  "indicators": {
    "enabled": true,
    "rejected": false,
    "forwarded": true,
    "local_handled": false,
    "interrupted": true
  }
}
```

### Формат `speech_gate_patterns.json`

Файл описывает списки паттернов, которыми оперирует гейт при расчёте `rules`-скора. Структура представляет собой объект с массивами строк. Пример:

```json
{
  "command_verbs": ["включи", "останови", "покажи"],
  "politeness_markers": ["пожалуйста", "будь добра"],
  "question_markers": ["как", "почему", "зачем"],
  "modal_markers": ["можешь", "нужно ли", "давай"],
  "continuation_patterns": ["и ещё", "а также", "тогда"],
  "local_mute_commands": ["замолчи", "помолчи"],
  "local_normal_commands": ["говори", "слушай"],
  "local_standby_commands": ["выключись", "не слушай"],
  "local_abort_commands": ["стоп", "хватит"],
  "local_barge_in_commands": ["нет", "не так", "подожди", "точнее"]
}
```

Имя ассистента не хранится в `speech_gate_patterns.json`: поле
`assistant_names` в этом файле игнорируется. Основной источник имени:
`IDENTITY.md` в workspace OpenClaw. Listener пытается найти его автоматически:
сначала через переменные окружения `OPENCLAW_IDENTITY_FILE`, `OPENCLAW_WORKSPACE`,
`OPENCLAW_STATE_DIR`, `OPENCLAW_CONFIG_PATH`, затем через конфиги
`~/.openclaw/openclaw.json`, `~/.openclaw-dev/openclaw.json` и
`~/.openclaw-*/openclaw.json`. Если OpenClaw хранит workspace нестандартно,
задайте путь вручную:

```json
{
  "speech_gate": {
    "identity_file": "/path/to/openclaw/workspace/IDENTITY.md"
  }
}
```

Пример содержимого identity-файла:

```markdown
Name: Kissa
Имя: Кисса
```

Можно опустить любые поля. Listener умеет объединять списки из файла с
дополнительными inline-паттернами из `config/config.json`, если они там заданы,
но в проекте рекомендуется держать основной словарь именно в
`config/speech_gate_patterns.json`, а inline-списки в `config/config.json`
оставлять пустыми. Так проще избегать двойных определений и смысловых
пересечений между обычными командными глаголами и локальными control-фразами.
Если файл по указанному пути не найден или не читается, в лог пишется
предупреждение, а гейт продолжает работу только с inline-паттернами из
`config/config.json` и именем из identity-файла.

Локальные команды управления режимами и stop-команды принимаются только в форме
`<имя ассистента> + команда`, то есть имя должно быть в начале распознанной
фразы: `Марина, помолчи`, `Марина, говори`, `Марина, стоп`. Слова вроде
`включи` или `говори` без имени в начале не переключают режим.

Это же правило важно для `mute`: в этом режиме гейт пропускает только фразы,
которые начинаются с имени ассистента. Вхождение имени в середине реплики не
достаточно.

Есть отдельное исключение для локальной voice-команды `Марина, отключись`:
внутренний `SpeechGateAgent` может переключить Listener в `standby` без TTL,
потому что эта команда обрабатывается локально до отправки в OpenClaw. Выход из
такого состояния — например, `Марина, говори` или ручное
`listenerctl.py normal`.

`local_barge_in_commands` описывает явные фразы перебивки. Они тоже требуют имя
ассистента в начале: `Марина, нет, я имел в виду...`, `Марина, точнее...`.
Обычные обращения вроде `Марина, какая погода?` не отправляются через
`sessions.steer` и идут в OpenClaw обычным `chat.send`.

`command_verbs` стоит использовать только для обычных пользовательских задач
вроде `покажи`, `найди`, `объясни`, `напомни`. Фразы управления Listener вроде
`замолчи`, `слушай`, `стоп`, `остановись` лучше держать только в
`local_*_commands`, чтобы они не дублировались в общем rules-скоре.

### Параметры исполнения

Объект `StreamingTranscriberConfig` управляет поведением транскрайбера во время
работы. Все значения по умолчанию наследуются из `WhisperSttCfg`, но могут быть
переопределены в рантайме.

### Жизненный цикл

1. Создайте `BufferedSpeechWriter` и передайте его в конструктор
   `WhisperStreamingTranscriber` вместе с конфигурацией STT и, при необходимости,
   объектом `EventBus`.
2. Вызовите `await transcriber.start()`, чтобы запустить фоновую задачу. С этого
   момента транскрайбер будет потреблять сегменты из `writer.queue`.
3. По завершении работы вызовите `await transcriber.stop()` или используйте
   контекстный менеджер `async with`, чтобы дождаться публикации всех финальных
   гипотез и корректно очистить состояние.

Все публикации выполняются асинхронно с защитой от исключений, поэтому сбои
подписчиков не останавливают транскрайбер. В случае переполнения `llm_queue`
финальная фраза отбрасывается с записью предупреждения в лог.

### Whisper blacklist

Если задан `audio.stt.blacklist_path`, Listener читает blacklist из этого
файла. Формат поддерживает две секции:

- `[phrases]` - точные фразы. Распознанная фраза отбрасывается целиком только
  если после удаления пунктуации и нормализации регистра она полностью
  совпадает с записью.
- `[words]` - отдельные слова. Эти слова вырезаются из распознанного текста по
  границам слов; остальная часть фразы остаётся.

Матчинг выполняется в нормализованном виде:

- регистр игнорируется;
- знаки препинания и символы не влияют на сравнение;
- слова не матчятся как подстроки: `1988` не совпадает с `19880`.

Пример:

```text
[phrases]
Спасибо
Всем пока

[words]
1988
```

В таком режиме `Спасибо!` и `Всем пока...` будут отброшены целиком, но
`Спасибо, Марина!` и `Спасибо, спасибо!` пройдут дальше. Фраза
`1988, 1988! Ура! Здорово! 1888!` превратится в
`Ура! Здорово! 1888!`; чтобы убрать `1888`, его нужно добавить в `[words]`.

## Аудио-агент

Для удобства интеграции все компоненты объединены агентом
`agents.audio_agent.AudioAgent`. Он запускает микрофонный поток, обрабатывает
аудио и управляет транскрипцией, публикуя события в системную шину.

### Архитектура пайплайна

1. **MicrophoneStream** — захватывает PCM с выбранного устройства и транслирует
   кадры в шину `cfg.events.audio.raw_frame` (по умолчанию `audio/raw_frame`).
2. **AudioProcessor** — принимает кадры через `submit()` и выполняет VAD,
   шумоподавление и другую обработку, публикуя результат как
   `cfg.events.audio.processed_frame` и события `cfg.events.audio.voice_activity`.
3. **BufferedSpeechWriter** — подписывается на события процессора, собирает
   сегменты речи с прероллом/построллом и кладёт их в очередь `writer.queue`.
4. **WhisperStreamingTranscriber** — в отдельной задаче читает сегменты из
   очереди, вызывает `WhisperEngine` и отправляет промежуточные гипотезы
   (`cfg.audio.stt.partial_topic`, то есть `cfg.events.audio.stt_partial`) и
   финальные фразы (`cfg.audio.stt.final_topic`, то есть
   `cfg.events.audio.stt_final`). В стандартном `AudioAgent` финальный текст
   затем дополнительно публикуется в `cfg.events.llm.input_text` для языковой
   модели.

## Управление состоянием

* `await AudioAgent.pause()` — приостанавливает пайплайн, закрывая активные
  компоненты.
* `await AudioAgent.resume()` — возобновляет работу, заново инициализируя поток
  и перезапуская обработку аудио.
* `await AudioAgent.close()` — окончательно завершает работу агента и
  освобождает все ресурсы.

Методы защищены от повторных вызовов и корректно обрабатывают отмену фоновых
задач. Это позволяет безопасно встроить аудиопайплайн в управляющий
оркестратор системы и гибко управлять режимами захвата речи.
