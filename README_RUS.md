# Listener

[English version](README.md)

`Listener` — это локальная система голосового ввода для OpenClaw. Она захватывает звук
с микрофона, применяет обработку звука/VAD/AEC, распознаёт речь с
помощью Whisper, фильтрует фразы по правилам направленной речи и пересылает принятый
текст в OpenClaw.

Проект в первую очередь ориентирован на Linux, а совместимость с Windows поддерживается
посредством примера конфигурации для конкретной платформы в
`config/config.windows.example.json`.

## Возможности

- Захват микрофона через `sounddevice`.
- Loopback-захват через Linux PipeWire/PulseAudio для источников мониторинга AEC.
- Поддержка WASAPI loopback в Windows.
- LiveKit AEC, дополнительный NS/HPF/AGC, пользовательское шумоподавление и VAD.
- Гибридный конвейер VAD: WebRTC + Silero.
- Whisper STT через `faster-whisper`.
- Фильтрация SpeechGate с именем помощника, загруженным из OpenClaw `IDENTITY.md`.
- API управления SpeechGate во время работы и `utils/listenerctl.py`.
- Дополнительные короткие звуковые индикаторы отклоненных, перенаправленных и локальных
  событий управления.
- Встроенная локальная озвучка через Piper, VoxCPM2 или CosyVoice3.
- Файловая озвучка нейронным TTS в WAV через тот же worker без второго процесса модели.
- Входящие в комплект навыки OpenClaw для управления Listener и создания WAV.

## Конвейер

```text
Microphone -> AudioProcessor -> BufferedSpeechWriter -> WhisperStreamingTranscriber
           -> llm/input_text -> SpeechGateAgent -> llm/accepted_phrase
           -> OpenClawInputAgent -> OpenClaw
```

Путь ответа, если голосовые ответы включены:

```text
OpenClaw Gateway -> SpeakerAgent -> SpeechEngine -> local playback
OpenClaw skill   -> Control API  -> same neural worker -> WAV file
```

Основные модули:

- `agents/` — оркестрация компонентов во время работы.
- `audio/` — микрофон, обработка, буферизация и STT.
- `core/` — конфиг, шина событий и логирование.
- `llm/` — SpeechGate логика направленной речи.
- `utils/` — CLI диагностики и управления.
- `openclaw/skills/` — OpenClaw набор навыков рабочего пространства.

## Быстрый старт

Полное руководство по установке см. в [INSTALL_RUS.md](INSTALL_RUS.md).
Дополнительное единое окружение Python 3.12 для Listener, VoxCPM2 и CosyVoice3
описано в [docs/neural-tts_RUS.md](docs/neural-tts_RUS.md).

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

В другом терминале:

```bash
curl -s http://127.0.0.1:18790/ | jq
.venv/bin/python utils/listenerctl.py status
```

## Запуск как службы

В Linux рекомендуется запускать Listener как службу `systemd --user` после успешной
первой ручной smoke-проверки:

```bash
.venv/bin/python utils/install_user_service.py --start
.venv/bin/python utils/listenerctl.py ready
journalctl --user -u listener.service -f
```

Установщик создаёт unit-файл для пути текущего checkout. См.
[docs/service_RUS.md](docs/service_RUS.md) для установки, проверок готовности, журналов,
перезапуска, удаления и поведения строгого запуска.

## Конфигурация

Основные настройки времени выполнения находятся в `config/config.json`.

Важные разделы:

- `control` — HTTP API локальной среды выполнения для управления режимом SpeechGate.
- `openclaw` — настройки пересылки через CLI/шлюз OpenClaw.
- `speaker` — встроенная TTS-озвучка ответов OpenClaw, Piper и управление приглушением.
- `indicators` — короткие сигналы уведомлений о событиях SpeechGate/OpenClaw.
- `speech_gate` — правила направленной речи, идентификационный файл и настройки
  классификатора.
- `audio.input` — частота дискретизации микрофона, каналы, размер чанка и устройство.
- `audio.processing` — AEC, VAD, AGC, фильтр верхних частот и подавление шума.
- `audio.buffer` — буферизация речевого сегмента перед STT.
- `audio.stt` — модель Whisper и настройки декодирования.
- `events` — внутренние имена тем EventBus.

Конфигурация репозитория содержит локальный пример `speaker` с путями внутри
`/home/re/src/Speaker`. В новом клоне следует выполнить одно из действий:

- установить `speaker.enabled=false` для первого запуска; либо
- заменить `speaker.piper.command`, `speaker.piper.model` и, при необходимости,
  `speaker.gateway.*` значениями, действительными на вашем компьютере.

Основным источником шаблона SpeechGate является `config/speech_gate_patterns.json`.
Встроенные массивы шаблонов в `config/config.json` поддерживаются как переопределения,
но в конфигурации проекта по умолчанию они остаются пустыми, чтобы избежать дублирования
определений.

Когда `indicators.enabled=true`, Listener воспроизводит короткие сигналы в четырех
случаях:

- фраза отклонена SpeechGate;
- фраза пересылается в OpenClaw;
- локальная голосовая команда обрабатывается внутри Listener;
- успешное прерывание/останов отправлено в OpenClaw.

Каждый из них можно переключать независимо с помощью `indicators.rejected`,
`indicators.forwarded`, `indicators.local_handled` и `indicators.interrupted`.

`indicators.ducking` может приглушать другие потоки PulseAudio/PipeWire во время
воспроизведения каждого короткого сигнала. В отличие от собственного приглушения TTS
Speaker, приглушение индикатора также приглушает любой воспроизводимый в данный момент
поток Speaker, поэтому слышны локальные звуковые сигналы остановки/прерывания.

`requirements.txt` по умолчанию настроен на CUDA 12.8 PyTorch. Для машин только с ЦП
установите `audio.stt.device="cpu"` и `speech_gate.model.device="cpu"` в
`config/config.json`.

## Диагностика производительности

Listener имеет дополнительные структурированные журналы задержки. Включите их в
`config/config.json`:

```json
"performance": {
  "enabled": true,
  "log_level": "info",
  "include_text_preview": true,
  "text_preview_chars": 80
}
```

Затем запустите Listener и найдите компактные строки производительности:

```bash
.venv/bin/python main.py 2>&1 | tee /tmp/listener-perf.log
rg "perf\\.(input|stt|speech_gate|openclaw|speaker)" /tmp/listener-perf.log
rg "stage=(speech_to_openclaw|tts_segment)" /tmp/listener-perf.log
```

Полезными первыми метриками являются `speech_to_openclaw_ms`, `stt_ms`,
`speech_gate_ms`, `openclaw_send_ms`, `synth_ms` и `playback_start_delay_ms`. В качестве
грубой цели следует избегать многосекундных пауз перед отправкой коротких локальных фраз
в OpenClaw; при озвучке одного ответа приглушение не должно сниматься, а громкость —
затухать между предложениями.

Для тестирования задержки только ввода временно установите `speaker.enabled=false`.
Чтобы изолировать STT/OpenClaw от стоимости классификатора SpeechGate, установите
`speech_gate.mode="chatty"`. Для сравнения минимальной задержки VAD установите
`audio.processing.vad.pipeline="webrtc"`.

## Модели

Веса моделей намеренно не отслеживаются в git.

Ожидаемые пути по умолчанию:

- `models/silero_vad_v6.jit`
- `models/directed-ruElectra-small-fp16`
- `models/whisper`
- `config/blacklist.txt`

Скачать Silero VAD:

```bash
.venv/bin/python utils/silero_vad_model_downloader.py
```

Для Whisper либо поместите локальный снимок в `models/whisper`, временно установите
`audio.stt.local_files_only=false`, либо отключите STT на время тестирования остальной
части конвейера.

Для встроенного Speaker также требуются путь к модели Piper и рабочая точка входа
Piper. Самая простая автономная установка:

- установите `requirements-optional.txt` в Listener `.venv`;
- установите для `speaker.piper.command` значение `.venv/bin/python3`;
- установите `speaker.piper.model` на вашу местную модель голоса `.onnx`;
- или временно установите `speaker.enabled=false`.

## Настройка звука в Linux

Список устройств:

```bash
.venv/bin/python utils/list_devices.py
```

Список источников мониторинга PipeWire/PulseAudio для loopback/AEC:

```bash
.venv/bin/python utils/list_devices.py --monitors
```

Рекомендуемые настройки AEC для Linux по умолчанию:

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

Полезная диагностика:

```bash
.venv/bin/python utils/meter_fast.py --duration 10
.venv/bin/python utils/AEC_meter.py --aec --pulse \
  --mic-source @DEFAULT_SOURCE@ \
  --loopback-source @DEFAULT_MONITOR@ \
  --duration 30
```

Подробнее: [docs/audio_RUS.md](docs/audio_RUS.md).

## Настройка OpenClaw

Включите переадресацию OpenClaw:

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

Установите входящие в комплект навыки OpenClaw:

```bash
OPENCLAW_WORKSPACE="$(openclaw config get agents.defaults.workspace)"
mkdir -p "$OPENCLAW_WORKSPACE/skills"
for skill in listener-control listener-speaker-off listener-tts-file; do
  rm -rf "$OPENCLAW_WORKSPACE/skills/$skill"
  cp -R "openclaw/skills/$skill" "$OPENCLAW_WORKSPACE/skills/"
done
```

Добавьте локальные примечания Listener к OpenClaw `TOOLS.md`:

```bash
OPENCLAW_WORKSPACE="$(openclaw config get agents.defaults.workspace)"
cat >> "$OPENCLAW_WORKSPACE/TOOLS.md" <<EOF

### Listener
- LISTENER_HOME=$(pwd)
- Control URL: http://127.0.0.1:18790
- Use: \$LISTENER_HOME/.venv/bin/python \$LISTENER_HOME/utils/listenerctl.py
EOF
```

Запустите команду из корня репозитория Listener, чтобы `LISTENER_HOME` был записан как
текущий путь проекта.

Чтобы позволить OpenClaw управлять стилем нейронного TTS с ведущими эмодзи, один раз
добавьте прилагаемый фрагмент подсказки к его инструкциям в рабочей области:

```bash
OPENCLAW_WORKSPACE="$(openclaw config get agents.defaults.workspace)"
sed -n '1,$p' openclaw/prompts/listener-tts-style.md >> "$OPENCLAW_WORKSPACE/AGENTS.md"
```

Listener использует только разрешённый список: произвольные эмодзи никогда не
становятся инструкциями модели. Ведущий эмодзи меняет стиль текущего ответа;
завершающий эмодзи остается только для отображения.

Listener автоматически обнаруживает имя помощника OpenClaw из рабочей области
`IDENTITY.md` с помощью `Name:` или `Имя:`. Полное руководство:
[docs/openclaw_RUS.md](docs/openclaw_RUS.md).

## Speaker и вывод эмодзи

Встроенный Speaker подписывается на события чата шлюза OpenClaw и локально озвучивает
ответы помощника. Он намеренно независим от входного пути: OpenClaw генерирует текст,
Listener принимает поток, разбивает его на произносимые сегменты, синтезирует их с
помощью Piper, VoxCPM2 или CosyVoice3 и воспроизводит их по порядку. Нейронные
worker-процессы работают в изолированных постоянных средах, напрямую передают PCM-поток
и переключаются на Piper после сбоев запуска или генерации. В конфигурации репозитория
по умолчанию остаётся Piper. Установка отдельных сред, загрузка моделей, оценка
ресурсов, настройка стилей и smoke-тесты описаны в
[docs/neural-tts_RUS.md](docs/neural-tts_RUS.md).

Нормализация русских числовых фрагментов общая для Piper, VoxCPM2 и CosyVoice3,
включая резервный Piper и создание WAV нейронной моделью. Она включается параметром
`speaker.tts.normalize_numbers=true`; установите основной `requirements.txt`, чтобы
в окружении Listener был доступен `rutextnorm`.

При постоянном бэкенде VoxCPM2 или CosyVoice3 тот же загруженный worker может сохранять
текст в WAV без второго процесса модели:

```bash
.venv/bin/python utils/listenerctl.py tts-file render \
  --text '😌 Спокойное сообщение для записи.' --filename calm-message --wait
```

Renderer использует ограниченную очередь, атомарно пишет внутрь
`speaker.file_render.output_dir` и применяет тот же разрешённый список стилей по
ведущему эмодзи, что и обычная озвучка. Чтобы модель OpenClaw вызывала этот путь,
установите навык `listener-tts-file`. Подробности и маршруты API приведены в
[docs/neural-tts_RUS.md](docs/neural-tts_RUS.md#создание-wav-без-второго-tts-процесса).

В Linux Piper воспроизводит готовые WAV преимущественно через `paplay`. Постоянные
нейронные TTS передают PCM в изолированный процесс `pacat` (или резервный `pw-cat`),
не загружая PortAudio внутрь Listener. Один лёгкий проигрыватель используется для всех
сегментов ответа OpenClaw; worker модели CosyVoice3/VoxCPM2 по-прежнему остаётся один.
Prebuffer по умолчанию 150 мс применяется один раз на ответ и сглаживает короткие паузы
генерации. `sounddevice` сохранён как явный legacy-бэкенд для платформ без этих команд.

Для Speaker требуется всё следующее:

- шлюз OpenClaw, доступный по адресу `speaker.gateway.url` (по умолчанию
  `ws://127.0.0.1:18789`);
- Среда Python с `piper` доступна через `speaker.piper.command`;
- действующая модель голоса по адресу `speaker.piper.model`;
- команда воспроизведения, такая как `/usr/bin/paplay`.

Типичная форма конфигурации:

```json
{
  "speaker": {
    "enabled": true,
    "tts_mode": "persistent",
    "queue_size": 4,
    "tts": {
      "normalize_numbers": true
    },
    "piper": {
      "command": "/home/re/src/Listener/.venv/bin/python3",
      "model": "/path/to/voice.onnx",
      "volume": 1.0
    },
    "playback": {
      "backend": "auto",
      "command": "/usr/bin/paplay",
      "streaming_backend": "auto",
      "prebuffer_ms": 150,
      "latency_ms": 100,
      "queue_ms": 2000,
      "restart_attempts": 1,
      "ducking": {
        "enabled": true
      }
    }
  }
}
```

Когда OpenClaw передаёт длинный ответ, Listener ставит в очередь сегменты размером с
предложение, а выбранный постоянный worker остаётся прогретым. Нейронный playback
включает приглушение только после заполнения prebuffer, сохраняет один аудиопроцесс
между сегментами и восстанавливает громкость после фактического слива последнего PCM.
Падение проигрывателя может оборвать текущий сегмент, но не завершает Listener и не
перезагружает TTS-модель.

Эмодзи обрабатываются до того, как текст достигнет Piper. Listener удаляет эмодзи из
произнесенного текста, при необходимости отправляет извлеченные символы отдельному
HTTP-демону `emoji-display` и продолжает говорить, даже если демон недоступен. Сегменты,
содержащие только эмодзи, можно отображать без создания
пустого воспроизведения TTS.

```json
{
  "speaker": {
    "emoji_display": {
      "enabled": false,
      "url": "http://127.0.0.1:18791",
      "send": "last",
      "mode": "replace",
      "hold_ms": 2200,
      "clear_on_interrupt": true
    }
  }
}
```

`speaker.emoji_display.send` может быть `last`, `first` или `none`; устаревшее значение
`all` принимается как `last`. Listener никогда не ставит в очередь отображение символов:
если текстовый сегмент содержит несколько эмодзи, отправляется только последний
извлеченный эмодзи. Listener взаимодействует только со службой HTTP и сам не открывает
последовательные/COM-порты.

Быстрые проверки:

```bash
.venv/bin/python utils/listenerctl.py speaker status
curl -s http://127.0.0.1:18790/speaker/status | jq
```

Если на этапе первоначальной настройки голосовые ответы не нужны, отключите их:

```json
{
  "speaker": {
    "enabled": false
  }
}
```

## Управление временем выполнения SpeechGate

Когда `main.py` работает, режимы SpeechGate можно менять без перезапуска:

```bash
.venv/bin/python utils/listenerctl.py status
.venv/bin/python utils/listenerctl.py mute --reason "quiet mode"
.venv/bin/python utils/listenerctl.py chatty --ttl 600
.venv/bin/python utils/listenerctl.py standby --ttl 300
.venv/bin/python utils/listenerctl.py normal
.venv/bin/python utils/listenerctl.py speech_gate_reset --reason "recover voice"
.venv/bin/python utils/listenerctl.py speaker status
.venv/bin/python utils/listenerctl.py speaker off
.venv/bin/python utils/listenerctl.py speaker on
```

Строка состояния включает режим, временное/постоянное состояние, время истечения срока
действия и режим восстановления.

Если из-за неудачного перебивания или прерывания собственный голос Listener или звуковые
сигналы приглушены, запустите `speech_gate_reset`. Он принудительно возвращает
`speech_gate` в `normal`, повторно включает `speaker`, прерывает зависшее
воспроизведение и восстанавливает запомненные уровни громкости sink-input
PulseAudio/PipeWire. В PipeWire он также нормализует настройки маршрута Speaker/Listener,
хранящиеся в
WirePlumber, что охватывает случаи, когда активный поток воспроизведения уже исчез.

HTTP-примеры:

```bash
curl -s http://127.0.0.1:18790/speech-gate/status | jq
curl -s -X POST http://127.0.0.1:18790/speech-gate/mode \
  -H 'Content-Type: application/json' \
  -d '{"mode":"chatty","ttl_seconds":60,"source":"curl"}' | jq
curl -s -X POST http://127.0.0.1:18790/speech-gate/reset \
  -H 'Content-Type: application/json' \
  -d '{"source":"curl","reason":"recover voice"}' | jq
curl -s http://127.0.0.1:18790/speaker/status | jq
```

Режимы:

- `normal` — обычная направленная фильтрация речи.
- `mute` - проходят только вызовы по имени помощника.
- `chatty` — проходят все непустые фразы.
- `standby` — все фразы заблокированы; требуется TTL.

Listener также может локально обрабатывать небольшой набор команд голосового режима,
прежде чем что-либо будет перенаправлено на OpenClaw:

- `Имя, помолчи` -> `mute`
- `Имя, говори` -> `normal`
- `Имя, отключись` -> `standby`
- `Имя, включи озвучку` или `Имя, верни озвучку` -> голосовые ответы `on`
- `Имя, отключи озвучку` или `Имя, выключи озвучку` -> голосовые ответы `off`
- `Имя, стоп` -> OpenClaw `chat.abort` для настроенного `session_key`

Эти локальные команды обрабатываются `SpeechGateAgent` и не пересылаются как обычный
ввод в чат.

Если встроенный Speaker включён, `Имя, стоп` и явные фразы перебивания также прерывают
локальное воспроизведение TTS и очищают голосовые сегменты в очереди. OpenClaw может
переключать голосовые ответы с помощью встроенного навыка: `speaker on`,
`speaker off` и `speaker status`. Специальный навык `listener-speaker-off` также включен
для узкого сценария «перестать говорить ответы вслух».

## Устранение неполадок Speaker

Самый полезный первый сигнал — `speaker status`:

- `agent=running gateway=connected` означает, что Listener подписан на шлюз OpenClaw;
- `speaker=off` означает, что голосовые ответы отключены конфигурацией или управлением
  во время выполнения;
- `error=...` указывает на последнюю ошибку шлюза, Piper или воспроизведения;
- `queue` и `current` показывают, поставлена ли речь в очередь или активно
  воспроизводится.

Для диагностики во время выполнения запустите Listener в режиме DEBUG и просмотрите
журналы Speaker:

```bash
.venv/bin/python main.py 2>&1 | tee /tmp/listener-speaker.log
rg "SpeakerAgent: (connected|final event needs history check|history check produced|queued speech segment|speaking assistant reply|speech failed|interrupted|dropped)" /tmp/listener-speaker.log
rg "EmojiDisplay|extracted .* emoji|emoji-only" /tmp/listener-speaker.log
```

Это особенно полезно, когда последнее предложение видно в OpenClaw, но не было
произнесено: цепочка журналов показывает, отсутствовал ли хвост в потоковой передаче
шлюза, был ли он удален из очереди, прерван или произошел сбой в Piper/воспроизведении.

Если другие приложения остаются тихими после прерывания, используйте endpoint
восстановления:

```bash
.venv/bin/python utils/listenerctl.py speech_gate_reset --reason "recover ducking"
```

Для пользовательской службы тот же путь восстановления предоставляется как перезагрузка:

```bash
systemctl --user reload listener.service
```

Если сам Speaker звучит тихо, сначала проверьте `speaker.piper.volume`. Если WAV в
норме, но прямая трансляция тихая, проверьте громкость потока PipeWire/PulseAudio и
настройки маршрута; `speech_gate_reset` нормализует поток Speaker до 100 % и
восстанавливает сохранённые базовые уровни приглушения.

## Тесты

```bash
. .venv/bin/activate
python -m pytest -q
```

Ожидаемый результат: пройден весь набор тестов.

## Документация

- [INSTALL_RUS.md](INSTALL_RUS.md) — новая установка и первый запуск.
- [docs/audio_RUS.md](docs/audio_RUS.md) — обработка звука, VAD и AEC.
- [docs/stt_RUS.md](docs/stt_RUS.md) — Whisper STT и SpeechGate.
- [docs/openclaw_RUS.md](docs/openclaw_RUS.md) — пересылка OpenClaw и навык управления.
- [docs/neural-tts_RUS.md](docs/neural-tts_RUS.md) — инструкции по установке, размеру и стилю
  VoxCPM2/CosyVoice3.
- [docs/service_RUS.md](docs/service_RUS.md) — запуск Listener как службы `systemd --user`.
- [docs/release_RUS.md](docs/release_RUS.md) — контрольный список выпуска.

## Лицензия

MIT. См. [LICENSE](LICENSE).
