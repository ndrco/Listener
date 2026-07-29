# Интеграция с OpenClaw

[English version](openclaw.md)

Listener интегрируется с OpenClaw в нескольких направлениях:

1. Принятые голосовые фразы отправляются в OpenClaw через `openclaw gateway call
   chat.send`.
2. OpenClaw может управлять режимом SpeechGate Listener с помощью встроенного навыка
   рабочего пространства `listener-control` и `utils/listenerctl.py`.
3. Listener может озвучивать ответы OpenClaw локально через встроенного агента Speaker и
   позволяет OpenClaw включать и выключать голосовые ответы.
4. OpenClaw может попросить уже запущенный нейронный TTS worker сохранить текст в
   локальный WAV через навык `listener-tts-file`.

Путь ответа:

```text
OpenClaw Gateway chat events -> SpeakerAgent -> selected TTS -> local audio playback
OpenClaw skill -> Listener control API -> existing neural worker -> WAV file
```

## Отправка голосовых фраз на OpenClaw

Включите OpenClaw в `config/config.json`:

```json
{
  "openclaw": {
    "enabled": true,
    "command": "openclaw",
    "source_topic": "llm/accepted_phrase",
    "session_key": "main",
    "gateway_url": null,
    "gateway_token": null
  }
}
```

В Windows с OpenClaw внутри WSL используйте `config/config.windows.example.json` в
качестве отправной точки.

## Имя/личность помощника

SpeechGate не хранит имена помощников в `config/speech_gate_patterns.json`. Вместо
этого Listener автоматически обнаруживает файл идентификации рабочей области OpenClaw:

- `OPENCLAW_IDENTITY_FILE`
- `OPENCLAW_WORKSPACE/IDENTITY.md`
- `OPENCLAW_STATE_DIR/workspace/IDENTITY.md`
- `OPENCLAW_CONFIG_PATH`
- `~/.openclaw/openclaw.json`
- `~/.openclaw-dev/openclaw.json`
- `~/.openclaw-*/openclaw.json`

Идентификационный файл должен содержать одно из:

```markdown
Name: Marina
Имя: Марина
```

Если автоматическое обнаружение неверно, установите:

```json
{
  "speech_gate": {
    "identity_file": "/path/to/openclaw/workspace/IDENTITY.md"
  }
}
```

## API управления во время работы

Listener запускает локальный API управления HTTP, когда `control.enabled=true`:

```text
GET  /
GET  /health
GET  /speech-gate/status
POST /speech-gate/mode
POST /speech-gate/reset
GET  /speaker/status
POST /speaker/enabled
POST /tts/files
GET  /tts/files
GET  /tts/files/{job_id}
POST /tts/files/{job_id}/cancel
```

Пример:

```bash
curl -s http://127.0.0.1:18790/speech-gate/status | jq
```

Режимы переключения:

```bash
curl -s -X POST http://127.0.0.1:18790/speech-gate/mode \
  -H 'Content-Type: application/json' \
  -d '{"mode":"chatty","ttl_seconds":600,"source":"curl"}' | jq
curl -s -X POST http://127.0.0.1:18790/speech-gate/reset \
  -H 'Content-Type: application/json' \
  -d '{"source":"curl","reason":"recover voice"}' | jq
curl -s -X POST http://127.0.0.1:18790/speaker/enabled \
  -H 'Content-Type: application/json' \
  -d '{"enabled":false,"source":"curl","reason":"quiet"}' | jq
curl -s -X POST http://127.0.0.1:18790/tts/files \
  -H 'Content-Type: application/json' \
  -d '{"text":"😔 Иногда тишина всё объясняет.","filename":"thought"}' | jq
```

`POST /speech-gate/reset` — endpoint восстановления для редкого случая, когда
из-за застрявшего состояния перебивания или прерванного ответа собственная озвучка
Listener и звуковые сигналы остаются приглушёнными. Он возвращает `speech_gate` в `normal`, повторно
включает `speaker`, прерывает воспроизведение активного ответа и восстанавливает
отслеживаемую громкость sink-input.

Поддерживаемые режимы:

- `normal` — обычная направленная фильтрация речи.
- `mute` - проходят только прямые вызовы по имени помощника.
- `chatty` — проходят все непустые фразы.
- `standby` — все фразы заблокированы; требуется TTL.

`chatty` и другие временные режимы оцениваются по времени начала сегмента STT, поэтому
фраза, начавшаяся внутри окна TTL, по-прежнему использует этот режим, даже если Whisper
завершается после истечения срока TTL.

## Работа Speaker

Для работы голосовых ответов Listener необходимы:

- шлюз OpenClaw, доступный по адресу `speaker.gateway.url` и соответствующий
  `speaker.gateway.session_key`;
- `websockets` установлен в Listener `.venv` через `requirements.txt`;
- `piper` доступен через `speaker.piper.command`;
- действующая модель голоса по адресу `speaker.piper.model`;
- рабочая команда воспроизведения, например `/usr/bin/paplay`.

Дополнительная поддержка отображения эмодзи настраивается в `speaker.emoji_display`.
Listener удаляет эмодзи из текста, отправленного на Piper, а затем пересылает
извлечённые символы во внешнюю HTTP-службу, например, в родственный проект
`emoji-display`. Аппаратное/COM-соединение намеренно остается за пределами Listener.

В потоковом режиме Listener произносит полные фрагменты, похожие на предложения, по мере
их поступления от OpenClaw и выполняет короткую финальную проверку `chat.history`, чтобы
восстановить любой хвост, который был виден в пользовательском интерфейсе, но
отсутствовал в дельтах шлюза. При `speaker.tts_mode="persistent"` worker Piper остаётся
прогретым, и синтез следующего сегмента в очереди может идти параллельно с воспроизведением
текущего. В Linux `speaker.playback.backend="auto"` предпочитает `paplay`, поэтому поток
воспроизведения имеет стабильные метаданные PulseAudio/PipeWire для приглушения и
восстановления громкости.

Приглушение Speaker действует в пределах одного ответа OpenClaw: Listener снижает
громкость других приложений во время озвучки и восстанавливает её после последнего
сегмента в очереди. Если после прерывания или перебивания звук остаётся тихим,
вызовите `listenerctl.py speech_gate_reset` или `systemctl --user reload
listener.service`, чтобы сбросить SpeechGate/Speaker и восстановить сохранённые уровни громкости
PipeWire/PulseAudio.

Репозиторий `config/config.json` в настоящее время содержит машинно-специфичный пример,
указывающий на соседний checkout `/home/re/src/Speaker`. На другом компьютере
следует либо заменить эти пути, либо установить `speaker.enabled=false`, пока ваша
локальная установка Piper не будет готова.

Полезные проверки во время выполнения:

```bash
.venv/bin/python utils/listenerctl.py speaker status
curl -s http://127.0.0.1:18790/speaker/status | jq
```

Ключевые поля статуса:

- `speaker=on|off` — включены ли голосовые ответы;
- `agent=running|stopped` — жив ли `SpeakerAgent` внутри Listener;
- `gateway=connected|disconnected` — подписан ли Listener на шлюз OpenClaw;
- `queue` и `current` — поставленные в очередь или активные речевые сегменты;
- `last_interrupt` — причина последней остановки или перебивания;
- `error` — последняя ошибка шлюза, Piper или воспроизведения.

## Локальные голосовые команды

Listener также может локально перехватывать несколько голосовых команд с именем
помощника, прежде чем фраза будет перенаправлена в OpenClaw:

- `Имя, помолчи` -> переключает SpeechGate на `mute`
- `Имя, говори` -> переключает SpeechGate на `normal`
- `Имя, отключись` -> переключает SpeechGate на `standby`
- `Имя, включи озвучку` или `Имя, верни озвучку` -> включает голосовые ответы.
- `Имя, отключи озвучку` или `Имя, выключи озвучку` -> отключает голосовые ответы.
- `Имя, стоп` -> вызывает OpenClaw `chat.abort` для настроенного `openclaw.session_key`

Эти локальные команды намеренно обрабатываются Listener и не отправляются как обычный ввод
`chat.send`. Собственный навык управления OpenClaw по-прежнему полезен для ввода команд,
более сложных изменений режима, таких как временный `chatty`, и ручной проверки через
`listenerctl`.

Когда встроенный Speaker включен, `Имя, стоп` также прерывает текущее воспроизведение
TTS и очищает очередь речи. Явные фразы перебивания, пересылаемые через
`sessions.steer`, прерывают воспроизведение Speaker до того, как запрос на управление
ожидает OpenClaw.

## Устранение неполадок Speaker

Запустите Listener с журналированием уровня DEBUG, если нужно отследить потерянные или
прерванные голосовые ответы:

```bash
.venv/bin/python main.py 2>&1 | tee /tmp/listener-speaker.log
```

Ищите цепочку Speaker:

```bash
rg "SpeakerAgent: (connected|final event needs history check|history check produced|queued speech segment|speaking assistant reply|speech failed|interrupted|dropped)" /tmp/listener-speaker.log
rg "EmojiDisplay|extracted .* emoji|emoji-only" /tmp/listener-speaker.log
```

Как это читать:

- `history check produced ... final segment(s)` означает, что Listener должен был
  восстановить последний хвост из `chat.history`;
- `queued speech segment` означает, что текст достиг очереди воспроизведения Speaker;
- `speaking assistant reply` означает Piper/воспроизведение началось;
- `interrupted` означает локальную остановку, перебивание или очистку воспроизведения
  при прерывании ответа OpenClaw;
- `speech failed` указывает на Piper или сбои команды воспроизведения.

Это основной рабочий процесс для ошибок, когда последнее предложение отображается в
OpenClaw, но не произносится локально.

## Установка навыка OpenClaw

Из репозитория Listener:

```bash
OPENCLAW_WORKSPACE="$(openclaw config get agents.defaults.workspace)"
mkdir -p "$OPENCLAW_WORKSPACE/skills"
for skill in listener-control listener-speaker-off listener-tts-file; do
  rm -rf "$OPENCLAW_WORKSPACE/skills/$skill"
  cp -R "openclaw/skills/$skill" "$OPENCLAW_WORKSPACE/skills/"
done
```

Добавьте примечания к локальному пути к OpenClaw `TOOLS.md`:

```bash
OPENCLAW_WORKSPACE="$(openclaw config get agents.defaults.workspace)"
cat >> "$OPENCLAW_WORKSPACE/TOOLS.md" <<EOF

### Listener
- LISTENER_HOME=$(pwd)
- Control URL: http://127.0.0.1:18790
- Use: $OPENCLAW_WORKSPACE/skills/listener-control/scripts/listener-control
EOF
```

Запустите команду из корня репозитория Listener, чтобы в `LISTENER_HOME` был записан
текущий путь проекта. Скрипт навыка также проверяет переменные окружения,
Listener `config/config.json` и общим локальным путям.

При желании добавьте короткое постоянное примечание к OpenClaw `AGENTS.md`, чтобы агент
распознавал, что некоторые сообщения чата могут поступать от Listener в виде голосовых
расшифровок:

```bash
OPENCLAW_WORKSPACE="$(openclaw config get agents.defaults.workspace)"
cat >> "$OPENCLAW_WORKSPACE/AGENTS.md" <<'EOF'

## Listener Voice Input

Some messages may come from Listener as voice transcripts through OpenClaw
chat.send. When the user asks to change listening behavior, use the
listener-control skill: chatty for conversation mode, mute for name-only mode,
standby only with TTL, and normal to return to default filtering. For disabling
spoken replies, use listener-speaker-off or `listener-control speaker off`.
EOF
```

Если Listener использует VoxCPM2 или CosyVoice3, также установите соглашение о стиле
ответа:

```bash
OPENCLAW_WORKSPACE="$(openclaw config get agents.defaults.workspace)"
sed -n '1,$p' openclaw/prompts/listener-tts-style.md >> "$OPENCLAW_WORKSPACE/AGENTS.md"
```

Ведущий emoji используется намеренно: Listener ставит завершённые предложения в
очередь, пока OpenClaw ещё передаёт ответ, поэтому завершающий emoji
появляется слишком поздно, чтобы стилизовать это предложение. Стили выбираются из списка
разрешённых и сохраняются только внутри одного OpenClaw `runId`; прерывание, ошибка,
завершение и перебивание очищают состояние текущего ответа.

## Ручные команды `listenerctl`

```bash
.venv/bin/python utils/listenerctl.py status
.venv/bin/python utils/listenerctl.py chatty --ttl 600
.venv/bin/python utils/listenerctl.py mute
.venv/bin/python utils/listenerctl.py standby --ttl 300
.venv/bin/python utils/listenerctl.py normal
.venv/bin/python utils/listenerctl.py speaker status
.venv/bin/python utils/listenerctl.py speaker off
.venv/bin/python utils/listenerctl.py speaker on
.venv/bin/python utils/listenerctl.py tts-file render \
  --text '🙂 Добро пожаловать!' --filename welcome --wait
.venv/bin/python utils/listenerctl.py tts-file list
```

`listenerctl` читает переменные:

- `LISTENER_CONTROL_URL`
- `LISTENER_CONTROL_TOKEN`

Его удобочитаемый вывод включает текущий режим, временный или постоянный, время
истечения срока действия, если он временный, и режим восстановления.

Помощник по навыкам OpenClaw
(`openclaw/skills/listener-control/scripts/listener-control`) также обнаруживает
`LISTENER_HOME`, URL и токен управления из окружения, OpenClaw `TOOLS.md` или
Listener `config/config.json` перед делегированием `listenerctl.py`.

Если API управления доступен не только через loopback-интерфейс, настройте непустой
`control.token`.

## Навык OpenClaw для файлового рендера

Входящий в комплект навык `listener-tts-file` создаёт WAV через уже работающий worker
VoxCPM2 или CosyVoice3. Для поиска корня Listener, URL control API и токена он использует
установленный helper `listener-control`. Навык не активирует окружение модели и не
запускает второй worker.

Типичная команда навыка:

```bash
openclaw/skills/listener-tts-file/scripts/listener-tts-file render \
  --text '😌 Спокойный текст для записи.' \
  --filename calm-note \
  --wait --json
```

После завершения задача содержит `output_path`. Ведущий эмодзи из разрешённого списка
выбирает ту же фиксированную безопасную инструкцию стиля, что и для обычных ответов.
Стиль можно задать явно через `--style calm`; произвольные инструкции не принимаются.
Каталог вывода, лимит текста, размер очереди и сегмента настраиваются в
`speaker.file_render`. Жизненный цикл, планирование, расход диска и поведение при ошибке
описаны в разделе
[`neural-tts_RUS.md`](neural-tts_RUS.md#создание-wav-без-второго-tts-процесса).

## Соответствие команд навыка управления Speaker

Входящий в комплект навык `listener-control` предоставляет элементы управления голосовым
ответом:

- «выключить голосовые ответы», «не читать ответы вслух» -> `speaker off`
- «включить голосовые ответы», «снова прочитать ответы вслух» -> `speaker on`
- голосовой ответ/статус голосового вывода -> `speaker status`

В узком случае «отключить речь прямо сейчас» встроенный навык `listener-speaker-off`
вызывает выделенного помощника `scripts/listener-speaker-off`.

После изменения состояния Speaker навык должен запустить `speaker status` и сообщить,
включены ли голосовые ответы.
