# Установка

[English version](INSTALL.md)

В этом руководстве рассматривается новая настройка Linux для Listener и дополнительная
интеграция OpenClaw. Windows поддерживается через пример конфигурации в
`config/config.windows.example.json`, но Linux PipeWire/PulseAudio является основным
тестируемым путем.

## 1. Системные требования

- Python 3.12
- Среда выполнения и заголовки PortAudio
- PipeWire или PulseAudio для loopback/AEC в Linux
- `pactl` для диагностики устройства/источника
- Дополнительно: драйвер NVIDIA и совместимая с CUDA 12.8 сборка PyTorch для STT на GPU.
- Дополнительно: CLI OpenClaw в `PATH`.
- `pacat` (пакет `pulseaudio-utils`) или `pw-cat` (пакет `pipewire-bin`) для
  изолированного воспроизведения нейронного TTS.
- Дополнительно: команда `paplay` для воспроизведения WAV от Piper.

Пакеты Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv python3-pip \
  libportaudio2 portaudio19-dev pulseaudio-utils pipewire-bin jq
```

## 2. Клонирование и создание `.venv`

```bash
git clone <repository-url>
cd Listener
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
```

Установите зависимости:

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

`requirements.txt` по умолчанию настроен на CUDA 12.8 PyTorch. На машине только с ЦП вы
все равно можете запустить аудиоконвейер, но установите для устройств модели STT и
речевого шлюза значение `cpu` в `config/config.json`.

`requirements-optional.txt` теперь также устанавливает `piper-tts`, который используется
встроенным Speaker, когда включены голосовые ответы.

## 3. Модели

Веса моделей намеренно не отслеживаются в git. Конфигурация по умолчанию ожидает:

- Silero VAD: `models/silero_vad_v6.jit`
- Классификатор SpeechGate: `models/directed-ruElectra-small-fp16`
- Кэш/корень Whisper: `models/whisper`

Скачать Silero VAD:

```bash
.venv/bin/python utils/silero_vad_model_downloader.py
```

Для Whisper либо:

- поместите совместимый локальный снимок под `models/whisper`;
- установите `audio.stt.local_files_only=false` для первой загрузки модели;
- или временно установите `audio.stt.enabled=false` во время тестирования остальной
  части приложения.

Для Speaker предоставьте файл `.onnx` модели Piper и убедитесь, что
`speaker.piper.command` и `speaker.piper.model` указывают на реальные локальные пути.
Конфигурация репозитория включает пример, подключённый к соседнему checkout
`/home/re/src/Speaker`; на другом компьютере вам следует либо заменить эти пути, либо
отключить Speaker при первом запуске:

```json
{
  "speaker": {
    "enabled": false
  }
}
```

Для дополнительных бэкендов VoxCPM2 или Fun-CosyVoice3-0.5B не устанавливайте их
зависимости в `.venv` Listener. Каждому бэкенду необходима собственная среда постоянного
worker-процесса. Полное проверенное руководство по установке и оценке ресурсов —
[docs/neural-tts_RUS.md](docs/neural-tts_RUS.md).

Если вам нужна автономная установка Listener, используйте виртуальную среду Listener в
качестве точки входа Piper:

```json
{
  "speaker": {
    "enabled": true,
    "piper": {
      "command": ".venv/bin/python3",
      "model": "/absolute/path/to/voice-model.onnx"
    }
  }
}
```

Для первого запуска только с процессором используйте:

```json
{
  "speech_gate": {
    "model": {
      "device": "cpu"
    }
  },
  "audio": {
    "stt": {
      "device": "cpu",
      "compute_type": "int8"
    }
  }
}
```

## 4. Аудиоустройства

Перечислите все аудиоустройства и источники PipeWire/PulseAudio:

```bash
.venv/bin/python utils/list_devices.py
```

Список только кандидатов на monitor/loopback:

```bash
.venv/bin/python utils/list_devices.py --monitors
```

Для Linux AEC `config/config.json` может использовать псевдонимы источника
Pulse/PipeWire:

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

Быстрая проверка микрофона:

```bash
.venv/bin/python utils/meter_fast.py --duration 10
```

Проверка AEC:

```bash
.venv/bin/python utils/AEC_meter.py --aec --pulse \
  --mic-source @DEFAULT_SOURCE@ \
  --loopback-source @DEFAULT_MONITOR@ \
  --duration 30
```

## 5. Запуск Listener

```bash
.venv/bin/python main.py
```

В другом терминале проверьте API локального управления:

```bash
curl -s http://127.0.0.1:18790/ | jq
curl -s http://127.0.0.1:18790/speech-gate/status | jq
curl -s http://127.0.0.1:18790/speaker/status | jq
```

Переключите SpeechGate во временный режим общения:

```bash
.venv/bin/python utils/listenerctl.py chatty --ttl 60
```

Вернитесь в нормальное состояние:

```bash
.venv/bin/python utils/listenerctl.py normal
```

Проверьте состояние выполнения Speaker:

```bash
.venv/bin/python utils/listenerctl.py speaker status
```

## 6. Установка как службы

Запустите Listener вручную один раз, прежде чем устанавливать его в качестве службы. Это
значительно упрощает просмотр ошибок аудиоустройства, модели и конфигурации OpenClaw
непосредственно в терминале.

После успешной ручной smoke-проверки установите службу Linux `systemd --user` для
текущего checkout:

```bash
.venv/bin/python utils/install_user_service.py
systemctl --user start listener.service
```

Проверьте работоспособность и готовность:

```bash
.venv/bin/python utils/listenerctl.py health
.venv/bin/python utils/listenerctl.py ready
```

Следите за журналами:

```bash
journalctl --user -u listener.service -f
```

Корректно остановите работающую службу:

```bash
.venv/bin/python utils/listenerctl.py stop --reason manual
```

Полный рабочий процесс службы, включая запуск при входе в систему, перезапуск, удаление,
пользовательские пути checkout и строгий режим запуска, см. в
[docs/service_RUS.md](docs/service_RUS.md).

Изменения режима выполнения сохраняются в `state/runtime_state.json` на установленном
компьютере. Свежие выпуски не включают этот файл, поэтому новая установка начинается с
`config/config.json`.

### Необязательное обновление PipeWire в Ubuntu 24.04

Ubuntu 24.04 (Noble) пока оставляет PipeWire `1.0.5` в поддерживаемых apt-репозиториях.
Listener не требует обновления дистрибутива для локализации сбоя аудиодрайвера:
воспроизведение neural PCM и индикаторов уже вынесено в дочерние процессы
`pacat`/`pw-cat`. Тем не менее на боевой машине запущен официальный upstream-тег
PipeWire `1.6.8` (`b741e0c74f5436f0c925f7741140db0efd32cf4e`) как версионированный
пользовательский runtime:

```text
~/.local/opt/pipewire-1.6.8
~/.config/systemd/user/pipewire.service.d/10-local-1.6.8.conf
~/.config/systemd/user/pipewire-pulse.service.d/10-local-1.6.8.conf
```

Сборка содержит daemon, Pulse-протокол, ALSA/udev SPA-модули, поддержку D-Bus и
RTKit. Системный `wireplumber` остаётся менеджером сессии. Ubuntu-пакеты `1.0.5`
под `/usr` намеренно сохранены для отката; достоверна версия работающего сервера,
а не вывод `/usr/bin/pipewire --version`:

```bash
pw-cli info 0 | grep 'version:'
pactl info | grep 'Server Name'
systemctl --user show pipewire.service pipewire-pulse.service wireplumber.service \
  -p Id -p MainPID -p ActiveState -p NRestarts
```

Для отката без удаления любой из установок:

```bash
systemctl --user stop listener.service
mv ~/.config/systemd/user/pipewire.service.d/10-local-1.6.8.conf \
  ~/.config/systemd/user/pipewire.service.d/10-local-1.6.8.conf.disabled
mv ~/.config/systemd/user/pipewire-pulse.service.d/10-local-1.6.8.conf \
  ~/.config/systemd/user/pipewire-pulse.service.d/10-local-1.6.8.conf.disabled
systemctl --user daemon-reload
systemctl --user restart pipewire.service wireplumber.service pipewire-pulse.service
systemctl --user start listener.service
```

После любого перезапуска daemon PipeWire долгоживущим настольным клиентам вроде
EasyEffects может потребоваться перезапуск, чтобы вновь создать виртуальные sink/source.

## 7. Интеграция OpenClaw

Listener отправляет принятые фразы на OpenClaw через:

```bash
openclaw gateway call chat.send
```

Минимальная конфигурация:

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

После этого OpenClaw может использовать навык `listener-control` для таких фраз, как:

- "включить режим разговора" -> `chatty --ttl 600`
- "тихий режим" -> `mute`
- "не слушать пять минут" -> `standby --ttl 300`
- «вернуться к обычному режиму прослушивания» -> `normal`
- "включить активное прослушивание" -> `chatty --ttl 600`
- "выключить активное прослушивание" -> `mute`

При постоянном бэкенде VoxCPM2 или CosyVoice3 навык `listener-tts-file` позволяет
OpenClaw сохранять текст в WAV через уже существующий worker Listener. Для поиска
локального пути, URL и токена он использует установленный helper `listener-control`.
См. [docs/neural-tts_RUS.md](docs/neural-tts_RUS.md#создание-wav-без-второго-tts-процесса).

Listener также автоматически считывает имя помощника OpenClaw из файла идентификации
области OpenClaw `IDENTITY.md`. Поддерживаемые ключи: `Name:` и `Имя:`.

Голосовые ответы также зависят от событий чата шлюза OpenClaw. Здоровое состояние – это:

- шлюз OpenClaw доступен по адресу `speaker.gateway.url`;
- `listenerctl speaker status` показывает `agent=running gateway=connected`;
- нет поля `error=...` в выводе состояния.

## 8. Тесты

```bash
. .venv/bin/activate
python -m pytest -q
```

Ожидаемый результат: пройден весь набор тестов.
