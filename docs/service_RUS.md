# Listener как пользовательская служба

[English version](service.md)

Listener следует запускать как foreground-процесс Python под контролем systemd. Используйте
пользовательскую службу в Linux, поскольку Listener нужен доступ к микрофону
пользователя, сеансу PipeWire/PulseAudio и локальной среде OpenClaw.

## Перед установкой

Сначала завершите обычную настройку:

```bash
git clone <repository-url>
cd Listener
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-optional.txt
.venv/bin/python utils/silero_vad_model_downloader.py
```

Затем настройте `config/config.json` для этой машины:

- `audio.input.device_index`, если микрофон по умолчанию неправильный.
- `audio.processing.aec.loopback_source_name`, если AEC должен использовать определенный
  monitor-источник.
- `audio.stt.device` и `speech_gate.model.device`, особенно на машинах только с
  CPU.
- `speaker.enabled=false` для первого запуска или установите реальные пути
  `speaker.piper.command` и `speaker.piper.model`.
- `openclaw.gateway_url`, `openclaw.gateway_token` и `openclaw.session_key`, если ваш
  шлюз OpenClaw не использует настройки по умолчанию.

Перед созданием службы выполните ручной smoke-тест:

```bash
.venv/bin/python main.py
```

В другом терминале:

```bash
.venv/bin/python utils/listenerctl.py health
.venv/bin/python utils/listenerctl.py ready
```

Остановите ручной процесс с помощью `Ctrl+C` или:

```bash
.venv/bin/python utils/listenerctl.py stop --reason smoke-test
```

## Установка

Самый простой путь — скрипт установки. Он считывает встроенный шаблон, перезаписывает
его для текущего checkout, устанавливает в `~/.config/systemd/user`,
перезагружает systemd и включает службу:

```bash
.venv/bin/python utils/install_user_service.py
```

Запустите сразу во время установки:

```bash
.venv/bin/python utils/install_user_service.py --start
```

Просмотрите сгенерированный unit-файл, ничего не записывая:

```bash
.venv/bin/python utils/install_user_service.py --dry-run
```

Необработанный шаблон находится по адресу `deploy/systemd/listener.service`. Он содержит
пути для этого checkout и служит читаемым образцом. Если вы копируете его
вручную, отредактируйте эти поля в `~/.config/systemd/user/listener.service`:

- `WorkingDirectory`
- `ExecStart`
- `ExecReload`
- `ExecStop`

## Запуск

```bash
systemctl --user start listener.service
systemctl --user status listener.service
.venv/bin/python utils/listenerctl.py health
.venv/bin/python utils/listenerctl.py ready
```

`health` только проверяет работоспособность API управления. `ready` проверяет, успешно
ли запустились критически важные компоненты Listener.

Если `ready` печатает `listener=not_ready`, проверьте список компонентов и последнюю
ошибку:

```bash
.venv/bin/python utils/listenerctl.py ready --json
```

## Журналы

```bash
journalctl --user -u listener.service -f
```

Listener пишет в stdout/stderr, а journald собирает и ротирует журнал.

## Остановить и перезапустить

```bash
.venv/bin/python utils/listenerctl.py speech_gate_reset --reason manual
systemctl --user reload listener.service
.venv/bin/python utils/listenerctl.py stop --reason manual
systemctl --user restart listener.service
systemctl --user stop listener.service
```

Unit systemd использует `listenerctl stop` для корректного завершения работы.
Команда `ExecStop` работает в режиме best effort, поэтому unit не переходит в состояние
строя, если Listener уже остановился через `/shutdown`. Если процесс не завершится в
течение `TimeoutStopSec`, systemd завершит его.

`systemctl --user reload listener.service` подключен к пути мягкого восстановления
Listener. Он запускает `listenerctl.py speech_gate_reset --reason systemd-reload`,
который возвращает `speech_gate` в `normal`, повторно включает `speaker`, прерывает
воспроизведение зависшего ответа и принудительно восстанавливает громкость без перезапуска
процесса Python. В системах PipeWire/WirePlumber он также восстанавливает сохраненные
базовые уровни приглушения из `state/ducking_state.json` и нормализует настройки
выходного маршрута Speaker/Listener, поэтому перезагрузка — это первая команда
восстановления, которую следует попробовать после неудачного перебивания или
прерванного длительного ответа OpenClaw.

## Строгий запуск

По умолчанию Listener сохраняет существующее поведение запуска в режиме best effort.
Чтобы запуск службы завершался сбоем при невозможности запустить критический
компонента, включите это в `config/config.json`:

```json
"service": {
  "strict_startup": true
}
```

Важнейшими компонентами являются аудиовход, SpeechGate, переадресация ввода OpenClaw и
Speaker при `speaker.enabled=true`.

Изменения режима выполнения сохраняются локально в `state/runtime_state.json`. Этот файл
создается Listener на установленном компьютере и намеренно не поставляется в выпусках.

Рекомендуемое внедрение:

```bash
.venv/bin/python utils/listenerctl.py ready
systemctl --user restart listener.service
journalctl --user -u listener.service -n 80
```

## Удаление

```bash
systemctl --user disable --now listener.service
rm ~/.config/systemd/user/listener.service
systemctl --user daemon-reload
```

## Поиск неисправностей

- `connection_failed` из `listenerctl`: служба не запущена, произошел сбой при запуске
  или `control.host`/`control.port` отличается от значения по умолчанию.
- `listener=not_ready`: процесс активен, но один или несколько критических компонентов
  вышли из строя. Используйте `listenerctl ready --json` и `journalctl`.
- Нет микрофонного входа: запустите `utils/list_devices.py`, затем установите
  `audio.input.device_index` или устройство ввода системы по умолчанию.
- Нет голосовых ответов: проверьте пути `speaker.enabled`, Piper и `listenerctl speaker
  status`.
- OpenClaw не получает фраз: проверьте режим `speech_gate`, настройки шлюза OpenClaw и
  `listenerctl ready`.
