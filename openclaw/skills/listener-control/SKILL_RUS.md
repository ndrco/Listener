---
name: listener-control
description: Управление локальным SpeechGate Listener через listenerctl. Используйте, когда пользователь просит OpenClaw изменить режим прослушивания, включить тихий, разговорный или ожидающий режим, вернуть обычную маршрутизацию голоса либо показать текущий статус.
---

# Управление Listener

Используйте этот навык, когда пользователь хочет изменить или проверить, как Listener
принимает голосовой ввод.

## Предпочтительная команда

Используйте встроенный скрипт `scripts/listener-control` (путь разрешается
относительно этого `SKILL.md`). Он определяет:

- `LISTENER_HOME` из окружения, рабочего пространства OpenClaw `TOOLS.md`, исходного
  пути навыка или общих локальных путей.
- управляющий URL-адрес из `LISTENER_CONTROL_URL`, `TOOLS.md` (`Control URL:`) или
  Listener `config/config.json`.
- токен управления из `LISTENER_CONTROL_TOKEN`, `TOOLS.md` или Listener
  `config/config.json`.

Примеры:

```bash
scripts/listener-control status
```

```bash
scripts/listener-control speaker status
```

```bash
scripts/listener-control normal --reason "normal listening"
```

```bash
scripts/listener-control mute --reason "quiet mode"
```

```bash
scripts/listener-control chatty --ttl 600 --reason "conversation mode"
```

```bash
scripts/listener-control standby --reason "standby requested"
```

```bash
scripts/listener-control speech_gate_reset --reason "recover voice"
```

```bash
scripts/listener-control speaker off --reason "disable spoken replies"
```

```bash
scripts/listener-control speaker on --reason "enable spoken replies"
```

Скрипт делегирует выполнение `listenerctl.py`. Если известен `LISTENER_HOME`, можно
также вызвать его напрямую, например:
`$LISTENER_HOME/.venv/bin/python $LISTENER_HOME/utils/listenerctl.py status`.

## Сопоставление намерений

- Режим разговора, прослушивание всего, имя пробуждения не требуется, активное
  прослушивание включено -> `chatty --ttl 600`, если пользователь не укажет
  продолжительность.
- Тихий режим, режим только имени, прекращение прослушивания фоновой речи, активное
  прослушивание отключено -> `mute`.
- Не слушать, полностью прекратить слушать, режим ожидания -> `standby`. Если
  пользователь указывает продолжительность, вы можете использовать `--ttl`, например
  `--ttl 600`.
- Обычный режим, вернитесь, слушайте нормально, выйдите из активного режима
  прослушивания -> `normal`.
- Восстановить застрявшую приглушённую озвучку или звуковые сигналы Listener после
  неудачного прерывания или перебивания -> `speech_gate_reset`.
- Отключить голосовые ответы, отключить голосовой вывод, прекратить читать ответы вслух,
  не произносить ответы -> `speaker off`.
- Включить голосовые ответы, включить голосовой вывод, снова прочитать ответы вслух ->
  `speaker on`.
- Если пользователь спрашивает о статусе голосового ответа/голосового вывода, запустите
  `speaker status`.
- Если пользователь спрашивает об активности/статусе прослушивания, сначала запустите
  `status` и сообщите текущий режим.

После изменения режима запустите `status` и кратко подведите итоги полученного режима.
Выходные данные CLI включают режим, постоянное/временное состояние, время истечения
срока действия и режим восстановления. Не используйте `chatty` без `--ttl`. После
изменения состояния Speaker запустите `speaker status` и сообщите, включены ли
голосовые ответы.
