# Рабочий процесс выпуска

[English version](release.md)

Используйте этот контрольный список перед публикацией выпуска Listener.

## Чистота репозитория

- `git status --short` содержит только преднамеренные изменения выпуска.
- Никакие веса моделей, `.venv`, кэши, локальные `.openclaw` или пути, специфичные для
  машины, не отслеживаются.
- `README.md`, `INSTALL.md`, `docs/audio.md`, `docs/stt.md` и `docs/openclaw.md`
  описывают текущее поведение.
- `LICENSE` присутствует.

## Проверка

```bash
. .venv/bin/activate
python -m py_compile main.py agents/control_agent.py agents/openclaw_input_agent.py \
  agents/speaker_agent.py agents/speech_gate_agent.py audio/ducking.py \
  llm/speech_gate.py speaker/*.py utils/listenerctl.py
python -m pytest -q
```

Ручная smoke-проверка:

```bash
.venv/bin/python main.py
curl -s http://127.0.0.1:18790/ | jq
.venv/bin/python utils/listenerctl.py speech-gate set-mode chatty --ttl 10
.venv/bin/python utils/listenerctl.py speech-gate status
.venv/bin/python utils/listenerctl.py speaker status
.venv/bin/python utils/listenerctl.py speaker off
.venv/bin/python utils/listenerctl.py speaker on
```

## Следующий тег

```bash
git tag -a v0.2.7 -m "Listener v0.2.7"
git push origin main --tags
```

Предлагаемое название выпуска:

```text
Listener v0.2.7 - OpenClaw gateway v4 and emoji display while speaker is off
```

Предлагаемые примечания к выпуску:

- Клиенты WebSocket Listener для OpenClaw обновлены для протокола шлюза v4 — как при
  пересылке входных фраз, так и при чтении истории и событий Speaker.
- Отображение emoji продолжает работать даже при отключённых голосовых ответах, поэтому
  emoji по-прежнему могут выводиться на внешний дисплей, пока локальный TTS
  остается выключенным.
- Добавлен fallback только для loopback-подключения, читающий локальный токен шлюза
  OpenClaw; это устраняет ошибку `device identity required` в WebSocket-клиенте Listener.
- В шаблон systemd Listener явно добавлен `/home/re/.local/bin` в `PATH`, чтобы
  резервный CLI-вызов при необходимости находил `openclaw`.
- Расширено покрытие регрессионными тестами совместимости с Gateway v4 и отображения
  emoji при `speaker.enabled=false`.
- Версия среды выполнения изменена на `0.2.7`.
