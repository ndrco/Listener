---
name: listener-tts-file
description: Создаёт локальный WAV-файл из текста через уже запущенную в Listener модель VoxCPM2 или CosyVoice3. Используй, когда пользователь просит OpenClaw озвучить текст или сохранить его как аудиофайл.
---

# Файловая озвучка Listener

Используй комплектный helper, разрешив его путь относительно каталога установки этого
навыка. Никогда не запускай голый `scripts/listener-tts-file` из корня workspace.
Обычный путь команды:

```bash
TTS_FILE_HELPER="${OPENCLAW_WORKSPACE:-$HOME/.openclaw/workspace}/skills/listener-tts-file/scripts/listener-tts-file"
```

Если этот `SKILL.md` прочитан из другого пути workspace, используй каталог `scripts`
рядом именно с этим файлом. Helper обращается к локальному control API Listener и сам
не запускает процесс модели.

Создание WAV с ожиданием результата:

```bash
"$TTS_FILE_HELPER" render --text "🙂 Добро пожаловать!" --filename welcome --wait --json
```

Для длинного текста передай существующий UTF-8-файл:

```bash
"$TTS_FILE_HELPER" render --text-file /path/to/text.txt --filename narration --wait --json
```

Готовая задача содержит `output_path`. Верни этот путь пользователю или используй
его в следующей запрошенной операции с локальным файлом. Не запускай Python из
окружений CosyVoice3/VoxCPM2 напрямую и не создавай второй TTS worker.

## Стиль

Предпочтительно ставить один разрешённый эмодзи в начале текста. Listener удалит
его из произносимого текста и преобразует в безопасную инструкцию модели:

- `🙂` / `😊` — тепло;
- `😄` / `🎉` — радостно;
- `😌` — спокойно;
- `🤔` — задумчиво;
- `😔` / `😢` — грустно и сочувственно;
- `😠` — твёрдо;
- `😮` — удивлённо;
- `😏` — игриво;
- `⚠️` — срочно;
- `😂` — с улыбкой.

Можно явно указать разрешённый стиль, например `--style calm`. Произвольные
инструкции модели передавать нельзя. Доступны `neutral`, `warm`, `cheerful`,
`calm`, `thoughtful`, `sad`, `firm`, `surprised`, `playful`, `urgent`, `amused`.

## Управление задачами

```bash
"$TTS_FILE_HELPER" list --json
"$TTS_FILE_HELPER" status JOB_ID --json
"$TTS_FILE_HELPER" cancel JOB_ID --json
```

Если рендер недоступен, сообщи, что Listener должен работать с
`speaker.tts_mode=persistent` и `speaker.tts.backend=cosyvoice3` либо `voxcpm2`.
