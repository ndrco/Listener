# Whisper STT engine

[Russian version](stt_RUS.md)

The `audio.stt.whisper_engine.WhisperEngine` module encapsulates loading and running
[Whisper](https://github.com/openai/whisper) models through `faster-whisper`. It is
configured by the `audio.stt` section of `config/config.json` and performs the
following supporting tasks:

* initializes the model by name (`tiny`, `base`, `small`, `medium`, `large`) or local
  directory and selects the inference device and compute type;
* resamples an incoming PCM stream (`int16`) to the Whisper target sample rate
  (`audio.stt.sample_rate`, 16 kHz by default) with a streaming resampler;
* converts samples to the `[-1, 1]` range and passes them to
  `WhisperModel.transcribe` with the configured decoder options;
* returns recognized text fragments, one item for each segment returned by the model.

## Engine Lifecycle

1. **Initialize.** Instance is created with `WhisperSttCfg` configuration. When
the `enabled` flag is active, the engine immediately loads the weights and is ready
for inference.
2. **Model preparation.** `WhisperEngine` creates `WhisperModel`, passing
parameters `device`, `compute_type`, `cpu_threads`, `num_workers` and
`download_root`. If `faster-whisper` is unavailable, it raises
   `RuntimeError`.
3. **Resampling.** The `transcribe` method accepts arbitrary PCM blocks with
specified sampling rate. If `sample_rate` differs from
`audio.stt.sample_rate`, data passes through streaming resampler
   `StreamingResampler`.
4. **Inference.** After normalizing the signal, the engine calls
   `WhisperModel.transcribe` with decoder parameters such as beam search,
   temperature, and the VAD filter. Leading and trailing whitespace is removed
   from the resulting strings.

## Method `transcribe`

```python
WhisperEngine.transcribe(batch_pcm: np.ndarray | bytes | Iterable[int], sample_rate: int) -> list[str]
```

* `batch_pcm` — mono PCM `int16` as bytes, a NumPy array, or an iterable sequence.
  The caller is responsible for mixing multichannel audio.
* `sample_rate` is the sample rate of the received block.

The method returns a list of text hypotheses. When the module is disabled
(`enabled=false`), it returns an empty list.

## `audio.stt` Configuration

### Basic parameters

| Key | Purpose | Default |
|------|------------|-----------------------|
| `enabled` | Enables Whisper STT. | `false` |
| `model` | A `faster-whisper` model name such as `avazir/faster-distil-whisper-large-v3-ru`, `large-v3`, or `large-v3-turbo`, or a local weights directory. | `"small"` |
| `device` | Inference device (`"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, ...). With `null`, the library picks it up automatically. | `null` |
| `compute_type` | Calculation type (`"default"`, `"int8"`, `"int8_float16"`, ...). | `null` |
| `download_root` | Directory for the model cache. | `null` |
| `blacklist_path` | Path to the text blacklist used to post-filter Whisper phrases. Relative paths are resolved from the project root. | `"config/blacklist.txt"` |
| `local_files_only` | Prevents automatic model download and forces `faster-whisper` to only work with local files. | `false` |
| `cpu_threads` | Number of threads for CPU-inference. Ignored if `null`. | `null` |
| `num_workers` | Number of parallel `faster-whisper` workers. | `null` |
| `language` | Language ISO code (e.g. `"ru"`, `"en"`). With `null`, Whisper tries to detect the language automatically. | `null` |
| `task` | Job type (`"transcribe"` or `"translate"`). | `"transcribe"` |
| `sample_rate` | Target audio sampling rate (Hz). | `16000` |
| `partial_topic` / `final_topic` | EventBus topics for partial and final hypotheses (taken from `cfg.events.audio`). | `"audio/stt/partial"` / `"audio/stt/final"` |
| `min_confidence` | Minimum confidence to commit a phrase. | `0.35` |
| `stability_timeout_s` | Timeout waiting for new updates. | `1.2` sec |
| `queue_wait_s` | Timeout waiting for new segments from `BufferedSpeechWriter`. | `0.2` sec |
| `enable_punctuation` | Add a final punctuation mark when publishing the final text. | `true` |

### Decoder Options

The following keys are passed directly to `WhisperModel.transcribe` if specified in
presets:

* `beam_size`, `best_of`, `patience`, `length_penalty` — beam search control;
* `temperature`, `temperature_increment_on_fallback`,
`prompt_reset_on_temperature` — temperature parameters;
* `initial_prompt` — prefix for the first hypothesis;
* `condition_on_previous_text` — whether to inherit the previous text in the following
windows;
* `compression_ratio_threshold`, `logprob_threshold`,
  `no_speech_threshold`, `max_initial_timestamp` — stopping and filtering controls;
* `suppress_tokens`, `suppress_blank` — token suppression management;
* `vad_filter`, `vad_parameters` — activation of VAD filter inside
  `faster-whisper`;
* `word_timestamps`, `without_timestamps` — enable time stamps.

Set fields to `null` to use `faster-whisper` defaults.

## Streaming Transcriber

Module `audio.stt.streaming.WhisperStreamingTranscriber` links
`BufferedSpeechWriter`, `WhisperEngine` and system event bus for continuous
transcription. It works in an asynchronous task, reads segments from the queue
`BufferedSpeechWriter.queue`, transfers them to Whisper and manages the accumulation
of partial hypotheses.

Key responsibilities:

1. **Buffering and state.** The transcriber tracks the current hypothesis,
update timestamps and segment metadata (duration, VAD confidence,
the boundaries of the segments).
2. **Partial publications.** After each update, the transcriber publishes an event
`audio/stt/partial` with fields `text`, `raw_text`, `is_final=false` and
segment metadata.
3. **Finalization.** After the stability timeout or a forced
finalization, it generates the final phrase, applies post-processing (whitespace
normalization, capitalization, and optional punctuation), and publishes an event
   `audio/stt/final`.
4. **Integration with LLM.** The final text is placed in the asynchronous queue
`llm_queue` as a string; optionally, you can pass callback `on_final`,
which is called for each final utterance. In the standard
`AudioAgent`, this callback publishes the result to `cfg.events.llm.input_text`.

The internal callback payload contains the `pcm_data` of the original segment, but in
event `audio/stt/final` this field is not published: before sending to EventBus
it is removed.

## Directed-Speech Gating

Before sending the recognized speech to the LLM, a two-stage gate is triggered
`llm.speech_gate.SpeechDirectionGate`:

* **Rules and scoring.** The text is checked for the assistant name from the OpenClaw
identity-file (lines `Name:` / `Имя:`) and address markers: command verbs,
interrogative and modal words, polite formulas. Markers are read once
when starting from a `speech_gate.patterns_file` file (e.g.,
`config/speech_gate_patterns.json`), in its absence are used
inline lists from the configuration. If the rule score reaches
`speech_gate.rules_threshold` (0.7 default), the request is considered
addressed and skipped without ML-check.
* **ML classifier.** For uncertain utterances, the model
`models/directed-ruElectra-small-fp16` (parameters and device are set in
`speech_gate.model`). The final score is calculated as
`0.6 * ml + 0.4 * rules`, and below `speech_gate.final_threshold`
(0.5) the phrase is discarded with diagnostics in the logs in DEBUG mode.

After a phrase passes the gate, "attention mode" is enabled for several seconds
(`speech_gate.attention_window_seconds`), allowing subsequent utterances through
without filtering. If a phrase ends with a continuation marker
(`speech_gate.continuation_patterns`, for example, "and more", "as well as"), window
renews on `speech_gate.attention_extension_seconds`.

### Gate Operating Modes

Modes (`standby/mute/normal/chatty`) can be switched externally. Purpose:

- **normal** — standard mode. Rules are triggered, if necessary
the ML classifier is invoked; a successful request enables
  «attention mode».
- **standby** — standby mode: the gate blocks all utterances regardless of
rules and ML.
- **mute** — quiet mode: only phrases beginning with the assistant's name pass,
and all other utterances are blocked.
- **chatty** — chatty mode: utterances pass without filtering; the gate
is actually always open.

### Runtime Mode Switching

During operation, `main.py` starts the local control API at
`http://127.0.0.1:18790`. It is easier to use it through the CLI:

```bash
.venv/bin/python utils/listenerctl.py speech-gate status
.venv/bin/python utils/listenerctl.py speech-gate set-mode mute --reason "quiet mode"
.venv/bin/python utils/listenerctl.py speech-gate set-mode chatty --ttl 600
.venv/bin/python utils/listenerctl.py speech-gate set-mode standby --ttl 300
.venv/bin/python utils/listenerctl.py speech-gate set-mode normal
.venv/bin/python utils/listenerctl.py speech-gate reset --reason "recover voice"
```

`normal` cancels a temporary mode. `mute` and `chatty` can be permanent or
temporary. Via HTTP API and `listenerctl` mode `standby` is accepted only with
TTL to keep voice control from locking. Any runtime switching
resets the attention window.

Section `control` in `config/config.json`:

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

The CLI reads `LISTENER_CONTROL_URL` and `LISTENER_CONTROL_TOKEN`. If `host` is not
loopback, Listener requires a non-empty `control.token`.

The same control API also provides runtime management of the built-in Speaker:

```bash
.venv/bin/python utils/listenerctl.py speaker status
.venv/bin/python utils/listenerctl.py speaker off
.venv/bin/python utils/listenerctl.py speaker on
```

The `speech-gate reset` command is needed as a recovery button: it returns
`speech_gate` in `normal`, re-enables `speaker` and force
restores all remembered ducking volumes if Listener's voice or beeps remain
quiet after barge-in or interrupted generation.

OpenClaw integration v1 is implemented as a workspace skill:

```bash
mkdir -p "$(openclaw config get agents.defaults.workspace)/skills"
cp -R openclaw/skills/listener-control \
  "$(openclaw config get agents.defaults.workspace)/skills/"
```

In OpenClaw `TOOLS.md`, it is convenient to add a local note:

```markdown
### Listener
- LISTENER_HOME=<path-to-Listener>
- Control URL: http://127.0.0.1:18790
- Use: $LISTENER_HOME/.venv/bin/python $LISTENER_HOME/utils/listenerctl.py
```

### `speech_gate` Setup

`speech_gate` section keys in `config/config.json` and their purpose:

| Key | Purpose | Default |
|------|----------|--------------|
| `enable` | Enables or disables the gate. When `false`, utterances pass without rule or ML checks and attention mode is unused. | `true` |
| `rules_threshold` | Rule threshold. If the score from dictionaries (name, verbs, and markers) exceeds it, the phrase is treated as directed speech without running ML. | `0.7` |
| `final_threshold` | Final threshold after combining rules and ML (`0.6 * ml + 0.4 * rules`). Utterances below it are ignored. | `0.5` |
| `attention_window_seconds` | Duration of attention mode after a successful directed utterance; subsequent utterances pass without filtering. | `8.0` |
| `attention_extension_seconds` | The attention window is extended by how many seconds when continuation markers are detected ("and more", "as well as", etc.). | `3.0` |
| `patterns_file` | Path to JSON token lists (`command_verbs`, `continuation_patterns`, etc.). Relative paths are resolved from the project root. `assistant_names` in this file is ignored. | `"config/speech_gate_patterns.json"` |
| `identity_file` | Path to the OpenClaw identity Markdown. `null` or `"auto"` enables discovery through `OPENCLAW_IDENTITY_FILE`, `OPENCLAW_WORKSPACE`, `OPENCLAW_STATE_DIR`, `OPENCLAW_CONFIG_PATH`, `~/.openclaw/openclaw.json`, and `~/.openclaw-*` profile directories. Explicit relative paths are resolved from the Listener root. | `null` |
| `model.path` | `directed-ruElectra-small-fp16` model directory for speech direction classifier. | `"models/directed-ruElectra-small-fp16"` |
| `model.device` | Classifier inference device (`cpu`, `cuda`, `cuda:0`, etc.). | `"cpu"` |
| `model.threshold` | The probability threshold for the model response (before mixing with the rule). | `0.7` |
| `model.max_length` | The maximum length of the input text tokenization for the classifier. | `64` |

### Audible Indicators

Listener can play short notification tones for key voice-path transitions. The
configuration is stored in the `indicators` section of `config/config.json`.

| Key | Purpose | Default |
|------|----------|--------------|
| `enabled` | Turns on/off the audible indicators. | `true` |
| `backend` | `auto`, `sounddevice`, `winsound`, or `none`. On Linux, `sounddevice` is usually used, on Windows, fallback to `winsound` is possible. | `"auto"` |
| `output_device_index` | Output audio-device index for signals. `null` uses the default system device. | `null` |
| `sample_rate` | Sampling frequency of synthesized signals. | `24000` |
| `volume` | The signal volume is in the range `0.0..1.0`. | `0.18` |
| `queue_maxsize` | Maximum signals in the playback queue. In case of overflow, new signals are discarded. | `8` |
| `rejected` | Play a signal when a phrase is rejected by SpeechGate. | `true` |
| `forwarded` | Play a tone when a phrase is successfully sent to OpenClaw. | `true` |
| `local_handled` | Play a tone when Listener handles a local control command. | `true` |
| `interrupted` | Play a signal for a successful barge-in or stop in OpenClaw. | `true` |

By default, there are four different short beeps:

1. the phrase was rejected by SpeechGate.
2. the phrase passed SpeechGate and was successfully sent to OpenClaw;
3. service voice command (`mute`, `normal`, `standby`) processed inside Listener;
4. interrupt or stop command successfully reached OpenClaw.

You can turn off the signal types individually, for example:

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

### Format `speech_gate_patterns.json`

The file contains pattern lists used by the gate to calculate the `rules` score.
Its top-level value is an object containing arrays of strings. Example:

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
  "local_speaker_on_commands": ["включи озвучку", "верни озвучку"],
  "local_speaker_off_commands": ["отключи озвучку", "выключи озвучку"],
  "local_abort_commands": ["стоп", "хватит"],
  "local_barge_in_commands": ["нет", "не так", "подожди", "точнее"]
}
```

The assistant name is not stored in `speech_gate_patterns.json`; the
`assistant_names` field is ignored. The primary name source is `IDENTITY.md` in
the OpenClaw workspace. Listener attempts to locate it automatically:
first through the environment variables `OPENCLAW_IDENTITY_FILE`, `OPENCLAW_WORKSPACE`,
`OPENCLAW_STATE_DIR`, `OPENCLAW_CONFIG_PATH`, then via configs
`~/.openclaw/openclaw.json`, `~/.openclaw-dev/openclaw.json` and
`~/.openclaw-*/openclaw.json`. If OpenClaw stores its workspace elsewhere,
set the path manually:

```json
{
  "speech_gate": {
    "identity_file": "/path/to/openclaw/workspace/IDENTITY.md"
  }
}
```

Example of identity file contents:

```markdown
Name: Kissa
Имя: Кисса
```

Any fields may be omitted. Listener can combine lists from the file with
additional inline patterns from `config/config.json`, if defined there,
but the project recommends keeping the main dictionary in
`config/speech_gate_patterns.json` and leaving inline lists in `config/config.json`
empty. This avoids duplicate definitions and semantic
intersections between ordinary command verbs and local control phrases.
If the configured file is missing or unreadable, Listener logs a warning and the gate
continues using only inline patterns from
`config/config.json` and the name from the identity file.

Local mode-control and stop commands are accepted only in the form
`<assistant name> + command`: the name must begin the recognized phrase, for example
`Марина, помолчи`, `Марина, говори`, `Марина, включи озвучку`,
`Марина, отключи озвучку`, or `Марина, стоп`. Words such as `включи` or `говори`
without a name at the beginning do not switch the mode.

The same rule applies to `mute`: the gate passes only phrases that begin with the
assistant's name. An occurrence in the middle of an utterance is insufficient.

There is a separate exception for the local voice command `Марина, отключись`:
the internal `SpeechGateAgent` can switch Listener to `standby` without a TTL,
because this command is handled locally before being sent to OpenClaw. Quit
this state with, for example, `Марина, говори` or the manual command
`listenerctl.py normal`.

`local_barge_in_commands` describes explicit interrupt phrases. They also require a name
at the beginning: `Марина, нет, я имел в виду...`, `Марина, точнее...`.
Ordinary inquiries like `Марина, какая погода?` are not sent via
`sessions.steer` and go to OpenClaw with the usual `chat.send`.

`command_verbs` should only be used for normal user tasks
like `покажи`, `найди`, `объясни`, `напомни`. Listener control phrases like
`замолчи`, `слушай`, `стоп`, and `остановись` should remain only in
`local_*_commands`, so that they are not duplicated in the general rules.

### Execution Parameters

The `StreamingTranscriberConfig` entity controls the behavior of the transcriber during
work. All default values are inherited from `WhisperSttCfg`, but can be
redefined in runtime.

### Lifecycle

1. Create `BufferedSpeechWriter` and pass it to the constructor
`WhisperStreamingTranscriber` together with the STT configuration and, if needed, an
`EventBus` object.
2. Call `await transcriber.start()` to start the background task. From this
moment, the transcriber will consume segments from `writer.queue`.
3. When finished, call `await transcriber.stop()` or use
context manager `async with` to wait for the publication of all final
hypotheses and correctly clear the state.

All publications are asynchronous and exception-safe, so transcriber failures do not
stop subscribers. If `llm_queue` overflows,
the final phrase is discarded with a warning entry in the log.

### Whisper Blacklist

If set to `audio.stt.blacklist_path`, Listener reads the blacklist from this
file. The format supports two sections:

- `[phrases]` — exact phrases. A recognized phrase is discarded in full only when,
  after punctuation removal and case normalization, it exactly matches an entry.
- `[words]` - individual words. These words are cut from the recognized text by
word boundaries; the rest of the phrase remains.

Matching is performed in a normalized form:

- letter case is ignored;
- punctuation marks and symbols do not affect the comparison;
- words will not match as substrings: `1988` does not match `19880`.

Example:

```text
[phrases]
Спасибо
Всем пока

[words]
1988
```

In this mode, `Спасибо!` and `Всем пока...` will be discarded entirely, but
`Спасибо, Марина!` and `Спасибо, спасибо!` will go further. Phrase
`1988, 1988! Ура! Здорово! 1888!` will turn into
`Ура! Здорово! 1888!`; to remove `1888`, it must be added to `[words]`.

## Audio Agent

For ease of integration, all components are combined by the agent
`agents.audio_agent.AudioAgent`. It starts the microphone stream, processes
audio and controls transcription by publishing events to the system bus.

### Pipeline Architecture

1. **MicrophoneStream** — captures PCM from the selected device and broadcasts
frames to the `cfg.events.audio.raw_frame` bus (default is `audio/raw_frame`).
2. **AudioProcessor** — accepts frames via `submit()` and performs VAD,
noise reduction and other processing, publishing the result as
`cfg.events.audio.processed_frame` and `cfg.events.audio.voice_activity` events.
3. **BufferedSpeechWriter** — subscribes to processed-frame and VAD events, collects
preroll/postroll speech segments and puts them in the `writer.queue` queue.
4. **WhisperStreamingTranscriber** — reads segments from the
queue, calls `WhisperEngine` and sends intermediate hypotheses
(`cfg.audio.stt.partial_topic`, i.e. `cfg.events.audio.stt_partial`) and
final phrases (`cfg.audio.stt.final_topic`, i.e.
`cfg.events.audio.stt_final`). In the standard `AudioAgent`, the final text is
also published to `cfg.events.llm.input_text` for language-model consumers.

## State Control

* `await AudioAgent.pause()` — suspends the pipeline, closing the active
components.
* `await AudioAgent.resume()` — Resumes operation by reinitializing the flow
and restarting audio processing.
* `await AudioAgent.close()` — terminates the agent and
frees up all resources.

Methods are protected from repeated calls and correctly handle the cancellation of background
tasks. This allows you to safely embed an audio pipeline into the manager
system orchestrator and flexibly control speech capture modes.
