# Audio Subsystem

[Russian version](audio_RUS.md)

This document describes the purpose and design of the project's audio subsystem, from
capturing microphone PCM frames through processing, optional storage, and integration
with other components. Its configuration lives in the `audio` section of
`config/config.json` and is exposed through `core.config.cfg`.

All event topics published by the audio subsystem are listed in the `events.audio` section of the `config/config.json` file and are available in runtime via `cfg.events.audio.*`. The values below (e.g. `audio/raw_frame`) describe the default values.

## Input Capture

### `audio.microphone.MicrophoneStream`

`MicrophoneStream` is an asynchronous context manager that uses the backend `sounddevice` (or compatible) to read PCM audio from the selected input device. Each received block is published to the internal queue and to the system event bus (`core.bus.EventBus`) as a `cfg.events.audio.raw_frame` event (by default `audio/raw_frame`). This event carries the frame itself (`bytes`), the sampling rate, and the number of channels, allowing other subsystems to subscribe to raw audio without being directly dependent on the microphone.

Constructor parameters default to values from `cfg.audio.input` and can be overridden:

* `input_sample_rate` — input stream sampling frequency (Hz, by default taken from `cfg.audio.input.input_sample_rate`);
* `chunk_size` — block/buffer size in samples;
* `channels` — number of channels (mono by default);
* `device_index` — `sounddevice` device index (or `None` for default selection);
* `queue_maxsize` — buffer size limit before issuing to `async for`.

The `audio.input.output_sample_rate` setting controls the output sample rate before
subsequent processing. When it is `null` or empty, the input rate is retained. When it
is lower than the input rate, `AudioProcessor` applies a polyphase low-pass resampler
immediately after AEC so downstream stages operate at the new rate.

The same keys are available directly in `config/config.json` and allow you to configure the capture without changing the code.

## Audio Processing

<details>
<summary>Audio Processing Scheme</summary>

```
+-----------------------------+
| AudioProcessor              |
+-----------------------------+

Core stream:
  Input PCM (int16, SR_in, CH_in)
            |
            v
  [AEC: Acoustic Echo Canceller]
    - removes speaker echo (near-end cleaned)
    - LiveKit APM, 10 ms frames
    - stream_delay_ms = HINT for AEC3
    - optional APM flags: NS / HPF / AGC
            |
            v
  [Resampler (optional)]
    - if SR_out < SR_in -> downsample
    - align to output_sample_rate
            |
            v
  [High-Pass / DC-block]
    - remove DC offset and sub-100 Hz rumble
    - configurable cutoff (~100 Hz)
            |
            v
  [Noise Suppression]
    - suppresses background noise
    - local NoiseSuppressor config (frame_ms, etc.)
            |
            v
  [VAD: Voice Activity Detection]
    - energy gate (RMS -> dBFS threshold)
    - WebRTC VAD (10/20/30 ms frames)
    - Silero VAD escalation (hybrid mode)
    - segmentation: min_speech/ms, min_silence/ms,
      pad/ms, hangover/ms
            |
            v
  [AGC: Automatic Gain Control]
    - target_level_dbFS, max_gain_db
    - limiter (attack/release), headroom
    - applied AFTER VAD/NS
            |
            v
  Output: ProcessedAudioFrame
    - data (PCM), sample_rate, channels
    - voice_detected, vad_probability, stats
    - publishes `cfg.events.audio.processed_frame`

Side event:
  VAD publishes `cfg.events.audio.voice_activity`

Far-end (playback) sources -> AEC.submit_farend():
  - Event Bus topic `cfg.events.audio.playback_frame`  -> PCM -> AEC
  - Windows WASAPI Loopback (sounddevice, loopback) -> PCM -> AEC
  - Linux PipeWire/PulseAudio monitor source (`parec` or sounddevice input) -> PCM -> AEC
  - Manual feed (submit_playback(bytes))     -> PCM -> AEC
```

</details>


The main processor is `audio.processing.AudioProcessor`. The legacy
`WindowsAudioProcessor` name remains as a compatibility alias. It accepts incoming PCM
blocks, for example from `MicrophoneStream`, and applies normalization, voice-activity
detection, and auxiliary statistics. Behavior is controlled by `cfg.audio.processing`.
The processing result is a `ProcessedAudioFrame` object comprising:
* `data` — processed PCM data (bytes);
* `sample_rate` — sampling frequency (Hz);
* `channels` — number of channels;
* `voice_detected` — final state of VAD for a frame taking into account hangover;
* `timestamp` — frame timestamp (seconds since the epoch);
* `vad_probability` — final score/probability of VAD;
* `vad_speech_frames` / `vad_total_frames` — the number of speech-positive and total VAD subframes;
* `voice_active_duration` — duration of the current voice segment (in seconds);
* `webrtc_probability` — WebRTC VAD contribution;
* `silero_probability` — Silero VAD contribution;
* `silero_invocations` — number of Silero invocations for this frame.

After processing, each frame is published to the event bus as
`cfg.events.audio.processed_frame` (default is `audio/processed_frame`).
The event payload additionally contains `voice_activity` as alias for
`voice_detected` and `segment_duration` for the current speech segment.

If voice activity publishing (`audio.processing.vad.publish_voice_activity`) is enabled, a `cfg.events.audio.voice_activity` event is also generated with the fields `active`, `timestamp`, `vad_probability`, `vad_speech_frames`, `vad_total_frames`, `webrtc_probability`, `silero_probability`, `silero_invocations`, `voice_active_duration`, and `segment_duration`.

### Acoustic Echo Cancellation (AEC)

`AudioProcessor` can use the LiveKit AudioProcessingModule for acoustic echo
cancellation. The `audio.processing.aec` section controls this optional feature, which
is disabled by default to avoid requiring `livekit` and consuming extra resources.
When enabled, AEC expects a mono 16 kHz—or otherwise LiveKit-compatible—stream and a
sequence of far-end frames selected by `playback_source`: `event_bus`, `manual`, or
`loopback`.

Main setting keys:

* `enabled` — Enables or disables the AEC.
* `frame_duration_ms` is the size of the frame that LiveKit APM works with (10 ms by default).
* `stream_delay_ms` — delay between far-end and near-end streams; it is important to choose it for the delay of the audio path.
* `noise_suppression`, `high_pass_filter`, `auto_gain_control` are optional flags of the built-in LiveKit modules. By default, they are disabled so as not to conflict with filters already used in the pipeline.
* Far-end frames come from the `cfg.events.audio.playback_frame` topic, configured in
  the `events` section. This subscription is used when `playback_source` is `event_bus`.
* `playback_source` — selection of the far-end audio source: `event_bus` (subscription to the topic from `cfg.events.audio.playback_frame`), `loopback` (automatic capture of system sound) or `manual` for manual transmission via `submit_playback`.
* `loopback_backend` — backend loopback capture: `auto`, `wasapi`, `pipewire`, `pulse`, `sounddevice_monitor`. In `auto`, Windows uses WASAPI; on Linux, Listener prefers PipeWire/Pulse source via `parec`, and can search for monitor/source device via `sounddevice` if necessary.
* `loopback_device_index` is the index of the loopback device. On Windows, this is the playback device for WASAPI loopback; on Linux, this is the input device monitor/source from PipeWire/PulseAudio.
* `loopback_source_name` is the name of PipeWire/PulseAudio source for Linux loopback capture. `@DEFAULT_MONITOR@` and `@DEFAULT_SOURCE@` aliases are supported.
* `loopback_device_name_contains` is an optional substring for auto-searching a Linux monitor device.
* `loopback_frame_duration_ms` is the frame size of the loopback stream; if not set, the AEC window is used.

AEC runs before the high-pass filter, noise suppression, and AGC. If LiveKit is unavailable or initialization fails, the pipeline automatically falls back to its previous behavior without AEC.

On Linux, first look at the available monitor sources:

```bash
python3 utils/list_devices.py --monitors
```

If you want to capture the name Pulse/PipeWire source, leave
`audio.processing.aec.loopback_source_name="@DEFAULT_MONITOR@"` or specify
specific source name. If you want to force sounddevice
monitor, set `audio.processing.aec.loopback_device_index`.

If no loopback source is found or opened, the pipeline will continue to work
without loopback capture and will write a warning in the log. To start without AEC
set `audio.processing.aec.enabled=false`.

For OpenClaw on Linux, it is usually enough to:

```json
{
  "openclaw": {
    "enabled": true,
    "command": "openclaw"
  }
}
```

If OpenClaw is not needed, set `openclaw.enabled=false`.

The Windows-specific example with the OpenClaw WSL command and WASAPI loopback lies in
`config/config.windows.example.json`.

### High-Pass Filter and DC Offset Removal

Before noise reduction and AGC, the simplest first-order high-frequency filter `DCBlockingHighPass` is included, which eliminates zero line drift and emphasizes speech. Its parameters are next to the AGC settings:

* `audio.processing.highpass.enabled` — enables or disables the filter (by default `true`).
* `audio.processing.highpass.cutoff_hz` — cutoff frequency in Hertz; 80–120 Hz values are well suited for human speech, but can be reduced to 0 if necessary (to completely disable HPF).

The filter stabilizes the RMS level before further processing, reducing the low-frequency hum and equalizing the sensitivity of the VAD.

### Noise Suppression

Noise cancellation module (`audio.processing.noise_suppression.NoiseSuppressor`) is built into `AudioProcessor` and starts up to VAD and AGC stages. It works with short RMS windows, adaptively tracking the background level and attenuating noise without heavy spectral transformations, which keeps the CPU load minimal.

Settings are in `audio.processing.noise_suppression`:

* `enabled` — turns the unit on or off;
* `frame_duration_ms` is the length of the analyzed window in milliseconds (at least 5 ms); increasing the value makes the reaction smoother and reduces the frequency of recalculations;
* `energy_threshold_ratio` is the ratio of the frame energy to the noise background, in which the fragment is considered speech;
* `suppression_factor` — how aggressively the noise level is subtracted when calculating the gain (1.0–1.5 give soft suppression, higher values muffle the background more);
* `noise_learning_rate` — noise-estimate learning rate in quiet regions (0...1); values around 0.9–0.98 provide smooth adaptation;
* `noise_release_rate` — rate at which the noise threshold relaxes when speech appears; small values reduce the risk of treating speech as noise;
* `gain_smoothing` — smoothing the gain between windows (0...0.999), reducing the "pump" effect;
* `min_gain` is the lower attenuation limit, which prevents complete zeroing of the signal.

The algorithm does not create large buffers and does not require third-party libraries. If necessary, you can increase `frame_duration_ms` and `gain_smoothing` to further reduce the number of recalculations at the cost of a more inertial reaction.

### Voice Activity Detection (VAD)

The `audio.processing` section controls the behavior of the hybrid VAD (WebRTC + Silero) and the accompanying event publishing logic.

* `enabled` — globally enables the processing unit.
* `audio.processing.vad.enabled` — enables/disables VAD operation.
* `audio.processing.vad.pipeline` — selects the processing mode: `"hybrid"` (default, WebRTC Silero → cascade), `"webrtc"` (classic VAD only) or `"silero"` (neural network model only).
* `audio.processing.vad` is a nested section with detector settings, including:
* `mode` — WebRTC VAD sensitivity mode (0–3, where 3 is the most aggressive).
* `frame_duration_ms` is the size of the VAD window in milliseconds (10, 20 or 30).
* `energy_threshold_db` is the threshold of the energy gate before the VAD call.
* `hangover_ms` - the duration of holding the "there is a voice" state after the last triggering; until the specified time expires, VAD continues to return `True` and postpones the publication of the `active=False` event.
* `active_republish_interval_ms` — how often to republish the `active=True` event with long fragments of speech; helps consumers of VAD events to receive actual "pulses" of activity. If the value is not set to zero or an empty string, `hangover_ms` is used as the default interval.
* `publish_voice_activity` — whether to publish events about voice activity.
* `probability_threshold` — minimum speech probability used by the detector; frames below the threshold are reset and do not contribute to segment accumulation.
* `min_speech_duration_ms` is the minimum speech duration required for detection.
* `min_silence_duration_ms` is the minimum duration of silence to complete the segment.
* `speech_pad_ms` — margin (padding) at the edges of the speech segment.

VAD accumulates the results of `webrtcvad.is_speech` frame by frame, so short single pulses less than `vad.min_speech_duration_ms` do not activate the "voice" state. The active segment continues until the total silence exceeds `vad.min_silence_duration_ms`. When publishing events and calculating `voice_active_duration`/`segment_duration`, `vad.speech_pad_ms` padding is used: the beginning of the segment is shifted backward by the specified value (but not before the zero mark), and the end is shifted forward. Thus, reports on voice activity reflect the actual duration of the segment, taking into account the protective fields at the edges.

#### Hybrid WebRTC + Silero Pipeline

The hybrid mode (`audio.processing.vad.pipeline="hybrid"`) combines the advantages of two approaches:

1. Each incoming PCM passes the `vad.energy_threshold_db` energy gate. If the level is below the threshold, the frame is immediately considered silence.
2. Audio that passes the energy gate is split into `vad.frame_duration_ms` windows and analyzed by WebRTC VAD, when the library is available. The speech probability is the ratio of speech-positive windows to all windows, additionally filtered by `vad.probability_threshold`.
3. If WebRTC is disabled or returns no decision, Silero is used. Audio frames are
   accumulated into windows of `vad.silero_cadence_ms` (or `vad.frame_duration_ms` when
   cadence is unset) until `vad.silero_min_activation_duration_ms` provides enough
   context. Silero then evaluates the buffer and supplies the final speech probability.
4. The resulting state (`True/False`) enters the publication logic, taking into account the minimum duration of speech and silence, padding and hangover.

If only one of the detectors is required, configure `audio.processing.vad.pipeline`:

* `"webrtc"` - the processor will be limited to the WebRTC library. Silero is not loaded or called, even if the paths to the model are set.
* `"silero"` — all decisions are made by Silero. WebRTC is ignored, which is useful when a consistent neural score is required or the `webrtcvad` library is unavailable.

##### Hybrid VAD Configuration Keys

* `webrtc_escalation_low_threshold` (default is `0.35`) is the lower confidence limit for WebRTC. If the probability drops lower, the segment is forcibly considered silent and Silero is not called, which saves resources on obviously empty areas.
* `webrtc_escalation_high_threshold` (default is `0.85`) is the upper confidence limit of WebRTC. A value above the threshold fixes the voice without starting Silero. The interval between low and high threshold is an area of uncertainty in which you can escalate to Silero.
* `vad.silero_cadence_ms` (default `null`, resulting in the use of `vad.frame_duration_ms`) — the length of the averaging window/frame step to accumulate audio before calling Silero. The larger the cadence, the less often the model is accessed and the larger the batches are obtained.
* `vad.silero_min_activation_duration_ms` (default `60.0`) is the minimum total duration of audio in the buffer required to run Silero.
* `vad.silero_device` (default `null`, equivalent to `"cpu"`) is an explicit choice of Silero execution device (`"cpu"`, `"cuda:0"`, `"mps"`, etc.).

##### Resource consumption and tuning

By default, the hybrid pipeline runs on the CPU: WebRTC is CPU-only, and Silero uses the CPU when `vad.silero_device` is unset. To reduce load on slower hardware:

* increase `vad.silero_cadence_ms` or `vad.silero_min_activation_duration_ms` to invoke the model less often and process larger batches;
* raise `webrtc_escalation_low_threshold` to classify uncertain regions as silence more often;
* if necessary, completely disconnect Silero by installing `audio.processing.vad.pipeline="webrtc"` or removing the paths to the model.

On powerful workstations, Silero can run on a GPU (`vad.silero_device="cuda:0"`) or another accelerator. Reducing cadence improves detection latency, while lowering the upper escalation threshold invokes neural analysis more often. GPU execution increases memory requirements; for multiple concurrent streams, tune cadence and minimum duration to balance latency and VRAM usage.

### Automatic Gain Control (AGC)

AGC parameters allow to equalize the input signal level to the set volume and are also configured via `cfg.audio.processing`:

* `audio.processing.agc.enabled` — enables/disables AGC.
* `audio.processing.agc.target_level_dbfs` is the target level in decibels relative to full scale (range −100…0 dBFS). The closer to zero, the louder the result.
* `audio.processing.agc.max_gain_db` — maximum allowable gain (0...60 dB). Limits how much the AGC can raise the level.
* `audio.processing.agc.attack_ms` is the attack time in milliseconds (1...1000). The lower the value, the faster the system reacts to the increase in volume.
* `audio.processing.agc.release_ms` is the release time in milliseconds (not less than `audio.processing.agc.attack_ms`, maximum 5000). It controls the return to normal gain after loud passages.
* `audio.processing.agc.headroom_db` — headroom maintained before clipping; values from 0 to 12 dB help prevent peak saturation.
* `audio.processing.agc.limiter_attack_ms` is a limiter attack that controls how quickly it reduces gain during overload (0.1...100 ms).
* `audio.processing.agc.limiter_release_ms` — release of the limiter; should be not less than the attack and usually lies in the range of 10...5000 ms.

If the values are out of range, they are automatically corrected when the configuration is loaded. The final gain is applied to all frames entering the processor, which allows maintaining a stable signal level for subsequent modules (speech recognition, recording, etc.).

## Speech Buffering for STT

`audio.writer.BufferedSpeechWriter` subscribes to `cfg.events.audio.processed_frame` and `cfg.events.audio.voice_activity` events, accumulating PCM frames before, during, and after the VAD event. This allows you to send segments to the speech recognition engine without cropped beginnings and ends. The behavior is configured through the `audio.buffer` section in `config/config.json` (and, accordingly, `cfg.audio.buffer`):

* `pre_roll_ms` — how many milliseconds of PCM to keep in the ring buffer before voice activation;
* `post_roll_ms` — post-buffer duration after voice deactivation until the end of the segment;
* `max_silence_ms` is the maximum permissible silence between active frames, after which the segment ends;
* `max_segment_duration_ms` — limit on the duration of one segment (milliseconds);
* `max_segment_bytes` — maximum segment size in bytes (limits the accumulated PCM);
* `queue_maxsize` — limit on the number of prepared segments in the internal queue (0 — no limit).

You can create an instance through `BufferedSpeechWriter.from_config()`, which automatically reads values from `cfg.audio.buffer`.

## `audio.emotion`

In `core.config` there is a section `audio.emotion`, but in the current code
the main runtime (`agents.audio_agent.AudioAgent`) does not create or run
a separate emotion-analyzer. That is, it is still a configuration reserve, not
active stage of the Listener pipeline.

## Related Components and Tests

* The `audio` package contains `microphone`, `processing`, `stt`, `writer` modules and service files like `config/silero_vad_config.json`.
* Integration and unit checks lie in `tests/test_microphone.py`, `tests/test_audio_processing.py`, `tests/test_vad_pipeline.py`, `tests/test_vad_pipeline_local_vad.py`, `tests/test_windows_audio_processing.py`, `tests/test_audio_stt.py`, `tests/test_whisper_engine.py`, `tests/test_silero_vad_helper.py`, `tests/test_record.py`.
* The configuration and type diagrams are described in `core.config`.

Use this document as an entry point when configuring, extending, and debugging the audio subsystem.
