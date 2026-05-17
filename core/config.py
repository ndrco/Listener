from __future__ import annotations
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
import json
from typing import Any

from speaker.config import SpeakerConfig


def _clean_topic_value(value: object, current: str) -> str:
    if value in (None, ""):
        return current
    try:
        text = str(value)
    except Exception:
        return current
    text = text.strip()
    return text or current


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _resolve_project_path(value: object, root: Path) -> str | None:
    if value in (None, ""):
        return None
    try:
        path = Path(str(value).strip()).expanduser()
    except Exception:
        return None
    if not str(path):
        return None
    if not path.is_absolute():
        path = root / path
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _apply_event_topics(section: dict[str, Any], cfg_obj: Any) -> None:
    for field_info in getattr(cfg_obj, "__dataclass_fields__", {}).values():
        name = field_info.name
        current_value = getattr(cfg_obj, name)
        if is_dataclass(current_value):
            nested_section = section.get(name, {})
            if isinstance(nested_section, dict):
                _apply_event_topics(nested_section, current_value)
        else:
            setattr(
                cfg_obj,
                name,
                _clean_topic_value(section.get(name, current_value), current_value),
            )


@dataclass
class Paths:
    """Paths to important files and directories."""
    root: Path


@dataclass
class AudioEventTopics:
    raw_frame: str = "audio/raw_frame"
    processed_frame: str = "audio/processed_frame"
    voice_activity: str = "audio/voice_activity"
    playback_frame: str = "audio/playback_frame"
    stt_partial: str = "audio/stt/partial"
    stt_final: str = "audio/stt/final"


@dataclass
class LlmEventTopics:
    input_text: str = "llm/input_text"
    accepted_phrase: str = "llm/accepted_phrase"


@dataclass
class AppEventTopics:
    stop: str = "app/stop"


@dataclass
class SystemEventTopics:
    session: str = "system/session"


@dataclass
class EventsCfg:
    audio: AudioEventTopics = field(default_factory=AudioEventTopics)
    llm: LlmEventTopics = field(default_factory=LlmEventTopics)
    app: AppEventTopics = field(default_factory=AppEventTopics)
    system: SystemEventTopics = field(default_factory=SystemEventTopics)


@dataclass
class OpenClawInputCfg:
    """Settings for forwarding voice input into OpenClaw chat."""

    enabled: bool = False
    command: str = "openclaw"
    transport: str = "gateway_ws"
    source_topic: str = "llm/accepted_phrase"
    session_key: str = "main"
    gateway_url: str | None = None
    gateway_token: str | None = None
    call_timeout_s: float = 12.0


@dataclass
class PerformanceCfg:
    """Structured runtime performance logging settings."""

    enabled: bool = False
    log_level: str = "info"
    include_text_preview: bool = True
    text_preview_chars: int = 80


@dataclass
class DuckingCfg:
    """System output volume ducking settings."""

    enabled: bool = False
    volume_scale: float = 0.35
    fade_in_ms: int = 20
    fade_out_ms: int = 80


@dataclass
class SoundIndicatorsCfg:
    """Short audio cues for SpeechGate/OpenClaw workflow events."""

    enabled: bool = True
    backend: str = "auto"
    output_device_index: int | None = None
    sample_rate: int = 24000
    volume: float = 0.18
    queue_maxsize: int = 8
    ducking: DuckingCfg = field(default_factory=DuckingCfg)
    rejected: bool = True
    forwarded: bool = True
    local_handled: bool = True
    interrupted: bool = True


@dataclass
class ControlCfg:
    """Local runtime control API settings."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 18790
    token: str | None = None
    max_ttl_seconds: float = 86400.0
    state_path: str | None = "state/runtime_state.json"


@dataclass
class ServiceCfg:
    """Service/supervisor integration settings."""

    strict_startup: bool = False


@dataclass
class AudioInputCfg:
    """Input audio stream settings."""
    input_sample_rate: int = 16000
    output_sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    device_index: int | None = None


@dataclass
class AudioPlaybackCfg:
    """Audio playback settings."""

    enabled: bool = True
    sample_rate: int = 24000
    channels: int = 1
    chunk_size: int = 1024
    device_index: int | None = None
    queue_maxsize: int = 0
    volume: float = 1.0


@dataclass
class NoiseSuppressionCfg:
    """Noise suppression module settings."""

    enabled: bool = False
    frame_duration_ms: int = 20
    energy_threshold_ratio: float = 1.6
    suppression_factor: float = 1.2
    noise_learning_rate: float = 0.95
    noise_release_rate: float = 0.01
    gain_smoothing: float = 0.6
    min_gain: float = 0.1


@dataclass
class AudioVadCfg:
    """Voice activity detection settings."""

    enabled: bool = True
    pipeline: str = "hybrid"
    mode: int = 3
    frame_duration_ms: int = 30
    energy_threshold_db: float = -45.0
    hangover_ms: int = 250
    active_republish_interval_ms: int = 250
    publish_voice_activity: bool = True
    model_path: str | None = None
    model_config_path: str | None = str(
        Path(__file__).resolve().parents[1] / "config" / "silero_vad_config.json"
    )
    probability_threshold: float = 0.5
    webrtc_escalation_low_threshold: float = 0.35
    webrtc_escalation_high_threshold: float = 0.85
    silero_cadence_ms: float | None = None
    silero_min_activation_duration_ms: float = 60.0
    silero_device: str | None = None
    min_speech_duration_ms: int = 150
    min_silence_duration_ms: int = 250
    speech_pad_ms: int = 30


@dataclass
class AudioAgcCfg:
    """Automatic gain control settings."""

    enabled: bool = False
    target_level_dbfs: float = -20.0
    max_gain_db: float = 20.0
    attack_ms: int = 10
    release_ms: int = 200
    headroom_db: float = 0.8
    limiter_attack_ms: float = 0.75
    limiter_release_ms: float = 60.0


@dataclass
class AudioHighpassCfg:
    """High-pass filter settings."""

    enabled: bool = True
    cutoff_hz: float = 100.0


@dataclass
class AcousticEchoCancellationCfg:
    """Acoustic echo cancellation settings."""

    enabled: bool = False
    frame_duration_ms: int = 10
    stream_delay_ms: float = 80.0
    noise_suppression: bool = False
    high_pass_filter: bool = False
    auto_gain_control: bool = False
    playback_event_topic: str | None = "audio/playback_frame"
    playback_source: str = "event_bus"
    loopback_backend: str = "auto"
    loopback_device_index: int | None = None
    loopback_source_name: str | None = None
    loopback_device_name_contains: str | None = None
    loopback_frame_duration_ms: int | None = None


@dataclass
class AudioBufferCfg:
    """Speech buffering settings used by recording modules."""

    pre_roll_ms: float = 300.0
    post_roll_ms: float = 400.0
    max_silence_ms: float = 1000.0
    max_segment_duration_ms: float = 30_000.0
    max_segment_bytes: int = 5 * 1024 * 1024
    queue_maxsize: int = 0


@dataclass
class AudioProcessingCfg:
    """Audio stream processing settings (VAD and event publishing)."""

    enabled: bool = True
    vad: AudioVadCfg = field(default_factory=AudioVadCfg)
    agc: AudioAgcCfg = field(default_factory=AudioAgcCfg)
    highpass: AudioHighpassCfg = field(default_factory=AudioHighpassCfg)
    noise_suppression: NoiseSuppressionCfg = field(default_factory=NoiseSuppressionCfg)
    aec: AcousticEchoCancellationCfg = field(default_factory=AcousticEchoCancellationCfg)

@dataclass
class AudioEmotionCfg:
    """Audio emotion analysis settings."""

    enabled: bool = False
    topic: str = "audio/emotion"
    acoustic_model_preset: str = "wav2vec2_xlsr"
    acoustic_model: str | None = None
    acoustic_device: str | None = None
    acoustic_weight: float = 0.6
    model_root: str | None = "models/audio/emotion"
    text_model: str | None = "IlyaGusev/rubert-base-cased-emotion"
    text_device: str | None = None
    text_weight: float = 0.4
    min_emit_prob: float = 0.0
    min_text_length: int = 4


@dataclass
class WhisperSttCfg:
    """Whisper STT settings."""

    enabled: bool = False
    model: str = "small"
    device: str | None = None
    compute_type: str | None = None
    download_root: str | None = None
    blacklist_path: str | None = "config/blacklist.txt"
    local_files_only: bool = False
    cpu_threads: int | None = None
    num_workers: int | None = None
    language: str | None = None
    task: str = "transcribe"
    beam_size: int | None = None
    best_of: int | None = None
    patience: float | None = None
    temperature: float | tuple[float, ...] | None = 0.0
    temperature_increment_on_fallback: float | None = None
    initial_prompt: str | None = None
    condition_on_previous_text: bool = False
    compression_ratio_threshold: float | None = None
    logprob_threshold: float | None = None
    no_speech_threshold: float | None = None
    length_penalty: float | None = None
    max_initial_timestamp: float | None = None
    suppress_tokens: str | None = None
    suppress_blank: bool = True
    prompt_reset_on_temperature: bool = False
    vad_filter: bool = False
    vad_parameters: dict[str, float] | None = None
    word_timestamps: bool = False
    without_timestamps: bool = True
    sample_rate: int = 16_000
    partial_topic: str = "audio/stt/partial"
    final_topic: str = "audio/stt/final"
    min_confidence: float = 0.35
    stability_timeout_s: float = 1.2
    queue_wait_s: float = 0.2
    enable_punctuation: bool = True


@dataclass
class SpeechGateModelCfg:
    path: str = "models/directed-ruElectra-small-fp16"
    device: str = "cpu"
    threshold: float = 0.7
    max_length: int = 64


@dataclass
class SpeechGateCfg:
    enable: bool = True
    mode: str = "normal"
    rules_threshold: float = 0.7
    final_threshold: float = 0.5
    attention_window_seconds: float = 8.0
    attention_extension_seconds: float = 3.0
    patterns_file: str | None = "config/speech_gate_patterns.json"
    identity_file: str | None = None
    assistant_names: list[str] = field(default_factory=list)
    command_verbs: list[str] = field(default_factory=list)
    politeness_markers: list[str] = field(default_factory=list)
    question_markers: list[str] = field(default_factory=list)
    modal_markers: list[str] = field(default_factory=list)
    continuation_patterns: list[str] = field(default_factory=list)
    model: SpeechGateModelCfg = field(default_factory=SpeechGateModelCfg)


@dataclass
class AudioCfg:
    input: AudioInputCfg = field(default_factory=AudioInputCfg)
    playback: AudioPlaybackCfg = field(default_factory=AudioPlaybackCfg)
    processing: AudioProcessingCfg = field(default_factory=AudioProcessingCfg)
    buffer: AudioBufferCfg = field(default_factory=AudioBufferCfg)
    stt: WhisperSttCfg = field(default_factory=WhisperSttCfg)
    emotion: AudioEmotionCfg = field(default_factory=AudioEmotionCfg)


class Config:
    def __init__(self) -> None:
        self.debug: bool = False
        self.info: bool = False
        self.preview_width: int = 960
        root = Path(__file__).resolve().parents[1]
        self.paths = Paths(root=root)
        self.performance = PerformanceCfg()
        self.openclaw = OpenClawInputCfg()
        self.indicators = SoundIndicatorsCfg()
        self.control = ControlCfg()
        self.service = ServiceCfg()
        self.events = EventsCfg()
        self.speech_gate = SpeechGateCfg()
        self.audio = AudioCfg()
        self.speaker = SpeakerConfig.from_openclaw_defaults()
        self.apply_event_topic_defaults()

    def apply_event_topic_defaults(self) -> None:
        audio_events = self.events.audio
        self.audio.processing.aec.playback_event_topic = audio_events.playback_frame
        self.audio.stt.partial_topic = audio_events.stt_partial
        self.audio.stt.final_topic = audio_events.stt_final
        if not str(self.openclaw.source_topic or "").strip():
            self.openclaw.source_topic = self.events.llm.accepted_phrase

cfg = Config()

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def load(path: str | None = None) -> None:
    """Load config/config.json (if present) and update ``cfg``."""
    root = cfg.paths.root
    p = Path(path) if path else (root / "config" / "config.json")
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8-sig")) or {}

        # debug
        cfg.debug = bool(data.get("debug", cfg.debug))
        cfg.info = bool(data.get("info", cfg.info))

        # preview_width
        cfg.preview_width = int(data.get("preview_width", 960))
        # openclaw
        openclaw_section = _as_dict(data.get("openclaw"))
        if openclaw_section:
            openclaw_cfg = cfg.openclaw
            if "enabled" in openclaw_section:
                openclaw_cfg.enabled = bool(openclaw_section.get("enabled", openclaw_cfg.enabled))

            command_value = openclaw_section.get("command", openclaw_cfg.command)
            if isinstance(command_value, str) and command_value.strip():
                openclaw_cfg.command = command_value.strip()

            transport_value = openclaw_section.get("transport", openclaw_cfg.transport)
            if isinstance(transport_value, str) and transport_value.strip():
                transport = transport_value.strip().lower()
                if transport in {"gateway_ws", "cli"}:
                    openclaw_cfg.transport = transport

            source_topic_value = openclaw_section.get("source_topic", openclaw_cfg.source_topic)
            if isinstance(source_topic_value, str) and source_topic_value.strip():
                openclaw_cfg.source_topic = source_topic_value.strip()

            session_key_value = openclaw_section.get("session_key", openclaw_cfg.session_key)
            if isinstance(session_key_value, str) and session_key_value.strip():
                openclaw_cfg.session_key = session_key_value.strip()

            gateway_url_value = openclaw_section.get("gateway_url", openclaw_cfg.gateway_url)
            if gateway_url_value in ("", None):
                openclaw_cfg.gateway_url = None
            elif isinstance(gateway_url_value, str):
                openclaw_cfg.gateway_url = gateway_url_value.strip().rstrip("/")

            gateway_token_value = openclaw_section.get("gateway_token", openclaw_cfg.gateway_token)
            if gateway_token_value in ("", None):
                openclaw_cfg.gateway_token = None
            elif isinstance(gateway_token_value, str):
                openclaw_cfg.gateway_token = gateway_token_value.strip()

            if "call_timeout_s" in openclaw_section:
                try:
                    timeout_val = float(openclaw_section.get("call_timeout_s", openclaw_cfg.call_timeout_s))
                except (TypeError, ValueError):
                    timeout_val = openclaw_cfg.call_timeout_s
                if timeout_val > 0:
                    openclaw_cfg.call_timeout_s = timeout_val

        # performance
        performance_section = _as_dict(data.get("performance"))
        if performance_section:
            perf_cfg = cfg.performance
            if "enabled" in performance_section:
                perf_cfg.enabled = bool(performance_section.get("enabled", perf_cfg.enabled))
            log_level = performance_section.get("log_level", perf_cfg.log_level)
            if isinstance(log_level, str) and log_level.strip():
                normalized = log_level.strip().lower()
                if normalized in {"debug", "info", "warning", "error"}:
                    perf_cfg.log_level = normalized
            if "include_text_preview" in performance_section:
                perf_cfg.include_text_preview = bool(
                    performance_section.get(
                        "include_text_preview", perf_cfg.include_text_preview
                    )
                )
            if "text_preview_chars" in performance_section:
                try:
                    preview_chars = int(
                        performance_section.get(
                            "text_preview_chars", perf_cfg.text_preview_chars
                        )
                    )
                except (TypeError, ValueError):
                    preview_chars = perf_cfg.text_preview_chars
                perf_cfg.text_preview_chars = max(0, preview_chars)

        # speaker
        speaker_section = _as_dict(data.get("speaker"))
        if speaker_section:
            cfg.speaker = cfg.speaker.merge_dict(speaker_section)
        speaker_gateway_defaults: dict[str, Any] = {}
        speaker_gateway_section = _as_dict(speaker_section.get("gateway"))
        if "session_key" not in speaker_gateway_section:
            speaker_gateway_defaults["session_key"] = cfg.openclaw.session_key
        if "url" not in speaker_gateway_section and cfg.openclaw.gateway_url:
            speaker_gateway_defaults["url"] = cfg.openclaw.gateway_url
        if "token" not in speaker_gateway_section and cfg.openclaw.gateway_token:
            speaker_gateway_defaults["token"] = cfg.openclaw.gateway_token
        if speaker_gateway_defaults:
            cfg.speaker = cfg.speaker.merge_dict({"gateway": speaker_gateway_defaults})

        # sound indicators
        indicators_section = _as_dict(data.get("indicators"))
        if indicators_section:
            indicators_cfg = cfg.indicators
            if "enabled" in indicators_section:
                indicators_cfg.enabled = bool(
                    indicators_section.get("enabled", indicators_cfg.enabled)
                )

            backend_value = indicators_section.get("backend", indicators_cfg.backend)
            if isinstance(backend_value, str) and backend_value.strip():
                indicators_cfg.backend = backend_value.strip().lower()

            device_value = indicators_section.get(
                "output_device_index", indicators_cfg.output_device_index
            )
            if device_value in ("", None):
                indicators_cfg.output_device_index = None
            else:
                try:
                    indicators_cfg.output_device_index = int(device_value)
                except (TypeError, ValueError):
                    pass

            if "sample_rate" in indicators_section:
                try:
                    sample_rate = int(
                        indicators_section.get("sample_rate", indicators_cfg.sample_rate)
                    )
                except (TypeError, ValueError):
                    sample_rate = indicators_cfg.sample_rate
                if sample_rate > 0:
                    indicators_cfg.sample_rate = sample_rate

            if "volume" in indicators_section:
                try:
                    volume = float(indicators_section.get("volume", indicators_cfg.volume))
                except (TypeError, ValueError):
                    volume = indicators_cfg.volume
                indicators_cfg.volume = _clip(volume, 0.0, 1.0)

            if "queue_maxsize" in indicators_section:
                try:
                    queue_maxsize = int(
                        indicators_section.get("queue_maxsize", indicators_cfg.queue_maxsize)
                    )
                except (TypeError, ValueError):
                    queue_maxsize = indicators_cfg.queue_maxsize
                if queue_maxsize >= 1:
                    indicators_cfg.queue_maxsize = queue_maxsize

            ducking_section = _as_dict(indicators_section.get("ducking"))
            if ducking_section:
                ducking_cfg = indicators_cfg.ducking
                if "enabled" in ducking_section:
                    ducking_cfg.enabled = bool(
                        ducking_section.get("enabled", ducking_cfg.enabled)
                    )
                if "volume_scale" in ducking_section:
                    try:
                        volume_scale = float(
                            ducking_section.get("volume_scale", ducking_cfg.volume_scale)
                        )
                    except (TypeError, ValueError):
                        volume_scale = ducking_cfg.volume_scale
                    ducking_cfg.volume_scale = _clip(volume_scale, 0.0, 1.0)
                if "fade_in_ms" in ducking_section:
                    try:
                        fade_in_ms = int(
                            ducking_section.get("fade_in_ms", ducking_cfg.fade_in_ms)
                        )
                    except (TypeError, ValueError):
                        fade_in_ms = ducking_cfg.fade_in_ms
                    ducking_cfg.fade_in_ms = max(0, fade_in_ms)
                if "fade_out_ms" in ducking_section:
                    try:
                        fade_out_ms = int(
                            ducking_section.get("fade_out_ms", ducking_cfg.fade_out_ms)
                        )
                    except (TypeError, ValueError):
                        fade_out_ms = ducking_cfg.fade_out_ms
                    ducking_cfg.fade_out_ms = max(0, fade_out_ms)

            for key in ("rejected", "forwarded", "local_handled", "interrupted"):
                if key in indicators_section:
                    setattr(
                        indicators_cfg,
                        key,
                        bool(indicators_section.get(key, getattr(indicators_cfg, key))),
                    )

        # control API
        control_section = _as_dict(data.get("control"))
        if control_section:
            control_cfg = cfg.control
            if "enabled" in control_section:
                control_cfg.enabled = bool(control_section.get("enabled", control_cfg.enabled))

            host_value = control_section.get("host", control_cfg.host)
            if isinstance(host_value, str) and host_value.strip():
                control_cfg.host = host_value.strip()

            try:
                port_value = int(control_section.get("port", control_cfg.port))
                if 0 < port_value <= 65535:
                    control_cfg.port = port_value
            except (TypeError, ValueError):
                pass

            token_value = control_section.get("token", control_cfg.token)
            if token_value in ("", None):
                control_cfg.token = None
            elif isinstance(token_value, str):
                control_cfg.token = token_value.strip() or None
            else:
                control_cfg.token = str(token_value)

            try:
                max_ttl_value = float(
                    control_section.get("max_ttl_seconds", control_cfg.max_ttl_seconds)
                )
                if max_ttl_value > 0:
                    control_cfg.max_ttl_seconds = max_ttl_value
            except (TypeError, ValueError):
                pass

            if "state_path" in control_section:
                state_path = _resolve_project_path(control_section.get("state_path"), root)
                control_cfg.state_path = state_path

        # service integration
        service_section = _as_dict(data.get("service"))
        if service_section:
            service_cfg = cfg.service
            if "strict_startup" in service_section:
                service_cfg.strict_startup = bool(
                    service_section.get("strict_startup", service_cfg.strict_startup)
                )

        # speech gate
        speech_gate_section = _as_dict(data.get("speech_gate"))
        if speech_gate_section:
            sg_cfg = cfg.speech_gate

            if "enable" in speech_gate_section:
                sg_cfg.enable = bool(speech_gate_section.get("enable", sg_cfg.enable))

            mode_value = speech_gate_section.get("mode", sg_cfg.mode)
            if isinstance(mode_value, str) and mode_value.strip():
                sg_cfg.mode = mode_value.strip().lower()

            try:
                sg_cfg.rules_threshold = _clip(
                    float(speech_gate_section.get("rules_threshold", sg_cfg.rules_threshold)),
                    0.0,
                    1.0,
                )
            except (TypeError, ValueError):
                pass

            try:
                sg_cfg.final_threshold = _clip(
                    float(speech_gate_section.get("final_threshold", sg_cfg.final_threshold)),
                    0.0,
                    1.0,
                )
            except (TypeError, ValueError):
                pass

            try:
                sg_cfg.attention_window_seconds = max(
                    0.0,
                    float(
                        speech_gate_section.get(
                            "attention_window_seconds",
                            sg_cfg.attention_window_seconds,
                        )
                    ),
                )
            except (TypeError, ValueError):
                pass

            try:
                sg_cfg.attention_extension_seconds = max(
                    0.0,
                    float(
                        speech_gate_section.get(
                            "attention_extension_seconds",
                            sg_cfg.attention_extension_seconds,
                        )
                    ),
                )
            except (TypeError, ValueError):
                pass

            if "patterns_file" in speech_gate_section:
                patterns_file_value = speech_gate_section.get(
                    "patterns_file", sg_cfg.patterns_file
                )
                sg_cfg.patterns_file = _resolve_project_path(patterns_file_value, root)

            if "identity_file" in speech_gate_section:
                identity_file_value = speech_gate_section.get(
                    "identity_file", sg_cfg.identity_file
                )
                if isinstance(identity_file_value, str) and identity_file_value.strip().lower() == "auto":
                    sg_cfg.identity_file = None
                else:
                    sg_cfg.identity_file = _resolve_project_path(identity_file_value, root)

            def _clean_str_list(value: object, current: list[str]) -> list[str]:
                if isinstance(value, list):
                    cleaned = [str(item).strip() for item in value if str(item).strip()]
                    return cleaned
                return list(current)

            sg_cfg.assistant_names = _clean_str_list(
                speech_gate_section.get("assistant_names", sg_cfg.assistant_names),
                sg_cfg.assistant_names,
            )
            sg_cfg.command_verbs = _clean_str_list(
                speech_gate_section.get("command_verbs", sg_cfg.command_verbs),
                sg_cfg.command_verbs,
            )
            sg_cfg.politeness_markers = _clean_str_list(
                speech_gate_section.get(
                    "politeness_markers", sg_cfg.politeness_markers
                ),
                sg_cfg.politeness_markers,
            )
            sg_cfg.question_markers = _clean_str_list(
                speech_gate_section.get("question_markers", sg_cfg.question_markers),
                sg_cfg.question_markers,
            )
            sg_cfg.modal_markers = _clean_str_list(
                speech_gate_section.get("modal_markers", sg_cfg.modal_markers),
                sg_cfg.modal_markers,
            )
            sg_cfg.continuation_patterns = _clean_str_list(
                speech_gate_section.get(
                    "continuation_patterns", sg_cfg.continuation_patterns
                ),
                sg_cfg.continuation_patterns,
            )

            model_section = _as_dict(speech_gate_section.get("model"))
            model_cfg = sg_cfg.model
            if model_section:
                if "path" in model_section:
                    model_path = _resolve_project_path(
                        model_section.get("path", model_cfg.path),
                        root,
                    )
                    if model_path is not None:
                        model_cfg.path = model_path
                if "device" in model_section:
                    model_cfg.device = str(model_section.get("device", model_cfg.device))
                try:
                    model_cfg.threshold = _clip(
                        float(model_section.get("threshold", model_cfg.threshold)),
                        0.0,
                        1.0,
                    )
                except (TypeError, ValueError):
                    pass
                try:
                    model_cfg.max_length = max(
                        1,
                        int(model_section.get("max_length", model_cfg.max_length)),
                    )
                except (TypeError, ValueError):
                    pass

        # events/topics
        events_section = _as_dict(data.get("events"))
        if events_section:
            _apply_event_topics(events_section, cfg.events)
        cfg.apply_event_topic_defaults()
        # audio
        a = _as_dict(data.get("audio"))
        ai = _as_dict(a.get("input"))
        input_cfg = cfg.audio.input
        input_rate_source = ai.get("input_sample_rate", input_cfg.input_sample_rate)
        try:
            input_cfg.input_sample_rate = int(input_rate_source)
        except (TypeError, ValueError):
            input_cfg.input_sample_rate = int(input_cfg.input_sample_rate)

        output_rate_source = ai.get(
            "output_sample_rate", input_cfg.output_sample_rate
        )
        if output_rate_source in ("", None):
            input_cfg.output_sample_rate = int(input_cfg.input_sample_rate)
        else:
            try:
                input_cfg.output_sample_rate = int(output_rate_source)
            except (TypeError, ValueError):
                input_cfg.output_sample_rate = int(input_cfg.input_sample_rate)
        input_cfg.chunk_size = int(ai.get("chunk_size", input_cfg.chunk_size))
        input_cfg.channels = int(ai.get("channels", input_cfg.channels))
        device_idx = ai.get("device_index", input_cfg.device_index)
        if device_idx in ("", None):
            input_cfg.device_index = None
        else:
            input_cfg.device_index = int(device_idx)

        playback_section = _as_dict(a.get("playback"))
        playback_cfg = cfg.audio.playback
        if playback_section:
            playback_cfg.enabled = bool(playback_section.get("enabled", playback_cfg.enabled))
            if "sample_rate" in playback_section:
                try:
                    playback_cfg.sample_rate = int(
                        playback_section.get("sample_rate", playback_cfg.sample_rate)
                    )
                except (TypeError, ValueError):
                    playback_cfg.sample_rate = int(playback_cfg.sample_rate)
            if "channels" in playback_section:
                try:
                    playback_cfg.channels = int(
                        playback_section.get("channels", playback_cfg.channels)
                    )
                except (TypeError, ValueError):
                    playback_cfg.channels = int(playback_cfg.channels)
            if "chunk_size" in playback_section:
                try:
                    playback_cfg.chunk_size = int(
                        playback_section.get("chunk_size", playback_cfg.chunk_size)
                    )
                except (TypeError, ValueError):
                    playback_cfg.chunk_size = int(playback_cfg.chunk_size)
            device_idx = playback_section.get("device_index", playback_cfg.device_index)
            if device_idx in ("", None):
                playback_cfg.device_index = None
            else:
                playback_cfg.device_index = int(device_idx)
            if "queue_maxsize" in playback_section:
                try:
                    playback_cfg.queue_maxsize = int(
                        playback_section.get("queue_maxsize", playback_cfg.queue_maxsize)
                    )
                except (TypeError, ValueError):
                    playback_cfg.queue_maxsize = int(playback_cfg.queue_maxsize)
            if "volume" in playback_section:
                try:
                    volume = float(playback_section.get("volume", playback_cfg.volume))
                except (TypeError, ValueError):
                    volume = float(playback_cfg.volume)
                playback_cfg.volume = _clip(volume, 0.0, 2.0)

        ap = _as_dict(a.get("processing"))
        proc_cfg = cfg.audio.processing
        proc_cfg.enabled = bool(ap.get("enabled", proc_cfg.enabled))

        vad_section = _as_dict(ap.get("vad"))
        vad_cfg = proc_cfg.vad
        vad_cfg.enabled = bool(vad_section.get("enabled", vad_cfg.enabled))

        current_pipeline = vad_cfg.pipeline
        pipeline_raw = vad_section.get("pipeline", current_pipeline)
        try:
            pipeline = str(pipeline_raw).strip().lower()
        except Exception:
            pipeline = str(current_pipeline).strip().lower()
        if pipeline not in {"hybrid", "webrtc", "silero"}:
            pipeline = current_pipeline
        vad_cfg.pipeline = pipeline

        vad_cfg.mode = _clip(int(vad_section.get("mode", vad_cfg.mode)), 0, 3)

        frame_duration_source = vad_section.get(
            "frame_duration_ms", vad_cfg.frame_duration_ms
        )
        try:
            vad_cfg.frame_duration_ms = int(frame_duration_source)
        except (TypeError, ValueError):
            vad_cfg.frame_duration_ms = int(vad_cfg.frame_duration_ms)
        if vad_cfg.frame_duration_ms not in (10, 20, 30):
            vad_cfg.frame_duration_ms = 30

        energy_source = vad_section.get("energy_threshold_db", vad_cfg.energy_threshold_db)
        try:
            vad_cfg.energy_threshold_db = float(energy_source)
        except (TypeError, ValueError):
            vad_cfg.energy_threshold_db = float(vad_cfg.energy_threshold_db)

        try:
            vad_cfg.hangover_ms = int(
                vad_section.get("hangover_ms", vad_cfg.hangover_ms)
            )
        except (TypeError, ValueError):
            vad_cfg.hangover_ms = int(vad_cfg.hangover_ms)

        active_interval_source = vad_section.get(
            "active_republish_interval_ms", vad_cfg.hangover_ms
        )
        if active_interval_source in ("", None):
            active_interval = vad_cfg.hangover_ms
        else:
            try:
                active_interval = int(active_interval_source)
            except (TypeError, ValueError):
                active_interval = vad_cfg.hangover_ms
        vad_cfg.active_republish_interval_ms = active_interval

        vad_cfg.publish_voice_activity = bool(
            vad_section.get("publish_voice_activity", vad_cfg.publish_voice_activity)
        )

        model_path_value = vad_section.get("model_path", vad_cfg.model_path)
        vad_cfg.model_path = _resolve_project_path(model_path_value, root)

        model_cfg_value = vad_section.get(
            "model_config_path", vad_cfg.model_config_path
        )
        vad_cfg.model_config_path = _resolve_project_path(model_cfg_value, root)

        probability_source = vad_section.get(
            "probability_threshold", vad_cfg.probability_threshold
        )
        try:
            probability = float(probability_source)
        except (TypeError, ValueError):
            probability = float(vad_cfg.probability_threshold)
        vad_cfg.probability_threshold = _clip(probability, 0.0, 1.0)

        low_threshold_raw = vad_section.get(
            "webrtc_escalation_low_threshold",
            vad_cfg.webrtc_escalation_low_threshold,
        )
        try:
            low_threshold = float(low_threshold_raw)
        except (TypeError, ValueError):
            low_threshold = float(vad_cfg.webrtc_escalation_low_threshold)
        vad_cfg.webrtc_escalation_low_threshold = _clip(low_threshold, 0.0, 1.0)

        high_threshold_raw = vad_section.get(
            "webrtc_escalation_high_threshold",
            vad_cfg.webrtc_escalation_high_threshold,
        )
        try:
            high_threshold = float(high_threshold_raw)
        except (TypeError, ValueError):
            high_threshold = float(vad_cfg.webrtc_escalation_high_threshold)
        high_threshold = _clip(high_threshold, 0.0, 1.0)
        if high_threshold < vad_cfg.webrtc_escalation_low_threshold:
            high_threshold = vad_cfg.webrtc_escalation_low_threshold
        vad_cfg.webrtc_escalation_high_threshold = high_threshold

        cadence_source = vad_section.get("silero_cadence_ms", vad_cfg.silero_cadence_ms)
        if cadence_source in (None, ""):
            cadence_ms: float | None = None
        else:
            try:
                cadence_ms = float(cadence_source)
            except (TypeError, ValueError):
                cadence_ms = vad_cfg.silero_cadence_ms
        if cadence_ms is not None and cadence_ms <= 0.0:
            cadence_ms = None
        vad_cfg.silero_cadence_ms = cadence_ms

        silero_device = vad_section.get("silero_device", vad_cfg.silero_device)
        if silero_device in (None, ""):
            vad_cfg.silero_device = None
        else:
            vad_cfg.silero_device = str(silero_device)

        min_activation_raw = vad_section.get(
            "silero_min_activation_duration_ms",
            vad_cfg.silero_min_activation_duration_ms,
        )
        try:
            min_activation_ms = float(min_activation_raw)
        except (TypeError, ValueError):
            min_activation_ms = float(vad_cfg.silero_min_activation_duration_ms)
        if min_activation_ms < 0.0:
            min_activation_ms = 0.0
        vad_cfg.silero_min_activation_duration_ms = min_activation_ms

        try:
            min_speech = int(
                vad_section.get("min_speech_duration_ms", vad_cfg.min_speech_duration_ms)
            )
        except (TypeError, ValueError):
            min_speech = int(vad_cfg.min_speech_duration_ms)
        vad_cfg.min_speech_duration_ms = max(0, min_speech)

        try:
            min_silence = int(
                vad_section.get(
                    "min_silence_duration_ms", vad_cfg.min_silence_duration_ms
                )
            )
        except (TypeError, ValueError):
            min_silence = int(vad_cfg.min_silence_duration_ms)
        vad_cfg.min_silence_duration_ms = max(0, min_silence)

        try:
            speech_pad = int(vad_section.get("speech_pad_ms", vad_cfg.speech_pad_ms))
        except (TypeError, ValueError):
            speech_pad = int(vad_cfg.speech_pad_ms)
        vad_cfg.speech_pad_ms = max(0, speech_pad)

        ns = _as_dict(ap.get("noise_suppression"))
        ns_cfg = proc_cfg.noise_suppression
        ns_cfg.enabled = bool(ns.get("enabled", ns_cfg.enabled))
        ns_cfg.frame_duration_ms = max(
            5, int(ns.get("frame_duration_ms", ns_cfg.frame_duration_ms))
        )
        ns_cfg.energy_threshold_ratio = max(
            1.0,
            float(ns.get("energy_threshold_ratio", ns_cfg.energy_threshold_ratio)),
        )
        ns_cfg.suppression_factor = max(
            0.0, float(ns.get("suppression_factor", ns_cfg.suppression_factor))
        )
        ns_cfg.noise_learning_rate = float(
            max(0.0, min(1.0, ns.get("noise_learning_rate", ns_cfg.noise_learning_rate)))
        )
        ns_cfg.noise_release_rate = float(
            max(0.0, min(1.0, ns.get("noise_release_rate", ns_cfg.noise_release_rate)))
        )
        ns_cfg.gain_smoothing = float(
            max(0.0, min(0.999, ns.get("gain_smoothing", ns_cfg.gain_smoothing)))
        )
        ns_cfg.min_gain = float(
            max(0.0, min(1.0, ns.get("min_gain", ns_cfg.min_gain)))
        )

        aec = _as_dict(ap.get("aec"))
        aec_cfg = proc_cfg.aec
        aec_cfg.enabled = bool(aec.get("enabled", aec_cfg.enabled))
        try:
            aec_cfg.frame_duration_ms = max(5, int(aec.get("frame_duration_ms", aec_cfg.frame_duration_ms)))
        except (TypeError, ValueError):
            aec_cfg.frame_duration_ms = max(5, int(aec_cfg.frame_duration_ms))
        try:
            aec_cfg.stream_delay_ms = float(aec.get("stream_delay_ms", aec_cfg.stream_delay_ms))
        except (TypeError, ValueError):
            aec_cfg.stream_delay_ms = float(aec_cfg.stream_delay_ms)
        aec_cfg.noise_suppression = bool(
            aec.get("noise_suppression", aec_cfg.noise_suppression)
        )
        aec_cfg.high_pass_filter = bool(aec.get("high_pass_filter", aec_cfg.high_pass_filter))
        aec_cfg.auto_gain_control = bool(aec.get("auto_gain_control", aec_cfg.auto_gain_control))
        ps_raw = aec.get("playback_source", aec_cfg.playback_source)
        if ps_raw in ("", None):
            ps_clean = None
        else:
            try:
                ps_norm = str(ps_raw).strip()
            except Exception:
                ps_norm = ""
            if ps_norm == "":
                ps_clean = None
            else:
                ps_clean = ps_norm.lower()
        aec_cfg.playback_source = ps_clean
        backend_raw = aec.get("loopback_backend", aec_cfg.loopback_backend)
        try:
            backend_clean = str(backend_raw or "auto").strip().lower()
        except Exception:
            backend_clean = "auto"
        if backend_clean not in {"auto", "wasapi", "pipewire", "pulse", "sounddevice_monitor"}:
            backend_clean = "auto"
        aec_cfg.loopback_backend = backend_clean
        lbi_raw = aec.get("loopback_device_index", aec_cfg.loopback_device_index)
        lbi_clean = aec_cfg.loopback_device_index
        if lbi_raw in ("", None):
            lbi_clean = None
        else:
            try:
                _lbi = int(lbi_raw)
                lbi_clean = None if _lbi < 0 else _lbi
            except (TypeError, ValueError):
                # Conversion error: keep current value.
                pass
        aec_cfg.loopback_device_index = lbi_clean
        source_raw = aec.get(
            "loopback_source_name",
            aec_cfg.loopback_source_name,
        )
        if source_raw in ("", None):
            aec_cfg.loopback_source_name = None
        else:
            try:
                source_clean = str(source_raw).strip()
            except Exception:
                source_clean = ""
            aec_cfg.loopback_source_name = source_clean or None
        name_raw = aec.get(
            "loopback_device_name_contains",
            aec_cfg.loopback_device_name_contains,
        )
        if name_raw in ("", None):
            aec_cfg.loopback_device_name_contains = None
        else:
            try:
                name_clean = str(name_raw).strip()
            except Exception:
                name_clean = ""
            aec_cfg.loopback_device_name_contains = name_clean or None
        lfd_raw = aec.get("loopback_frame_duration_ms", aec_cfg.loopback_frame_duration_ms)
        lfd_clean = aec_cfg.loopback_frame_duration_ms
        if lfd_raw in ("", None):
            lfd_clean = None
        else:
            try:
                _lfd = int(lfd_raw)
                lfd_clean = None if _lfd <= 0 else _lfd
            except (TypeError, ValueError):
                # Conversion error: keep current value.
                pass
        aec_cfg.loopback_frame_duration_ms = lfd_clean

        agc_section = _as_dict(ap.get("agc"))
        agc_cfg = proc_cfg.agc
        agc_cfg.enabled = bool(agc_section.get("enabled", agc_cfg.enabled))
        agc_cfg.target_level_dbfs = float(
            agc_section.get("target_level_dbfs", agc_cfg.target_level_dbfs)
        )
        agc_cfg.target_level_dbfs = _clip(agc_cfg.target_level_dbfs, -100.0, 0.0)
        agc_cfg.max_gain_db = float(agc_section.get("max_gain_db", agc_cfg.max_gain_db))
        agc_cfg.max_gain_db = _clip(agc_cfg.max_gain_db, 0.0, 60.0)
        agc_cfg.attack_ms = int(agc_section.get("attack_ms", agc_cfg.attack_ms))
        agc_cfg.attack_ms = max(1, min(1000, agc_cfg.attack_ms))
        agc_cfg.release_ms = int(agc_section.get("release_ms", agc_cfg.release_ms))
        agc_cfg.release_ms = max(agc_cfg.attack_ms, min(5000, agc_cfg.release_ms))

        agc_cfg.headroom_db = float(agc_section.get("headroom_db", agc_cfg.headroom_db))
        agc_cfg.headroom_db = _clip(agc_cfg.headroom_db, 0.0, 12.0)

        limiter_attack = float(
            agc_section.get("limiter_attack_ms", agc_cfg.limiter_attack_ms)
        )
        limiter_attack = max(0.1, min(100.0, limiter_attack))
        agc_cfg.limiter_attack_ms = limiter_attack

        limiter_release = float(
            agc_section.get("limiter_release_ms", agc_cfg.limiter_release_ms)
        )
        limiter_release = max(limiter_attack, min(5000.0, limiter_release))
        agc_cfg.limiter_release_ms = limiter_release

        highpass_section = _as_dict(ap.get("highpass"))
        highpass_cfg = proc_cfg.highpass
        highpass_cfg.enabled = bool(highpass_section.get("enabled", highpass_cfg.enabled))
        highpass_cfg.cutoff_hz = float(
            highpass_section.get("cutoff_hz", highpass_cfg.cutoff_hz)
        )
        highpass_cfg.cutoff_hz = max(0.0, min(2000.0, highpass_cfg.cutoff_hz))

        # buffer
        buffer_section = _as_dict(a.get("buffer"))
        buffer_cfg = cfg.audio.buffer

        def _as_float(source: object, default: float) -> float:
            try:
                return float(source)
            except (TypeError, ValueError):
                return float(default)

        def _as_int(source: object, default: int) -> int:
            try:
                return int(source)
            except (TypeError, ValueError):
                return int(default)

        buffer_cfg.pre_roll_ms = max(
            0.0, _as_float(buffer_section.get("pre_roll_ms", buffer_cfg.pre_roll_ms), buffer_cfg.pre_roll_ms)
        )
        buffer_cfg.post_roll_ms = max(
            0.0, _as_float(buffer_section.get("post_roll_ms", buffer_cfg.post_roll_ms), buffer_cfg.post_roll_ms)
        )
        buffer_cfg.max_silence_ms = max(
            0.0, _as_float(buffer_section.get("max_silence_ms", buffer_cfg.max_silence_ms), buffer_cfg.max_silence_ms)
        )
        buffer_cfg.max_segment_duration_ms = max(
            0.0,
            _as_float(
                buffer_section.get("max_segment_duration_ms", buffer_cfg.max_segment_duration_ms),
                buffer_cfg.max_segment_duration_ms,
            ),
        )
        buffer_cfg.max_segment_bytes = max(
            0,
            _as_int(
                buffer_section.get("max_segment_bytes", buffer_cfg.max_segment_bytes),
                buffer_cfg.max_segment_bytes,
            ),
        )
        buffer_cfg.queue_maxsize = max(
            0,
            _as_int(buffer_section.get("queue_maxsize", buffer_cfg.queue_maxsize), buffer_cfg.queue_maxsize),
        )

        stt_section = _as_dict(a.get("stt"))
        stt_cfg = cfg.audio.stt
        stt_cfg.enabled = bool(stt_section.get("enabled", stt_cfg.enabled))

        def _clean_required_string(value: object, current: str) -> str:
            if value in (None, ""):
                return current
            try:
                text = str(value).strip()
            except Exception:
                return current
            return text or current

        def _clean_optional_string(value: object, current: str | None) -> str | None:
            if value in (None, ""):
                return None
            try:
                text = str(value).strip()
            except Exception:
                return current
            return text or None

        def _parse_optional_positive_int(value: object, current: int | None) -> int | None:
            if value in (None, ""):
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return current
            return parsed if parsed > 0 else (current if current and current > 0 else None)

        def _parse_optional_float(value: object, current: float | None) -> float | None:
            if value in (None, ""):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return current

        def _parse_bool(value: object, current: bool) -> bool:
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "on"}:
                    return True
                if lowered in {"0", "false", "no", "off"}:
                    return False
                return current
            if value in (None, ""):
                return current
            return bool(value)

        def _parse_temperature(value: object, current: float | tuple[float, ...] | None):
            if value is None:
                return None
            if value == "":
                return None if current is None else current
            if isinstance(value, (list, tuple)):
                cleaned: list[float] = []
                for item in value:
                    try:
                        cleaned.append(float(item))
                    except (TypeError, ValueError):
                        continue
                return tuple(cleaned)
            if isinstance(value, str) and "," in value:
                parts = [part.strip() for part in value.split(",") if part.strip()]
                cleaned = []
                for part in parts:
                    try:
                        cleaned.append(float(part))
                    except ValueError:
                        continue
                if cleaned:
                    return tuple(cleaned)
            try:
                return float(value)
            except (TypeError, ValueError):
                return current

        stt_model_value = _clean_required_string(
            stt_section.get("model", stt_cfg.model),
            stt_cfg.model,
        )
        stt_model_path = _resolve_project_path(stt_model_value, root)
        if stt_model_path is not None and Path(stt_model_path).exists():
            stt_cfg.model = stt_model_path
        else:
            stt_cfg.model = stt_model_value
        stt_cfg.device = _clean_optional_string(stt_section.get("device", stt_cfg.device), stt_cfg.device)
        stt_cfg.compute_type = _clean_optional_string(
            stt_section.get("compute_type", stt_cfg.compute_type), stt_cfg.compute_type
        )
        stt_download_root = _clean_optional_string(
            stt_section.get("download_root", stt_cfg.download_root), stt_cfg.download_root
        )
        stt_cfg.download_root = _resolve_project_path(stt_download_root, root)
        stt_blacklist_path = _clean_optional_string(
            stt_section.get("blacklist_path", stt_cfg.blacklist_path),
            stt_cfg.blacklist_path,
        )
        stt_cfg.blacklist_path = _resolve_project_path(stt_blacklist_path, root)
        stt_cfg.local_files_only = _parse_bool(
            stt_section.get("local_files_only", stt_cfg.local_files_only), stt_cfg.local_files_only
        )
        stt_cfg.cpu_threads = _parse_optional_positive_int(
            stt_section.get("cpu_threads", stt_cfg.cpu_threads), stt_cfg.cpu_threads
        )
        stt_cfg.num_workers = _parse_optional_positive_int(
            stt_section.get("num_workers", stt_cfg.num_workers), stt_cfg.num_workers
        )

        stt_cfg.language = _clean_optional_string(stt_section.get("language", stt_cfg.language), stt_cfg.language)
        stt_cfg.task = _clean_required_string(stt_section.get("task", stt_cfg.task), stt_cfg.task)
        stt_cfg.beam_size = _parse_optional_positive_int(
            stt_section.get("beam_size", stt_cfg.beam_size), stt_cfg.beam_size
        )
        stt_cfg.best_of = _parse_optional_positive_int(
            stt_section.get("best_of", stt_cfg.best_of), stt_cfg.best_of
        )
        stt_cfg.patience = _parse_optional_float(stt_section.get("patience", stt_cfg.patience), stt_cfg.patience)
        stt_cfg.temperature = _parse_temperature(stt_section.get("temperature", stt_cfg.temperature), stt_cfg.temperature)
        stt_cfg.temperature_increment_on_fallback = _parse_optional_float(
            stt_section.get("temperature_increment_on_fallback", stt_cfg.temperature_increment_on_fallback),
            stt_cfg.temperature_increment_on_fallback,
        )
        stt_cfg.initial_prompt = _clean_optional_string(
            stt_section.get("initial_prompt", stt_cfg.initial_prompt), stt_cfg.initial_prompt
        )
        stt_cfg.condition_on_previous_text = _parse_bool(
            stt_section.get("condition_on_previous_text", stt_cfg.condition_on_previous_text),
            stt_cfg.condition_on_previous_text,
        )
        stt_cfg.compression_ratio_threshold = _parse_optional_float(
            stt_section.get("compression_ratio_threshold", stt_cfg.compression_ratio_threshold),
            stt_cfg.compression_ratio_threshold,
        )
        stt_cfg.logprob_threshold = _parse_optional_float(
            stt_section.get("logprob_threshold", stt_cfg.logprob_threshold), stt_cfg.logprob_threshold
        )
        stt_cfg.no_speech_threshold = _parse_optional_float(
            stt_section.get("no_speech_threshold", stt_cfg.no_speech_threshold), stt_cfg.no_speech_threshold
        )
        stt_cfg.length_penalty = _parse_optional_float(
            stt_section.get("length_penalty", stt_cfg.length_penalty), stt_cfg.length_penalty
        )
        stt_cfg.max_initial_timestamp = _parse_optional_float(
            stt_section.get("max_initial_timestamp", stt_cfg.max_initial_timestamp), stt_cfg.max_initial_timestamp
        )
        stt_cfg.suppress_tokens = _clean_optional_string(
            stt_section.get("suppress_tokens", stt_cfg.suppress_tokens), stt_cfg.suppress_tokens
        )
        stt_cfg.suppress_blank = _parse_bool(
            stt_section.get("suppress_blank", stt_cfg.suppress_blank), stt_cfg.suppress_blank
        )
        stt_cfg.prompt_reset_on_temperature = _parse_bool(
            stt_section.get("prompt_reset_on_temperature", stt_cfg.prompt_reset_on_temperature),
            stt_cfg.prompt_reset_on_temperature,
        )
        stt_cfg.vad_filter = _parse_bool(stt_section.get("vad_filter", stt_cfg.vad_filter), stt_cfg.vad_filter)

        vad_params_value = stt_section.get("vad_parameters", stt_cfg.vad_parameters)
        if isinstance(vad_params_value, dict):
            stt_cfg.vad_parameters = dict(vad_params_value)
        elif vad_params_value in (None, ""):
            stt_cfg.vad_parameters = None

        stt_cfg.word_timestamps = _parse_bool(
            stt_section.get("word_timestamps", stt_cfg.word_timestamps), stt_cfg.word_timestamps
        )
        stt_cfg.without_timestamps = _parse_bool(
            stt_section.get("without_timestamps", stt_cfg.without_timestamps), stt_cfg.without_timestamps
        )

        sample_rate_value = stt_section.get("sample_rate", stt_cfg.sample_rate)
        try:
            parsed_rate = int(sample_rate_value)
        except (TypeError, ValueError):
            parsed_rate = stt_cfg.sample_rate
        else:
            if parsed_rate > 0:
                stt_cfg.sample_rate = parsed_rate

        def _as_float(value: object, current: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(current)

        stt_cfg.min_confidence = max(
            0.0,
            min(1.0, _as_float(stt_section.get("min_confidence", stt_cfg.min_confidence), stt_cfg.min_confidence)),
        )
        stt_cfg.stability_timeout_s = max(
            0.0,
            _as_float(
                stt_section.get("stability_timeout_s", stt_cfg.stability_timeout_s),
                stt_cfg.stability_timeout_s,
            ),
        )
        stt_cfg.queue_wait_s = max(
            0.0,
            _as_float(stt_section.get("queue_wait_s", stt_cfg.queue_wait_s), stt_cfg.queue_wait_s),
        )
        stt_cfg.enable_punctuation = _parse_bool(
            stt_section.get("enable_punctuation", stt_cfg.enable_punctuation), stt_cfg.enable_punctuation
        )

        emotion_section = _as_dict(a.get("emotion"))
        emotion_cfg = cfg.audio.emotion
        emotion_cfg.enabled = bool(emotion_section.get("enabled", emotion_cfg.enabled))
        emotion_cfg.acoustic_model_preset = _clean_required_string(
            emotion_section.get("acoustic_model_preset", emotion_cfg.acoustic_model_preset),
            emotion_cfg.acoustic_model_preset,
        )
        emotion_cfg.acoustic_model = _clean_optional_string(
            emotion_section.get("acoustic_model", emotion_cfg.acoustic_model), emotion_cfg.acoustic_model
        )
        emotion_cfg.acoustic_device = _clean_optional_string(
            emotion_section.get("acoustic_device", emotion_cfg.acoustic_device), emotion_cfg.acoustic_device
        )
        emotion_cfg.acoustic_weight = max(
            0.0, _as_float(emotion_section.get("acoustic_weight", emotion_cfg.acoustic_weight), emotion_cfg.acoustic_weight)
        )
        emotion_cfg.text_model = _clean_optional_string(
            emotion_section.get("text_model", emotion_cfg.text_model), emotion_cfg.text_model
        )
        emotion_cfg.text_device = _clean_optional_string(
            emotion_section.get("text_device", emotion_cfg.text_device), emotion_cfg.text_device
        )
        emotion_cfg.text_weight = max(
            0.0, _as_float(emotion_section.get("text_weight", emotion_cfg.text_weight), emotion_cfg.text_weight)
        )
        emotion_cfg.min_emit_prob = max(
            0.0, min(1.0, _as_float(emotion_section.get("min_emit_prob", emotion_cfg.min_emit_prob), emotion_cfg.min_emit_prob))
        )
        try:
            min_text_length = int(emotion_section.get("min_text_length", emotion_cfg.min_text_length))
        except (TypeError, ValueError):
            min_text_length = emotion_cfg.min_text_length
        emotion_cfg.min_text_length = max(0, min_text_length)


