from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

DEFAULT_OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "speaker.json"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
LEGACY_PIPER_DIR = PROJECT_ROOT / "piper"
DEFAULT_PLAYER_COMMAND = "/usr/bin/paplay"
SPEAKER_MODES = {"streaming", "final"}
TTS_MODES = {"persistent", "subprocess"}


def default_piper_command() -> str:
    return str(PROJECT_ROOT / ".venv" / "bin" / "python3")


def default_piper_model() -> str:
    primary = DEFAULT_MODELS_DIR / "ru_RU-irina-medium.onnx"
    legacy = LEGACY_PIPER_DIR / "ru_RU-irina-medium.onnx"
    if primary.exists() or not legacy.exists():
        return str(primary)
    return str(legacy)


@dataclass(slots=True)
class GatewayConfig:
    url: str = "ws://127.0.0.1:18789"
    token: str | None = None
    session_key: str = "main"
    history_limit: int = 8
    history_max_chars: int = 12000
    request_timeout_s: float = 10.0
    connect_timeout_s: float = 10.0

    def matches_session(self, value: str) -> bool:
        event_key = (value or "").strip()
        wanted = (self.session_key or "main").strip() or "main"
        if not event_key:
            return wanted == "main"
        if event_key == wanted:
            return True
        if wanted == "main" and event_key == "agent:main:main":
            return True
        return event_key.endswith(f":{wanted}")


@dataclass(slots=True)
class PiperConfig:
    command: str = field(default_factory=default_piper_command)
    model: str = field(default_factory=default_piper_model)
    volume: float = 1.0
    sentence_silence: float = 0.25
    timeout_s: float = 120.0
    extra_args: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DuckingConfig:
    enabled: bool = False
    volume_scale: float = 0.35
    fade_in_ms: int = 20
    fade_out_ms: int = 60


@dataclass(slots=True)
class PlaybackConfig:
    backend: str = "auto"
    command: str = DEFAULT_PLAYER_COMMAND
    client_name: str = "Speaker"
    stream_name: str = "Speaker TTS"
    timeout_s: float = 120.0
    ducking: DuckingConfig = field(default_factory=DuckingConfig)


@dataclass(slots=True)
class StreamingConfig:
    chunking: str = "sentence"
    min_chars: int = 40
    max_chars: int = 700
    flush_on_final: bool = True


@dataclass(slots=True)
class RuntimeConfig:
    mode: str = "streaming"
    tts_mode: str = "persistent"
    speak_existing_on_start: bool = False
    queue_size: int = 32
    streaming: StreamingConfig = field(default_factory=StreamingConfig)


@dataclass(slots=True)
class EmojiDisplayConfig:
    enabled: bool = False
    url: str = "http://127.0.0.1:18791"
    token: str | None = None
    timeout_s: float = 0.25
    hold_ms: int = 1200
    mode: str = "replace"
    source: str = "listener"
    send: str = "all"
    clear_on_interrupt: bool = True


@dataclass(slots=True)
class SpeakerConfig:
    enabled: bool = False
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    piper: PiperConfig = field(default_factory=PiperConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    speaker: RuntimeConfig = field(default_factory=RuntimeConfig)
    emoji_display: EmojiDisplayConfig = field(default_factory=EmojiDisplayConfig)

    @classmethod
    def load(cls, path: str | None = None) -> "SpeakerConfig":
        config = cls.from_openclaw_defaults()
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        config = config.merge_json(config_path)
        return config.apply_env()

    @classmethod
    def from_openclaw_defaults(cls, path: Path = DEFAULT_OPENCLAW_CONFIG) -> "SpeakerConfig":
        config = cls()
        data = _read_json_object(path)
        gateway = data.get("gateway") if isinstance(data, dict) else None
        if not isinstance(gateway, dict):
            return config

        port = gateway.get("port")
        if isinstance(port, int) and port > 0:
            config.gateway.url = f"ws://127.0.0.1:{port}"

        auth = gateway.get("auth")
        if isinstance(auth, dict):
            token = auth.get("token")
            if isinstance(token, str) and token.strip():
                config.gateway.token = token.strip()
        return config

    def merge_json(self, path: Path) -> "SpeakerConfig":
        data = _read_json_object(path)
        if not data:
            return self
        return self.merge_dict(data)

    def merge_dict(self, data: dict[str, Any]) -> "SpeakerConfig":
        if not isinstance(data, dict) or not data:
            return self
        runtime_data = data.get("speaker")
        if not isinstance(runtime_data, dict):
            runtime_keys = set(self.speaker.__dataclass_fields__.keys())
            runtime_data = {key: data[key] for key in runtime_keys if key in data}
        return replace(
            self,
            enabled=_parse_bool_value(data.get("enabled"), self.enabled),
            gateway=_merge_dataclass(self.gateway, data.get("gateway")),
            piper=_merge_dataclass(self.piper, data.get("piper")),
            playback=_merge_dataclass(self.playback, data.get("playback")),
            speaker=_merge_dataclass(self.speaker, runtime_data),
            emoji_display=_merge_dataclass(self.emoji_display, data.get("emoji_display")),
        )

    def apply_env(self) -> "SpeakerConfig":
        gateway = self.gateway
        piper = self.piper
        playback = self.playback
        speaker = self.speaker
        emoji_display = self.emoji_display

        if enabled := os.getenv("SPEAKER_ENABLED"):
            enabled_value = _parse_bool_value(enabled, self.enabled)
        else:
            enabled_value = self.enabled

        if url := os.getenv("SPEAKER_GATEWAY_URL"):
            gateway = replace(gateway, url=_normalize_gateway_url(url))
        else:
            gateway = replace(gateway, url=_normalize_gateway_url(gateway.url))

        token = os.getenv("SPEAKER_GATEWAY_TOKEN") or os.getenv("OPENCLAW_GATEWAY_TOKEN")
        if token:
            gateway = replace(gateway, token=token)
        if session_key := os.getenv("SPEAKER_SESSION_KEY"):
            gateway = replace(gateway, session_key=session_key)
        if mode := os.getenv("SPEAKER_MODE"):
            speaker = replace(speaker, mode=mode)
        if tts_mode := os.getenv("SPEAKER_TTS_MODE"):
            speaker = replace(speaker, tts_mode=tts_mode)
        if command := os.getenv("SPEAKER_PIPER_COMMAND"):
            piper = replace(piper, command=command)
        if model := os.getenv("SPEAKER_PIPER_MODEL"):
            piper = replace(piper, model=model)
        if volume := os.getenv("SPEAKER_PIPER_VOLUME"):
            piper = replace(piper, volume=float(volume))
        if player := os.getenv("SPEAKER_PLAYER_COMMAND"):
            playback = replace(playback, command=player)
        if backend := os.getenv("SPEAKER_PLAYBACK_BACKEND"):
            playback = replace(playback, backend=backend)
        if fade_in_ms := os.getenv("SPEAKER_DUCKING_FADE_IN_MS") or os.getenv("SPEAKER_FADE_IN_MS"):
            playback = replace(
                playback,
                ducking=replace(playback.ducking, fade_in_ms=int(fade_in_ms)),
            )
        if fade_out_ms := os.getenv("SPEAKER_DUCKING_FADE_OUT_MS") or os.getenv("SPEAKER_FADE_OUT_MS"):
            playback = replace(
                playback,
                ducking=replace(playback.ducking, fade_out_ms=int(fade_out_ms)),
            )
        if ducking_enabled := os.getenv("SPEAKER_DUCKING_ENABLED"):
            playback = replace(
                playback,
                ducking=replace(playback.ducking, enabled=_parse_bool(ducking_enabled)),
            )
        if ducking_scale := os.getenv("SPEAKER_DUCKING_VOLUME_SCALE"):
            playback = replace(
                playback,
                ducking=replace(playback.ducking, volume_scale=float(ducking_scale)),
            )
        if emoji_enabled := os.getenv("EMOJI_DISPLAY_ENABLED"):
            emoji_display = replace(
                emoji_display,
                enabled=_parse_bool_value(emoji_enabled, emoji_display.enabled),
            )
        if emoji_url := os.getenv("EMOJI_DISPLAY_URL"):
            emoji_display = replace(emoji_display, url=emoji_url)
        if emoji_token := os.getenv("EMOJI_DISPLAY_TOKEN"):
            emoji_display = replace(emoji_display, token=emoji_token)
        if emoji_timeout := os.getenv("EMOJI_DISPLAY_TIMEOUT_S"):
            emoji_display = replace(emoji_display, timeout_s=float(emoji_timeout))
        if emoji_hold := os.getenv("EMOJI_DISPLAY_HOLD_MS"):
            emoji_display = replace(emoji_display, hold_ms=int(emoji_hold))
        if emoji_mode := os.getenv("EMOJI_DISPLAY_MODE"):
            emoji_display = replace(emoji_display, mode=emoji_mode)
        if emoji_source := os.getenv("EMOJI_DISPLAY_SOURCE"):
            emoji_display = replace(emoji_display, source=emoji_source)
        if emoji_send := os.getenv("EMOJI_DISPLAY_SEND"):
            emoji_display = replace(emoji_display, send=emoji_send)
        if emoji_clear := os.getenv("EMOJI_DISPLAY_CLEAR_ON_INTERRUPT"):
            emoji_display = replace(
                emoji_display,
                clear_on_interrupt=_parse_bool_value(
                    emoji_clear,
                    emoji_display.clear_on_interrupt,
                ),
            )

        return replace(
            self,
            enabled=enabled_value,
            gateway=gateway,
            piper=piper,
            playback=_normalize_playback_config(playback),
            speaker=_normalize_runtime_config(speaker),
            emoji_display=_normalize_emoji_display_config(emoji_display),
        )

    def to_redacted_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("gateway", {}).get("token"):
            data["gateway"]["token"] = "<redacted>"
        if data.get("emoji_display", {}).get("token"):
            data["emoji_display"]["token"] = "<redacted>"
        return data


def _merge_dataclass(current: Any, raw: Any) -> Any:
    if not isinstance(raw, dict):
        return current
    allowed = set(current.__dataclass_fields__.keys())
    values = {key: value for key, value in raw.items() if key in allowed}
    if isinstance(current, GatewayConfig) and "url" in values:
        values["url"] = _normalize_gateway_url(str(values["url"]))
    if isinstance(current, PiperConfig):
        if "volume" in values:
            values["volume"] = max(0.0, float(values["volume"]))
        if "sentence_silence" in values:
            values["sentence_silence"] = max(0.0, float(values["sentence_silence"]))
        if "timeout_s" in values:
            values["timeout_s"] = max(1.0, float(values["timeout_s"]))
    if isinstance(current, DuckingConfig):
        if "volume_scale" in values:
            values["volume_scale"] = min(1.0, max(0.0, float(values["volume_scale"])))
        if "fade_in_ms" in values:
            values["fade_in_ms"] = max(0, int(values["fade_in_ms"]))
        if "fade_out_ms" in values:
            values["fade_out_ms"] = max(0, int(values["fade_out_ms"]))
    if isinstance(current, PlaybackConfig):
        if "timeout_s" in values:
            values["timeout_s"] = max(1.0, float(values["timeout_s"]))
        ducking = current.ducking
        if "ducking" in raw:
            ducking = _merge_dataclass(current.ducking, raw.get("ducking"))
        legacy_ducking: dict[str, Any] = {}
        if "fade_in_ms" in raw:
            legacy_ducking["fade_in_ms"] = raw["fade_in_ms"]
        if "fade_out_ms" in raw:
            legacy_ducking["fade_out_ms"] = raw["fade_out_ms"]
        if legacy_ducking:
            ducking = _merge_dataclass(ducking, legacy_ducking)
        if "ducking" in raw or legacy_ducking:
            values["ducking"] = ducking
    if isinstance(current, RuntimeConfig) and "streaming" in values:
        values["streaming"] = _merge_dataclass(current.streaming, values["streaming"])
    if isinstance(current, EmojiDisplayConfig):
        if "enabled" in values:
            values["enabled"] = _parse_bool_value(values["enabled"], current.enabled)
        if "timeout_s" in values:
            values["timeout_s"] = float(values["timeout_s"])
        if "hold_ms" in values:
            values["hold_ms"] = int(values["hold_ms"])
        if "clear_on_interrupt" in values:
            values["clear_on_interrupt"] = _parse_bool_value(
                values["clear_on_interrupt"],
                current.clear_on_interrupt,
            )
    updated = replace(current, **values)
    if isinstance(updated, PlaybackConfig):
        return _normalize_playback_config(updated)
    if isinstance(updated, EmojiDisplayConfig):
        return _normalize_emoji_display_config(updated)
    return _normalize_runtime_config(updated)


def _normalize_runtime_config(config: Any) -> Any:
    if not isinstance(config, RuntimeConfig):
        return config
    mode = str(config.mode or "streaming").strip().casefold()
    if mode not in SPEAKER_MODES:
        raise ValueError(f"speaker.mode must be one of: {', '.join(sorted(SPEAKER_MODES))}")
    tts_mode = str(getattr(config, "tts_mode", "persistent") or "persistent").strip().casefold()
    if tts_mode not in TTS_MODES:
        raise ValueError(f"speaker.tts_mode must be one of: {', '.join(sorted(TTS_MODES))}")
    streaming = config.streaming
    chunking = str(streaming.chunking or "sentence").strip().casefold()
    if chunking != "sentence":
        raise ValueError("speaker.streaming.chunking must be 'sentence'")
    normalized_streaming = replace(
        streaming,
        chunking=chunking,
        min_chars=max(0, int(streaming.min_chars)),
        max_chars=max(1, int(streaming.max_chars)),
        flush_on_final=bool(streaming.flush_on_final),
    )
    return replace(config, mode=mode, tts_mode=tts_mode, streaming=normalized_streaming)


def _normalize_playback_config(config: PlaybackConfig) -> PlaybackConfig:
    backend = str(getattr(config, "backend", "auto") or "auto").strip().casefold()
    if backend not in {"auto", "sounddevice", "paplay", "subprocess"}:
        raise ValueError("speaker.playback.backend must be one of: auto, sounddevice, paplay, subprocess")
    return replace(
        config,
        backend=backend,
        timeout_s=max(1.0, float(config.timeout_s)),
        ducking=replace(
            config.ducking,
            enabled=bool(config.ducking.enabled),
            volume_scale=min(1.0, max(0.0, float(config.ducking.volume_scale))),
            fade_in_ms=max(0, int(config.ducking.fade_in_ms)),
            fade_out_ms=max(0, int(config.ducking.fade_out_ms)),
        ),
    )


def _normalize_emoji_display_config(config: EmojiDisplayConfig) -> EmojiDisplayConfig:
    mode = str(config.mode or "replace").strip().casefold()
    if mode not in {"replace", "queue"}:
        raise ValueError("speaker.emoji_display.mode must be 'replace' or 'queue'")
    send = str(config.send or "all").strip().casefold()
    if send not in {"all", "first", "none"}:
        raise ValueError("speaker.emoji_display.send must be 'all', 'first', or 'none'")
    url = str(config.url or "http://127.0.0.1:18791").strip().rstrip("/")
    if not url:
        url = "http://127.0.0.1:18791"
    token = config.token
    if isinstance(token, str):
        token = token.strip() or None
    source = str(config.source or "listener").strip() or "listener"
    return replace(
        config,
        enabled=bool(config.enabled),
        url=url,
        token=token,
        timeout_s=max(0.05, float(config.timeout_s)),
        hold_ms=max(0, int(config.hold_ms)),
        mode=mode,
        source=source,
        send=send,
        clear_on_interrupt=bool(config.clear_on_interrupt),
    )


def _normalize_gateway_url(value: str) -> str:
    text = str(value or "").strip() or "ws://127.0.0.1:18789"
    if text.startswith("http://"):
        return "ws://" + text[len("http://") :]
    if text.startswith("https://"):
        return "wss://" + text[len("https://") :]
    if "://" not in text:
        return f"ws://{text}"
    return text


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a JSON object: {path}")
    return data


def _parse_bool(value: str) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def _parse_bool_value(value: Any, current: bool) -> bool:
    if value in (None, ""):
        return bool(current)
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return bool(current)
    return bool(value)
