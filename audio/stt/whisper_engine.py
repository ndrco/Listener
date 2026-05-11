"""Whisper speech-to-text engine wrapper."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np

from audio.processing.resampler import StreamingResampler
from core.config import WhisperSttCfg

try:  # pragma: no cover - optional dependency
    from faster_whisper import WhisperModel  # type: ignore
except Exception:  # pragma: no cover - faster-whisper is optional at runtime
    WhisperModel = None  # type: ignore


log = logging.getLogger(__name__)
_BLACKLIST_SPACE_RE = re.compile(r"\s+")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class WhisperEngine:
    """High-level wrapper around Whisper STT models."""

    def __init__(self, config: WhisperSttCfg, debug: bool = False) -> None:
        self._config = config
        self._debug = debug
        self._enabled = bool(config.enabled)
        self._target_sample_rate = max(1, int(getattr(config, "sample_rate", 16_000)))
        self._model: Any | None = None
        self._resamplers: Dict[int, StreamingResampler] = {}
        self._transcribe_options = self._build_transcribe_options(config)
        self._blacklist: Set[str] = set()
        self._requested_device = str(config.device or "auto")
        self._active_device = self._requested_device
        self._active_compute_type = str(config.compute_type) if config.compute_type else None
        self._resolved_model_name = self._resolve_model_name(config.model or "small")

        if self._enabled:
            self._initialise_model()
            self._load_blacklist()

    # ------------------------------------------------------------------
    # Model initialisation helpers
    # ------------------------------------------------------------------
    def _initialise_model(self) -> None:
        if WhisperModel is None:
            raise RuntimeError(
                "audio.stt.whisper: faster-whisper is not available; install the 'faster-whisper' package"
            )

        cfg = self._config
        model_name = self._resolved_model_name
        init_kwargs = self._build_init_kwargs(cfg)

        try:
            self._model = WhisperModel(model_name, **init_kwargs)
        except Exception as exc:  # pragma: no cover - runtime error surfaces to caller
            if self._should_retry_on_cpu(exc, init_kwargs):
                fallback_kwargs = self._build_init_kwargs(cfg, device_override="cpu")
                log.warning(
                    "audio.stt.whisper: CUDA model load ran out of memory; "
                    "retrying on CPU (model=%s, compute_type=%s)",
                    model_name,
                    init_kwargs.get("compute_type", "default"),
                )
                try:
                    self._model = WhisperModel(model_name, **fallback_kwargs)
                except Exception as fallback_exc:  # pragma: no cover - runtime error surfaces
                    raise RuntimeError(
                        "audio.stt.whisper: failed to load Whisper model "
                        f"'{model_name}' after CUDA OOM fallback to CPU with args {fallback_kwargs}"
                    ) from fallback_exc
                self._active_device = str(fallback_kwargs.get("device", "cpu"))
                self._active_compute_type = (
                    str(fallback_kwargs["compute_type"])
                    if "compute_type" in fallback_kwargs
                    else None
                )
            else:
                raise RuntimeError(
                    f"audio.stt.whisper: failed to load Whisper model '{model_name}' with args {init_kwargs}"
                ) from exc
        else:
            self._active_device = str(init_kwargs.get("device", self._requested_device))
            self._active_compute_type = (
                str(init_kwargs["compute_type"]) if "compute_type" in init_kwargs else None
            )

        log.debug(
            "audio.stt.whisper: model %s initialised (device=%s, compute_type=%s)",
            model_name,
            self._active_device,
            self._active_compute_type or "default",
        )

    @staticmethod
    def _should_retry_on_cpu(exc: Exception, init_kwargs: Dict[str, Any]) -> bool:
        device = str(init_kwargs.get("device", "") or "").strip().lower()
        if not device.startswith("cuda"):
            return False
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "out of memory",
                "cuda failed with error out of memory",
                "cuda out of memory",
                "failed to allocate memory",
            )
        )

    @staticmethod
    def _build_init_kwargs(
        cfg: WhisperSttCfg,
        *,
        device_override: str | None = None,
    ) -> Dict[str, Any]:
        init_kwargs: Dict[str, Any] = {
            "device": device_override or cfg.device or "auto",
        }

        if cfg.compute_type:
            init_kwargs["compute_type"] = str(cfg.compute_type)
        if cfg.download_root:
            init_kwargs["download_root"] = str(cfg.download_root)
        init_kwargs["local_files_only"] = bool(getattr(cfg, "local_files_only", False))

        cpu_threads = getattr(cfg, "cpu_threads", None)
        if cpu_threads:
            try:
                threads_value = int(cpu_threads)
            except (TypeError, ValueError):
                threads_value = None
            else:
                if threads_value > 0:
                    init_kwargs["cpu_threads"] = threads_value

        num_workers = getattr(cfg, "num_workers", None)
        if num_workers:
            try:
                workers_value = int(num_workers)
            except (TypeError, ValueError):
                workers_value = None
            else:
                if workers_value > 0:
                    init_kwargs["num_workers"] = workers_value
        return init_kwargs

    @property
    def active_device(self) -> str:
        return self._active_device

    @property
    def active_compute_type(self) -> str | None:
        return self._active_compute_type

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        text = str(model_name or "").strip() or "small"
        path = Path(text).expanduser()
        candidates = [path] if path.is_absolute() else [Path.cwd() / path, _PROJECT_ROOT / path]
        for candidate in candidates:
            try:
                if candidate.exists():
                    return str(candidate.resolve())
            except OSError:
                continue
        return text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def transcribe(
        self, batch_pcm: np.ndarray | bytes | Iterable[int], *, sample_rate: int
    ) -> List[str]:
        """Transcribe ``batch_pcm`` and return a list of recognised phrases."""

        if not self._enabled:
            return []

        model = self._model
        if model is None:
            log.debug("audio.stt.whisper: engine is disabled or model not initialised")
            return []

        pcm = self._to_int16(batch_pcm)
        if pcm.size == 0:
            return []

        src_rate = int(sample_rate) if sample_rate and sample_rate > 0 else self._target_sample_rate
        if src_rate != self._target_sample_rate:
            resampler = self._get_resampler(src_rate)
            pcm = resampler.process(pcm)

        if pcm.size == 0:
            return []

        audio = pcm.astype(np.float32) / 32768.0
        if audio.size == 0:
            return []

        try:
            return self._transcribe_audio(model, audio)
        except Exception as exc:  # pragma: no cover - runtime path
            if self._should_retry_inference_on_cpu(exc):
                log.warning(
                    "audio.stt.whisper: CUDA transcription ran out of memory; "
                    "reloading model on CPU and retrying (model=%s, compute_type=%s)",
                    self._resolved_model_name,
                    self._active_compute_type or "default",
                )
                try:
                    cpu_model = self._reload_model_on_cpu()
                    return self._transcribe_audio(cpu_model, audio)
                except Exception:  # pragma: no cover - delegate errors to log and continue
                    log.exception(
                        "audio.stt.whisper: transcription failed after CUDA OOM fallback to CPU"
                    )
                    return []
            log.exception("audio.stt.whisper: transcription failed")
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_resampler(self, input_rate: int) -> StreamingResampler:
        resampler = self._resamplers.get(input_rate)
        if resampler is None:
            resampler = StreamingResampler(input_rate, self._target_sample_rate)
            self._resamplers[input_rate] = resampler
        return resampler

    def _transcribe_audio(self, model: Any, audio: np.ndarray) -> List[str]:
        segments, _ = model.transcribe(audio, **self._transcribe_options)
        results: List[str] = []
        for segment in segments:
            text = getattr(segment, "text", "")
            if not text:
                continue
            cleaned = text.strip()
            if not cleaned:
                continue
            if self._is_blacklisted(cleaned):
                log.debug("audio.stt.whisper: ignoring blacklisted phrase %r", cleaned)
                continue
            results.append(cleaned)
        return results

    def _should_retry_inference_on_cpu(self, exc: Exception) -> bool:
        return self._should_retry_on_cpu(exc, {"device": self._active_device})

    def _reload_model_on_cpu(self) -> Any:
        cfg = self._config
        fallback_kwargs = self._build_init_kwargs(cfg, device_override="cpu")
        self._model = WhisperModel(self._resolved_model_name, **fallback_kwargs)
        self._active_device = str(fallback_kwargs.get("device", "cpu"))
        self._active_compute_type = (
            str(fallback_kwargs["compute_type"]) if "compute_type" in fallback_kwargs else None
        )
        log.info(
            "audio.stt.whisper: model reloaded on CPU after CUDA OOM (model=%s, compute_type=%s)",
            self._resolved_model_name,
            self._active_compute_type or "default",
        )
        return self._model

    @staticmethod
    def _to_int16(batch_pcm: np.ndarray | bytes | Iterable[int]) -> np.ndarray:
        if isinstance(batch_pcm, np.ndarray):
            arr = np.asarray(batch_pcm).reshape(-1)
            if arr.dtype != np.int16:
                arr = arr.astype(np.int16)
            else:
                arr = np.ascontiguousarray(arr)
            return arr

        if isinstance(batch_pcm, (bytes, bytearray, memoryview)):
            buffer = batch_pcm
            if isinstance(buffer, memoryview):
                buffer = buffer.tobytes()
            return np.frombuffer(buffer, dtype=np.int16).copy()

        try:
            iterator = iter(batch_pcm)
        except TypeError as exc:
            raise TypeError("audio.stt.whisper: unsupported audio buffer type") from exc

        return np.fromiter(iterator, dtype=np.int16)

    def _build_transcribe_options(self, cfg: WhisperSttCfg) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "task": (cfg.task or "transcribe").strip() or "transcribe",
            "condition_on_previous_text": bool(cfg.condition_on_previous_text),
            "without_timestamps": bool(cfg.without_timestamps),
            "word_timestamps": bool(cfg.word_timestamps),
            "vad_filter": bool(cfg.vad_filter),
            "suppress_blank": bool(getattr(cfg, "suppress_blank", True)),
        }

        if cfg.language:
            options["language"] = cfg.language
        if cfg.beam_size and cfg.beam_size > 0:
            options["beam_size"] = int(cfg.beam_size)
        if cfg.best_of and cfg.best_of > 0:
            options["best_of"] = int(cfg.best_of)
        if cfg.patience is not None:
            try:
                options["patience"] = float(cfg.patience)
            except (TypeError, ValueError):
                pass

        temperature = getattr(cfg, "temperature", None)
        if temperature is not None:
            parsed = self._parse_temperature(temperature)
            if parsed is not None:
                options["temperature"] = parsed

        increment = getattr(cfg, "temperature_increment_on_fallback", None)
        if increment is not None:
            try:
                options["temperature_increment_on_fallback"] = float(increment)
            except (TypeError, ValueError):
                pass

        if cfg.initial_prompt:
            options["initial_prompt"] = str(cfg.initial_prompt)

        if cfg.compression_ratio_threshold is not None:
            try:
                options["compression_ratio_threshold"] = float(cfg.compression_ratio_threshold)
            except (TypeError, ValueError):
                pass

        if cfg.logprob_threshold is not None:
            try:
                options["log_prob_threshold"] = float(cfg.logprob_threshold)
            except (TypeError, ValueError):
                pass

        if cfg.no_speech_threshold is not None:
            try:
                options["no_speech_threshold"] = float(cfg.no_speech_threshold)
            except (TypeError, ValueError):
                pass

        if cfg.length_penalty is not None:
            try:
                options["length_penalty"] = float(cfg.length_penalty)
            except (TypeError, ValueError):
                pass

        if cfg.max_initial_timestamp is not None:
            try:
                options["max_initial_timestamp"] = float(cfg.max_initial_timestamp)
            except (TypeError, ValueError):
                pass

        options["prompt_reset_on_temperature"] = bool(
            getattr(cfg, "prompt_reset_on_temperature", False)
        )

        suppress_tokens = self._parse_suppress_tokens(getattr(cfg, "suppress_tokens", None))
        if suppress_tokens is not None:
            options["suppress_tokens"] = suppress_tokens

        vad_parameters = getattr(cfg, "vad_parameters", None)
        if isinstance(vad_parameters, dict):
            options["vad_parameters"] = dict(vad_parameters)

        return options

    @staticmethod
    def _parse_temperature(value: Any) -> float | tuple[float, ...] | None:
        if isinstance(value, (list, tuple)):
            cleaned: list[float] = []
            for item in value:
                try:
                    cleaned.append(float(item))
                except (TypeError, ValueError):
                    continue
            return tuple(cleaned)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_suppress_tokens(value: Any) -> Any:
        if value in (None, ""):
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text == "-1":
                return "-1"
            tokens: list[int] = []
            for part in text.replace(",", " ").split():
                try:
                    tokens.append(int(part))
                except ValueError:
                    continue
            return tokens or None
        if isinstance(value, Iterable):
            tokens = []
            for item in value:
                try:
                    tokens.append(int(item))
                except (TypeError, ValueError):
                    continue
            return tokens or None
        return None

    # ------------------------------------------------------------------
    # Blacklist helpers
    # ------------------------------------------------------------------
    def _load_blacklist(self) -> None:
        self._blacklist.clear()

        path = self._get_blacklist_path()
        if path is None:
            log.info("audio.stt.whisper: blacklist disabled (download_root not configured)")
            return

        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            log.info("audio.stt.whisper: blacklist file not found at %s", path)
            return
        except OSError:
            log.exception("audio.stt.whisper: failed to read blacklist at %s", path)
            return

        phrases = []
        for raw_line in content.splitlines():
            normalized = self._normalize_blacklist_phrase(raw_line)
            if normalized:
                phrases.append(normalized)
        self._blacklist.update(phrases)
        log.info(
            "audio.stt.whisper: blacklist loaded from %s (%d entries)", path, len(self._blacklist)
        )

    def _get_blacklist_path(self) -> Path | None:
        cfg = self._config
        download_root = getattr(cfg, "download_root", None)
        if not download_root:
            return None
        try:
            base = Path(download_root).expanduser()
        except (TypeError, ValueError):
            return None
        return base.joinpath("blacklist.txt")

    def _is_blacklisted(self, phrase: str) -> bool:
        if not self._blacklist:
            return False
        candidate = self._normalize_blacklist_phrase(phrase)
        if not candidate:
            return False
        return candidate in self._blacklist

    @staticmethod
    def _normalize_blacklist_phrase(phrase: str) -> str:
        """Normalize text for blacklist matching.

        Rules:
        - trim and collapse whitespace,
        - compare case-insensitively,
        - ignore Unicode punctuation marks.
        """

        if not phrase:
            return ""

        collapsed = _BLACKLIST_SPACE_RE.sub(" ", str(phrase)).strip().casefold()
        if not collapsed:
            return ""

        no_punctuation = "".join(
            ch for ch in collapsed if not unicodedata.category(ch).startswith("P")
        )
        return _BLACKLIST_SPACE_RE.sub(" ", no_punctuation).strip()


__all__ = ["WhisperEngine"]

