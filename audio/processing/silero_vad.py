"""Helpers for working with the Silero Voice Activity Detector."""
from __future__ import annotations

import ast
import json
import logging
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from core.config import AudioProcessingCfg, cfg as global_cfg

log = logging.getLogger(__name__)

_INT16_MAX_ABS = max(abs(np.iinfo(np.int16).min), np.iinfo(np.int16).max)
_INT16_SCALE = 1.0 / float(_INT16_MAX_ABS)


class SileroVADHelper:
    """Streaming helper that wraps the Silero VAD model.

    Parameters
    ----------
    cfg:
        Audio processing configuration describing VAD runtime behaviour.
    """

    def __init__(self, config: AudioProcessingCfg, *, debug: Optional[bool] = None) -> None:
        self._cfg = config
        self._vad_cfg = getattr(config, "vad", None)
        self._debug = global_cfg.debug if debug is None else bool(debug)

        model_path = getattr(self._vad_cfg, "model_path", None)
        self._model_path = Path(model_path) if model_path else None
        model_config_path = getattr(self._vad_cfg, "model_config_path", None)
        self._config_path = Path(model_config_path) if model_config_path else None

        self._model: Optional[torch.jit.ScriptModule] = None
        self._model_config: Optional[dict[str, Any]] = None

        self._model_id: Optional[str] = None
        self._default_device_name: str = "cpu"
        self._expected_pcm_dtype: str = "int16"
        self._expected_pcm_channels: int = 1
        self._pcm_scale: float = float(_INT16_SCALE)
        self._supported_sample_rates: Optional[tuple[int, ...]] = None
        self._frame_samples_by_rate: dict[int, int] = {}
        self._model_frame_duration_ms: Optional[int] = None

        self._load_model_config(require=False)

        device_name = self._select_device_name(config)
        self.device = torch.device(device_name)

        if self._debug:
            log.debug(
                "audio.silero_vad: helper initialised (device=%s, model_path=%s)",
                self.device,
                self._model_path,
            )

        self._sample_rate: Optional[int] = None
        self._frame_samples: int = 0
        self._frame_duration_sec: float = 0.0
        self._frame_buffer: Optional[torch.Tensor] = None
        self._buffer_fill: int = 0

        self._last_probability: Optional[float] = None
        self._last_timestamp: Optional[float] = None

    def predict(self, pcm: np.ndarray, sample_rate: int) -> float:
        """Return voice probability for the provided PCM chunk.

        The helper accumulates audio until the configured frame size is
        available and invokes the Silero model only once per frame.
        """

        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be a positive integer (got {sample_rate}).")

        if not isinstance(pcm, np.ndarray):
            raise TypeError("pcm must be a numpy.ndarray containing mono int16 audio samples.")

        expected_dtype = self._expected_pcm_dtype.lower()
        pcm_dtype_name = str(pcm.dtype).lower()
        if expected_dtype in ("int16", "<class 'numpy.int16'>"):
            if pcm.dtype != np.int16:
                raise TypeError(
                    f"pcm must have dtype int16 according to Silero config (got {pcm.dtype})."
                )
        elif pcm_dtype_name != expected_dtype:
            raise TypeError(
                f"pcm must have dtype {self._expected_pcm_dtype!s} according to Silero config (got {pcm.dtype})."
            )

        if pcm.ndim == 2:
            expected_channels = self._expected_pcm_channels
            if expected_channels != 1 and pcm.shape[1] != expected_channels:
                raise ValueError(
                    "PCM channel count does not match Silero configuration; "
                    f"expected {expected_channels}, got {pcm.shape[1]}."
                )
            if pcm.shape[1] != 1:
                raise ValueError("Only mono PCM audio is supported.")
            pcm = pcm.reshape(-1)
        elif pcm.ndim != 1:
            raise ValueError("pcm must be a one-dimensional mono signal.")

        if self._expected_pcm_channels != 1 and pcm.ndim == 1:
            raise ValueError(
                "Silero configuration expects multi-channel audio, which is not supported by the helper."
            )

        if not pcm.flags.c_contiguous:
            pcm = np.ascontiguousarray(pcm)

        now = time.monotonic()

        if self._supported_sample_rates and sample_rate not in self._supported_sample_rates:
            supported = ", ".join(str(rate) for rate in self._supported_sample_rates)
            raise ValueError(
                "Sample rate must match Silero configuration; "
                f"supported rates: {supported} (got {sample_rate})."
            )

        frame_samples = self._frame_samples_by_rate.get(int(sample_rate))
        if frame_samples is None:
            frame_duration_ms = self._resolve_frame_duration_ms()
            frame_samples = (sample_rate * frame_duration_ms) // 1000
        if frame_samples <= 0:
            raise ValueError(
                "VAD frame duration configuration results in zero frame samples; "
                "increase vad_frame_duration_ms or the sample rate."
            )

        self._ensure_buffers(sample_rate, frame_samples)

        if pcm.size == 0:
            probability = self._cached_probability(now)
            if self._debug:
                log.debug(
                    "audio.silero_vad: empty chunk, returning cached probability %.3f",
                    probability,
                )
            return probability

        self._ensure_model_loaded()

        # Create an independent copy directly on the target device and dtype.
        chunk_tensor = torch.tensor(pcm.reshape(-1), dtype=torch.float32, device=self.device)
        chunk_tensor.mul_(self._pcm_scale)

        produced_probability = False
        offset = 0
        total_samples = int(chunk_tensor.shape[0])
        while offset < total_samples:
            space_left = self._frame_samples - self._buffer_fill
            take = min(space_left, total_samples - offset)
            assert self._frame_buffer is not None  # for mypy like tools
            self._frame_buffer[self._buffer_fill : self._buffer_fill + take] = chunk_tensor[offset : offset + take]
            self._buffer_fill += take
            offset += take

            if self._buffer_fill == self._frame_samples:
                probability = self._run_inference(sample_rate)
                self._last_probability = probability
                self._last_timestamp = time.monotonic()
                self._buffer_fill = 0
                produced_probability = True

        if produced_probability:
            assert self._last_probability is not None
            if self._debug:
                log.debug(
                    "audio.silero_vad: inference probability %.3f (sample_rate=%d)",
                    self._last_probability,
                    sample_rate,
                )
            return self._last_probability

        probability = self._cached_probability(now)
        if self._debug:
            log.debug(
                "audio.silero_vad: returning cached probability %.3f (buffer_fill=%d/%d)",
                probability,
                self._buffer_fill,
                self._frame_samples,
            )
        return probability

    def _select_device_name(self, cfg: AudioProcessingCfg) -> str:
        vad_cfg = getattr(cfg, "vad", None)
        for attr in ("silero_device", "vad_device", "device"):
            value = getattr(vad_cfg, attr, None)
            if value is None:
                value = getattr(cfg, attr, None)
            if value:
                if isinstance(value, str):
                    value = value.strip()
                if value:
                    return str(value)

        return self._default_device_name or "cpu"

    def _resolve_frame_duration_ms(self) -> int:
        frame_duration = getattr(self._vad_cfg, "frame_duration_ms", None)
        try:
            frame_ms = int(frame_duration) if frame_duration is not None else 0
        except (TypeError, ValueError):
            frame_ms = 0

        if frame_ms <= 0 and self._model_frame_duration_ms:
            frame_ms = int(self._model_frame_duration_ms)

        if frame_ms <= 0:
            frame_ms = 30

        return frame_ms

    def _cached_probability(self, now: float) -> float:
        if self._last_probability is None:
            return 0.0
        if self._last_timestamp is None:
            return self._last_probability
        if now - self._last_timestamp <= self._frame_duration_sec:
            return self._last_probability
        return self._last_probability

    def _ensure_buffers(self, sample_rate: int, frame_samples: int) -> None:
        needs_reset = (
            self._frame_buffer is None
            or self._sample_rate != sample_rate
            or self._frame_samples != frame_samples
        )

        if needs_reset:
            self._sample_rate = sample_rate
            self._frame_samples = frame_samples
            self._frame_duration_sec = frame_samples / float(sample_rate)
            self._frame_buffer = torch.zeros(frame_samples, dtype=torch.float32, device=self.device)
            self._buffer_fill = 0
            self._last_probability = None
            self._last_timestamp = None
            if self._model is not None and hasattr(self._model, "reset_states"):
                self._model.reset_states()
            if self._debug:
                log.debug(
                    "audio.silero_vad: frame buffer reset (sample_rate=%d, frame_samples=%d)",
                    sample_rate,
                    frame_samples,
                )

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        if self._model_path is None:
            raise FileNotFoundError(
                "AudioVadCfg.model_path must point to Silero VAD weights file."
            )
        if not self._model_path.is_file():
            raise FileNotFoundError(f"Silero VAD model weights not found: {self._model_path}")

        if self._debug:
            log.debug(
                "audio.silero_vad: loading model weights from %s", self._model_path
            )

        self._load_model_config(require=True)

        try:
            self._model = torch.jit.load(str(self._model_path), map_location=self.device)
            self._model = self._model.eval()
            if hasattr(self._model, "to"):
                self._model = self._model.to(self.device)
            if hasattr(self._model, "reset_states"):
                self._model.reset_states()
        except Exception:
            if self._debug:
                log.exception(
                    "audio.silero_vad: failed to load Silero VAD model from %s",
                    self._model_path,
                )
            raise
        else:
            if self._debug:
                log.debug(
                    "audio.silero_vad: model %s loaded on %s",
                    self._model_id or self._model_path.name,
                    self.device,
                )

    def _load_model_config(self, *, require: bool) -> Optional[dict[str, Any]]:
        if self._model_config is not None:
            return self._model_config

        if self._config_path is None:
            if self._debug and require:
                log.debug("audio.silero_vad: model config path not provided")
            if require:
                raise FileNotFoundError(
                    "AudioVadCfg.model_config_path must point to the Silero VAD configuration file."
                )
            return None

        if not self._config_path.is_file():
            if self._debug:
                log.debug(
                    "audio.silero_vad: model config file %s is missing",
                    self._config_path,
                )
            if require:
                raise FileNotFoundError(f"Silero VAD model config not found: {self._config_path}")
            return None

        if self._debug:
            log.debug(
                "audio.silero_vad: loading model config from %s",
                self._config_path,
            )

        try:
            config = self._read_config(self._config_path)
        except Exception:
            if self._debug:
                log.exception(
                    "audio.silero_vad: failed to parse config %s",
                    self._config_path,
                )
            raise
        self._apply_model_config(config)
        return self._model_config

    def _apply_model_config(self, config: Mapping[str, Any]) -> None:
        data = dict(config)
        self._model_config = data

        model_id = data.get("model_id") or data.get("name")
        if isinstance(model_id, str) and model_id.strip():
            self._model_id = model_id.strip()
            if self._debug:
                log.debug("audio.silero_vad: model id set to %s", self._model_id)

        default_device = data.get("default_device")
        if isinstance(default_device, str) and default_device.strip():
            self._default_device_name = default_device.strip()

        pcm_cfg = data.get("pcm") or data.get("pcm_format") or {}
        if isinstance(pcm_cfg, Mapping):
            dtype = pcm_cfg.get("dtype") or pcm_cfg.get("type")
            if isinstance(dtype, str) and dtype.strip():
                self._expected_pcm_dtype = dtype.strip()

            channels = pcm_cfg.get("channels") or pcm_cfg.get("num_channels")
            try:
                if channels is not None:
                    parsed_channels = int(channels)
                    if parsed_channels > 0:
                        self._expected_pcm_channels = parsed_channels
            except (TypeError, ValueError):  # pragma: no cover - invalid config
                pass

            normalization = pcm_cfg.get("normalization_factor")
            if normalization is None:
                scale = pcm_cfg.get("scale") or pcm_cfg.get("max_abs_value")
                if scale is not None:
                    try:
                        value = float(scale)
                    except (TypeError, ValueError):  # pragma: no cover - invalid config
                        value = None
                    else:
                        if value > 0:
                            normalization = 1.0 / value
            if normalization is not None:
                try:
                    factor = float(normalization)
                except (TypeError, ValueError):  # pragma: no cover - invalid config
                    pass
                else:
                    if factor > 0:
                        self._pcm_scale = factor

            sample_rates = pcm_cfg.get("sample_rates") or data.get("sample_rates")
            if isinstance(sample_rates, (list, tuple, set)):
                rates: list[int] = []
                for value in sample_rates:
                    try:
                        rate = int(value)
                    except (TypeError, ValueError):  # pragma: no cover - invalid config
                        continue
                    if rate > 0:
                        rates.append(rate)
                if rates:
                    self._supported_sample_rates = tuple(sorted(set(rates)))

        frame_cfg = data.get("frame") or {}
        if isinstance(frame_cfg, Mapping):
            duration = frame_cfg.get("duration_ms") or frame_cfg.get("frame_length_ms")
            try:
                if duration is not None:
                    parsed_duration = int(duration)
                    if parsed_duration > 0:
                        self._model_frame_duration_ms = parsed_duration
            except (TypeError, ValueError):  # pragma: no cover - invalid config
                pass

            samples_map = frame_cfg.get("samples_per_rate") or frame_cfg.get("samples_by_rate")
            if isinstance(samples_map, Mapping):
                parsed: dict[int, int] = {}
                for key, value in samples_map.items():
                    try:
                        rate = int(key)
                        samples = int(value)
                    except (TypeError, ValueError):  # pragma: no cover - invalid config
                        continue
                    if rate > 0 and samples > 0:
                        parsed[rate] = samples
                if parsed:
                    self._frame_samples_by_rate = parsed
                    existing = set(self._supported_sample_rates or ())
                    existing.update(parsed.keys())
                    if existing:
                        self._supported_sample_rates = tuple(sorted(existing))

    def _read_config(self, path: Path) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {}

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml  # type: ignore
            except ModuleNotFoundError:
                try:
                    data = ast.literal_eval(text)
                except (SyntaxError, ValueError) as exc:  # pragma: no cover - defensive branch
                    raise ValueError(
                        "Failed to parse Silero VAD config; provide JSON or install PyYAML."
                    ) from exc
            else:
                data = yaml.safe_load(text)

        if not isinstance(data, dict):
            raise ValueError("Silero VAD config must deserialize to a mapping/dict.")

        return data

    def _run_inference(self, sample_rate: int) -> float:
        assert self._model is not None
        assert self._frame_buffer is not None

        try:
            with torch.inference_mode():
                output = self._model(self._frame_buffer.unsqueeze(0), int(sample_rate))
        except RuntimeError as exc:  # pragma: no cover - runtime guard
            detail = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
            raise RuntimeError(
                "Silero VAD inference failed"
                f" ({detail}); ensure the frame duration matches the model's expected "
                "window size and the selected torch/CUDA runtime is available."
            ) from exc

        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Unexpected output type from Silero VAD model.")

        if output.numel() == 0:
            raise RuntimeError("Silero VAD model returned an empty tensor.")

        return float(output.squeeze().item())
