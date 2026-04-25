"""Lightweight noise suppression module with low CPU load."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing support during development
    from core.config import NoiseSuppressionCfg


@dataclass(slots=True)
class _RuntimeParams:
    """Precomputed parameters for fast access during processing."""

    frame_samples: int
    min_gain: float
    suppression_factor: float
    energy_threshold_ratio: float
    noise_learning_rate: float
    noise_release_rate: float
    gain_smoothing: float


class NoiseSuppressor:
    """Lightweight noise suppression based on adaptive noise thresholding.

    The algorithm measures RMS in short windows and adaptively tracks the
    background level. This helps reduce stationary noise without heavy FFT
    calculations or extra dependencies.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        config: NoiseSuppressionCfg,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        frame_duration_ms = max(5.0, float(getattr(config, "frame_duration_ms", 20)))
        frame_samples = max(1, int(sample_rate * frame_duration_ms / 1000.0))

        self._params = _RuntimeParams(
            frame_samples=frame_samples,
            min_gain=float(np.clip(getattr(config, "min_gain", 0.1), 0.0, 1.0)),
            suppression_factor=max(0.0, float(getattr(config, "suppression_factor", 1.0))),
            energy_threshold_ratio=max(
                1.0, float(getattr(config, "energy_threshold_ratio", 1.6))
            ),
            noise_learning_rate=float(
                np.clip(getattr(config, "noise_learning_rate", 0.95), 0.0, 1.0)
            ),
            noise_release_rate=float(
                np.clip(getattr(config, "noise_release_rate", 0.01), 0.0, 1.0)
            ),
            gain_smoothing=float(
                np.clip(getattr(config, "gain_smoothing", 0.6), 0.0, 0.999)
            ),
        )

        self._sample_rate = int(sample_rate)
        self._channels = int(channels)
        self._buffer = np.empty((0, self._channels), dtype=np.float32)
        self._noise_rms: float | None = None
        self._last_gain: float = 1.0
        self._eps: float = 1e-8

    def reset(self) -> None:
        """Reset internal filter state."""

        self._buffer = np.empty((0, self._channels), dtype=np.float32)
        self._noise_rms = None
        self._last_gain = 1.0

    def process(self, pcm: np.ndarray) -> np.ndarray:
        """Apply noise suppression to PCM16 array.

        ``pcm`` is not modified in-place: a new ``np.int16`` array is returned.
        """

        if pcm.size == 0:
            return pcm

        samples = pcm.astype(np.float32).reshape(-1, self._channels) / 32768.0

        buffered_frames = self._buffer.shape[0]
        input_frames = pcm.size // self._channels

        if self._buffer.size:
            samples = np.vstack((self._buffer, samples))

        processed = samples.copy()
        frame_samples = self._params.frame_samples
        idx = 0
        length = samples.shape[0]

        while idx + frame_samples <= length:
            frame = samples[idx : idx + frame_samples]
            gain = self._compute_gain(frame)
            processed[idx : idx + frame_samples] *= gain
            idx += frame_samples

        # Keep remainder for the next call.
        if idx < length:
            self._buffer = samples[idx:].copy()
        else:
            self._buffer = np.empty((0, self._channels), dtype=np.float32)

        # Keep unprocessed tail unchanged to avoid extra latency.
        if idx < length:
            processed[idx:] = samples[idx:]

        scaled = np.rint(np.clip(processed, -1.0, 1.0) * 32768.0).astype(np.int16)
        scaled = scaled.reshape(-1, self._channels)

        if buffered_frames:
            scaled = scaled[buffered_frames:]

        scaled = scaled[:input_frames]

        return scaled.reshape(pcm.shape)

    # === internal ==========================================================

    def _compute_gain(self, frame: np.ndarray) -> float:
        """Compute gain factor for a frame."""

        if frame.size == 0:
            return 1.0

        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float32)))
        if not np.isfinite(rms):
            rms = 0.0

        if rms <= self._eps:
            self._noise_rms = self._noise_rms if self._noise_rms is not None else 0.0
            gain = 1.0
        else:
            if self._noise_rms is None:
                self._noise_rms = rms
            threshold = self._noise_rms * self._params.energy_threshold_ratio

            if rms <= threshold:
                # Frame looks noise-like: update noise floor faster.
                self._noise_rms = (
                    self._params.noise_learning_rate * self._noise_rms
                    + (1.0 - self._params.noise_learning_rate) * rms
                )
                noise_like = True
            else:
                # Speech is present: raise noise floor very slowly.
                self._noise_rms = (
                    (1.0 - self._params.noise_release_rate) * self._noise_rms
                    + self._params.noise_release_rate * rms
                )
                noise_like = False

            self._noise_rms = max(self._noise_rms, self._eps)

            if noise_like:
                excess = rms - self._params.suppression_factor * self._noise_rms
            else:
                excess = rms - self._params.suppression_factor * self._noise_rms

            if excess <= 0.0:
                gain = self._params.min_gain
            else:
                target_rms = excess
                gain = np.sqrt(target_rms / (rms + self._eps))
                gain = float(np.clip(gain, self._params.min_gain, 1.0))

        # Smoothing to avoid abrupt gain jumps.
        gain = (
            self._params.gain_smoothing * self._last_gain
            + (1.0 - self._params.gain_smoothing) * gain
        )
        gain = float(np.clip(gain, self._params.min_gain, 1.0))
        self._last_gain = gain
        return gain

