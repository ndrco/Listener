"""Automatic gain control with limiter and headroom."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class AGCSettings:
    """Automatic gain control settings."""

    enabled: bool
    target_level_dbfs: float
    max_gain_db: float
    attack_ms: float
    release_ms: float


class AutomaticGainControl:
    """Automatic gain control with limiter protection."""

    def __init__(
        self,
        *,
        headroom_db: float = 0.8,
        limiter_attack_ms: float = 0.75,
        limiter_release_ms: float = 60.0,
    ) -> None:
        self._gain: float = 1.0
        self._limiter_gain: float = 1.0
        self._headroom_linear = 10.0 ** (-float(headroom_db) / 20.0)
        self._limiter_attack_ms = max(0.1, float(limiter_attack_ms))
        self._limiter_release_ms = max(self._limiter_attack_ms, float(limiter_release_ms))

    @staticmethod
    def _compute_rms(samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

    @staticmethod
    def _rms_to_db(rms: float) -> float:
        if rms <= 0.0:
            return -120.0
        return 20.0 * math.log10(rms / 32768.0)

    def reset(self) -> None:
        self._gain = 1.0
        self._limiter_gain = 1.0

    def process(
        self,
        pcm: np.ndarray,
        *,
        sample_rate: int,
        channels: int,
        settings: AGCSettings,
    ) -> np.ndarray:
        if not settings.enabled:
            self.reset()
            return pcm

        if pcm.size == 0 or sample_rate <= 0 or channels <= 0:
            self.reset()
            return pcm

        samples = pcm.astype(np.float32)
        rms = self._compute_rms(samples)
        current_db = self._rms_to_db(rms)
        gain_db = float(settings.target_level_dbfs) - current_db
        gain_db = min(gain_db, max(0.0, float(settings.max_gain_db)))
        target_gain = 10.0 ** (gain_db / 20.0)

        frame_samples = max(1, pcm.size // max(1, int(channels)))
        frame_duration_ms = (frame_samples / float(sample_rate)) * 1000.0

        attack_ms = max(1.0, float(settings.attack_ms))
        release_ms = max(attack_ms, float(settings.release_ms))
        current_gain = float(self._gain)
        time_const = attack_ms if target_gain > current_gain else release_ms
        if time_const <= 0:
            new_gain = target_gain
        else:
            smoothing = math.exp(-frame_duration_ms / time_const)
            new_gain = current_gain * smoothing + target_gain * (1.0 - smoothing)
        if not math.isfinite(new_gain):
            new_gain = 1.0
        self._gain = float(new_gain)

        if abs(new_gain - 1.0) < 1e-6:
            scaled = samples
        else:
            scaled = samples * new_gain

        limit = 32767.0 * self._headroom_linear
        max_abs = float(np.max(np.abs(scaled))) if scaled.size else 0.0
        limiter_gain = float(self._limiter_gain)
        if max_abs > limit and max_abs > 0.0:
            desired_gain = limit / max_abs
            smoothing = math.exp(-frame_duration_ms / self._limiter_attack_ms)
            limiter_gain = min(limiter_gain * smoothing + desired_gain * (1.0 - smoothing), desired_gain)
        else:
            smoothing = math.exp(-frame_duration_ms / self._limiter_release_ms)
            limiter_gain = limiter_gain * smoothing + (1.0 - smoothing)
            limiter_gain = min(limiter_gain, 1.0)
        if not math.isfinite(limiter_gain) or limiter_gain <= 0.0:
            limiter_gain = 1.0
        self._limiter_gain = limiter_gain

        if limiter_gain < 1.0 - 1e-6:
            scaled = scaled * limiter_gain

        np.clip(scaled, -limit, limit, out=scaled)
        pcm[:] = np.rint(scaled).astype(np.int16)
        return pcm


__all__ = ["AGCSettings", "AutomaticGainControl"]
