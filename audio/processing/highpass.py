"""High-pass filtering and DC-offset removal utilities."""

from __future__ import annotations

import math

import numpy as np


class DCBlockingHighPass:
    """Simple first-order high-pass filter with DC removal."""

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        cutoff_hz: float = 100.0,
    ) -> None:
        self.sample_rate = max(1, int(sample_rate))
        self.channels = max(1, int(channels))
        self.cutoff_hz = max(0.0, float(cutoff_hz))
        self._pole = self._compute_pole(self.cutoff_hz)
        self._prev_input = np.zeros(self.channels, dtype=np.float32)
        self._prev_output = np.zeros(self.channels, dtype=np.float32)

    def _compute_pole(self, cutoff_hz: float) -> float:
        if cutoff_hz <= 0.0 or self.sample_rate <= 0:
            return 0.0
        pole = 1.0 - (2.0 * math.pi * cutoff_hz) / float(self.sample_rate)
        return float(min(max(pole, 0.0), 0.9999))

    def reset(self) -> None:
        self._prev_input.fill(0.0)
        self._prev_output.fill(0.0)

    def process(self, pcm: np.ndarray, *, enabled: bool = True) -> np.ndarray:
        if not enabled or pcm.size == 0:
            return pcm

        channels = max(1, self.channels)
        if channels <= 0:
            return pcm

        frames = pcm.size // channels
        if frames == 0:
            return pcm

        data = pcm.astype(np.float32).reshape(frames, channels)
        prev_in = self._prev_input
        prev_out = self._prev_output
        pole = self._pole

        for idx in range(frames):
            x = data[idx].copy()
            y = x - prev_in + pole * prev_out
            data[idx] = y
            prev_in[:] = x
            prev_out[:] = y

        np.clip(data, -32768.0, 32767.0, out=data)
        processed = np.rint(data).astype(np.int16)
        pcm[:] = processed.reshape(-1)
        return pcm


__all__ = ["DCBlockingHighPass"]
