"""Streaming-friendly audio resampling utilities."""

from __future__ import annotations

from fractions import Fraction

import numpy as np


class StreamingResampler:
    """Polyphase down/up-sampler for streaming int16 PCM.

    The implementation is based on a windowed-sinc low-pass filter with a
    Kaiser window. It keeps internal convolution state so that successive
    ``process`` calls operate as a continuous stream without introducing
    artefacts at chunk boundaries.
    """

    def __init__(self, input_rate: int, output_rate: int) -> None:
        if input_rate <= 0 or output_rate <= 0:
            raise ValueError("sample rates must be positive")

        self.input_rate = int(input_rate)
        self.output_rate = int(output_rate)

        if self.input_rate == self.output_rate:
            self._disabled = True
            self._up = 1
            self._down = 1
            self._filter = np.ones(1, dtype=np.float32)
            self._state = np.zeros(0, dtype=np.float32)
            self._mod = 0
            self._pending_discard = 0
            return

        self._disabled = False
        fraction = Fraction(self.output_rate, self.input_rate).limit_denominator(1024)
        self._up = max(1, int(fraction.numerator))
        self._down = max(1, int(fraction.denominator))

        self._filter = self._design_filter(self._up, self._down)
        state_len = max(0, self._filter.size - 1)
        self._state = np.zeros(state_len, dtype=np.float32)
        self._mod = 0
        # Group delay of a linear-phase FIR = (N - 1) / 2.
        self._pending_discard = (self._filter.size - 1) // 2

    @staticmethod
    def _design_filter(up: int, down: int) -> np.ndarray:
        max_rate = max(up, down)
        taps = 16 * max_rate + 1
        if taps % 2 == 0:
            taps += 1
        cutoff = 0.5 / max_rate
        n = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
        sinc = 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
        window = np.kaiser(taps, 5.0)
        coeffs = sinc * window
        coeffs /= np.sum(coeffs)
        coeffs *= up
        return coeffs.astype(np.float32)

    def reset(self) -> None:
        if self._disabled:
            return
        self._state.fill(0.0)
        self._mod = 0
        self._pending_discard = (self._filter.size - 1) // 2

    def process(self, pcm: np.ndarray) -> np.ndarray:
        """Resample ``pcm`` from ``input_rate`` to ``output_rate``.

        Parameters
        ----------
        pcm:
            Input mono PCM signal (any integer dtype). For multi-channel data
            the caller should flatten the array beforehand.
        """

        if pcm.size == 0:
            return np.zeros(0, dtype=np.int16)

        data = np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)

        if self._disabled:
            return data.astype(np.int16, copy=True)

        up = self._up
        curr = np.zeros(data.size * up, dtype=np.float32)
        curr[::up] = data

        padded = np.concatenate([self._state, curr])
        filtered = np.convolve(padded, self._filter, mode="valid")
        self._state = padded[-(self._filter.size - 1) :]

        if self._pending_discard > 0:
            discard = int(min(self._pending_discard, filtered.size))
            filtered = filtered[discard:]
            self._pending_discard -= discard
            self._mod = (self._mod + discard) % self._down

        if filtered.size == 0:
            return np.zeros(0, dtype=np.int16)

        start = (-self._mod) % self._down
        decimated = filtered[start:: self._down]
        self._mod = (self._mod + curr.size) % self._down

        return np.clip(np.rint(decimated), -32768, 32767).astype(np.int16)


__all__ = ["StreamingResampler"]
