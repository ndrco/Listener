"""Audio processing components (VAD and related metrics)."""

from .agc import AGCSettings, AutomaticGainControl
from .highpass import DCBlockingHighPass
from .noise_suppression import NoiseSuppressor
from .resampler import StreamingResampler
from .silero_vad import SileroVADHelper
from .windows_processing import AudioProcessor, ProcessedAudioFrame, WindowsAudioProcessor

__all__ = [
    "ProcessedAudioFrame",
    "AudioProcessor",
    "WindowsAudioProcessor",
    "SileroVADHelper",
    "NoiseSuppressor",
    "AutomaticGainControl",
    "AGCSettings",
    "DCBlockingHighPass",
    "StreamingResampler",
]
