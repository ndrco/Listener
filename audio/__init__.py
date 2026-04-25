"""Audio package exports for the voice-input runtime."""

from .microphone import MicrophoneStream
from .processing import AudioProcessor, ProcessedAudioFrame, WindowsAudioProcessor
from .stt import WhisperEngine
from .writer import BufferedSpeechWriter, SpeechSegment

__all__ = [
    "MicrophoneStream",
    "ProcessedAudioFrame",
    "AudioProcessor",
    "WindowsAudioProcessor",
    "BufferedSpeechWriter",
    "SpeechSegment",
    "WhisperEngine",
]
