"""Audio-source implementations for the transcription engine.

Each source wraps a different origin of audio data (files, raw bytes,
microphone) and conforms to the :class:`transcribe.core.interfaces.AudioSource`
abstract interface.
"""

from .file_source import FileSource
from .bytes_source import BytesSource
from .microphone_source import MicrophoneSource

__all__ = [
    "FileSource",
    "BytesSource",
    "MicrophoneSource",
]
