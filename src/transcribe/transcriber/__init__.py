"""
Transcriber implementations for the transcription engine.

This package provides concrete ``Transcriber`` subclasses:

- :class:`LocalWhisperTranscriber` — runs Whisper models locally
  via ``faster-whisper``.
- :class:`CloudAPITranscriber` — delegates transcription to a
  cloud API (OpenAI Whisper, etc.).
- :class:`BaseTranscriber` — shared base class used by both.
"""

from .base import BaseTranscriber
from .local_whisper import LocalWhisperTranscriber
from .cloud_api import CloudAPITranscriber

__all__ = [
    "BaseTranscriber",
    "LocalWhisperTranscriber",
    "CloudAPITranscriber",
]
