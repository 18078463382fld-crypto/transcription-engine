"""
Utility package for common transcription-engine helpers.

All modules in this package use **only the Python standard library**
— no third-party dependencies.

Modules
-------
audio
    ``AudioConverter``, ``resample``, ``to_pcm`` helpers for raw PCM
    audio manipulation (resampling, format conversion, normalisation).

logging
    ``setup_logging``, ``ColoredFormatter`` for consistent, colourful
    log output in console applications.
"""

from .audio import AudioConverter, resample, to_pcm
from .logging import ColoredFormatter, setup_logging

__all__ = [
    "AudioConverter",
    "resample",
    "to_pcm",
    "ColoredFormatter",
    "setup_logging",
]
