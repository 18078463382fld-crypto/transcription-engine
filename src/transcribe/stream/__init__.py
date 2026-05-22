"""Stream-processing handlers for real-time audio pipelines.

Each handler conforms to the :class:`transcribe.core.interfaces.StreamHandler`
abstract interface and implements a single pipeline stage such as voice
activity detection (VAD) or audio buffering.
"""

from .stream_processor import VADStreamHandler, BufferingStreamHandler

__all__ = [
    "VADStreamHandler",
    "BufferingStreamHandler",
]
