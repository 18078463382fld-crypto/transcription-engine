"""
transcribe-engine
=================
A high-extensibility transcription engine supporting:

- **Real-time** streaming transcription (microphone -> text)
- **File** transcription (audio / video)
- **Local** Whisper or **cloud** API (OpenAI, etc.)
- **Plugin** system — register custom transcribers, sources, processors
- **Async API** for concurrent workloads

Quick start::

    from transcribe import TranscriptionEngine, FileSource
    from transcribe.transcriber import LocalWhisperTranscriber

    engine = TranscriptionEngine(transcriber=LocalWhisperTranscriber("base"))
    result = await engine.transcribe(FileSource("speech.mp3"))
    print(result.text)
"""

from .core.interfaces import Transcriber, AudioSource, StreamHandler, TranscriptionPlugin
from .core.models import TranscriptResult, TranscriptSegment, EngineConfig
from .core.engine import TranscriptionEngine
from .api.async_api import AsyncTranscriptionAPI

__all__ = [
    "Transcriber",
    "AudioSource",
    "StreamHandler",
    "TranscriptionPlugin",
    "TranscriptResult",
    "TranscriptSegment",
    "EngineConfig",
    "TranscriptionEngine",
    "AsyncTranscriptionAPI",
]
