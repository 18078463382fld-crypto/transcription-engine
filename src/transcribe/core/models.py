"""
Data models used throughout the transcription engine.

All models are plain dataclasses with clear field semantics so consumers
(plugins, callbacks, API responses) always know what they are working with.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TranscriberBackend(str, Enum):
    """Which kind of backend is handling transcription."""

    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"  # local fallback with cloud primary, or vice versa


class AudioSourceType(str, Enum):
    """The origin of audio data submitted for transcription."""

    FILE = "file"
    MICROPHONE = "microphone"
    STREAM = "stream"
    BYTES = "bytes"  # raw in-memory bytes
    URL = "url"  # remote audio URL


@dataclass
class TranscriptSegment:
    """
    A single segment of transcribed text with timing information.

    Attributes:
        start:     Start time in seconds (float)
        end:       End time in seconds (float)
        text:      Transcribed text for this segment
        confidence: Confidence score between 0.0 and 1.0 (if available)
        speaker:   Optional speaker label (if diarization is enabled)
        metadata:  Arbitrary key-value metadata from plugins or backends
    """

    start: float
    end: float
    text: str
    confidence: float = 1.0
    speaker: Optional[str] = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return self.end - self.start


@dataclass
class TranscriptResult:
    """
    The complete result of a transcription operation.

    Attributes:
        text:              Full concatenated transcript text
        segments:          List of timed segments
        language:          Detected or requested language code (e.g. \"en\", \"zh\")
        backend:           Which backend produced this result
        source_type:       Origin of the audio data
        duration_seconds:  Total audio duration (if known)
        processing_time:   Wall-clock time spent processing (seconds)
        metadata:          Arbitrary key-value metadata
    """

    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = "en"
    backend: str = "unknown"
    source_type: AudioSourceType = AudioSourceType.FILE
    duration_seconds: Optional[float] = None
    processing_time: Optional[float] = None
    metadata: dict[str, object] = field(default_factory=dict)

    def segment_count(self) -> int:
        return len(self.segments)


@dataclass
class EngineConfig:
    """
    Global configuration for the TranscriptionEngine.

    Attributes:
        backend:         Preferred backend type
        model_name:      Model name / size (e.g. \"base\", \"large-v3\", \"whisper-1\")
        device:          Torch device for local models (\"cpu\", \"cuda\", \"mps\")
        compute_type:    Precision for local models (\"float16\", \"int8\", \"float32\")
        language:        Default language hint (None = auto-detect)
        sample_rate:     Target sample rate in Hz (default 16000)
        chunk_seconds:   Streaming chunk duration (default 0.5)
        max_retries:     Cloud API retry count on transient failures
        timeout_seconds: Request timeout for cloud APIs
    """

    backend: TranscriberBackend = TranscriberBackend.LOCAL
    model_name: str = "base"
    device: str = "cpu"
    compute_type: str = "float32"
    language: Optional[str] = None
    sample_rate: int = 16000
    chunk_seconds: float = 0.5
    max_retries: int = 3
    timeout_seconds: int = 60

    @classmethod
    def local_defaults(cls) -> "EngineConfig":
        """Returns a config tuned for local Whisper inference."""
        return cls(
            backend=TranscriberBackend.LOCAL, device="cpu", compute_type="float32"
        )

    @classmethod
    def cloud_defaults(cls) -> "EngineConfig":
        """Returns a config tuned for cloud API usage."""
        return cls(
            backend=TranscriberBackend.CLOUD,
            model_name="whisper-1",
            timeout_seconds=120,
        )
