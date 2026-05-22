"""Tests for data models and EngineConfig factories.

All tests in this module operate on pure dataclass logic — no GPU,
no cloud APIs, no audio files required.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pytest

from transcribe.core.models import (
    AudioSourceType,
    EngineConfig,
    TranscriberBackend,
    TranscriptResult,
    TranscriptSegment,
)


# ═══════════════════════════════════════════════════════════════════
# TranscriptSegment
# ═══════════════════════════════════════════════════════════════════


class TestTranscriptSegment:
    """Verify ``TranscriptSegment`` field defaults, properties, and mutability."""

    def test_minimal_segment(self) -> None:
        """Only ``start`` and ``end`` are required; all other fields have sane defaults."""
        seg = TranscriptSegment(start=0.0, end=2.5)
        assert seg.text == ""
        assert seg.confidence == 1.0
        assert seg.speaker is None
        assert seg.metadata == {}
        assert seg.duration == 2.5

    def test_full_segment(self) -> None:
        """All fields can be provided explicitly."""
        seg = TranscriptSegment(
            start=1.0,
            end=3.0,
            text="hello world",
            confidence=0.95,
            speaker="speaker_0",
            metadata={"channel": 0},
        )
        assert seg.text == "hello world"
        assert seg.confidence == 0.95
        assert seg.speaker == "speaker_0"
        assert seg.metadata == {"channel": 0}
        assert seg.duration == 2.0

    def test_duration_property(self) -> None:
        """``duration`` is always ``end - start``."""
        seg = TranscriptSegment(start=1.5, end=4.0)
        assert seg.duration == 2.5

    def test_duration_negative_span(self) -> None:
        """Backwards time spans are permitted (edge case)."""
        seg = TranscriptSegment(start=5.0, end=3.0)
        assert seg.duration == -2.0

    def test_mutable_fields(self) -> None:
        """Dataclass fields are mutable after construction."""
        seg = TranscriptSegment(start=0.0, end=1.0, text="foo")
        seg.text = "bar"
        seg.confidence = 0.5
        seg.metadata["key"] = "val"
        assert seg.text == "bar"
        assert seg.confidence == 0.5
        assert seg.metadata["key"] == "val"

    def test_metadata_default_is_new_dict_per_instance(self) -> None:
        """Each segment gets its own ``metadata`` dict (no mutable default sharing)."""
        seg1 = TranscriptSegment(start=0.0, end=1.0)
        seg2 = TranscriptSegment(start=0.0, end=1.0)
        seg1.metadata["x"] = 1
        assert "x" not in seg2.metadata


# ═══════════════════════════════════════════════════════════════════
# TranscriptResult
# ═══════════════════════════════════════════════════════════════════


class TestTranscriptResult:
    """Verify ``TranscriptResult`` aggregation and field defaults."""

    def test_minimal_result(self) -> None:
        """Only ``text`` is required; all other fields default sensibly."""
        result = TranscriptResult(text="hello world")
        assert result.text == "hello world"
        assert result.segments == []
        assert result.language == "en"
        assert result.backend == "unknown"
        assert result.source_type == AudioSourceType.FILE
        assert result.duration_seconds is None
        assert result.processing_time is None
        assert result.metadata == {}

    def test_segment_count_empty(self) -> None:
        """``segment_count`` returns 0 when no segments are present."""
        result = TranscriptResult(text="")
        assert result.segment_count() == 0

    def test_segment_count_multiple(self) -> None:
        """``segment_count`` returns the correct number of segments."""
        result = TranscriptResult(
            text="a b c",
            segments=[
                TranscriptSegment(start=0.0, end=1.0, text="a"),
                TranscriptSegment(start=1.0, end=2.0, text="b"),
            ],
        )
        assert result.segment_count() == 2

    def test_result_with_segments(self) -> None:
        """Segments are stored and accessible."""
        seg = TranscriptSegment(start=0.0, end=2.0, text="hello")
        result = TranscriptResult(
            text="hello",
            segments=[seg],
            language="en",
            backend="test",
            source_type=AudioSourceType.BYTES,
            duration_seconds=2.0,
            processing_time=1.5,
            metadata={"model": "test"},
        )
        assert result.segments[0] is seg
        assert result.backend == "test"
        assert result.source_type == AudioSourceType.BYTES
        assert result.duration_seconds == 2.0
        assert result.processing_time == 1.5


# ═══════════════════════════════════════════════════════════════════
# EngineConfig / TranscriberBackend / AudioSourceType enums
# ═══════════════════════════════════════════════════════════════════


class TestTranscriberBackend:
    def test_enum_values(self) -> None:
        assert TranscriberBackend.LOCAL.value == "local"
        assert TranscriberBackend.CLOUD.value == "cloud"
        assert TranscriberBackend.HYBRID.value == "hybrid"

    def test_enum_membership(self) -> None:
        assert "local" in {e.value for e in TranscriberBackend}


class TestAudioSourceType:
    def test_enum_values(self) -> None:
        assert AudioSourceType.FILE.value == "file"
        assert AudioSourceType.MICROPHONE.value == "microphone"
        assert AudioSourceType.STREAM.value == "stream"
        assert AudioSourceType.BYTES.value == "bytes"
        assert AudioSourceType.URL.value == "url"


class TestEngineConfig:
    """Verify ``EngineConfig`` field defaults and factory methods."""

    def test_default_config(self) -> None:
        """Defaults match the documented values."""
        cfg = EngineConfig()
        assert cfg.backend == TranscriberBackend.LOCAL
        assert cfg.model_name == "base"
        assert cfg.device == "cpu"
        assert cfg.compute_type == "float32"
        assert cfg.language is None
        assert cfg.sample_rate == 16000
        assert cfg.chunk_seconds == 0.5
        assert cfg.max_retries == 3
        assert cfg.timeout_seconds == 60

    def test_custom_values(self) -> None:
        """All fields are settable via constructor."""
        cfg = EngineConfig(
            backend=TranscriberBackend.CLOUD,
            model_name="whisper-1",
            device="cuda",
            compute_type="float16",
            language="zh",
            sample_rate=48000,
            chunk_seconds=1.0,
            max_retries=5,
            timeout_seconds=120,
        )
        assert cfg.backend == TranscriberBackend.CLOUD
        assert cfg.model_name == "whisper-1"
        assert cfg.device == "cuda"
        assert cfg.compute_type == "float16"
        assert cfg.language == "zh"
        assert cfg.sample_rate == 48000
        assert cfg.chunk_seconds == 1.0
        assert cfg.max_retries == 5
        assert cfg.timeout_seconds == 120

    def test_local_defaults_factory(self) -> None:
        """``local_defaults()`` returns a config tuned for local Whisper."""
        cfg = EngineConfig.local_defaults()
        assert cfg.backend == TranscriberBackend.LOCAL
        assert cfg.device == "cpu"
        assert cfg.compute_type == "float32"
        # Other fields retain their class defaults
        assert cfg.model_name == "base"

    def test_cloud_defaults_factory(self) -> None:
        """``cloud_defaults()`` returns a config tuned for cloud API."""
        cfg = EngineConfig.cloud_defaults()
        assert cfg.backend == TranscriberBackend.CLOUD
        assert cfg.model_name == "whisper-1"
        assert cfg.timeout_seconds == 120
        # Other fields retain their class defaults
        assert cfg.device == "cpu"

    def test_config_is_dataclass(self) -> None:
        """``EngineConfig`` is a dataclass (frozen-by-convention, mutable)."""
        assert dataclasses.is_dataclass(EngineConfig)
        cfg = EngineConfig()
        cfg.model_name = "large-v3"
        assert cfg.model_name == "large-v3"

    def test_language_can_be_none(self) -> None:
        """Language can be set to ``None`` for auto-detection."""
        cfg = EngineConfig(language=None)
        assert cfg.language is None
        cfg.language = "fr"
        assert cfg.language == "fr"
