"""Tests for ``TranscriptionEngine`` with mocked transcriber and source.

All tests use fake async implementations — no real GPU, microphone,
or cloud API calls are made.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from transcribe.core.engine import TranscriptionEngine
from transcribe.core.interfaces import AudioSource, Transcriber, TranscriptionEvent
from transcribe.core.models import (
    AudioSourceType,
    EngineConfig,
    TranscriberBackend,
    TranscriptResult,
    TranscriptSegment,
)


# ═══════════════════════════════════════════════════════════════════
# Mock implementations
# ═══════════════════════════════════════════════════════════════════


class MockTranscriber(Transcriber):
    """A fake transcriber that returns canned results."""

    def __init__(self, result: TranscriptResult | None = None) -> None:
        self._result = result or TranscriptResult(
            text="mock transcript",
            segments=[TranscriptSegment(start=0.0, end=1.0, text="mock transcript")],
        )
        self.initialize_called = False
        self.shutdown_called = False
        self.last_audio: bytes | None = None

    async def initialize(self, config: EngineConfig) -> None:
        self.initialize_called = True

    async def transcribe(self, audio: bytes) -> TranscriptResult:
        self.last_audio = audio
        return self._result

    async def transcribe_stream(
        self, stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptResult]:
        chunks = b""
        async for chunk in stream:
            chunks += chunk
        self.last_audio = chunks
        yield self._result

    async def shutdown(self) -> None:
        self.shutdown_called = True

    @property
    def backend_name(self) -> str:
        return "MockTranscriber"


class MockAudioSource(AudioSource):
    """A fake audio source that returns predefined PCM bytes."""

    def __init__(
        self,
        data: bytes = b"\x00\x00\x00\x00\x00\x00\x00\x00",
        source_type: AudioSourceType = AudioSourceType.FILE,
    ) -> None:
        self._data = data
        self._source_type = source_type
        self.closed = False

    @property
    def source_type(self) -> AudioSourceType:
        return self._source_type

    async def read(self) -> bytes:
        return self._data

    async def stream(self) -> AsyncIterator[bytes]:
        chunk_size = 4
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i : i + chunk_size]

    async def close(self) -> None:
        self.closed = True


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_transcriber() -> MockTranscriber:
    return MockTranscriber()


@pytest.fixture
def mock_source() -> MockAudioSource:
    return MockAudioSource(b"\x01\x00\x02\x00\x03\x00\x04\x00")


@pytest.fixture
def engine(mock_transcriber: MockTranscriber) -> TranscriptionEngine:
    return TranscriptionEngine(
        transcriber=mock_transcriber,
        config=EngineConfig(backend=TranscriberBackend.LOCAL),
    )


# ═══════════════════════════════════════════════════════════════════
# Construction & properties
# ═══════════════════════════════════════════════════════════════════


class TestTranscriptionEngineConstruction:
    def test_default_config_when_none(self) -> None:
        """When no config is supplied a default ``EngineConfig()`` is used."""
        engine = TranscriptionEngine(transcriber=MockTranscriber())
        assert engine.config.backend == TranscriberBackend.LOCAL

    def test_custom_config(self) -> None:
        """A user-supplied config is honoured."""
        cfg = EngineConfig(backend=TranscriberBackend.CLOUD)
        engine = TranscriptionEngine(transcriber=MockTranscriber(), config=cfg)
        assert engine.config.backend == TranscriberBackend.CLOUD

    def test_backend_name(self, engine: TranscriptionEngine) -> None:
        """``backend_name`` delegates to the transcriber."""
        assert engine.backend_name == "MockTranscriber"

    def test_empty_plugins_and_handlers(self) -> None:
        """Default lists are empty, not ``None``."""
        engine = TranscriptionEngine(transcriber=MockTranscriber())
        assert engine._plugins == []
        assert engine._stream_handlers == []


# ═══════════════════════════════════════════════════════════════════
# Lifecycle
# ═══════════════════════════════════════════════════════════════════


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_initialize(self, engine: TranscriptionEngine, mock_transcriber: MockTranscriber) -> None:
        """``initialize`` calls the transcriber's ``initialize``."""
        assert not engine._initialized
        await engine.initialize()
        assert engine._initialized
        assert mock_transcriber.initialize_called

    @pytest.mark.asyncio
    async def test_shutdown(self, engine: TranscriptionEngine, mock_transcriber: MockTranscriber) -> None:
        """``shutdown`` calls the transcriber's ``shutdown``."""
        await engine.initialize()
        await engine.shutdown()
        assert not engine._initialized
        assert mock_transcriber.shutdown_called

    @pytest.mark.asyncio
    async def test_initialize_called_automatically_on_transcribe(
        self, engine: TranscriptionEngine, mock_transcriber: MockTranscriber, mock_source: MockAudioSource
    ) -> None:
        """If not yet initialised, ``transcribe`` calls ``initialize`` first."""
        assert not engine._initialized
        await engine.transcribe(mock_source)
        assert engine._initialized
        assert mock_transcriber.initialize_called


# ═══════════════════════════════════════════════════════════════════
# Batch transcription
# ═══════════════════════════════════════════════════════════════════


class TestBatchTranscribe:
    @pytest.mark.asyncio
    async def test_transcribe_returns_result(
        self, engine: TranscriptionEngine, mock_source: MockAudioSource
    ) -> None:
        """``transcribe`` returns the result from the transcriber."""
        result = await engine.transcribe(mock_source)
        assert isinstance(result, TranscriptResult)
        assert result.text == "mock transcript"
        assert result.source_type == AudioSourceType.FILE

    @pytest.mark.asyncio
    async def test_transcribe_passes_audio_bytes(
        self, engine: TranscriptionEngine, mock_source: MockAudioSource, mock_transcriber: MockTranscriber
    ) -> None:
        """The audio bytes from ``source.read()`` are passed to the transcriber."""
        await engine.transcribe(mock_source)
        assert mock_transcriber.last_audio == b"\x01\x00\x02\x00\x03\x00\x04\x00"

    @pytest.mark.asyncio
    async def test_transcribe_sets_processing_time(
        self, engine: TranscriptionEngine, mock_source: MockAudioSource
    ) -> None:
        """``processing_time`` is populated after transcription."""
        result = await engine.transcribe(mock_source)
        assert result.processing_time is not None
        assert isinstance(result.processing_time, float)
        assert result.processing_time >= 0.0

    @pytest.mark.asyncio
    async def test_transcribe_closes_source(
        self, engine: TranscriptionEngine, mock_source: MockAudioSource
    ) -> None:
        """The audio source is closed after batch transcription."""
        await engine.transcribe(mock_source)
        assert mock_source.closed

    @pytest.mark.asyncio
    async def test_transcribe_language_override(
        self, engine: TranscriptionEngine, mock_source: MockAudioSource
    ) -> None:
        """A language override is applied and then restored."""
        engine._config.language = "en"
        result = await engine.transcribe(mock_source, language="fr")
        assert result is not None
        # config restored
        assert engine._config.language == "en"

    @pytest.mark.asyncio
    async def test_transcribe_bytes_method(
        self, engine: TranscriptionEngine, mock_transcriber: MockTranscriber
    ) -> None:
        """``transcribe_bytes`` wraps audio in ``BytesSource`` and transcribes."""
        result = await engine.transcribe_bytes(b"\x01\x00\x02\x00")
        assert isinstance(result, TranscriptResult)
        assert mock_transcriber.last_audio == b"\x01\x00\x02\x00"


# ═══════════════════════════════════════════════════════════════════
# Streaming transcription
# ═══════════════════════════════════════════════════════════════════


class TestStreamTranscribe:
    @pytest.mark.asyncio
    async def test_transcribe_stream_yields_results(
        self, engine: TranscriptionEngine, mock_source: MockAudioSource
    ) -> None:
        """``transcribe_stream`` yields ``TranscriptResult`` objects."""
        results = []
        async for result in engine.transcribe_stream(mock_source):
            results.append(result)
        assert len(results) == 1
        assert results[0].text == "mock transcript"

    @pytest.mark.asyncio
    async def test_transcribe_stream_closes_source(
        self, engine: TranscriptionEngine, mock_source: MockAudioSource
    ) -> None:
        """The audio source is closed after streaming completes."""
        async for _ in engine.transcribe_stream(mock_source):
            pass
        assert mock_source.closed

    @pytest.mark.asyncio
    async def test_transcribe_stream_language_restored(
        self, engine: TranscriptionEngine, mock_source: MockAudioSource
    ) -> None:
        """Language override is restored after streaming finishes."""
        engine._config.language = "en"
        async for _ in engine.transcribe_stream(mock_source, language="de"):
            pass
        assert engine._config.language == "en"


# ═══════════════════════════════════════════════════════════════════
# Event system
# ═══════════════════════════════════════════════════════════════════


class TestEventSystem:
    @pytest.mark.asyncio
    async def test_on_initialized_event(self, engine: TranscriptionEngine) -> None:
        """The ``initialized`` event fires after ``initialize()``."""
        events: list[str] = []

        async def handler(event: TranscriptionEvent) -> None:
            events.append(event.type)

        engine.on("initialized", handler)
        await engine.initialize()
        assert "initialized" in events

    @pytest.mark.asyncio
    async def test_on_final_event_in_batch(
        self, engine: TranscriptionEngine, mock_source: MockAudioSource
    ) -> None:
        """The ``final`` event fires after batch transcription."""
        events: list[str] = []

        async def handler(event: TranscriptionEvent) -> None:
            events.append(event.type)

        engine.on("final", handler)
        await engine.transcribe(mock_source)
        assert "final" in events

    @pytest.mark.asyncio
    async def test_on_stopped_event_on_shutdown(self, engine: TranscriptionEngine) -> None:
        """The ``stopped`` event fires during ``shutdown()``."""
        events: list[str] = []

        async def handler(event: TranscriptionEvent) -> None:
            events.append(event.type)

        engine.on("stopped", handler)
        await engine.initialize()
        await engine.shutdown()
        assert "stopped" in events

    @pytest.mark.asyncio
    async def test_off_unsubscribes(self, engine: TranscriptionEngine) -> None:
        """After ``off()``, the handler no longer receives events."""
        events: list[str] = []

        async def handler(event: TranscriptionEvent) -> None:
            events.append(event.type)

        engine.on("initialized", handler)
        engine.off("initialized", handler)
        await engine.initialize()
        assert "initialized" not in events

    @pytest.mark.asyncio
    async def test_event_handler_exception_does_not_crash(
        self, engine: TranscriptionEngine, mock_source: MockAudioSource
    ) -> None:
        """An exception in an event handler is caught and logged."""

        async def failing_handler(event: TranscriptionEvent) -> None:
            raise RuntimeError("handler failure")

        engine.on("final", failing_handler)
        # This should not raise
        result = await engine.transcribe(mock_source)
        assert result.text == "mock transcript"
