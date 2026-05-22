"""
Abstract interfaces (ABCs) defining the transcription engine's plugin contract.

Every transcriber, audio source, stream handler, and plugin must implement
one of these ABCs.  This is the **extensibility layer** — third-party packages
can register concrete implementations through entry points or direct
instantiation.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from .models import TranscriptResult, EngineConfig, AudioSourceType


# ═══════════════════════════════════════════════════════════════
# AudioSource
# ═══════════════════════════════════════════════════════════════


class AudioSource(abc.ABC):
    """
    Abstract source of audio data.

    Implementations wrap files, microphone streams, in-memory bytes,
    or remote URLs.  The engine calls ``read()`` to get raw PCM data
    and ``stream()`` for chunked ("real-time") consumption.
    """

    @property
    @abc.abstractmethod
    def source_type(self) -> AudioSourceType:
        """What kind of source this is (file, microphone, stream, etc.)."""

    @abc.abstractmethod
    async def read(self) -> bytes:
        """
        Return the complete audio as raw PCM (16-bit mono, 16 kHz).

        Used for batch/one-shot transcription.  Implementations should
        decode and resample to the engine's expected format.
        """

    @abc.abstractmethod
    async def stream(self) -> AsyncIterator[bytes]:
        """
        Yield chunks of raw PCM data for real-time streaming.

        Each chunk should be ``EngineConfig.chunk_seconds`` worth of
        audio.  When the stream is exhausted iteration simply stops.
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """Release resources (file handles, microphone, network connections)."""


# ═══════════════════════════════════════════════════════════════
# Transcriber
# ═══════════════════════════════════════════════════════════════


class Transcriber(abc.ABC):
    """
    Core transcription engine plug-in.

    Subclasses implement either :meth:`transcribe` (batch) or
    :meth:`transcribe_stream` (real-time), or both.
    """

    @abc.abstractmethod
    async def initialize(self, config: EngineConfig) -> None:
        """
        One-time setup: load model weights, open API sessions, etc.

        Called once by the engine before any transcription work.
        """

    @abc.abstractmethod
    async def transcribe(self, audio: bytes) -> TranscriptResult:
        """
        Transcribe a complete audio buffer (batch mode).

        Args:
            audio: Raw PCM 16-bit mono audio at the configured sample rate.

        Returns:
            A ``TranscriptResult`` with full transcript and timed segments.
        """

    async def transcribe_stream(
        self, stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptResult]:
        """
        Transcribe a live audio stream (real-time mode).

        Yields partial ``TranscriptResult`` objects as each chunk or window
        is processed.  The iterator ends when the source stream is exhausted.

        The default implementation buffers all chunks and calls ``transcribe()``
        — override for true streaming behaviour (e.g. Whisper's streaming mode).
        """
        chunks = b""
        async for chunk in stream:
            chunks += chunk
        yield await self.transcribe(chunks)

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Cleanup: unload models, close connections, free GPU memory."""

    @property
    def backend_name(self) -> str:
        """Human-readable backend identifier (e.g. ``\"local-whisper\"``)."""
        return self.__class__.__name__


# ═══════════════════════════════════════════════════════════════
# StreamHandler
# ═══════════════════════════════════════════════════════════════


class StreamHandler(abc.ABC):
    """
    Real-time audio pipeline stage.

    A StreamHandler sits between the audio source and the transcriber,
    performing VAD (voice activity detection), noise reduction, re-sampling,
    or buffering.
    """

    @abc.abstractmethod
    async def process(self, chunk: bytes) -> bytes:
        """
        Process a single audio chunk.

        Return the processed chunk, or an empty ``b\"\"`` to indicate the
        chunk should be dropped (e.g. silence / non-speech).
        """

    @abc.abstractmethod
    async def reset(self) -> None:
        """Reset internal state (e.g. between utterances)."""


# ═══════════════════════════════════════════════════════════════
# TranscriptionPlugin
# ═══════════════════════════════════════════════════════════════

T = type("T", (), {})  # dummy for Generic[T] in docstrings


class TranscriptionPlugin(abc.ABC):
    """
    Extend the transcription pipeline with custom logic.

    Plugins can:
    - Modify audio **before** transcription (pre-processing)
    - Modify/filter **results** after transcription (post-processing)
    - Add custom **metadata** to results
    - Enrich results with NLP (entity extraction, translation, etc.)

    Type parameter ``T`` is an optional configuration dataclass stored
    as ``self.plugin_config``.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique plugin name (used for ordering and logging)."""

    async def setup(self, config: EngineConfig, plugin_config: dict | None = None) -> None:
        """Called once when the plugin is loaded."""

    async def pre_process(self, audio: bytes) -> bytes:
        """
        Modify raw audio before transcription.

        Return modified audio or the original unchanged.
        """
        return audio

    async def post_process(self, result: TranscriptResult) -> TranscriptResult:
        """
        Modify / enrich a transcript result after transcription.

        Typical uses: grammar correction, translation, keyword highlighting.
        """
        return result

    async def teardown(self) -> None:
        """Called when the engine shuts down."""


# ═══════════════════════════════════════════════════════════════
# Event types for the engine event system
# ═══════════════════════════════════════════════════════════════


@dataclass
class TranscriptionEvent:
    """
    Event emitted during the transcription lifecycle.

    Consumers can subscribe to events via ``engine.on(event_type, handler)``.

    Attributes:
        type:   Event type string (e.g. ``\"segment\"``, ``\"final\"``, ``\"error\"``)
        engine: Reference to the engine that emitted the event
        data:   Optional payload (``TranscriptResult``, ``Exception``, etc.)
    """

    type: str
    engine: "TranscriptionEngine"  # noqa: F821
    data: Optional[object] = None


# ═══════════════════════════════════════════════════════════════
# CompositeTranscriber — multi-backend fallback / ensemble
# ═══════════════════════════════════════════════════════════════


class CompositeTranscriber(Transcriber):
    """
    A transcriber that delegates to multiple backends and merges results.

    Useful for:
    - **fallback** (try cloud, fall back to local)
    - **ensemble** (combine outputs from multiple backends)

    Strategy options:
    - ``\"first\"`` — return the first successful result
    - ``\"best\"``  — return the result with the highest confidence
    - ``\"merge\"`` — concatenate all segments from all backends
    """

    def __init__(
        self, transcribers: list[Transcriber], merge_strategy: str = "first"
    ):
        self._transcribers = transcribers
        self._merge_strategy = merge_strategy

    async def initialize(self, config: EngineConfig) -> None:
        for t in self._transcribers:
            await t.initialize(config)

    async def transcribe(self, audio: bytes) -> TranscriptResult:
        results: list[TranscriptResult] = []
        for t in self._transcribers:
            try:
                results.append(await t.transcribe(audio))
            except Exception:
                continue

        if not results:
            raise RuntimeError("All backends failed to transcribe")

        if self._merge_strategy == "first":
            return results[0]
        if self._merge_strategy == "best":
            return max(
                results,
                key=lambda r: (
                    r.segments[0].confidence if r.segments else 0
                ),
            )
        # merge
        merged = results[0]
        for r in results[1:]:
            merged.text += "\n" + r.text
            merged.segments.extend(r.segments)
        return merged

    async def transcribe_stream(
        self, stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptResult]:
        for t in self._transcribers:
            try:
                async for r in t.transcribe_stream(stream):
                    yield r
                return
            except Exception:
                continue
        raise RuntimeError("All streaming backends failed")

    async def shutdown(self) -> None:
        for t in self._transcribers:
            await t.shutdown()

    @property
    def backend_name(self) -> str:
        return "+".join(t.backend_name for t in self._transcribers)
