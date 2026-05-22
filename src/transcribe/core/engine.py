"""
Main engine orchestrator.

``TranscriptionEngine`` ties together transcribers, audio sources, stream
handlers, and plugins into a coherent transcription pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, Callable, Coroutine

from .interfaces import (
    AudioSource,
    Transcriber,
    StreamHandler,
    TranscriptionPlugin,
    TranscriptionEvent,
)
from .models import TranscriptResult, EngineConfig

logger = logging.getLogger("transcribe.engine")

# ── Type alias ───────────────────────────────────────────────
EventHandler = Callable[[TranscriptionEvent], Coroutine[object, None, None]]


class TranscriptionEngine:
    """
    High-level orchestrator for transcription workloads.

    Typical usage::

        engine = TranscriptionEngine(
            transcriber=LocalWhisperTranscriber("base"),
            config=EngineConfig(device="cuda"),
        )
        await engine.initialize()

        result = await engine.transcribe(FileSource("meeting.mp3"))
        print(result.text)

        # Real-time:
        async for segment in engine.transcribe_stream(MicrophoneSource()):
            print(segment.text)

        await engine.shutdown()
    """

    def __init__(
        self,
        transcriber: Transcriber,
        config: EngineConfig | None = None,
        stream_handlers: list[StreamHandler] | None = None,
        plugins: list[TranscriptionPlugin] | None = None,
    ):
        """
        Args:
            transcriber:    The core Transcriber implementation.
            config:         Engine configuration (defaults to ``EngineConfig()``).
            stream_handlers: Optional list of stream pre-processors.
            plugins:        Optional list of pipeline plugins.
        """
        self._transcriber = transcriber
        self._config = config or EngineConfig()
        self._stream_handlers = stream_handlers or []
        self._plugins = plugins or []
        self._event_handlers: dict[str, list[EventHandler]] = {}
        self._initialized = False

    # ── Lifecycle ────────────────────────────────────────────

    async def initialize(self) -> None:
        """Load the transcriber and set up plugins."""
        logger.info(
            "Initializing engine with backend=%s model=%s",
            self._config.backend.value,
            self._config.model_name,
        )
        await self._transcriber.initialize(self._config)
        for plugin in self._plugins:
            await plugin.setup(self._config)
        self._initialized = True
        await self._emit("initialized")

    async def shutdown(self) -> None:
        """Release all resources."""
        logger.info("Shutting down engine")
        await self._emit("stopped")
        for plugin in reversed(self._plugins):
            await plugin.teardown()
        await self._transcriber.shutdown()
        self._initialized = False

    # ── Event system ─────────────────────────────────────────

    def on(self, event_type: str, handler: EventHandler) -> None:
        """
        Subscribe to engine events.

        Args:
            event_type: One of ``\"initialized\"``, ``\"started\"``, ``\"segment\"``,
                        ``\"final\"``, ``\"stopped\"``, ``\"error\"``.
            handler:    Async callable ``async def handler(event: TranscriptionEvent)``.
        """
        self._event_handlers.setdefault(event_type, []).append(handler)

    def off(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe an event handler."""
        handlers = self._event_handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def _emit(self, event_type: str, data: object = None) -> None:
        event = TranscriptionEvent(type=event_type, engine=self, data=data)
        for handler in self._event_handlers.get(event_type, []):
            try:
                await handler(event)
            except Exception:
                logger.exception("Event handler %s failed for %s", handler, event_type)

    # ── Batch transcription ──────────────────────────────────

    async def transcribe(
        self,
        source: AudioSource,
        language: str | None = None,
    ) -> TranscriptResult:
        """
        Transcribe a complete audio source (batch / file mode).

        Args:
            source:   An AudioSource implementation.
            language: Optional language override for this call.

        Returns:
            A ``TranscriptResult`` with the full transcript.
        """
        if not self._initialized:
            await self.initialize()

        await self._emit("started", source)

        # Read audio
        audio = await source.read()

        # Plugins: pre_process
        for plugin in self._plugins:
            audio = await plugin.pre_process(audio)

        # Override language if provided
        original_lang = self._config.language
        if language:
            self._config.language = language

        # Transcribe
        t0 = time.perf_counter()
        try:
            result = await self._transcriber.transcribe(audio)
        finally:
            if language:
                self._config.language = original_lang

        elapsed = time.perf_counter() - t0
        result.processing_time = elapsed
        result.source_type = source.source_type

        # Plugins: post_process
        for plugin in self._plugins:
            result = await plugin.post_process(result)

        await self._emit("final", result)
        await source.close()

        logger.info(
            "Transcribed %.1fs audio in %.2fs -> %d chars, %d segments",
            result.duration_seconds or 0,
            elapsed,
            len(result.text),
            len(result.segments),
        )
        return result

    # ── Streaming transcription ──────────────────────────────

    async def transcribe_stream(
        self,
        source: AudioSource,
        language: str | None = None,
    ) -> AsyncIterator[TranscriptResult]:
        """
        Transcribe a live audio stream (real-time mode).

        Yields ``TranscriptResult`` objects as partial segments are recognised.

        Args:
            source:   An AudioSource implementation that supports ``stream()``.
            language: Optional language override.
        """
        if not self._initialized:
            await self.initialize()

        original_lang = self._config.language
        if language:
            self._config.language = language

        try:
            await self._emit("started", source)

            # Build the processing pipeline
            stream = source.stream()
            # Wrap with stream handlers
            for handler in self._stream_handlers:
                stream = self._wrap_handler(stream, handler)

            async for result in self._transcriber.transcribe_stream(stream):
                # Plugins: post_process
                for plugin in self._plugins:
                    result = await plugin.post_process(result)
                await self._emit("segment", result)
                yield result

        finally:
            if language:
                self._config.language = original_lang
            await source.close()
            await self._emit("stopped")

    async def _wrap_handler(
        self, stream: AsyncIterator[bytes], handler: StreamHandler
    ) -> AsyncIterator[bytes]:
        """Apply a StreamHandler to each chunk in the stream."""
        async for chunk in stream:
            processed = await handler.process(chunk)
            if processed:
                yield processed

    # ── Convenience: transcribe from bytes ───────────────────

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: str | None = None,
    ) -> TranscriptResult:
        """
        Transcribe raw PCM audio bytes directly.

        This is a convenience shortcut that wraps the bytes in a
        ``BytesSource`` internally.
        """
        from ..source.bytes_source import BytesSource  # noqa: E402

        source = BytesSource(audio_bytes, self._config.sample_rate)
        return await self.transcribe(source, language=language)

    # ── Properties ───────────────────────────────────────────

    @property
    def config(self) -> EngineConfig:
        """Current engine configuration."""
        return self._config

    @property
    def backend_name(self) -> str:
        """Name of the active transcription backend."""
        return self._transcriber.backend_name
