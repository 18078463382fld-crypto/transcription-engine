"""
Shared base class for all transcriber backends.

``BaseTranscriber`` implements the boilerplate shared by local and cloud
transcribers: configuration storage, lifecycle logging, language override
helpers, and model-info properties.
"""

from __future__ import annotations

import abc
import logging
from typing import AsyncIterator, Optional

from ..core.interfaces import Transcriber
from ..core.models import EngineConfig, TranscriptResult

logger = logging.getLogger("transcribe.transcriber")


class BaseTranscriber(Transcriber):
    """
    Abstract base with shared initialisation and config management.

    Subclasses **must** implement:
        - :meth:`transcribe`
        - :meth:`shutdown`

    Subclasses **should** override:
        - :meth:`initialize` (call ``super().initialize(config)`` first)
        - :meth:`transcribe_stream` (the default buffers everything)
    """

    def __init__(self, model_name: str = "base") -> None:
        self._model_name = model_name
        self._config: EngineConfig | None = None
        self._initialized = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def initialize(self, config: EngineConfig) -> None:
        """
        Store the engine config and set the initialised flag.

        Subclasses that load model weights should call
        ``await super().initialize(config)`` **before** loading.
        """
        self._config = config
        self._initialized = True
        logger.info(
            "%s initialised (model=%s, device=%s, compute=%s)",
            self.backend_name,
            self._model_name,
            config.device,
            config.compute_type,
        )

    async def shutdown(self) -> None:
        """Mark as uninitialised. Subclasses should release resources."""
        self._initialized = False
        logger.info("%s shut down", self.backend_name)

    # ── Core transcription (must be overridden) ─────────────────────────

    @abc.abstractmethod
    async def transcribe(self, audio: bytes) -> TranscriptResult:
        """Transcribe a complete audio buffer."""

    # ── Streaming (optional override) ────────────────────────────────────

    async def transcribe_stream(
        self, stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptResult]:
        """
        Default streaming implementation — buffers all audio then calls
        :meth:`transcribe`. Override for true streaming behaviour.
        """
        chunks = b""
        async for chunk in stream:
            chunks += chunk
        yield await self.transcribe(chunks)

    # ── Helpers ─────────────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        return self.__class__.__name__

    @property
    def model_name(self) -> str:
        """Configured model identifier (e.g. ``\"base\"``, ``\"whisper-1\"``)."""
        return self._model_name

    def _assert_initialized(self) -> None:
        """Raise ``RuntimeError`` if :meth:`initialize` was never called."""
        if not self._initialized or self._config is None:
            raise RuntimeError(
                f"{self.backend_name} has not been initialised. "
                "Call await transcriber.initialize(config) first."
            )

    @property
    def config(self) -> EngineConfig:
        """The current engine configuration (only valid after initialise)."""
        self._assert_initialized()
        return self._config  # type: ignore[return-value]

    def _language_param(self) -> str | None:
        """Return the language from config, if set."""
        return self._config.language if self._config else None
