"""
Abstract plugin base with pre- / post-processing hooks and priority ordering.

All custom plugins **must** subclass :class:`AbstractPlugin` and provide at
least a unique :attr:`name`. Hooks are no-ops by default — override
:meth:`pre_process` and/or :meth:`post_process` to inject custom logic into
the transcription pipeline.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

from ..core.interfaces import TranscriptionPlugin
from ..core.models import EngineConfig, TranscriptResult

logger = logging.getLogger("transcribe.plugins")


class AbstractPlugin(TranscriptionPlugin):
    """
    Convenience base class for transcription pipeline plugins.

    Adds **priority-based ordering** so plugins run in a deterministic
    sequence (lower priority values execute first).  Subclasses need only
    implement :attr:`name` and whichever hooks they care about.

    Hook order during a batch transcription::

        for plugin in sorted_plugins:          # by priority ASC
            audio = await plugin.pre_process(audio)

        result = await transcriber.transcribe(audio)

        for plugin in sorted_plugins:          # by priority ASC
            result = await plugin.post_process(result)

    Attributes:
        priority:  Execution order (0 = first, higher = later). Default 100.
        enabled:   If ``False`` the plugin's hooks are skipped. Default ``True``.
    """

    # ── Plugin metadata ────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique, human-readable plugin name (e.g. ``\"vad-filter\"``)."""

    priority: int = 100
    enabled: bool = True

    # ── Lifecycle hooks ────────────────────────────────────────────

    async def setup(self, config: EngineConfig, plugin_config: dict | None = None) -> None:
        """
        Called once when the engine initialises.

        Override to perform one-time setup (load models, open connections,
        read configuration).  The default implementation logs the event.
        """
        logger.info("Plugin %s: setup complete (priority=%d)", self.name, self.priority)

    async def teardown(self) -> None:
        """
        Called when the engine shuts down.

        Override to release resources, close network sessions, etc.
        """
        logger.info("Plugin %s: teardown complete", self.name)

    # ── Pre / post hooks ───────────────────────────────────────────

    async def pre_process(self, audio: bytes) -> bytes:
        """
        Modify or inspect raw audio **before** transcription.

        Args:
            audio: Raw PCM 16-bit mono audio at the engine's sample rate.

        Returns:
            The (possibly modified) audio bytes.  Return ``b\"\"`` to
            signal that the audio should be discarded (plugin-level VAD).
        """
        return audio

    async def post_process(self, result: TranscriptResult) -> TranscriptResult:
        """
        Modify, filter, or enrich a ``TranscriptResult`` **after** transcription.

        Typical uses: grammar correction, entity extraction, translation,
        keyword highlighting, confidence re-scoring.

        Args:
            result: The raw transcript result from the transcriber.

        Returns:
            The (possibly modified) result.
        """
        return result

    # ── Optional convenience hooks ─────────────────────────────────

    async def on_error(self, exception: Exception, context: dict[str, Any] | None = None) -> None:
        """
        Called when an error occurs during transcription.

        Plugins can log, increment metrics, or perform fallback logic.
        The default implementation logs the exception.
        """
        logger.warning(
            "Plugin %s: error hook invoked — %s: %s",
            self.name,
            type(exception).__name__,
            exception,
        )

    # ── Representation ─────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"priority={self.priority}, "
            f"enabled={self.enabled})"
        )
