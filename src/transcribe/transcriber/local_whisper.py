"""
Local Whisper transcriber.

Runs ``faster-whisper`` as the local inference backend.  Supports both
batch (``transcribe``) and streaming (``transcribe_stream``) modes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, Optional

from ..core.interfaces import Transcriber
from ..core.models import EngineConfig, TranscriptResult, TranscriptSegment
from .base import BaseTranscriber

logger = logging.getLogger("transcribe.transcriber.local_whisper")

# Optional dependency — warn at import time if missing
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment,misc]

    def _import_warning() -> None:
        logger.warning(
            "faster-whisper is not installed. "
            "Install it with: pip install faster-whisper"
        )

else:
    _import_warning = lambda: None  # noqa: E731


class LocalWhisperTranscriber(BaseTranscriber):
    """
    Transcriber backed by a local ``faster-whisper`` model.

    Args:
        model_name: Model size or path (``\"tiny\"``, ``\"base\"``,
                    ``\"small\"``, ``\"medium\"``, ``\"large-v3\"``,
                    or a local ``.bin`` path).

    Example::

        transcriber = LocalWhisperTranscriber(\"base\")
        await transcriber.initialize(
            EngineConfig(device=\"cuda\", compute_type=\"float16\")
        )
        result = await transcriber.transcribe(audio_bytes)
        print(result.text)
    """

    def __init__(self, model_name: str = "base") -> None:
        super().__init__(model_name=model_name)
        self._model: WhisperModel | None = None
        self._loop = asyncio.get_event_loop()

    async def initialize(self, config: EngineConfig) -> None:
        """Load the Whisper model into memory."""
        if WhisperModel is None:
            _import_warning()
            raise ImportError(
                "faster-whisper is required for LocalWhisperTranscriber. "
                "Install: pip install faster-whisper"
            )

        await super().initialize(config)

        logger.info(
            "Loading Whisper model %s on %s (%s)...",
            self._model_name,
            config.device,
            config.compute_type,
        )
        t0 = time.perf_counter()

        # faster-whisper's constructor is blocking; run in executor
        self._model = await self._loop.run_in_executor(
            None,
            lambda: WhisperModel(
                self._model_name,
                device=config.device,
                compute_type=config.compute_type,
            ),
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Whisper model %s loaded in %.2fs", self._model_name, elapsed
        )

    async def transcribe(self, audio: bytes) -> TranscriptResult:
        """
        Transcribe a full audio buffer using the local Whisper model.

        Args:
            audio: Raw PCM 16-bit mono audio at ``self.config.sample_rate``.

        Returns:
            A ``TranscriptResult`` with full text and timed segments.
        """
        self._assert_initialized()
        if self._model is None:
            raise RuntimeError("Model not loaded; call initialize() first")

        # Convert bytes to a numpy array (int16 -> float32)
        import numpy as np  # noqa: E811

        audio_array: np.ndarray = (
            np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # Run transcription in executor (blocking C extension)
        segments, info = await self._loop.run_in_executor(
            None,
            lambda: self._model.transcribe(
                audio_array,
                language=self._language_param(),
                beam_size=5,
            ),
        )

        # Collect all segments
        seg_list: list[TranscriptSegment] = []
        full_text: list[str] = []
        for seg in segments:
            seg_list.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    confidence=seg.avg_logprob
                    if seg.avg_logprob is not None
                    else 0.0,
                )
            )
            full_text.append(seg.text.strip())

        return TranscriptResult(
            text=" ".join(full_text),
            segments=seg_list,
            language=info.language if info else "en",
            backend=self.backend_name,
            duration_seconds=info.duration if info else None,
        )

    async def shutdown(self) -> None:
        """Unload the model and free GPU memory."""
        self._model = None
        await super().shutdown()
