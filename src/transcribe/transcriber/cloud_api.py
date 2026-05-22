"""
Cloud API transcriber.

Delegates transcription to a remote API — supports OpenAI Whisper
(``whisper-1``) and any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, AsyncIterator, Optional

import aiohttp

from ..core.interfaces import Transcriber
from ..core.models import (
    AudioSourceType,
    EngineConfig,
    TranscriptResult,
    TranscriptSegment,
)
from .base import BaseTranscriber

logger = logging.getLogger("transcribe.transcriber.cloud_api")

# ── Defaults ───────────────────────────────────────────────────────────

DEFAULT_API_URL = "https://api.openai.com/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-1"


class CloudAPITranscriber(BaseTranscriber):
    """
    Transcriber backed by a cloud speech-to-text API.

    By default uses OpenAI's ``whisper-1`` endpoint but can be configured
    for any OpenAI-compatible API (e.g. Azure OpenAI, local proxy).

    Args:
        model_name:  API model name (e.g. ``\"whisper-1\"``).
        api_key:     API key.  Falls back to ``OPENAI_API_KEY`` env var.
        api_url:     Full URL of the transcription endpoint.
        language:    Default language hint (or ``None`` for auto-detect).

    Example::

        transcriber = CloudAPITranscriber(api_key=\"sk-...\")
        await transcriber.initialize(EngineConfig())
        result = await transcriber.transcribe(audio_bytes)
        print(result.text)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: str | None = None,
        api_url: str | None = None,
        language: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._api_url = api_url or DEFAULT_API_URL
        self._language = language
        self._session: aiohttp.ClientSession | None = None

    async def initialize(self, config: EngineConfig) -> None:
        """Create an ``aiohttp`` client session."""
        await super().initialize(config)

        if not self._api_key:
            raise ValueError(
                "No API key provided for CloudAPITranscriber. "
                "Pass api_key= or set the OPENAI_API_KEY environment variable."
            )

        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self._api_key}",
            },
            timeout=aiohttp.ClientTimeout(
                total=self._config.timeout_seconds  # type: ignore[union-attr]
            ),
        )
        logger.info(
            "Cloud API transcriber ready (url=%s, model=%s)",
            self._api_url,
            self._model_name,
        )

    async def transcribe(self, audio: bytes) -> TranscriptResult:
        """
        Send audio to the cloud API for transcription.

        Args:
            audio: Raw PCM 16-bit mono audio at the configured sample rate.
                   The audio is sent as a WAV file in the multipart request.

        Returns:
            A ``TranscriptResult`` with the transcribed text and segments.
        """
        self._assert_initialized()
        if self._session is None:
            raise RuntimeError("Session not created; call initialize() first")

        # Convert raw PCM to WAV in-memory for the API request
        wav_bytes = _pcm_to_wav(audio, sample_rate=self._config.sample_rate)

        # Build multipart form
        form = aiohttp.FormData()
        form.add_field(
            "file",
            wav_bytes,
            filename="audio.wav",
            content_type="audio/wav",
        )
        form.add_field("model", self._model_name)
        if lang := self._language_param():
            form.add_field("language", lang)
        form.add_field("response_format", "verbose_json")
        form.add_field("timestamp_granularities[]", "segment")

        t0 = time.perf_counter()
        async with self._session.post(self._api_url, data=form) as resp:
            elapsed = time.perf_counter() - t0

            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"Cloud API returned HTTP {resp.status}: {body}"
                )

            data: dict[str, Any] = await resp.json()

        # Parse response
        segments = [
            TranscriptSegment(
                start=s.get("start", 0.0),
                end=s.get("end", 0.0),
                text=s.get("text", "").strip(),
                confidence=float(s.get("confidence", 1.0)),
            )
            for s in data.get("segments", [])
        ]

        return TranscriptResult(
            text=data.get("text", "").strip(),
            segments=segments,
            language=data.get("language", self._language_param() or "en"),
            backend=self.backend_name,
            processing_time=elapsed,
        )

    async def transcribe_stream(
        self, stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptResult]:
        """
        Streaming mode for cloud API.

        The default implementation buffers all chunks and sends them
        in a single request.  Override this method if your cloud
        provider supports real-time streaming (e.g. WebSocket-based
        transcription).
        """
        async for result in super().transcribe_stream(stream):
            yield result

    async def shutdown(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        await super().shutdown()


# ── Utility: PCM → WAV converter ──────────────────────────────────────


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """
    Wrap raw PCM 16-bit mono audio in a WAV container.

    Most cloud APIs (including OpenAI) expect a proper container format
    (WAV, MP3, etc.) rather than raw PCM.
    """
    import struct
    import wave
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)  # mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)

    return buf.getvalue()
