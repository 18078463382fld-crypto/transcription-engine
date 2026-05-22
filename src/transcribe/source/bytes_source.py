"""Audio source backed by raw PCM bytes in memory.

Wraps pre-decoded PCM audio data so it can be fed into the engine
just like any other source.  Useful for programmatic use where audio
has already been loaded or generated in memory.

The :class:`~transcribe.core.engine.TranscriptionEngine.transcribe_bytes`
convenience method creates a ``BytesSource`` internally.
"""

from __future__ import annotations

import math
from typing import AsyncIterator, Optional

from ..core.interfaces import AudioSource
from ..core.models import AudioSourceType, EngineConfig

# ── Source implementation ─────────────────────────────────────


class BytesSource(AudioSource):
    """Audio source wrapping raw PCM 16-bit mono bytes.

    Args:
        data: Raw PCM audio data (16-bit signed, little-endian, mono).
        sample_rate: Sample rate of *data* (default 16000).
        chunk_seconds: Duration of each chunk yielded by :meth:`stream`
            (default 0.5).

    The caller is responsible for ensuring *data* is already in the
    correct format (16-bit mono PCM at the given sample rate) — no
    resampling or format conversion is performed.

    Example::

        source = BytesSource(pcm_bytes, sample_rate=16000)
        result = await engine.transcribe(source)
    """

    def __init__(
        self,
        data: bytes,
        sample_rate: int = 16000,
        chunk_seconds: float = 0.5,
    ) -> None:
        self._data = data
        self._sample_rate = sample_rate
        self._chunk_seconds = chunk_seconds

    # ── Properties ────────────────────────────────────────────

    @property
    def source_type(self) -> AudioSourceType:
        return AudioSourceType.BYTES

    @property
    def data(self) -> bytes:
        """The raw PCM bytes wrapped by this source."""
        return self._data

    @property
    def sample_rate(self) -> int:
        """Sample rate of the wrapped audio data."""
        return self._sample_rate

    @property
    def duration_seconds(self) -> float:
        """Total audio duration in seconds (float)."""
        # 2 bytes per sample, 1 channel
        return len(self._data) / (self._sample_rate * 2)

    # ── AudioSource interface ─────────────────────────────────

    async def read(self) -> bytes:
        return self._data

    async def stream(self) -> AsyncIterator[bytes]:
        chunk_size = self._compute_chunk_size()

        offset = 0
        while offset < len(self._data):
            yield self._data[offset : offset + chunk_size]
            offset += chunk_size

    async def close(self) -> None:
        """Release the internal data buffer."""
        self._data = b""

    # ── Internals ─────────────────────────────────────────────

    def _compute_chunk_size(self) -> int:
        return int(self._sample_rate * self._chunk_seconds * 2)

    # ── repr / str ────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"BytesSource({len(self._data)} bytes, "
            f"{self._sample_rate} Hz, "
            f"{self.duration_seconds:.2f}s)"
        )
