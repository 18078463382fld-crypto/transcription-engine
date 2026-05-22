"""Audio source backed by a file on disk.

Reads and decodes common audio formats (WAV, FLAC, MP3, M4A, OGG, etc.)
and converts them to raw PCM 16-bit mono at the configured sample rate.

Uses ``soundfile`` for natively supported formats and ``pydub`` (via
``AudioSegment``) for everything else.
"""

from __future__ import annotations

import io
import logging
import math
import os
from typing import AsyncIterator, Optional

import soundfile as sf

from ..core.interfaces import AudioSource
from ..core.models import AudioSourceType, EngineConfig

logger = logging.getLogger("transcribe.source.file")

# ── Format helpers ─────────────────────────────────────────────


def _read_audio(path: str, sample_rate: int) -> bytes:
    """Decode *path* and return raw PCM s16le mono at *sample_rate* Hz.

    Tries ``soundfile`` first (fast, memory-efficient for WAV/FLAC).
    Falls back to ``pydub.AudioSegment`` for MP3, M4A, OGG, etc.
    """
    # Attempt soundfile read first
    try:
        data, sr = sf.read(path, always_2d=False)
    except Exception as exc:
        logger.debug("soundfile failed for %s: %s — trying pydub", path, exc)
        return _read_with_pydub(path, sample_rate)

    # Convert to mono if needed
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != sample_rate:
        data = _resample(data, sr, sample_rate)

    # Normalise & convert to int16
    return _to_int16(data)


def _read_with_pydub(path: str, sample_rate: int) -> bytes:
    """Decode audio via pydub (handles MP3, M4A, OGG, etc.)."""
    from pydub import AudioSegment  # type: ignore[import-untyped]

    seg = AudioSegment.from_file(path)
    seg = seg.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
    return seg.raw_data


def _resample(data, orig_sr: int, target_sr: int):
    """Simple linear-interpolation resample for 1-D arrays.

    For production use ``scipy.signal.resample`` or ``torchaudio``;
    this is a lightweight fallback that works with only numpy.
    """
    import numpy as np

    ratio = target_sr / orig_sr
    n_out = int(round(len(data) * ratio))
    indices = np.linspace(0, len(data) - 1, n_out)
    return np.interp(indices, np.arange(len(data)), data)


def _to_int16(data) -> bytes:
    """Normalise float audio to int16 PCM bytes."""
    import numpy as np

    data = np.asarray(data, dtype=np.float64)

    # Clamp to [-1.0, 1.0] then scale to int16 range
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak

    int_data = (data * 32767).astype(np.int16)
    return int_data.tobytes()


# ── Source implementation ─────────────────────────────────────


class FileSource(AudioSource):
    """Audio source that reads from a local file.

    Args:
        path: Path to an audio file (WAV, FLAC, MP3, M4A, OGG, …).
        config: Optional engine config (uses defaults if omitted).

    The file is decoded to **raw PCM 16-bit mono at 16 kHz** (or the
    configured sample rate) on first call to :meth:`read` or :meth:`stream`.

    Example::

        source = FileSource("meeting.mp3")
        audio = await source.read()
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        config: EngineConfig | None = None,
    ) -> None:
        self._path = os.fspath(path)
        self._config = config or EngineConfig()
        self._pcm_bytes: bytes | None = None  # cached after first decode

    # ── Properties ────────────────────────────────────────────

    @property
    def source_type(self) -> AudioSourceType:
        return AudioSourceType.FILE

    @property
    def path(self) -> str:
        """The original file path provided at construction time."""
        return self._path

    # ── AudioSource interface ─────────────────────────────────

    async def read(self) -> bytes:
        if self._pcm_bytes is None:
            self._pcm_bytes = await self._decode()
        return self._pcm_bytes

    async def stream(self) -> AsyncIterator[bytes]:
        pcm = await self.read()
        chunk_size = self._chunk_size_bytes()

        offset = 0
        while offset < len(pcm):
            yield pcm[offset : offset + chunk_size]
            offset += chunk_size

    async def close(self) -> None:
        """Release the decoded PCM buffer (if any)."""
        self._pcm_bytes = None

    # ── Internals ─────────────────────────────────────────────

    async def _decode(self) -> bytes:
        """Decode the file (runs in a thread to avoid blocking the event loop)."""
        import asyncio

        logger.info("Decoding audio file: %s", self._path)
        return await asyncio.to_thread(
            _read_audio, self._path, self._config.sample_rate
        )

    def _chunk_size_bytes(self) -> int:
        """Number of bytes per streaming chunk based on config."""
        return int(
            self._config.sample_rate
            * self._config.chunk_seconds
            * 2  # 2 bytes per sample (int16)
        )

    # ── repr / str ────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"FileSource(path={self._path!r})"
