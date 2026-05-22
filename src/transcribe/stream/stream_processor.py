"""
Voice Activity Detection and buffering stream handlers.

This module provides two concrete ``StreamHandler`` implementations
designed to be composed in a real-time transcription pipeline:

1. :class:`VADStreamHandler` — filters out non-speech (silence) chunks.
2. :class:`BufferingStreamHandler` — accumulates audio chunks and emits
   them when a flush condition is reached (duration, silence gap, etc.).

Typical pipeline usage::

    handlers: list[StreamHandler] = [
        VADStreamHandler(threshold=0.025),
        BufferingStreamHandler(min_chunk_seconds=2.0, max_chunk_seconds=30.0),
    ]
    engine = TranscriptionEngine(transcriber=..., stream_handlers=handlers)

    async for segment in engine.transcribe_stream(source):
        print(segment.text)
"""

from __future__ import annotations

import math
import struct
from typing import Callable, Optional

from ..core.interfaces import StreamHandler

# ── Public API ──────────────────────────────────────────────────────────────

__all__ = [
    "VADStreamHandler",
    "BufferingStreamHandler",
]

# ═══════════════════════════════════════════════════════════════════════════════
# VADStreamHandler
# ═══════════════════════════════════════════════════════════════════════════════


class VADStreamHandler(StreamHandler):
    """Voice Activity Detection stream handler.

    Examines each incoming audio chunk and determines whether it contains
    speech. Non-speech (silence) chunks are dropped by returning ``b""``;
    speech chunks are passed through unchanged.

    By default a lightweight **energy-based** (RMS) detector is used, which
    works well for clean, close-mic audio.  For noisy environments or
    higher accuracy, pass an external ``detector`` callable (e.g. wrapping
    `webrtcvad <https://github.com/wiseman/py-webrtcvad>`_ or
    `Silero VAD <https://github.com/snakers4/silero-vad>`_).

    Parameters
    ----------
    threshold : float
        RMS energy threshold above which a chunk is considered speech.
        Only used when no external ``detector`` is supplied.  Default 0.02.
    sample_rate : int
        Audio sample rate in Hz (default 16000).
    sample_width : int
        Bytes per sample (default 2 for 16-bit PCM).
    frame_duration_ms : int
        Frame duration in milliseconds used for sub-frame analysis
        (default 30).  Only relevant for the built-in energy detector.
    detector : Callable[[bytes], bool] | None
        Optional external VAD callable.  Receives a raw PCM chunk and
        returns ``True`` if speech is present.  When set, ``threshold``
        is ignored.
    """

    def __init__(
        self,
        threshold: float = 0.02,
        sample_rate: int = 16000,
        sample_width: int = 2,
        frame_duration_ms: int = 30,
        detector: Callable[[bytes], bool] | None = None,
    ) -> None:
        if threshold <= 0:
            raise ValueError("VAD threshold must be positive")
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"Unsupported sample rate: {sample_rate}")
        if sample_width not in (1, 2, 4):
            raise ValueError(f"Unsupported sample width: {sample_width}")

        self._threshold = threshold
        self._sample_rate = sample_rate
        self._sample_width = sample_width
        self._frame_duration_ms = frame_duration_ms
        self._detector = detector

        # Internal state
        self._frame_size = (
            sample_rate * sample_width * frame_duration_ms // 1000
        )
        self._silence_frames: int = 0  # consecutive silence frame count

    # ── State ──────────────────────────────────────────────────────────

    @property
    def threshold(self) -> float:
        """RMS energy threshold for voice detection."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if value <= 0:
            raise ValueError("VAD threshold must be positive")
        self._threshold = value

    @property
    def silence_frame_count(self) -> int:
        """Number of consecutive silent frames since last speech."""
        return self._silence_frames

    # ── StreamHandler interface ────────────────────────────────────────

    async def process(self, chunk: bytes) -> bytes:
        """Return *chunk* if it contains speech, else ``b""``."""
        if not chunk:
            return b""

        if self._detector is not None:
            is_speech = self._detector(chunk)
        else:
            is_speech = self._energy_detector(chunk)

        if is_speech:
            self._silence_frames = 0
            return chunk

        self._silence_frames += max(
            1, math.ceil(len(chunk) / self._frame_size)
        )
        return b""

    async def reset(self) -> None:
        """Reset the silence counter."""
        self._silence_frames = 0

    # ── Internal helpers ───────────────────────────────────────────────

    def _energy_detector(self, chunk: bytes) -> bool:
        """Built-in RMS-energy-based voice detector.

        Operates on sub-frames of ``frame_duration_ms`` so that a single
        chunk containing both speech and silence is handled correctly.
        The chunk is considered speech if **any** sub-frame exceeds the
        threshold.
        """
        if len(chunk) < self._sample_width:
            return False

        # Normalise to 16-bit signed integers for RMS calculation
        offset = 0
        n_frames = max(1, len(chunk) // self._frame_size)

        for _ in range(n_frames):
            frame = chunk[offset : offset + self._frame_size]
            offset += self._frame_size

            if len(frame) < self._sample_width:
                continue

            rms = self._rms(frame)
            if rms >= self._threshold:
                return True

        return False

    @staticmethod
    def _rms(data: bytes, sample_width: int = 2) -> float:
        """Compute the RMS (root-mean-square) energy of a PCM frame.

        Normalised to a 0.0–1.0 range suitable for threshold comparison.
        """
        n = len(data) // sample_width
        if n == 0:
            return 0.0

        if sample_width == 1:
            # unsigned 8-bit
            fmt = f"{n}B"
            values = [(b - 128) / 128.0 for b in struct.unpack(fmt, data[:n])]
        elif sample_width == 2:
            # signed 16-bit little-endian
            fmt = f"<{n}h"
            values = [v / 32768.0 for v in struct.unpack(fmt, data[: n * 2])]
        elif sample_width == 4:
            # signed 32-bit little-endian
            fmt = f"<{n}i"
            values = [v / 2147483648.0 for v in struct.unpack(fmt, data[: n * 4])]
        else:
            return 0.0

        if not values:
            return 0.0

        mean_sq = sum(v * v for v in values) / n
        return math.sqrt(mean_sq)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"threshold={self._threshold:.4f}, "
            f"sr={self._sample_rate}, "
            f"detector={'custom' if self._detector else 'energy'})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# BufferingStreamHandler
# ═══════════════════════════════════════════════════════════════════════════════


class BufferingStreamHandler(StreamHandler):
    """Audio chunk accumulator with configurable flush conditions.

    Collects incoming audio chunks into an internal buffer and emits the
    accumulated buffer when a flush condition is met.  Between flushes the
    handler returns ``b""`` so the pipeline does not forward incomplete
    utterances to the transcriber.

    Flush conditions (any one triggers a flush):

    * **Max duration** — ``max_chunk_seconds`` of audio accumulated.
      This is the safety valve that prevents unbounded memory growth.
    * **Silence gap** — ``silence_threshold_seconds`` of consecutive
      silence in the incoming audio (only effective when the pipeline
      has *not* already dropped silence, i.e. when placed *before*
      a :class:`VADStreamHandler` in the handler chain, or when the
      buffering handler detects silence itself via an internal VAD).
    * **Min duration** (on flush) — the handler never emits a buffer
      smaller than ``min_chunk_seconds`` unless a force method is used.

    Parameters
    ----------
    min_chunk_seconds : float
        Minimum audio duration (seconds) before a flush is allowed.
        Default 2.0.
    max_chunk_seconds : float
        Maximum audio duration (seconds).  When reached the buffer is
        force-flushed regardless of other conditions.  Default 30.0.
    silence_threshold_seconds : float
        Duration of consecutive silence (seconds) that triggers an
        early flush at a natural utterance boundary.  Set to 0 to
        disable silence-based flushing.  Default 0.8.
    sample_rate : int
        Audio sample rate in Hz (default 16000).
    sample_width : int
        Bytes per sample (default 2 for 16-bit PCM).
    """

    def __init__(
        self,
        min_chunk_seconds: float = 2.0,
        max_chunk_seconds: float = 30.0,
        silence_threshold_seconds: float = 0.8,
        sample_rate: int = 16000,
        sample_width: int = 2,
    ) -> None:
        if min_chunk_seconds <= 0:
            raise ValueError("min_chunk_seconds must be positive")
        if max_chunk_seconds <= min_chunk_seconds:
            raise ValueError(
                "max_chunk_seconds must be greater than min_chunk_seconds"
            )

        self._min_chunk_seconds = min_chunk_seconds
        self._max_chunk_seconds = max_chunk_seconds
        self._silence_threshold_seconds = silence_threshold_seconds
        self._sample_rate = sample_rate
        self._sample_width = sample_width

        # Derived: bytes-per-second for duration calculations
        self._bytes_per_second = sample_rate * sample_width
        self._min_bytes = int(self._bytes_per_second * min_chunk_seconds)
        self._max_bytes = int(self._bytes_per_second * max_chunk_seconds)
        self._silence_bytes_threshold = int(
            self._bytes_per_second * silence_threshold_seconds
        )

        # Internal state
        self._buffer = bytearray()
        self._consecutive_silence_bytes: int = 0

    # ── Properties ────────────────────────────────────────────────────

    @property
    def buffer_duration(self) -> float:
        """Duration of audio currently in the buffer (seconds)."""
        return len(self._buffer) / self._bytes_per_second

    @property
    def buffered_bytes(self) -> int:
        """Number of bytes currently buffered."""
        return len(self._buffer)

    @property
    def is_flush_pending(self) -> bool:
        """``True`` if the accumulated buffer meets the flush threshold."""
        return self.buffer_duration >= self._min_chunk_seconds

    # ── StreamHandler interface ───────────────────────────────────────

    async def process(self, chunk: bytes) -> bytes:
        """Accumulate *chunk* and return ``b""`` until a flush condition
        is met, at which point the accumulated buffer is returned."""
        if not chunk:
            return b""

        # Accumulate into the buffer
        self._buffer.extend(chunk)

        # Track consecutive silence (energy-based) for early flush
        if self._silence_threshold_seconds > 0:
            if self._is_silence(chunk):
                self._consecutive_silence_bytes += len(chunk)
            else:
                self._consecutive_silence_bytes = 0

        # Check flush conditions
        buffer_seconds = len(self._buffer) / self._bytes_per_second

        # Condition 1: max duration reached (force flush)
        if buffer_seconds >= self._max_chunk_seconds:
            return self._flush()

        # Condition 2: silence gap exceeds threshold (early flush)
        if (
            self._silence_threshold_seconds > 0
            and self._consecutive_silence_bytes >= self._silence_bytes_threshold
            and buffer_seconds >= self._min_chunk_seconds
        ):
            return self._flush()

        return b""

    async def reset(self) -> None:
        """Discard the current buffer and reset silence tracking."""
        self._buffer.clear()
        self._consecutive_silence_bytes = 0

    # ── Public helpers ────────────────────────────────────────────────

    async def flush(self) -> bytes:
        """Explicitly request a flush of the current buffer.

        Returns the buffered audio (may be empty if nothing is buffered)
        and resets internal state.  Useful for signalling end-of-utterance
        from outside the hot path.
        """
        return self._flush()

    # ── Internal helpers ──────────────────────────────────────────────

    def _flush(self) -> bytes:
        """Emit the current buffer and reset."""
        if not self._buffer:
            return b""

        emitted = bytes(self._buffer)
        self._buffer.clear()
        self._consecutive_silence_bytes = 0
        return emitted

    def _is_silence(self, chunk: bytes) -> bool:
        """Quick energy-based check for silence detection.

        Uses a conservative default threshold (0.01) so that only
        very quiet audio is classified as silence for buffering
        boundary purposes.
        """
        if len(chunk) < self._sample_width:
            return True
        rms = VADStreamHandler._rms(chunk, self._sample_width)
        return rms < 0.01

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"min={self._min_chunk_seconds}s, "
            f"max={self._max_chunk_seconds}s, "
            f"silence={self._silence_threshold_seconds}s)"
        )
