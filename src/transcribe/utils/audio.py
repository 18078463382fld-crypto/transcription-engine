"""
Audio utility functions and the ``AudioConverter`` class.

All operations work on **raw PCM audio data** using only the Python
standard library (``struct``, ``array``, ``math``).  No third-party
dependencies are required.

Typical usage::

    from transcribe.utils.audio import AudioConverter

    converter = AudioConverter()

    # Convert float32 samples to int16 PCM bytes
    pcm = converter.to_pcm(float_samples, source_sample_width=4)

    # Resample from 48 kHz to 16 kHz
    resampled = converter.resample(pcm, orig_sr=48000, target_sr=16000)

    # Combined: normalise, mono downmix, resample, and convert to int16
    result = converter.convert(
        raw_bytes,
        source_sr=48000,
        target_sr=16000,
        source_channels=2,
        source_sample_width=2,  # int16
    )
"""

from __future__ import annotations

import array
import math
import struct
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

_INT16_MIN = -(2**15)
_INT16_MAX = 2**15 - 1
_INT24_MIN = -(2**23)
_INT24_MAX = 2**23 - 1
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1

# Default target format
_TARGET_SAMPLE_WIDTH = 2  # int16
_TARGET_SAMPLE_RATE = 16000
_TARGET_CHANNELS = 1  # mono


# ═══════════════════════════════════════════════════════════════
# Public helpers
# ═══════════════════════════════════════════════════════════════


def resample(
    data: bytes,
    orig_sr: int,
    target_sr: int,
    sample_width: int = 2,
) -> bytes:
    """Resample raw PCM audio using linear interpolation.

    Parameters
    ----------
    data:
        Raw PCM audio data.
    orig_sr:
        Original sample rate in Hz.
    target_sr:
        Desired sample rate in Hz.
    sample_width:
        Bytes per sample (1, 2, 3, or 4).  Default 2 (int16).

    Returns
    -------
        Resampled PCM audio data in the same sample format.

    Notes
    -----
    This is a **simple linear interpolation** resampler.  For
    production-quality resampling use ``scipy.signal.resample`` or
    ``torchaudio``; this is a lightweight stdlib-only fallback that
    works well for voice/transcription pipelines.
    """
    if orig_sr == target_sr:
        return data

    if not data:
        return b""

    samples = _unpack_samples(data, sample_width)
    n_in = len(samples)
    if n_in == 0:
        return b""

    ratio = target_sr / orig_sr
    n_out = max(1, int(round(n_in * ratio)))

    out_samples: list[float] = [0.0] * n_out

    for i in range(n_out):
        src_idx = i / ratio  # float index in source array
        lo = int(math.floor(src_idx))
        hi = min(lo + 1, n_in - 1)
        frac = src_idx - lo

        # Linear interpolation
        out_samples[i] = samples[lo] + (samples[hi] - samples[lo]) * frac

    return _pack_samples(out_samples, sample_width)


def to_pcm(
    data: bytes,
    *,
    source_sample_width: int = 2,
    source_channels: int = 1,
    source_signed: bool = True,
    source_endian: str = "little",
    normalize: bool = True,
    target_sample_width: int = 2,
    target_channels: int = 1,
    target_signed: bool = True,
    target_endian: str = "little",
) -> bytes:
    """Convert raw audio data to a target PCM format.

    Supports conversion between bit depths (8, 16, 24, 32 bit),
    channel counts (stereo → mono downmix), signed/unsigned, and
    endianness.

    Parameters
    ----------
    data:
        Raw audio data in the source format.
    source_sample_width:
        Bytes per source sample (1, 2, 3, or 4).  Default 2.
    source_channels:
        Number of source channels.  Default 1 (mono).
    source_signed:
        Whether source samples are signed.  Default ``True``.
        Only relevant for ``source_sample_width == 1`` (8-bit PCM
        is typically unsigned).
    source_endian:
        ``"little"`` or ``"big"``.  Default ``"little"``.
        Only relevant for widths >= 2.
    normalize:
        If ``True`` (default), normalise the signal to [-1.0, 1.0]
        before converting to the target format, avoiding clipping.
    target_sample_width:
        Bytes per target sample.  Default 2 (int16).
    target_channels:
        Number of target channels.  Default 1 (mono).
    target_signed:
        Whether target samples should be signed.  Default ``True``.
    target_endian:
        Target byte order.  Default ``"little"``.

    Returns
    -------
        PCM audio data in the target format as ``bytes``.
    """
    if not data:
        return b""

    # 1. Unpack to float64 samples (interleaved channels)
    samples = _unpack_samples(
        data, source_sample_width, signed=source_signed, endian=source_endian
    )

    # 2. Downmix to mono if needed
    if source_channels > 1 and target_channels == 1:
        samples = _downmix_to_mono(samples, source_channels)
    elif source_channels != target_channels:
        # If we can't sensibly convert, just take first channel
        samples = samples[::source_channels]

    # 3. Normalise (optional)
    if normalize and samples:
        peak = max(abs(s) for s in samples)
        if peak > 0:
            samples = [s / peak for s in samples]

    # 4. Pack to target format
    return _pack_samples(
        samples, target_sample_width, signed=target_signed, endian=target_endian
    )


# ═══════════════════════════════════════════════════════════════
# AudioConverter class
# ═══════════════════════════════════════════════════════════════


class AudioConverter:
    """High-level audio converter for transcription pipeline use.

    Wraps :func:`to_pcm` and :func:`resample` into a convenient
    single-call interface.  The default output format is **16-bit
    signed mono little-endian PCM at 16 kHz** — the standard format
    expected by Whisper and most speech recognition engines.

    Parameters
    ----------
    target_sr:
        Target sample rate in Hz (default 16000).
    target_sample_width:
        Bytes per output sample (default 2 for int16).
    target_channels:
        Number of output channels (default 1 for mono).
    """

    def __init__(
        self,
        target_sr: int = _TARGET_SAMPLE_RATE,
        target_sample_width: int = _TARGET_SAMPLE_WIDTH,
        target_channels: int = _TARGET_CHANNELS,
    ) -> None:
        if target_sr <= 0:
            raise ValueError(f"target_sr must be positive, got {target_sr}")
        if target_sample_width not in (1, 2, 3, 4):
            raise ValueError(
                f"target_sample_width must be 1, 2, 3, or 4, got {target_sample_width}"
            )
        if target_channels < 1:
            raise ValueError(
                f"target_channels must be >= 1, got {target_channels}"
            )

        self._target_sr = target_sr
        self._target_sample_width = target_sample_width
        self._target_channels = target_channels

    # ── Properties ────────────────────────────────────────────

    @property
    def target_sr(self) -> int:
        """Target sample rate in Hz."""
        return self._target_sr

    @property
    def target_sample_width(self) -> int:
        """Bytes per target sample."""
        return self._target_sample_width

    @property
    def target_channels(self) -> int:
        """Number of target channels."""
        return self._target_channels

    # ── Conversion methods ────────────────────────────────────

    def convert(
        self,
        data: bytes,
        source_sr: int,
        *,
        source_sample_width: int = 2,
        source_channels: int = 1,
        source_signed: bool = True,
        source_endian: str = "little",
        normalize: bool = True,
    ) -> bytes:
        """Convert raw audio to the configured target format in one step.

        This performs (in order):
        1. Format conversion (:meth:`to_pcm`) — downmix, signed/unsigned,
           bit depth change, normalisation.
        2. Resampling (:meth:`resample`) — if sample rates differ.

        Parameters
        ----------
        data:
            Raw audio data in the source format.
        source_sr:
            Sample rate of the source audio (Hz).
        source_sample_width:
            Bytes per source sample (default 2).
        source_channels:
            Number of source channels (default 1).
        source_signed:
            Whether source samples are signed (default ``True``).
        source_endian:
            Source endianness (default ``"little"``).
        normalize:
            Normalise before conversion (default ``True``).

        Returns
        -------
            PCM audio in the configured target format.
        """
        if not data:
            return b""

        # Step 1: format conversion (mono downmix, bit depth, signed, etc.)
        converted = self.to_pcm(
            data,
            source_sample_width=source_sample_width,
            source_channels=source_channels,
            source_signed=source_signed,
            source_endian=source_endian,
            normalize=normalize,
        )

        # Step 2: resample if needed
        if source_sr != self._target_sr:
            converted = self.resample(
                converted, source_sr, self._target_sr
            )

        return converted

    def to_pcm(
        self,
        data: bytes,
        *,
        source_sample_width: int = 2,
        source_channels: int = 1,
        source_signed: bool = True,
        source_endian: str = "little",
        normalize: bool = True,
    ) -> bytes:
        """Convert raw audio to the configured target PCM format.

        See :func:`to_pcm` for full parameter documentation.
        """
        return to_pcm(
            data,
            source_sample_width=source_sample_width,
            source_channels=source_channels,
            source_signed=source_signed,
            source_endian=source_endian,
            normalize=normalize,
            target_sample_width=self._target_sample_width,
            target_channels=self._target_channels,
            target_signed=True,
            target_endian="little",
        )

    def resample(
        self,
        data: bytes,
        orig_sr: int,
        target_sr: int,
    ) -> bytes:
        """Resample raw PCM audio.

        See :func:`resample` for full documentation.
        """
        return resample(data, orig_sr, target_sr, self._target_sample_width)

    def estimate_duration(self, data: bytes) -> float:
        """Estimate audio duration in seconds from PCM byte count.

        Uses the converter's target sample rate and sample width.
        For single-channel PCM::

            duration = len(data) / (sample_rate * sample_width)
        """
        if not data:
            return 0.0
        bytes_per_second = self._target_sr * self._target_sample_width * self._target_channels
        if bytes_per_second == 0:
            return 0.0
        return len(data) / bytes_per_second

    # ── repr ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AudioConverter("
            f"sr={self._target_sr}, "
            f"width={self._target_sample_width}, "
            f"channels={self._target_channels})"
        )


# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════


def _unpack_samples(
    data: bytes,
    sample_width: int,
    signed: bool = True,
    endian: str = "little",
) -> list[float]:
    """Unpack raw PCM bytes into a ``list[float]`` in [-1.0, 1.0] range.

    Parameters
    ----------
    data:
        Raw PCM bytes.
    sample_width:
        Bytes per sample (1, 2, 3, or 4).
    signed:
        Whether samples are signed (default ``True``).
        Only relevant for width == 1 (8-bit PCM is typically unsigned).
    endian:
        ``"little"`` or ``"big"``.  Default ``"little"``.

    Returns
    -------
        Normalised float samples in range [-1.0, 1.0).
    """
    n = len(data)
    if n == 0:
        return []

    if sample_width == 1:
        # 8-bit: typically unsigned [0, 255], centre at 128
        fmt = f"{n}B"
        raw = struct.unpack(fmt, data[:n])
        if signed:
            return [(b - 128) / 128.0 for b in raw]
        else:
            return [b / 255.0 for b in raw]

    elif sample_width == 2:
        # 16-bit signed little-endian
        endian_char = "<" if endian == "little" else ">"
        count = n // sample_width
        fmt = f"{endian_char}{count}h"
        raw = struct.unpack(fmt, data[: count * sample_width])
        return [v / 32768.0 for v in raw]

    elif sample_width == 3:
        # 24-bit signed (3 bytes per sample, no native struct format)
        endian_char = "<" if endian == "little" else ">"
        count = n // 3
        result: list[float] = []
        for i in range(count):
            offset = i * 3
            chunk = data[offset : offset + 3]
            if len(chunk) < 3:
                break
            # Reconstruct signed 24-bit integer
            if endian == "little":
                val = chunk[0] | (chunk[1] << 8) | (chunk[2] << 16)
            else:
                val = chunk[2] | (chunk[1] << 8) | (chunk[0] << 16)
            # Sign-extend
            if val & 0x800000:
                val -= 0x1000000
            result.append(val / 8388608.0)  # 2^23
        return result

    elif sample_width == 4:
        # 32-bit signed little-endian
        endian_char = "<" if endian == "little" else ">"
        count = n // sample_width
        fmt = f"{endian_char}{count}i"
        raw = struct.unpack(fmt, data[: count * sample_width])
        return [v / 2147483648.0 for v in raw]

    else:
        raise ValueError(f"Unsupported sample_width: {sample_width}")


def _pack_samples(
    samples: list[float],
    sample_width: int,
    signed: bool = True,
    endian: str = "little",
) -> bytes:
    """Pack a list of float samples into raw PCM bytes.

    Parameters
    ----------
    samples:
        Float samples in [-1.0, 1.0] range.
    sample_width:
        Bytes per output sample (1, 2, 3, or 4).
    signed:
        Whether output should be signed (default ``True``).
        Only relevant for width == 1.
    endian:
        ``"little"`` or ``"big"``.  Default ``"little"``.

    Returns
    -------
        Raw PCM bytes.
    """
    if not samples:
        return b""

    if sample_width == 1:
        # 8-bit
        if signed:
            raw = [max(0, min(255, int(round((s * 128.0) + 128)))) for s in samples]
        else:
            raw = [max(0, min(255, int(round(s * 255.0)))) for s in samples]
        return struct.pack(f"{len(raw)}B", *raw)

    elif sample_width == 2:
        # 16-bit signed little-endian
        endian_char = "<" if endian == "little" else ">"
        raw = [max(_INT16_MIN, min(_INT16_MAX, int(round(s * 32767)))) for s in samples]
        fmt = f"{endian_char}{len(raw)}h"
        return struct.pack(fmt, *raw)

    elif sample_width == 3:
        # 24-bit signed (3 bytes per sample)
        endian_char = "<" if endian == "little" else ">"
        buf = bytearray()
        for s in samples:
            val = max(_INT24_MIN, min(_INT24_MAX, int(round(s * 8388607))))
            # Mask to 24 bits and pack
            val = val & 0xFFFFFF
            if endian == "little":
                buf.extend([val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF])
            else:
                buf.extend([(val >> 16) & 0xFF, (val >> 8) & 0xFF, val & 0xFF])
        return bytes(buf)

    elif sample_width == 4:
        # 32-bit signed little-endian
        endian_char = "<" if endian == "little" else ">"
        raw = [max(_INT32_MIN, min(_INT32_MAX, int(round(s * 2147483647)))) for s in samples]
        fmt = f"{endian_char}{len(raw)}i"
        return struct.pack(fmt, *raw)

    else:
        raise ValueError(f"Unsupported sample_width: {sample_width}")


def _downmix_to_mono(samples: list[float], channels: int) -> list[float]:
    """Downmix interleaved multi-channel samples to mono by averaging.

    Parameters
    ----------
    samples:
        Interleaved float samples.
    channels:
        Number of interleaved channels.

    Returns
    -------
        Mono float samples.
    """
    if channels <= 1:
        return samples

    n_frames = len(samples) // channels
    mono: list[float] = [0.0] * n_frames

    for i in range(n_frames):
        frame_sum = 0.0
        for ch in range(channels):
            frame_sum += samples[i * channels + ch]
        mono[i] = frame_sum / channels

    return mono
