"""Audio source backed by a live microphone stream.

Captures audio from the default input device and provides it to the
engine in both batch (:meth:`read`) and streaming (:meth:`stream`) modes.

Uses ``sounddevice`` (optional dependency, see ``[stream]`` extra).

.. note::
    The ``read()`` method captures until :meth:`stop` is called or a
    timeout is reached; ``stream()`` yields chunks indefinitely until
    :meth:`close` is invoked.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

from ..core.interfaces import AudioSource
from ..core.models import AudioSourceType, EngineConfig

logger = logging.getLogger("transcribe.source.microphone")

# ── Constants ─────────────────────────────────────────────────

_BYTES_PER_SAMPLE = 2  # int16


# ── Source implementation ─────────────────────────────────────


class MicrophoneSource(AudioSource):
    """Audio source that captures from the default microphone.

    Args:
        config:        Engine config providing ``sample_rate`` and ``chunk_seconds``.
        device:        Optional sounddevice device index or substring.
                       ``None`` uses the default input device.
        blocksize:     Number of frames per audio callback block.
                       Defaults to ``chunk_seconds * sample_rate``.

    Example::

        source = MicrophoneSource()
        async for result in engine.transcribe_stream(source):
            print(result.text)
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        device: int | str | None = None,
        blocksize: int | None = None,
    ) -> None:
        self._config = config or EngineConfig()
        self._device = device
        self._blocksize = blocksize or int(
            self._config.sample_rate * self._config.chunk_seconds
        )

        # Thread-safe buffer — callbacks push, async tasks pull
        self._buffer: asyncio.Queue[bytes] = asyncio.Queue()
        self._stop_event = asyncio.Event()

        # Sounddevice state (lazily initialised)
        self._input_stream: object | None = None  # sd.InputStream
        self._stream_thread: threading.Thread | None = None
        self._exception: Exception | None = None

    # ── Properties ────────────────────────────────────────────

    @property
    def source_type(self) -> AudioSourceType:
        return AudioSourceType.MICROPHONE

    # ── AudioSource interface ─────────────────────────────────

    async def read(self) -> bytes:
        """Capture all microphone input until :meth:`stop` is called.

        If no explicit stop is signalled this will block indefinitely.
        Typical usage is to call ``read()`` concurrently with a timeout
        or a UI trigger that calls :meth:`stop`.
        """
        chunks: list[bytes] = []
        stop_task = asyncio.create_task(self._stop_event.wait())

        await self._start_stream()

        try:
            while True:
                # Wait for either a chunk or the stop signal
                get_task = asyncio.create_task(self._buffer.get())

                done, _ = await asyncio.wait(
                    [get_task, stop_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if stop_task in done:
                    # Stop requested — drain remaining
                    get_task.cancel()
                    break

                chunk = get_task.result()
                chunks.append(chunk)

                # Re-create stop_task for next iteration
                stop_task = asyncio.create_task(self._stop_event.wait())
        finally:
            stop_task.cancel()

        await self.close()
        return b"".join(chunks)

    async def stream(self) -> AsyncIterator[bytes]:
        """Yield chunks of microphone audio in real time.

        Iteration continues until :meth:`close` is called or the
        underlying stream encounters an error.
        """
        await self._start_stream()

        while not self._stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(
                    self._buffer.get(), timeout=0.1
                )
                yield chunk
            except asyncio.TimeoutError:
                # Timeout allows us to check _stop_event periodically
                continue
            except asyncio.CancelledError:
                break

        # Drain any remaining chunks
        while not self._buffer.empty():
            try:
                yield self._buffer.get_nowait()
            except asyncio.QueueEmpty:
                break

        await self.close()

    async def close(self) -> None:
        """Stop the microphone stream and release resources."""
        self._stop_event.set()
        await self._stop_stream()

    # ── Control ───────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the microphone to stop capturing.

        This is a **thread-safe** synchronous call — safe to invoke
        from UI callbacks, signal handlers, etc.
        """
        self._stop_event.set()

    # ── Internal stream management ────────────────────────────

    async def _start_stream(self) -> None:
        """Start the sounddevice InputStream in a background thread."""
        if self._input_stream is not None:
            return  # already started

        import sounddevice as sd  # type: ignore[import-untyped]

        self._stop_event.clear()
        self._buffer = asyncio.Queue()
        self._exception = None

        # Callback: runs in sounddevice's audio thread
        def _audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning("Audio callback status: %s", status)
            # indata shape: (frames, channels) — take mono (channel 0)
            chunk = indata[:, 0].tobytes()
            # Schedule the chunk onto the async queue via the event loop
            if self._loop is not None and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._buffer.put(chunk), self._loop
                )

        self._loop = asyncio.get_running_loop()

        self._input_stream = sd.InputStream(
            device=self._device,
            samplerate=self._config.sample_rate,
            blocksize=self._blocksize,
            channels=1,
            dtype="int16",
            callback=_audio_callback,
        )

        # Open in a thread (sd.InputStream.open is blocking)
        def _open():
            try:
                self._input_stream.open()  # type: ignore[union-attr]
            except Exception as exc:
                self._exception = exc
                self._stop_event.set()

        self._stream_thread = threading.Thread(target=_open, daemon=True)
        self._stream_thread.start()

        # Wait briefly for the stream to open
        await asyncio.sleep(0.05)
        if self._exception:
            raise RuntimeError(
                f"Failed to open microphone stream: {self._exception}"
            ) from self._exception

        logger.info(
            "Microphone stream started (%d Hz, %d frames/block)",
            self._config.sample_rate,
            self._blocksize,
        )

    async def _stop_stream(self) -> None:
        """Close the sounddevice stream and join the thread."""
        if self._input_stream is None:
            return

        stream: sd.InputStream = self._input_stream  # type: ignore[assignment]
        self._input_stream = None

        try:
            # Close in a thread since it may block
            await asyncio.to_thread(stream.close)
        except Exception as exc:
            logger.warning("Error closing microphone stream: %s", exc)

        if self._stream_thread is not None and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2)
            self._stream_thread = None

        logger.info("Microphone stream stopped")

    # ── repr / str ────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MicrophoneSource("
            f"device={self._device!r}, "
            f"rate={self._config.sample_rate})"
        )
