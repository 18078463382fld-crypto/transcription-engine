#!/usr/bin/env python3
"""
real_time.py — Microphone streaming transcription.

Captures audio from the default microphone and transcribes it in
real-time, printing partial results as they are recognised.

Press Ctrl+C to stop cleanly.

Usage:
    python examples/real_time.py

Dependencies:
    pip install faster-whisper sounddevice
"""

from __future__ import annotations

import asyncio
import signal
import sys

from transcribe import TranscriptionEngine, EngineConfig
from transcribe.source.microphone_source import MicrophoneSource
from transcribe.transcriber import LocalWhisperTranscriber


async def main() -> None:
    # ── Set up the engine ────────────────────────────────────────────
    print("[*] Loading Whisper model (tiny, CPU) ...")
    print("[*] Opening microphone ...")
    print("[*] Speak into your microphone.  Press Ctrl+C to stop.")
    print()

    transcriber = LocalWhisperTranscriber(model_name="tiny")

    engine = TranscriptionEngine(
        transcriber=transcriber,
        config=EngineConfig(
            device="cpu",
            compute_type="float32",
        ),
    )

    await engine.initialize()

    # ── Streaming transcription ──────────────────────────────────────
    source = MicrophoneSource()

    # Register signal handler for graceful shutdown
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        print("\n[!] Stopping microphone ...")
        stop_event.set()
        source.stop()

    # Hook SIGINT (Ctrl+C) and SIGTERM
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows: signals not fully supported on ProactorEventLoop
            pass

    try:
        async for result in engine.transcribe_stream(source):
            # Print each partial segment
            for seg in result.segments:
                print(f"  [{seg.start:6.2f}s] {seg.text}")

            if stop_event.is_set():
                break

    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
    finally:
        # ── Cleanup ──────────────────────────────────────────────────
        await engine.shutdown()
        print("[*] Engine shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] Exiting.")
        sys.exit(0)
