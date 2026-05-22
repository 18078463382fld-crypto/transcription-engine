#!/usr/bin/env python3
"""
basic_transcribe.py — File transcription with local Whisper.

Transcribes a single audio file using the local faster-whisper backend.

Usage:
    python examples/basic_transcribe.py <path/to/audio.mp3>

Example:
    python examples/basic_transcribe.py meeting.wav

Dependencies:
    pip install faster-whisper soundfile pydub
"""

from __future__ import annotations

import asyncio
import sys
import time

from transcribe import TranscriptionEngine, EngineConfig
from transcribe.source.file_source import FileSource
from transcribe.transcriber import LocalWhisperTranscriber


async def main() -> None:
    # ── Parse arguments ──────────────────────────────────────────────
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio-file>")
        print("Example: python examples/basic_transcribe.py meeting.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    # ── Set up the engine ────────────────────────────────────────────
    print(f"[*] Loading Whisper model (base, CPU) ...")
    print(f"[*] Transcribing: {audio_path}")

    transcriber = LocalWhisperTranscriber(model_name="base")

    engine = TranscriptionEngine(
        transcriber=transcriber,
        config=EngineConfig(
            device="cpu",
            compute_type="float32",
        ),
    )

    await engine.initialize()

    # ── Transcribe ───────────────────────────────────────────────────
    source = FileSource(audio_path)

    t0 = time.perf_counter()
    result = await engine.transcribe(source)
    elapsed = time.perf_counter() - t0

    # ── Print results ────────────────────────────────────────────────
    print()
    print("═" * 60)
    print("TRANSCRIPT")
    print("═" * 60)
    print(result.text)
    print("═" * 60)
    print(f"  Language:   {result.language}")
    print(f"  Segments:   {len(result.segments)}")
    print(f"  Duration:   {result.duration_seconds:.1f}s" if result.duration_seconds else "")
    print(f"  Time:       {elapsed:.2f}s")
    print(f"  Backend:    {result.backend}")
    print("═" * 60)

    # ── Print timed segments ─────────────────────────────────────────
    if result.segments:
        print()
        print("Timed segments:")
        for seg in result.segments:
            print(f"  [{seg.start:7.2f}s -> {seg.end:7.2f}s]  {seg.text}")

    # ── Cleanup ──────────────────────────────────────────────────────
    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
