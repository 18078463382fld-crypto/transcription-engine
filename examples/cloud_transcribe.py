#!/usr/bin/env python3
"""
cloud_transcribe.py — Transcribe an audio file using the OpenAI Whisper API.

Requires the OPENAI_API_KEY environment variable to be set.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/cloud_transcribe.py <path/to/audio.mp3>

Example:
    python examples/cloud_transcribe.py meeting.wav

Dependencies:
    pip install aiohttp
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from transcribe import TranscriptionEngine, EngineConfig
from transcribe.source.file_source import FileSource
from transcribe.transcriber import CloudAPITranscriber


async def main() -> None:
    # ── Check API key ────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY environment variable is not set.\n"
            "  export OPENAI_API_KEY=\"sk-...\"",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Parse arguments ──────────────────────────────────────────────
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio-file>")
        print("Example: python examples/cloud_transcribe.py speech.mp3")
        sys.exit(1)

    audio_path = sys.argv[1]

    # ── Set up the engine ────────────────────────────────────────────
    print(f"[*] Using OpenAI Whisper API (whisper-1)")
    print(f"[*] Transcribing: {audio_path}")

    transcriber = CloudAPITranscriber(
        model_name="whisper-1",
        api_key=api_key,
    )

    engine = TranscriptionEngine(
        transcriber=transcriber,
        config=EngineConfig.cloud_defaults(),  # timeout=120s, model=whisper-1
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
    print(f"  API time:   {elapsed:.2f}s")
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
