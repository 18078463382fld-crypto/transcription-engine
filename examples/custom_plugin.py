#!/usr/bin/env python3
"""
custom_plugin.py — Demonstrate the plugin system with a timestamp upper-case plugin.

This example defines a custom ``TimestampUppercasePlugin`` that:
  1. Leaves the audio unchanged (pre_process is a no-op).
  2. Uppercases every segment's text and prepends a "[UPPERCASED]" tag
     in the post_process hook — demonstrating how plugins can enrich or
     transform transcript results after transcription.

Usage:
    python examples/custom_plugin.py <path/to/audio.mp3>

Example:
    python examples/custom_plugin.py greeting.wav
"""

from __future__ import annotations

import asyncio
import sys
import time

from transcribe import (
    TranscriptionEngine,
    EngineConfig,
    TranscriptResult,
)
from transcribe.plugins.base import AbstractPlugin
from transcribe.source.file_source import FileSource
from transcribe.transcriber import LocalWhisperTranscriber


# ═══════════════════════════════════════════════════════════════════
# Custom plugin
# ═══════════════════════════════════════════════════════════════════


class TimestampUppercasePlugin(AbstractPlugin):
    """
    A post-processing plugin that uppercases all transcribed text and
    prepends a timestamp summary to the full transcript.

    This demonstrates the plugin hook system — no audio modification,
    but the result is enriched after transcription.
    """

    @property
    def name(self) -> str:
        return "timestamp-uppercase"

    # Run after most other plugins (higher number = later execution)
    priority: int = 200
    enabled: bool = True

    async def setup(
        self, config: EngineConfig, plugin_config: dict | None = None
    ) -> None:
        """Log that we're active."""
        print(f"[plugin:{self.name}] Plugin loaded and active.")
        await super().setup(config, plugin_config)

    async def pre_process(self, audio: bytes) -> bytes:
        """
        Pre-processing: pass audio through unchanged.
        (Could apply here e.g. gain adjustment, noise gate, etc.)
        """
        return audio

    async def post_process(
        self, result: TranscriptResult
    ) -> TranscriptResult:
        """
        Post-processing: uppercase all segment text and add a header.

        1. Uppercase every segment's text.
        2. Add a "[UPPERCASED]" tag to the full text.
        3. Stamp metadata so downstream consumers know it was modified.
        """
        # Uppercase each segment
        for seg in result.segments:
            seg.text = seg.text.upper()

        # Rebuild full text from uppercased segments
        uppercased_parts = [
            seg.text for seg in result.segments if seg.text.strip()
        ]
        result.text = "[UPPERCASED] " + " ".join(uppercased_parts)

        # Tag metadata
        result.metadata["plugin_applied"] = self.name
        result.metadata["uppercased"] = True

        return result

    async def teardown(self) -> None:
        print(f"[plugin:{self.name}] Plugin torn down.")
        await super().teardown()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


async def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio-file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    # ── Create the plugin instance ───────────────────────────────────
    uppercaser = TimestampUppercasePlugin()

    # ── Set up the engine WITH plugin ────────────────────────────────
    print(f"[*] Loading Whisper model (base, CPU) ...")
    print(f"[*] Plugin: {uppercaser.name} (priority={uppercaser.priority})")

    engine = TranscriptionEngine(
        transcriber=LocalWhisperTranscriber(model_name="base"),
        config=EngineConfig(device="cpu", compute_type="float32"),
        plugins=[uppercaser],
    )

    await engine.initialize()

    # ── Transcribe ───────────────────────────────────────────────────
    source = FileSource(audio_path)

    t0 = time.perf_counter()
    result = await engine.transcribe(source)
    elapsed = time.perf_counter() - t0

    # ── Print results (note: text is UPPERCASED by the plugin) ──────
    print()
    print("═" * 60)
    print("TRANSCRIPT (post-plugin)")
    print("═" * 60)
    print(result.text)
    print("═" * 60)
    print(f"  Language:   {result.language}")
    print(f"  Segments:   {len(result.segments)}")
    print(f"  Plugin:     {result.metadata.get('plugin_applied', 'N/A')}")
    print(f"  Uppercased: {result.metadata.get('uppercased', False)}")
    print(f"  Time:       {elapsed:.2f}s")
    print("═" * 60)

    # Show segmented output (all uppercased)
    if result.segments:
        print()
        print("Segments (uppercased):")
        for seg in result.segments:
            print(f"  [{seg.start:7.2f}s -> {seg.end:7.2f}s]  {seg.text}")

    # ── Cleanup ──────────────────────────────────────────────────────
    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
