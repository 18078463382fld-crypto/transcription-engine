#!/usr/bin/env python3
"""
batch_jobs.py — Use AsyncTranscriptionAPI for concurrent transcription jobs.

Submits multiple audio files for transcription concurrently using the
``AsyncTranscriptionAPI`` wrapper, demonstrating job submission, progress
callbacks, and result collection.

Usage:
    python examples/batch_jobs.py <audio-file-1> [<audio-file-2> ...]

Example:
    python examples/batch_jobs.py meeting1.wav meeting2.wav lecture.mp3

Dependencies:
    pip install faster-whisper soundfile pydub
"""

from __future__ import annotations

import asyncio
import sys
import time

from transcribe import AsyncTranscriptionAPI, EngineConfig
from transcribe.api.async_api import TranscriptionJob
from transcribe.source.file_source import FileSource
from transcribe.transcriber import LocalWhisperTranscriber


# ═══════════════════════════════════════════════════════════════════
# Progress callback
# ═══════════════════════════════════════════════════════════════════


async def on_job_progress(job: TranscriptionJob) -> None:
    """Called by AsyncTranscriptionAPI each time a job changes state."""
    status_icon = {
        "pending": "⏳",
        "running": "▶",
        "completed": "✓",
        "failed": "✗",
        "cancelled": "⊘",
    }.get(job.status.value, "?")

    elapsed = ""
    if job.created_at:
        dur = time.monotonic() - job.created_at
        elapsed = f"  (+{dur:.1f}s)"

    meta = ""
    if job.metadata.get("filename"):
        meta = f"  [{job.metadata['filename']}]"

    print(f"  {status_icon}  {job.job_id}: {job.status.value}{meta}{elapsed}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


async def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio-file-1> [<audio-file-2> ...]")
        print("Example: python examples/batch_jobs.py speech1.wav speech2.wav")
        sys.exit(1)

    audio_paths = sys.argv[1:]

    # ── Create audio sources ─────────────────────────────────────────
    sources: list[FileSource] = []
    for path in audio_paths:
        sources.append(FileSource(path))

    print(f"[*] Submitting {len(sources)} job(s) for concurrent transcription ...")
    print(f"[*] Max concurrent: 4")
    print()

    # ── Create the API ───────────────────────────────────────────────
    transcriber = LocalWhisperTranscriber(model_name="base")

    api = AsyncTranscriptionAPI(
        transcriber=transcriber,
        config=EngineConfig(device="cpu", compute_type="float32"),
        max_concurrent=4,
        progress_callback=on_job_progress,
    )

    await api.initialize()

    # ── Submit batch ─────────────────────────────────────────────────
    jobs = await api.transcribe_batch(
        sources,
        metadata={"batch": "example-batch"},
    )

    print("  Jobs submitted:")
    for job in jobs:
        print(f"    {job.job_id}: {job.source}  ({job.status.value})")
    print()

    # ── Wait for all jobs to complete ────────────────────────────────
    print("  Waiting for results ...")
    t0 = time.monotonic()

    completed_jobs = await api.await_all_jobs()

    total_time = time.monotonic() - t0
    print()

    # ── Print summary ────────────────────────────────────────────────
    success_count = sum(
        1 for j in completed_jobs if j.status.value == "completed"
    )
    fail_count = sum(
        1 for j in completed_jobs if j.status.value == "failed"
    )

    print("═" * 60)
    print(f"BATCH SUMMARY  ({success_count} ok / {fail_count} failed)")
    print("═" * 60)

    for job in completed_jobs:
        icon = "✓" if job.status.value == "completed" else "✗"
        filename = job.metadata.get("filename", str(job.source))

        if job.result is not None:
            duration_str = (
                f"  ({job.result.duration_seconds:.1f}s audio)"
                if job.result.duration_seconds
                else ""
            )
            print(
                f"  {icon}  {job.job_id}: "
                f"{job.result.text[:80].rstrip()}...{duration_str}"
            )
        elif job.error:
            print(f"  {icon}  {job.job_id}: FAILED — {job.error}")
        else:
            print(f"  {icon}  {job.job_id}: {job.status.value}")

    print("═" * 60)
    print(f"  Total wall time: {total_time:.2f}s")
    print(f"  Concurrent jobs: {len(sources)}")
    print("═" * 60)

    # ── Cleanup ──────────────────────────────────────────────────────
    await api.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
