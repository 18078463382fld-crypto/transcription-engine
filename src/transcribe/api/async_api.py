"""
High-level async API wrapper around the transcription engine.

``AsyncTranscriptionAPI`` provides a convenient, concurrent-friendly interface
for transcribing audio sources without managing engine lifecycle directly.
It supports batch queuing, per-job metadata, progress callbacks, and
graceful cancellation.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Callable, Coroutine, Optional

from ..core.engine import TranscriptionEngine
from ..core.interfaces import AudioSource, Transcriber
from ..core.models import EngineConfig, TranscriptResult

logger = logging.getLogger("transcribe.api.async_api")


# ═══════════════════════════════════════════════════════════════
# Job state enum
# ═══════════════════════════════════════════════════════════════


class JobStatus(str, Enum):
    """Status of a single transcription job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ═══════════════════════════════════════════════════════════════
# Job descriptor
# ═══════════════════════════════════════════════════════════════


@dataclass
class TranscriptionJob:
    """
    Descriptor for a single transcription job submitted to the API.

    Attributes:
        job_id:      Unique identifier for this job (auto-generated).
        source:      The audio source to transcribe.
        language:    Optional language override for this job.
        status:      Current job status.
        result:      The final ``TranscriptResult`` (populated on completion).
        error:       Exception message if the job failed.
        created_at:  Timestamp (monotonic) when the job was enqueued.
        completed_at: Timestamp when the job finished (or failed).
        metadata:    Arbitrary user-supplied metadata attached to this job.
    """

    job_id: str
    source: AudioSource
    language: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    result: Optional[TranscriptResult] = None
    error: Optional[str] = None
    created_at: float = 0.0
    completed_at: Optional[float] = None
    metadata: dict[str, object] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Progress callback type
# ═══════════════════════════════════════════════════════════════

ProgressCallback = Callable[
    [TranscriptionJob],
    Coroutine[object, None, None],
]

# ═══════════════════════════════════════════════════════════════
# AsyncTranscriptionAPI
# ═══════════════════════════════════════════════════════════════


class AsyncTranscriptionAPI:
    """
    High-level async wrapper for concurrent transcription workloads.

    Features:

    - **Single-shot transcribe** — transcribe one source and get the result.
    - **Batch queuing** — submit multiple jobs and await results.
    - **Progress callbacks** — receive a callback as each job completes.
    - **Cancellation** — cancel pending / running jobs by job ID.
    - **Engine lifecycle** — engine is initialised lazily and shut down
      when the API context exits.

    Usage::

        api = AsyncTranscriptionAPI(transcriber=my_transcriber)

        # Single file
        result = await api.transcribe(FileSource("speech.mp3"))

        # Batch
        jobs = await api.transcribe_batch(
            [FileSource("a.mp3"), FileSource("b.mp3")],
        )
        for job in jobs:
            print(job.job_id, job.result.text)

        await api.shutdown()

    Can also be used as an async context manager::

        async with AsyncTranscriptionAPI(transcriber=...) as api:
            result = await api.transcribe(FileSource("speech.mp3"))
    """

    def __init__(
        self,
        transcriber: Transcriber,
        config: EngineConfig | None = None,
        *,
        max_concurrent: int = 4,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """
        Args:
            transcriber:      The core Transcriber implementation.
            config:           Engine configuration (defaults to ``EngineConfig()``).
            max_concurrent:   Maximum number of jobs to process concurrently.
            progress_callback: Optional async callback invoked when each job
                               completes, fails, or is cancelled.
        """
        self._engine = TranscriptionEngine(
            transcriber=transcriber,
            config=config or EngineConfig(),
        )
        self._max_concurrent = max_concurrent
        self._progress_callback = progress_callback
        self._jobs: dict[str, TranscriptionJob] = {}
        self._semaphore: asyncio.Semaphore | None = None
        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialise the underlying engine (idempotent)."""
        if not self._initialized:
            await self._engine.initialize()
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
            self._initialized = True
            logger.info(
                "AsyncTranscriptionAPI initialised (max_concurrent=%d)",
                self._max_concurrent,
            )

    async def shutdown(self) -> None:
        """Shut down the engine and release all resources."""
        self._initialized = False
        # Cancel any pending jobs
        for job in list(self._jobs.values()):
            if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                job.status = JobStatus.CANCELLED
                self._notify_progress(job)
        self._jobs.clear()
        await self._engine.shutdown()
        logger.info("AsyncTranscriptionAPI shut down")

    async def __aenter__(self) -> "AsyncTranscriptionAPI":
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: object = None,
        exc_val: object = None,
        exc_tb: object = None,
    ) -> None:
        await self.shutdown()

    # ── Single-shot transcription ─────────────────────────────

    async def transcribe(
        self,
        source: AudioSource,
        language: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TranscriptResult:
        """
        Transcribe a single audio source and return the result immediately.

        This is a convenience wrapper around :meth:`submit_job` +
        :meth:`await_job`.

        Args:
            source:   An AudioSource implementation.
            language: Optional language override.
            metadata: Optional metadata attached to the job.

        Returns:
            A ``TranscriptResult`` with the full transcript.
        """
        job = await self.submit_job(source, language=language, metadata=metadata)
        return await self.await_job(job.job_id)

    # ── Batch transcription ───────────────────────────────────

    async def transcribe_batch(
        self,
        sources: list[AudioSource],
        language: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> list[TranscriptionJob]:
        """
        Submit multiple sources for concurrent transcription.

        Results are returned as a list of ``TranscriptionJob`` objects in
        submission order.  Call :meth:`await_job` on individual jobs if
        you need the actual result immediately, or use
        :meth:`await_all_jobs` to block until all finish.

        Args:
            sources:  List of AudioSource objects.
            language: Optional language override (applied to all jobs).
            metadata: Optional metadata attached to every job.

        Returns:
            List of ``TranscriptionJob`` descriptors.
        """
        jobs: list[TranscriptionJob] = []
        for source in sources:
            job = await self.submit_job(
                source,
                language=language,
                metadata=metadata,
            )
            jobs.append(job)
        return jobs

    # ── Job management ────────────────────────────────────────

    async def submit_job(
        self,
        source: AudioSource,
        *,
        language: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TranscriptionJob:
        """
        Submit a single job and start processing it (subject to concurrency
        limit).

        Args:
            source:   The audio source to transcribe.
            language: Optional language override.
            metadata: Optional user metadata.

        Returns:
            A ``TranscriptionJob`` descriptor.  The job begins processing
            as soon as a concurrency slot is available.
        """
        await self.initialize()

        job = TranscriptionJob(
            job_id=_generate_job_id(),
            source=source,
            language=language,
            status=JobStatus.PENDING,
            created_at=time.monotonic(),
            metadata=dict(metadata or {}),
        )
        self._jobs[job.job_id] = job

        # Schedule the job's execution in the background, bounded by
        # the concurrency semaphore.
        asyncio.get_running_loop().create_task(self._execute_job(job))
        return job

    async def await_job(self, job_id: str, *, timeout: float | None = None) -> TranscriptResult:
        """
        Wait for a specific job to complete and return its result.

        Args:
            job_id:  The job identifier returned by :meth:`submit_job`.
            timeout: Optional timeout in seconds.

        Returns:
            The ``TranscriptResult`` for the completed job.

        Raises:
            KeyError:    If ``job_id`` is not recognised.
            TimeoutError: If the timeout expires before completion.
            RuntimeError: If the job failed or was cancelled.
        """
        job = self._get_job(job_id)
        if job.status == JobStatus.COMPLETED and job.result is not None:
            return job.result
        if job.status == JobStatus.FAILED:
            raise RuntimeError(f"Job {job_id} failed: {job.error}")
        if job.status == JobStatus.CANCELLED:
            raise RuntimeError(f"Job {job_id} was cancelled")

        # Poll until done
        deadline = None if timeout is None else time.monotonic() + timeout
        while job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for job {job_id}")
            await asyncio.sleep(0.05)  # 50 ms polling interval

        if job.status == JobStatus.COMPLETED and job.result is not None:
            return job.result
        if job.status == JobStatus.FAILED:
            raise RuntimeError(f"Job {job_id} failed: {job.error}")
        raise RuntimeError(f"Job {job_id} was cancelled")

    async def await_all_jobs(self, *, timeout: float | None = None) -> list[TranscriptionJob]:
        """
        Wait for all currently tracked jobs to finish.

        Args:
            timeout: Optional timeout in seconds for the entire batch.

        Returns:
            List of ``TranscriptionJob`` objects in submission order.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        pending = [j for j in self._jobs.values() if j.status in (JobStatus.PENDING, JobStatus.RUNNING)]
        for job in pending:
            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for all jobs")
            await self.await_job(job.job_id, timeout=remaining)
        return list(self._jobs.values())

    def cancel_job(self, job_id: str) -> None:
        """
        Cancel a pending or running job by its ID.

        The actual running transcription call is **not** interrupted, but
        the job result will be discarded and the status set to CANCELLED.

        Args:
            job_id: The job identifier to cancel.

        Raises:
            KeyError: If ``job_id`` is not recognised.
        """
        job = self._get_job(job_id)
        if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            job.status = JobStatus.CANCELLED
            self._notify_progress(job)
            logger.info("Cancelled job %s", job_id)

    def list_jobs(
        self,
        status: JobStatus | None = None,
    ) -> list[TranscriptionJob]:
        """
        Return all tracked jobs, optionally filtered by status.

        Args:
            status: If provided, only return jobs with this status.

        Returns:
            List of ``TranscriptionJob`` objects.
        """
        if status is None:
            return list(self._jobs.values())
        return [j for j in self._jobs.values() if j.status == status]

    def get_job(self, job_id: str) -> TranscriptionJob:
        """Get a job descriptor by its ID (raises ``KeyError`` if not found)."""
        return self._get_job(job_id)

    # ── Internal helpers ──────────────────────────────────────

    async def _execute_job(self, job: TranscriptionJob) -> None:
        """
        Execute a single transcription job under the concurrency semaphore.
        """
        if self._semaphore is None:
            return

        async with self._semaphore:
            if job.status == JobStatus.CANCELLED:
                return

            job.status = JobStatus.RUNNING
            self._notify_progress(job)

            try:
                result = await self._engine.transcribe(job.source, language=job.language)
                job.result = result
                job.status = JobStatus.COMPLETED
            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
            except Exception as exc:
                logger.exception("Job %s failed", job.job_id)
                job.error = str(exc)
                job.status = JobStatus.FAILED
            finally:
                job.completed_at = time.monotonic()
                self._notify_progress(job)

    def _get_job(self, job_id: str) -> TranscriptionJob:
        """Look up a job by ID, raising ``KeyError`` with a helpful message."""
        try:
            return self._jobs[job_id]
        except KeyError:
            raise KeyError(f"Unknown job: {job_id}")

    def _notify_progress(self, job: TranscriptionJob) -> None:
        """Invoke the optional progress callback (fire-and-forget)."""
        if self._progress_callback is not None:
            try:
                asyncio.get_running_loop().create_task(self._progress_callback(job))
            except Exception:
                logger.exception("Progress callback failed for job %s", job.job_id)

    # ── Properties ────────────────────────────────────────────

    @property
    def engine(self) -> TranscriptionEngine:
        """Access the underlying engine (advanced use only)."""
        return self._engine

    @property
    def max_concurrent(self) -> int:
        """The maximum number of concurrent jobs allowed."""
        return self._max_concurrent

    @property
    def pending_count(self) -> int:
        """Number of jobs currently pending or running."""
        return len(
            [
                j
                for j in self._jobs.values()
                if j.status in (JobStatus.PENDING, JobStatus.RUNNING)
            ]
        )

    @property
    def is_initialized(self) -> bool:
        """Whether the API has been initialised."""
        return self._initialized


# ── Utility ─────────────────────────────────────────────────


def _generate_job_id() -> str:
    """Generate a short, unique job identifier."""
    return uuid.uuid4().hex[:12]
