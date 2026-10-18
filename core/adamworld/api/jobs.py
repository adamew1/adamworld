"""Job queue abstraction.

Starts as a simple in-process, single-worker queue (GPU work must serialize). The
``JobQueue`` interface is intentionally minimal so it can later be swapped for
Celery/RQ without touching the API endpoints.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

ProgressCb = Callable[[str, float], None]
JobFn = Callable[[ProgressCb], str]  # returns a result string (e.g. world id)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.QUEUED
    stage: str = ""
    progress: float = 0.0
    result: str | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.value,
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "result": self.result,
            "error": self.error,
        }


class JobQueue(ABC):
    @abstractmethod
    def submit(self, fn: JobFn) -> str: ...

    @abstractmethod
    def get(self, job_id: str) -> Job | None: ...


class InProcessJobQueue(JobQueue):
    def __init__(self, workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, fn: JobFn) -> str:
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job

        def _progress(stage: str, frac: float) -> None:
            with self._lock:
                job.stage = stage
                job.progress = frac
                if job.status == JobStatus.QUEUED:
                    job.status = JobStatus.RUNNING

        def _run() -> None:
            with self._lock:
                job.status = JobStatus.RUNNING
            try:
                result = fn(_progress)
                with self._lock:
                    job.result = result
                    job.status = JobStatus.DONE
                    job.progress = 1.0
            except Exception:  # noqa: BLE001
                with self._lock:
                    job.status = JobStatus.ERROR
                    job.error = traceback.format_exc()

        self._executor.submit(_run)
        return job.id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)
