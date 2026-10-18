"""FastAPI backend and swappable in-process job queue."""

from __future__ import annotations

from adamworld.api.app import create_app
from adamworld.api.jobs import InProcessJobQueue, Job, JobQueue, JobStatus

__all__ = ["InProcessJobQueue", "Job", "JobQueue", "JobStatus", "create_app"]
