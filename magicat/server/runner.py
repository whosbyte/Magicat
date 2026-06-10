# magicat/server/runner.py
"""Job execution behind a 2-method seam (the spec's Celery+Redis queue
slots in here as another JobRunner implementation - recorded M4 deviation).

LocalJobRunner runs jobs on a thread pool: ffmpeg/OCR/HTTP dominate the
work and release the GIL or block on subprocesses, so threads are fine
for a local single-host deployment.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from magicat.core.pipeline import run_job
from magicat.server.store import Job, JobStore

log = logging.getLogger(__name__)

RunFn = Callable[..., object]


@runtime_checkable
class JobRunner(Protocol):
    def submit(self, job: Job) -> None: ...

    def shutdown(self) -> None: ...


class LocalJobRunner:
    def __init__(self, store: JobStore, run_fn: RunFn = run_job,
                 max_workers: int = 1) -> None:
        self.store = store
        self.run_fn = run_fn
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="magicat-job")

    def submit(self, job: Job) -> None:
        self._pool.submit(self._execute, job)

    def _execute(self, job: Job) -> None:
        self.store.set_status(job.job_id, "running")
        try:
            self.run_fn(
                job.input_arg, Path(job.workdir), job_id=job.job_id,
                on_progress=lambda stage, state:
                    self.store.add_event(job.job_id, stage, state))
            self.store.set_status(job.job_id, "done")
        except Exception as exc:
            log.exception("job %s failed", job.job_id)
            self.store.add_event(job.job_id, "job", "failed")
            self.store.set_status(job.job_id, "failed", error=str(exc))

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
