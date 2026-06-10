# tests/test_worker.py
import time

import pytest

from magicat.server.runner import LocalJobRunner
from magicat.server.store import JobStore


@pytest.fixture()
def store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "jobs.db")


def wait_status(store, job_id, statuses=("done", "failed"),
                timeout=10.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.get_job(job_id)
        if job.status in statuses:
            return job.status
        time.sleep(0.02)
    raise TimeoutError(f"job stuck in {store.get_job(job_id).status}")


def test_successful_job_lifecycle(store, tmp_path):
    calls = {}

    def fake_run(input_arg, workdir, job_id=None, on_progress=None):
        calls["args"] = (input_arg, str(workdir), job_id)
        on_progress("ingest", "start")
        on_progress("ingest", "ok")
        on_progress("job", "done")

    runner = LocalJobRunner(store, run_fn=fake_run, max_workers=1)
    job = store.create_job("https://x/v/1", str(tmp_path / "j1"))
    runner.submit(job)
    assert wait_status(store, job.job_id) == "done"
    assert calls["args"] == ("https://x/v/1", str(tmp_path / "j1"),
                             job.job_id)
    stages = [(e.stage, e.state)
              for e in store.events_since(job.job_id, 0)]
    assert ("ingest", "ok") in stages
    runner.shutdown()


def test_failed_job_records_error(store, tmp_path):
    def fake_run(input_arg, workdir, job_id=None, on_progress=None):
        raise RuntimeError("ingest exploded")

    runner = LocalJobRunner(store, run_fn=fake_run, max_workers=1)
    job = store.create_job("x", str(tmp_path / "j1"))
    runner.submit(job)
    assert wait_status(store, job.job_id) == "failed"
    refreshed = store.get_job(job.job_id)
    assert "ingest exploded" in refreshed.error
    stages = [(e.stage, e.state)
              for e in store.events_since(job.job_id, 0)]
    assert ("job", "failed") in stages
    runner.shutdown()


def test_jobs_queue_when_pool_busy(store, tmp_path):
    order = []

    def slow_run(input_arg, workdir, job_id=None, on_progress=None):
        order.append(input_arg)
        time.sleep(0.2)

    runner = LocalJobRunner(store, run_fn=slow_run, max_workers=1)
    j1 = store.create_job("first", str(tmp_path / "a"))
    j2 = store.create_job("second", str(tmp_path / "b"))
    runner.submit(j1)
    runner.submit(j2)
    assert wait_status(store, j2.job_id) == "done"
    assert order == ["first", "second"]      # serialized on 1 worker
    runner.shutdown()


def test_running_status_set_before_run(store, tmp_path):
    seen = {}

    def check_run(input_arg, workdir, job_id=None, on_progress=None):
        seen["status_during_run"] = store.get_job(job_id).status

    runner = LocalJobRunner(store, run_fn=check_run, max_workers=1)
    job = store.create_job("x", str(tmp_path / "j"))
    runner.submit(job)
    wait_status(store, job.job_id)
    assert seen["status_during_run"] == "running"
    runner.shutdown()
