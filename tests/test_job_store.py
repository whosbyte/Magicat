# tests/test_job_store.py
import threading

import pytest

from magicat.server.store import JobStore


@pytest.fixture()
def store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "jobs.db")


def test_create_and_get_job(store, tmp_path):
    job = store.create_job(input_arg="https://x/v/1",
                           workdir=str(tmp_path / "j1"))
    assert job.status == "queued"
    assert len(job.job_id) == 32          # uuid hex
    fetched = store.get_job(job.job_id)
    assert fetched.input_arg == "https://x/v/1"
    assert fetched.workdir == str(tmp_path / "j1")
    assert fetched.created_at > 0


def test_get_missing_job_returns_none(store):
    assert store.get_job("nope") is None


def test_store_creates_missing_parent_dirs(tmp_path):
    store = JobStore(tmp_path / "not" / "yet" / "created" / "jobs.db")
    job = store.create_job("x", str(tmp_path))
    assert store.get_job(job.job_id) is not None


def test_status_transitions(store, tmp_path):
    job = store.create_job("x", str(tmp_path))
    store.set_status(job.job_id, "running")
    assert store.get_job(job.job_id).status == "running"
    store.set_status(job.job_id, "failed", error="boom")
    refreshed = store.get_job(job.job_id)
    assert refreshed.status == "failed"
    assert refreshed.error == "boom"


def test_events_ordered_and_resumable(store, tmp_path):
    job = store.create_job("x", str(tmp_path))
    store.add_event(job.job_id, "ingest", "start")
    store.add_event(job.job_id, "ingest", "ok")
    store.add_event(job.job_id, "shots", "start")
    events = store.events_since(job.job_id, after_seq=0)
    assert [(e.stage, e.state) for e in events] == [
        ("ingest", "start"), ("ingest", "ok"), ("shots", "start")]
    # resume from the middle
    later = store.events_since(job.job_id, after_seq=events[1].seq)
    assert [(e.stage, e.state) for e in later] == [("shots", "start")]


def test_list_jobs_newest_first(store, tmp_path):
    a = store.create_job("a", str(tmp_path / "a"))
    b = store.create_job("b", str(tmp_path / "b"))
    listed = store.list_jobs()
    assert [j.job_id for j in listed[:2]] == [b.job_id, a.job_id]


def test_thread_safety_smoke(store, tmp_path):
    job = store.create_job("x", str(tmp_path))

    def hammer(n):
        for i in range(25):
            store.add_event(job.job_id, f"t{n}", str(i))

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.events_since(job.job_id, 0)) == 100
