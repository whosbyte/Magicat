# Magicat M4 — SaaS Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A locally-runnable web service: paste a URL or upload a clip, watch live per-layer progress, then preview the reconstruction and download the deliverables — the launchable v1 shell.

**Architecture:** A `magicat/server/` package behind the existing contracts: a SQLite `JobStore` (jobs + ordered progress events), a `LocalJobRunner` executing `run_job` on a thread pool (the queue SEAM — Celery/Redis becomes a drop-in runner implementation at deployment), and a FastAPI app (factory pattern for testability) serving a JSON API, an SSE progress stream, allowlisted artifact downloads, and a no-build static UI. The pipeline gains an `on_progress` callback hook (analyzer/exporter stage events).

**Tech Stack:** FastAPI + uvicorn + python-multipart (the only new runtime deps), stdlib sqlite3 (WAL), vanilla HTML/JS UI served statically, httpx (dev, for TestClient).

**Spec:** `docs/superpowers/specs/2026-06-09-magicat-framework-design.md` §6 (orchestration), §13 M4 row ("web app, queue, progress, auth, downloads").

**Recorded deviations from the spec's cloud stack (CTO decisions — local-first launchable v1; each keeps a seam for the spec's deployment shape):**

| Spec says | M4 ships | Seam for later |
|---|---|---|
| Celery + Redis queue | `JobRunner` protocol + `LocalJobRunner` (ThreadPoolExecutor) | a `CeleryJobRunner` implements the same 2-method protocol |
| Postgres | SQLite (WAL) behind `JobStore` | store is interface-shaped; swap the connection layer |
| S3/R2 + presigned URLs | job-dir files + allowlisted `FileResponse` | artifact resolver function is the only place paths are made |
| Next.js frontend | no-build static HTML/JS served by FastAPI | the JSON/SSE API is the contract; any frontend can replace the static page |
| Clerk auth + Stripe credits | optional `MAGICAT_API_KEY` header gate | auth dependency is one FastAPI `Depends`; swap for Clerk middleware |
| users table + day-one `credits` column (spec §8) | no identity model in local-v1 (single shared key, jobs-only schema) | users/credits schema lands WITH the Clerk/Stripe integration — additive tables, no migration of existing data needed |

---

## File Structure

```
pyproject.toml                    # MODIFY: + fastapi, uvicorn, python-multipart; dev + httpx
magicat/
  core/pipeline.py                # MODIFY: run_job gains on_progress callback
  cli.py                          # MODIFY: + `magicat serve` command
  server/
    __init__.py                   # NEW (empty)
    store.py                      # NEW: JobStore (SQLite): jobs + events tables
    runner.py                     # NEW: JobRunner protocol + LocalJobRunner
    app.py                        # NEW: create_app() FastAPI factory (API + SSE + static)
    static/
      index.html                  # NEW: submit form / progress / results page
      app.js                      # NEW: fetch + EventSource glue
tests/
  test_job_store.py               # NEW
  test_progress_events.py         # NEW
  test_worker.py                  # NEW
  test_api.py                     # NEW
  test_api_e2e.py                 # NEW (one real-pipeline heavyweight test)
```

---

### Task 1: Dependencies + JobStore (SQLite)

**Files:**
- Modify: `pyproject.toml`
- Create: `magicat/server/__init__.py` (empty), `magicat/server/store.py`
- Test: `tests/test_job_store.py`

- [ ] **Step 1: Update `pyproject.toml`**

Add to `dependencies` (keep existing entries):

```toml
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "python-multipart>=0.0.9",
```

and extend the dev extra to `dev = ["pytest>=8", "opentimelineio>=0.18", "otio-fcp-adapter>=1.0", "httpx>=0.27"]`.

Run: `.venv/Scripts/python -m pip install -e .[dev]` — installs fastapi/uvicorn/python-multipart/httpx.

Also add `"MAGICAT_API_KEY"` to the env-var tuple in the autouse `_isolated_magicat_env` fixture in `tests/conftest.py` — the server reads it at request time, and a developer's ambient key must never 401 the test suite.

- [ ] **Step 2: Write the failing tests** — create `tests/test_job_store.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_job_store.py -v`
Expected: FAIL — ModuleNotFoundError: magicat.server.

- [ ] **Step 4: Implement** — create `magicat/server/store.py`:

```python
# magicat/server/store.py
"""SQLite-backed job store: job rows + ordered progress events.

Local stand-in for the spec's Postgres (recorded M4 deviation). WAL mode +
a connection per call keeps it safe across the API thread and worker
threads without a shared-connection lock.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

VALID_STATUSES = ("queued", "running", "done", "failed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    input_arg  TEXT NOT NULL,
    workdir    TEXT NOT NULL,
    error      TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq    INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    stage  TEXT NOT NULL,
    state  TEXT NOT NULL,
    ts     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_job ON events (job_id, seq);
"""


@dataclass
class Job:
    job_id: str
    status: str
    input_arg: str
    workdir: str
    error: str | None
    created_at: float
    updated_at: float


@dataclass
class Event:
    seq: int
    job_id: str
    stage: str
    state: str
    ts: float


class JobStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def create_job(self, input_arg: str, workdir: str) -> Job:
        now = time.time()
        job = Job(job_id=uuid.uuid4().hex, status="queued",
                  input_arg=input_arg, workdir=workdir, error=None,
                  created_at=now, updated_at=now)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job.job_id, job.status, job.input_arg, job.workdir,
                 job.error, job.created_at, job.updated_at))
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?",
                               (job_id,)).fetchone()
        return Job(**dict(row)) if row else None

    def set_status(self, job_id: str, status: str,
                   error: str | None = None) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status {status!r}")
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? "
                "WHERE job_id = ?",
                (status, error, time.time(), job_id))

    def add_event(self, job_id: str, stage: str, state: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (job_id, stage, state, ts) "
                "VALUES (?, ?, ?, ?)",
                (job_id, stage, state, time.time()))

    def events_since(self, job_id: str, after_seq: int) -> list[Event]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE job_id = ? AND seq > ? "
                "ORDER BY seq", (job_id, after_seq)).fetchall()
        return [Event(**dict(r)) for r in rows]

    def list_jobs(self, limit: int = 50) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC "
                "LIMIT ?", (limit,)).fetchall()
        return [Job(**dict(r)) for r in rows]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_job_store.py -v`
Expected: 6 PASS. Full suite: 156 passed, 2 skipped.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml magicat/server tests/test_job_store.py
git commit -m "feat: SQLite job store with ordered progress events"
```

---

### Task 2: Pipeline progress hook

**Files:**
- Modify: `magicat/core/pipeline.py`
- Test: `tests/test_progress_events.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_progress_events.py`:

```python
# tests/test_progress_events.py
from magicat.core.pipeline import run_job


def test_progress_callback_receives_stage_events(fixture_video, tmp_path):
    seen: list[tuple[str, str]] = []
    run_job(str(fixture_video), tmp_path / "job",
            on_progress=lambda stage, state: seen.append((stage, state)))

    assert seen[0] == ("ingest", "start")
    assert ("ingest", "ok") in seen
    assert ("cut_detection", "start") in seen
    assert ("cut_detection", "ok") in seen
    # music layer is skipped without provider keys (env isolated by fixture)
    assert ("audio_analysis", "skipped") in seen
    assert ("caption_analysis", "ok") in seen
    assert ("preview_mp4", "start") in seen
    assert ("preview_mp4", "ok") in seen
    assert ("premiere_resolve_zip", "ok") in seen
    assert seen[-1] == ("job", "done")
    # every start eventually resolves
    starts = {s for s, st in seen if st == "start"}
    resolved = {s for s, st in seen if st in ("ok", "failed", "skipped")}
    assert starts <= resolved | {"job"}


def test_progress_callback_reports_failure(fixture_video, tmp_path,
                                           monkeypatch):
    from magicat.core import registry

    def boom(manifest, ws):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(registry.get_analyzer("cut_detection"), "run", boom)
    seen: list[tuple[str, str]] = []
    run_job(str(fixture_video), tmp_path / "job",
            on_progress=lambda stage, state: seen.append((stage, state)))
    assert ("cut_detection", "failed") in seen
    assert seen[-1] == ("job", "done")


def test_progress_callback_optional(fixture_video, tmp_path):
    manifest = run_job(str(fixture_video), tmp_path / "job")
    assert manifest.layers_status["shots"].value == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_progress_events.py -v`
Expected: FAIL — TypeError: run_job() got an unexpected keyword argument 'on_progress'.

- [ ] **Step 3: Implement** — in `magicat/core/pipeline.py`:

Add the typing import `from typing import Callable` and extend the signature:

```python
ProgressFn = Callable[[str, str], None]


def _noop_progress(stage: str, state: str) -> None:
    return None


def run_job(input_arg: str, workdir: Path, job_id: str | None = None,
            on_progress: ProgressFn | None = None) -> Manifest:
    progress = on_progress or _noop_progress
```

Wire events INTO the existing loops — this is an ADDITIVE edit. CRITICAL: run_job currently contains TWO `manifest = apply_patch(manifest, {"report": build_report(manifest)})` lines — one BETWEEN the analyzer loop and the exporter loop, one AFTER the exporter loop before `ws.save_manifest(manifest)`. BOTH MUST BE PRESERVED exactly where they are (the report.html exporter, the zip, and the API's job payload all read `manifest.report`). Do not replace the loop region wholesale; insert the `progress(...)` lines shown below into the existing code:

```python
    # ingest is fatal on failure
    progress("ingest", "start")
    manifest = apply_patch(manifest, registry.get_analyzer("ingest")
                           .run(manifest, ws))
    progress("ingest", "ok")

    for name in ANALYZERS:
        analyzer = registry.get_analyzer(name)
        progress(name, "start")
        try:
            manifest = apply_patch(manifest, analyzer.run(manifest, ws))
            state = manifest.layers_status.get(analyzer.layer)
            progress(name, state.value if state else "ok")
        except Exception:
            log.exception("analyzer %s failed", name)
            manifest = apply_patch(
                manifest, {"layers_status": {analyzer.layer: "failed"}})
            progress(name, "failed")
```

and in the exporter loop:

```python
    for fmt in EXPORTERS:
        exporter = registry.get_exporter(fmt)
        progress(fmt, "start")
        try:
            artifact = exporter.export(manifest, ws)
            manifest = apply_patch(manifest, {
                "exports": [{"format": fmt, "artifact": str(artifact)}],
                "layers_status": {fmt: "ok"},
            })
            progress(fmt, "ok")
        except Exception:
            log.exception("exporter %s failed", fmt)
            manifest = apply_patch(
                manifest, {"layers_status": {fmt: "failed"}})
            progress(fmt, "failed")
```

and at the very end, after `ws.save_manifest(manifest)`:

```python
    progress("job", "done")
    return manifest
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_progress_events.py tests/test_pipeline.py -v`
Expected: all PASS. Full suite: 159 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat/core/pipeline.py tests/test_progress_events.py
git commit -m "feat: pipeline progress callback hook"
```

---

### Task 3: LocalJobRunner (the queue seam)

**Files:**
- Create: `magicat/server/runner.py`
- Test: `tests/test_worker.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_worker.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_worker.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement** — create `magicat/server/runner.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_worker.py -v`
Expected: 4 PASS. Full suite: 163 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat/server/runner.py tests/test_worker.py
git commit -m "feat: local thread-pool job runner behind the queue seam"
```

---

### Task 4: FastAPI app — jobs API, artifacts, auth

**Files:**
- Create: `magicat/server/app.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_api.py`:

```python
# tests/test_api.py
import json
import time

import pytest
from fastapi.testclient import TestClient

from magicat.server.app import create_app
from magicat.server.runner import LocalJobRunner
from magicat.server.store import JobStore


class InlineRunner:
    """Executes the job synchronously inside submit() - deterministic tests."""

    def __init__(self, store, run_fn):
        self.store = store
        self.run_fn = run_fn

    def submit(self, job):
        self.store.set_status(job.job_id, "running")
        try:
            self.run_fn(job.input_arg, job.workdir, job_id=job.job_id,
                        on_progress=lambda s, st:
                            self.store.add_event(job.job_id, s, st))
            self.store.set_status(job.job_id, "done")
        except Exception as exc:
            self.store.set_status(job.job_id, "failed", error=str(exc))

    def shutdown(self):
        pass


def fake_pipeline(input_arg, workdir, job_id=None, on_progress=None):
    from pathlib import Path
    wd = Path(workdir)
    (wd / "exports").mkdir(parents=True, exist_ok=True)
    (wd / "exports" / "report.html").write_text("<html>ok</html>",
                                                encoding="utf-8")
    (wd / "manifest.json").write_text(json.dumps(
        {"job_id": job_id, "report": {"shots": {"count": 3}}}),
        encoding="utf-8")
    on_progress("ingest", "ok")
    on_progress("job", "done")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("MAGICAT_API_KEY", raising=False)
    store = JobStore(tmp_path / "jobs.db")
    runner = InlineRunner(store, fake_pipeline)
    app = create_app(store=store, runner=runner,
                     jobs_root=tmp_path / "jobs")
    return TestClient(app)


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_create_job_from_url_and_fetch(client):
    r = client.post("/api/jobs", json={"url": "https://x/v/1"})
    assert r.status_code == 201
    job_id = r.json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done"
    assert job["report"]["shots"]["count"] == 3


def test_create_job_from_upload(client, tmp_path):
    payload = b"\x00fakevideo"
    r = client.post("/api/jobs",
                    files={"file": ("clip.mp4", payload, "video/mp4")})
    assert r.status_code == 201
    job_id = r.json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done"


def test_create_job_requires_url_or_file(client):
    assert client.post("/api/jobs", json={}).status_code == 422


def test_list_jobs(client):
    client.post("/api/jobs", json={"url": "https://x/v/1"})
    client.post("/api/jobs", json={"url": "https://x/v/2"})
    jobs = client.get("/api/jobs").json()["jobs"]
    assert len(jobs) == 2
    assert jobs[0]["input_arg"] == "https://x/v/2"   # newest first


def test_artifact_download_allowlisted(client):
    job_id = client.post("/api/jobs",
                         json={"url": "https://x/v/1"}).json()["job_id"]
    ok = client.get(f"/api/jobs/{job_id}/artifacts/report.html")
    assert ok.status_code == 200
    assert ok.text == "<html>ok</html>"
    missing = client.get(f"/api/jobs/{job_id}/artifacts/preview.mp4")
    assert missing.status_code == 404               # allowlisted but absent
    for evil in ("..%2f..%2fjobs.db", "manifest.yaml", "x.mp4"):
        assert client.get(
            f"/api/jobs/{job_id}/artifacts/{evil}").status_code == 404


def test_manifest_artifact(client):
    job_id = client.post("/api/jobs",
                         json={"url": "https://x/v/1"}).json()["job_id"]
    r = client.get(f"/api/jobs/{job_id}/artifacts/manifest.json")
    assert r.status_code == 200
    assert r.json()["report"]["shots"]["count"] == 3


def test_unknown_job_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404
    assert client.get(
        "/api/jobs/deadbeef/artifacts/report.html").status_code == 404


def test_api_key_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("MAGICAT_API_KEY", "sekrit")
    store = JobStore(tmp_path / "jobs.db")
    app = create_app(store=store,
                     runner=InlineRunner(store, fake_pipeline),
                     jobs_root=tmp_path / "jobs")
    client = TestClient(app)
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/jobs",
                      headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/jobs",
                      headers={"X-API-Key": "sekrit"}).status_code == 200
    assert client.get("/healthz").status_code == 200   # health is open
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_api.py -v`
Expected: FAIL — ModuleNotFoundError (magicat.server.app).

- [ ] **Step 3: Implement** — create `magicat/server/app.py`:

```python
# magicat/server/app.py
"""FastAPI shell: jobs API + SSE progress + artifact downloads + static UI.

create_app() is a factory so tests inject their own store/runner. Artifact
downloads are ALLOWLISTED filenames resolved inside the job's workdir -
never client-supplied paths.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    StreamingResponse,
)

from magicat.server.runner import JobRunner, LocalJobRunner
from magicat.server.store import JobStore

STATIC_DIR = Path(__file__).parent / "static"

ARTIFACTS = {
    "preview.mp4": ("exports/preview.mp4", "video/mp4"),
    "report.html": ("exports/report.html", "text/html"),
    "premiere_resolve.zip": ("exports/premiere_resolve.zip",
                             "application/zip"),
    "manifest.json": ("manifest.json", "application/json"),
}


def _require_api_key(request: Request) -> None:
    expected = os.environ.get("MAGICAT_API_KEY")
    if expected and request.headers.get("X-API-Key") != expected:
        raise HTTPException(status_code=401, detail="invalid API key")


def create_app(store: JobStore | None = None,
               runner: JobRunner | None = None,
               jobs_root: Path | str = "jobs") -> FastAPI:
    jobs_root = Path(jobs_root)
    store = store or JobStore(jobs_root / "jobs.db")
    runner = runner or LocalJobRunner(
        store, max_workers=int(os.environ.get("MAGICAT_MAX_JOBS", "1")))

    app = FastAPI(title="Magicat", version="0.1.0")
    api_key = Depends(_require_api_key)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/api/jobs", status_code=201, dependencies=[api_key])
    async def create_job(request: Request,
                         file: UploadFile | None = None) -> dict:
        input_arg: str | None = None
        # NOTE: the dir name is independent of the store's job_id (which the
        # store generates after the upload needs a home) - intentional; the
        # manifest's job_id still matches store.job_id via runner pass-through
        job_dir = jobs_root / uuid.uuid4().hex[:12]
        if file is not None:
            job_dir.mkdir(parents=True, exist_ok=True)
            upload_path = job_dir / "input.mp4"
            upload_path.write_bytes(await file.read())
            input_arg = str(upload_path)
        else:
            try:
                body = await request.json()
            except Exception:
                body = {}
            url = (body or {}).get("url")
            if url:
                input_arg = str(url)
        if not input_arg:
            raise HTTPException(status_code=422,
                                detail="provide a url or upload a file")
        job = store.create_job(input_arg=input_arg, workdir=str(job_dir))
        runner.submit(job)
        return {"job_id": job.job_id}

    def _job_payload(job) -> dict:
        payload = {
            "job_id": job.job_id, "status": job.status,
            "input_arg": job.input_arg, "error": job.error,
            "created_at": job.created_at, "updated_at": job.updated_at,
        }
        manifest_path = Path(job.workdir) / "manifest.json"
        if job.status == "done" and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["report"] = manifest.get("report", {})
        return payload

    @app.get("/api/jobs", dependencies=[api_key])
    def list_jobs() -> dict:
        return {"jobs": [_job_payload(j) for j in store.list_jobs()]}

    @app.get("/api/jobs/{job_id}", dependencies=[api_key])
    def get_job(job_id: str) -> dict:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return _job_payload(job)

    @app.get("/api/jobs/{job_id}/artifacts/{name}", dependencies=[api_key])
    def get_artifact(job_id: str, name: str) -> FileResponse:
        job = store.get_job(job_id)
        if job is None or name not in ARTIFACTS:
            raise HTTPException(status_code=404, detail="not found")
        rel, media_type = ARTIFACTS[name]
        path = Path(job.workdir) / rel
        if not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path, media_type=media_type, filename=name)

    @app.get("/api/jobs/{job_id}/events", dependencies=[api_key])
    async def job_events(job_id: str) -> StreamingResponse:
        if store.get_job(job_id) is None:
            raise HTTPException(status_code=404, detail="unknown job")

        async def stream():
            last_seq = 0
            job_frame_sent = False
            while True:
                events = store.events_since(job_id, last_seq)
                for event in events:
                    last_seq = event.seq
                    if event.stage == "job":
                        job_frame_sent = True
                    data = json.dumps(
                        {"stage": event.stage, "state": event.state})
                    yield f"data: {data}\n\n"
                job = store.get_job(job_id)
                if job.status in ("done", "failed") and not events:
                    # synthetic terminal frame ONLY if the pipeline never
                    # recorded its own job-stage event (e.g. crash before
                    # run_job's final progress call) - no duplicates
                    if not job_frame_sent:
                        yield ("data: " + json.dumps(
                            {"stage": "job", "state": job.status}) + "\n\n")
                    return
                await asyncio.sleep(0.3)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        index_path = STATIC_DIR / "index.html"
        if index_path.is_file():
            return index_path.read_text(encoding="utf-8")
        return "<html><body>Magicat API is running.</body></html>"

    @app.get("/static/app.js")
    def app_js() -> FileResponse:
        return FileResponse(STATIC_DIR / "app.js",
                            media_type="application/javascript")

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_api.py -v`
Expected: 9 PASS. Full suite: 172 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat/server/app.py tests/test_api.py
git commit -m "feat: FastAPI jobs API with allowlisted artifacts and API-key gate"
```

---

### Task 5: SSE stream test + static UI

**Files:**
- Create: `magicat/server/static/index.html`, `magicat/server/static/app.js`
- Test: `tests/test_api.py` (append 3 tests)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_api.py`:

```python
def test_sse_stream_yields_events_and_terminates(client):
    job_id = client.post("/api/jobs",
                         json={"url": "https://x/v/1"}).json()["job_id"]
    # job is already done (inline runner): the stream must replay events
    # then terminate. NOTE: TestClient BUFFERS the whole SSE body (frames
    # arrive together when the generator returns) - this asserts replay +
    # ordering + termination, NOT incremental delivery; live streaming is
    # smoke-verified via the manual uvicorn step in Task 6.
    frames = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(
            "text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    assert {"stage": "ingest", "state": "ok"} in frames
    assert frames[-1] == {"stage": "job", "state": "done"}


def test_sse_unknown_job_404(client):
    assert client.get("/api/jobs/deadbeef/events").status_code == 404


def test_index_page_served(client):
    html = client.get("/").text
    assert "magicat" in html.lower()
    assert "submit" in html.lower()
    js = client.get("/static/app.js")
    assert js.status_code == 200
    assert "EventSource" in js.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_api.py -v`
Expected: the SSE replay test PASSES already (Task 4 built the endpoint); `test_index_page_served` FAILS (no static files yet — the fallback page lacks "submit"). That is the expected red.

- [ ] **Step 3: Implement** — create `magicat/server/static/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Magicat</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;
       padding:0 1rem;color:#1a1a2e;background:#fafaff}
  h1{font-size:1.5rem} .card{background:#fff;border:1px solid #e4e4ef;
       border-radius:.75rem;padding:1rem 1.25rem;margin:1rem 0}
  input[type=url]{width:100%;padding:.5rem;border:1px solid #ccd;
       border-radius:.4rem}
  button{background:#5b4bdb;color:#fff;border:0;border-radius:.4rem;
       padding:.55rem 1.2rem;cursor:pointer;font-size:1rem}
  button:disabled{opacity:.5}
  #progress li{list-style:none;padding:.15rem 0}
  #progress .ok::before{content:"\2713  ";color:#2a9d4a}
  #progress .failed::before{content:"\2717  ";color:#d33}
  #progress .skipped::before{content:"\2014  ";color:#999}
  #progress .start::before{content:"\22EF  ";color:#888}
  video{width:100%;max-height:480px;border-radius:.5rem;background:#000}
  .downloads a{display:inline-block;margin-right:.75rem}
</style>
</head>
<body>
<h1>Magicat — deconstruct a short</h1>
<div class="card">
  <form id="submit-form">
    <p><input type="url" id="url" placeholder="https://www.tiktok.com/@user/video/..."></p>
    <p>or <input type="file" id="file" accept="video/*"></p>
    <button id="submit" type="submit">Deconstruct</button>
  </form>
</div>
<div class="card" id="progress-card" hidden>
  <h2>Progress</h2><ul id="progress"></ul>
</div>
<div class="card" id="results" hidden>
  <h2>Result</h2>
  <video id="preview" controls></video>
  <div id="summary"></div>
  <p class="downloads" id="downloads"></p>
</div>
<script src="/static/app.js"></script>
</body>
</html>
```

And `magicat/server/static/app.js`:

```javascript
const form = document.getElementById("submit-form");
const progressCard = document.getElementById("progress-card");
const progressList = document.getElementById("progress");
const results = document.getElementById("results");
const submitBtn = document.getElementById("submit");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  progressList.innerHTML = "";
  results.hidden = true;
  progressCard.hidden = false;

  const url = document.getElementById("url").value.trim();
  const fileInput = document.getElementById("file");
  let resp;
  if (fileInput.files.length > 0) {
    const data = new FormData();
    data.append("file", fileInput.files[0]);
    resp = await fetch("/api/jobs", { method: "POST", body: data });
  } else if (url) {
    resp = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
  } else {
    alert("Paste a URL or choose a file.");
    submitBtn.disabled = false;
    return;
  }
  if (!resp.ok) {
    alert("Submit failed: " + resp.status);
    submitBtn.disabled = false;
    return;
  }
  const { job_id } = await resp.json();
  watch(job_id);
});

function watch(jobId) {
  const items = {};
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  source.onmessage = (msg) => {
    const { stage, state } = JSON.parse(msg.data);
    if (stage === "job") {
      source.close();
      finish(jobId, state);
      return;
    }
    if (!items[stage]) {
      items[stage] = document.createElement("li");
      progressList.appendChild(items[stage]);
    }
    items[stage].textContent = stage;
    items[stage].className = state;
  };
  source.onerror = () => { source.close(); finish(jobId, "done"); };
}

async function finish(jobId, state) {
  submitBtn.disabled = false;
  const job = await (await fetch(`/api/jobs/${jobId}`)).json();
  if (job.status !== "done") {
    alert("Job " + job.status + (job.error ? ": " + job.error : ""));
    return;
  }
  results.hidden = false;
  document.getElementById("preview").src =
    `/api/jobs/${jobId}/artifacts/preview.mp4`;
  const report = job.report || {};
  const music = report.music || {};
  const captions = report.captions || {};
  const esc = (s) => String(s ?? "-").replace(/[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  let html = `<p>${esc((report.shots || {}).count)} shots</p>`;
  html += music.detected
    ? `<p>Music: <strong>${esc(music.title)}</strong> by ${esc(music.artist)}
       (via ${esc(music.identified_by)})</p>`
    : "<p>No music detected.</p>";
  html += captions.count
    ? `<p>${captions.count} caption(s), font: ${esc((captions.fonts || []).join(", ") || "uncertain")}</p>`
    : "<p>No captions.</p>";
  document.getElementById("summary").innerHTML = html;
  document.getElementById("downloads").innerHTML =
    `<a href="/api/jobs/${jobId}/artifacts/preview.mp4" download>Preview MP4</a>
     <a href="/api/jobs/${jobId}/artifacts/premiere_resolve.zip" download>Premiere/Resolve project</a>
     <a href="/api/jobs/${jobId}/artifacts/report.html" target="_blank">Report</a>
     <a href="/api/jobs/${jobId}/artifacts/manifest.json" download>Manifest</a>`;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_api.py -v`
Expected: 12 PASS. Full suite: 175 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat/server/static tests/test_api.py
git commit -m "feat: SSE progress stream coverage and no-build web UI"
```

---

### Task 6: `magicat serve` + e2e + README + ship

**Files:**
- Modify: `magicat/cli.py`, `README.md`
- Test: `tests/test_api_e2e.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_api_e2e.py`:

```python
# tests/test_api_e2e.py
"""One heavyweight test: the REAL pipeline through the REAL app factory."""
import json

from fastapi.testclient import TestClient

from magicat.server.app import create_app
from magicat.server.runner import LocalJobRunner
from magicat.server.store import JobStore


def test_full_pipeline_through_api(fixture_video, tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    runner = LocalJobRunner(store, max_workers=1)
    app = create_app(store=store, runner=runner,
                     jobs_root=tmp_path / "jobs")
    client = TestClient(app)

    with open(fixture_video, "rb") as f:
        r = client.post("/api/jobs",
                        files={"file": ("clip.mp4", f, "video/mp4")})
    assert r.status_code == 201
    job_id = r.json()["job_id"]

    # SSE blocks until the job finishes - collect every frame
    frames = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    assert frames[-1]["stage"] == "job"
    assert frames[-1]["state"] in ("done", "failed")

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done"
    assert job["report"]["shots"]["count"] == 3

    for name in ("preview.mp4", "report.html", "premiere_resolve.zip",
                 "manifest.json"):
        assert client.get(
            f"/api/jobs/{job_id}/artifacts/{name}").status_code == 200

    runner.shutdown()


def test_serve_command_exists():
    from typer.testing import CliRunner
    from magicat.cli import app
    result = CliRunner().invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_api_e2e.py -v`
Expected: the e2e PASSES if Tasks 1–5 landed correctly (it exercises only existing pieces); `test_serve_command_exists` FAILS (no serve command). The red here is the CLI test.

- [ ] **Step 3: Implement the serve command** — in `magicat/cli.py` add:

```python
@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8123, "--port"),
    jobs_root: Path = typer.Option(Path("jobs"), "--jobs-root"),
) -> None:
    """Run the Magicat web service (UI at http://HOST:PORT/)."""
    import uvicorn

    from magicat.server.app import create_app

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(jobs_root=jobs_root), host=host, port=port)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_api_e2e.py -v`
Expected: 2 PASS (the e2e takes ~30–60 s — real pipeline). Full suite: 177 passed, 2 skipped.

- [ ] **Step 5: Manual smoke.** Start the server in the background (`.venv/Scripts/magicat serve --port 8123`), then: `curl http://127.0.0.1:8123/healthz` → `{"status":"ok"}`; open the page HTML (`curl http://127.0.0.1:8123/`) and confirm the form markup; stop the server. (Full browser interaction is covered by the TestClient e2e; this just proves uvicorn boots.)

- [ ] **Step 6: README.** Replace the Status line with:

```markdown
**Status:** M4 — launchable v1: web service with live progress, plus the
full deconstruction pipeline (cuts, music, captions + fonts, NLE export).
```

Append:

```markdown
## Web service (M4)

    .venv/Scripts/magicat serve --port 8123

Open http://127.0.0.1:8123/ - paste a short-form URL or upload a clip,
watch per-layer progress live, then preview and download the deliverables.
JSON API: POST /api/jobs, GET /api/jobs/{id}, GET /api/jobs/{id}/events
(SSE), GET /api/jobs/{id}/artifacts/{name}. Optional auth: set
$env:MAGICAT_API_KEY and send X-API-Key. Local-first stack (SQLite +
thread pool + static UI) behind the same seams the cloud deployment
(Celery/Postgres/S3/Next.js) slots into - see the M4 plan's deviation
table.
```

- [ ] **Step 7: Commit**

```bash
git add magicat/cli.py README.md tests/test_api_e2e.py
git commit -m "feat: magicat serve command, API e2e, M4 README"
```

---

## Out of Scope (M4) — deliberate deviations recorded in the header table

- Celery/Redis, Postgres, S3/R2, Next.js, Clerk, Stripe billing — each has a named seam (see deviation table); they are deployment-shape work, not local-v1 work.
- Multi-user tenancy, job quotas, rate limits — post-launch.
- Burned-caption preview overlay (M3 deferral stands).
- Linux CI + the caption fixture font fallback (carried from M3).
- The `[separation]` extra in the prod image (carried; applies when a prod image exists).
- Font-render caching for server throughput (carried from the M3 final review; materializes once `MAGICAT_MAX_JOBS` > 1 runs concurrent caption-heavy pipelines) — deferred, the matcher is correct just not cached.
- UI summary shows the resolved font name(s) only; the full top-3 font candidates WITH confidences live in report.html and the manifest (spec §8/§9 partial — recorded deviation, one fetch away in the same payload).
- First-run note: RapidOCR downloads ~15 MB of models on a cold machine — the API e2e's ~30–60 s estimate assumes warmed caches.

## Self-Review Notes (already applied)

- **Spec coverage:** §13 M4 row — "web app" → Tasks 4/5; "queue" → Task 3 (seam + local impl); "progress" → Tasks 2/5 (SSE); "auth" → Task 4 (API-key gate; Clerk deviation recorded); "downloads" → Task 4 (allowlisted artifacts). §11 idempotency/poison-pill: per-task time/memory limits remain a Celery-deployment concern (recorded).
- **Security:** artifact downloads are an allowlist dict — client input never touches a path; API-key gate on /api/* with /healthz open; UI escapes all report values before innerHTML.
- **Type consistency:** JobStore.Job/Event dataclasses (T1) consumed by runner (T3) and app (T4); run_job's on_progress signature (T2) matches both the runner's lambda (T3) and the fake pipelines in tests (T4/T6); create_app(store, runner, jobs_root) consistent across T4/T5/T6.
- **Counts traced:** 150→156 (T1, +6) →159 (T2, +3) →163 (T3, +4) →172 (T4, +9) →175 (T5, +3) →177 (T6, +2); skips stay 2.


