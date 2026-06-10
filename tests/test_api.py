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


def test_upload_too_large_rejected(client, monkeypatch):
    from magicat.server import app as app_module
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/api/jobs",
                    files={"file": ("big.mp4", b"x" * 64, "video/mp4")})
    assert r.status_code == 413
