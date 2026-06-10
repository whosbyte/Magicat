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
