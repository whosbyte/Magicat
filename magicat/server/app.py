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
import secrets
import uuid
from contextlib import asynccontextmanager
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

MAX_UPLOAD_BYTES = 2 * 1024**3   # 2 GiB - generous for short-form clips

ARTIFACTS = {
    "preview.mp4": ("exports/preview.mp4", "video/mp4"),
    "report.html": ("exports/report.html", "text/html"),
    "premiere_resolve.zip": ("exports/premiere_resolve.zip",
                             "application/zip"),
    "manifest.json": ("manifest.json", "application/json"),
}


def _require_api_key(request: Request) -> None:
    expected = os.environ.get("MAGICAT_API_KEY")
    if expected:
        provided = request.headers.get("X-API-Key") or ""
        if not secrets.compare_digest(provided, expected):
            raise HTTPException(status_code=401, detail="invalid API key")


def create_app(store: JobStore | None = None,
               runner: JobRunner | None = None,
               jobs_root: Path | str = "jobs") -> FastAPI:
    jobs_root = Path(jobs_root)
    store = store or JobStore(jobs_root / "jobs.db")
    runner = runner or LocalJobRunner(
        store, max_workers=int(os.environ.get("MAGICAT_MAX_JOBS", "1")))

    # T3-review extra: shut the runner down on app teardown so uvicorn
    # reloads don't leak worker threads mid-job. FastAPI 0.136 deprecation-
    # warns on @app.on_event("shutdown"), so we use the lifespan pattern.
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        runner.shutdown()

    app = FastAPI(title="Magicat", version="0.1.0", lifespan=lifespan)
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
            written = 0
            with open(upload_path, "wb") as out:
                while chunk := await file.read(1 << 20):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        out.close()
                        upload_path.unlink(missing_ok=True)
                        raise HTTPException(status_code=413,
                                            detail="upload too large")
                    out.write(chunk)
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
