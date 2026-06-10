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
from contextlib import closing
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
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
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
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job.job_id, job.status, job.input_arg, job.workdir,
                 job.error, job.created_at, job.updated_at))
        return job

    def get_job(self, job_id: str) -> Job | None:
        with closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?",
                               (job_id,)).fetchone()
        return Job(**dict(row)) if row else None

    def set_status(self, job_id: str, status: str,
                   error: str | None = None) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status {status!r}")
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? "
                "WHERE job_id = ?",
                (status, error, time.time(), job_id))

    def add_event(self, job_id: str, stage: str, state: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO events (job_id, stage, state, ts) "
                "VALUES (?, ?, ?, ?)",
                (job_id, stage, state, time.time()))

    def events_since(self, job_id: str, after_seq: int) -> list[Event]:
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE job_id = ? AND seq > ? "
                "ORDER BY seq", (job_id, after_seq)).fetchall()
        return [Event(**dict(r)) for r in rows]

    def list_jobs(self, limit: int = 50) -> list[Job]:
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC "
                "LIMIT ?", (limit,)).fetchall()
        return [Job(**dict(r)) for r in rows]
