"""SQLite job store (stand-in for Postgres). One short-lived connection per call = thread safe."""
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

QUEUED, PROCESSING, COMPLETED, FAILED = "queued", "processing", "completed", "failed"


class JobStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, filename TEXT, audio_path TEXT, language TEXT,
                status TEXT, attempts INTEGER DEFAULT 0, progress TEXT, error TEXT,
                created_at REAL, updated_at REAL)""")

    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def create(self, filename: str, audio_path: str, language: Optional[str], job_id: str = None) -> str:
        job_id = job_id or uuid.uuid4().hex
        now = time.time()
        with self._conn() as c:
            c.execute("INSERT INTO jobs(id,filename,audio_path,language,status,created_at,updated_at)"
                      " VALUES(?,?,?,?,?,?,?)", (job_id, filename, audio_path, language, QUEUED, now, now))
        return job_id

    def update(self, job_id: str, **fields) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def get(self, job_id: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def incomplete(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM jobs WHERE status IN (?,?)", (QUEUED, PROCESSING)).fetchall()
        return [dict(r) for r in rows]
