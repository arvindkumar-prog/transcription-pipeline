"""Bounded worker pool with retry + exponential backoff. Jobs are resumable via chunk checkpoints."""
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config
from .errors import PermanentError
from .pipeline import transcribe_file
from .store import COMPLETED, FAILED, PROCESSING, QUEUED, JobStore
from .transcriber import Transcriber

log = logging.getLogger("worker")


class JobRunner:
    def __init__(self, store: JobStore, transcriber: Transcriber, data_dir: Path,
                 max_workers: int = 2, max_attempts: int = 3, backoff_base: float = 2.0):
        self.store, self.transcriber, self.data_dir = store, transcriber, data_dir
        self.max_attempts, self.backoff_base = max_attempts, backoff_base
        self.pool = ThreadPoolExecutor(max_workers=max_workers)   # backpressure: N jobs at a time

    def submit(self, job_id: str):
        return self.pool.submit(self._run, job_id)

    def recover(self) -> int:
        """On startup, re-enqueue anything that was queued/processing when the process died."""
        jobs = self.store.incomplete()
        for j in jobs:
            self.submit(j["id"])
        return len(jobs)

    def _run(self, job_id: str) -> None:
        job = self.store.get(job_id)
        workdir = self.data_dir / "work" / job_id
        result_path = self.data_dir / "transcripts" / f"{job_id}.json"
        for attempt in range(job["attempts"] + 1, self.max_attempts + 1):
            self.store.update(job_id, status=PROCESSING, attempts=attempt)
            try:
                result = transcribe_file(
                    Path(job["audio_path"]), workdir, self.transcriber,
                    language=job["language"],
                    chunk_target=config.CHUNK_TARGET_S, chunk_window=config.CHUNK_SEARCH_S,
                    on_progress=lambda d, t: self.store.update(job_id, progress=f"{d}/{t}"))
                result_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = result_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(result))
                os.replace(tmp, result_path)
                self.store.update(job_id, status=COMPLETED, error=None)
                return
            except PermanentError as e:                  # bad input: don't burn retries
                self.store.update(job_id, status=FAILED, error=str(e))
                return
            except Exception as e:                       # assume transient
                log.warning("job %s attempt %d failed: %s", job_id, attempt, e)
                self.store.update(job_id, error=str(e))
                if attempt == self.max_attempts:
                    self.store.update(job_id, status=FAILED)
                    return
                time.sleep(self.backoff_base ** attempt)

    def retry(self, job_id: str) -> None:
        self.store.update(job_id, status=QUEUED, attempts=0, error=None)   # checkpoints are kept
        self.submit(job_id)
