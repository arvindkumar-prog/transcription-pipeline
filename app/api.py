import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse

from . import config, formats
from .store import COMPLETED, FAILED, JobStore
from .transcriber import get_transcriber
from .worker import JobRunner


@asynccontextmanager
async def lifespan(app: FastAPI):
    data = config.DATA_DIR
    for d in ("uploads", "work", "transcripts"):
        (data / d).mkdir(parents=True, exist_ok=True)
    app.state.store = JobStore(data / "jobs.db")
    app.state.runner = JobRunner(app.state.store, get_transcriber(config.STT_BACKEND, config.WHISPER_MODEL),
                                 data, config.MAX_WORKERS, config.MAX_ATTEMPTS, backoff_base=2.0)
    app.state.runner.recover()
    yield
    app.state.runner.pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Transcription Pipeline", lifespan=lifespan)


@app.get("/healthz")
def health():
    return {"ok": True}


@app.post("/v1/transcriptions", status_code=202)
async def create(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    """Stream upload to disk (bounded memory), enqueue, return immediately."""
    store, runner = app.state.store, app.state.runner
    suffix = Path(file.filename or "audio").suffix[:10]
    job_id_dest = config.DATA_DIR / "uploads"
    tmp = job_id_dest / f"incoming_{id(file)}{suffix}"
    size, limit = 0, config.MAX_UPLOAD_MB * 1024 * 1024
    with tmp.open("wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                out.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {config.MAX_UPLOAD_MB} MB")
            out.write(chunk)
    job_id = store.create(file.filename or "audio", "", language)
    final = job_id_dest / f"{job_id}{suffix}"
    shutil.move(tmp, final)
    store.update(job_id, audio_path=str(final))
    runner.submit(job_id)
    return {"id": job_id, "status": "queued", "status_url": f"/v1/transcriptions/{job_id}"}


def _job_or_404(job_id: str) -> dict:
    job = app.state.store.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/v1/transcriptions/{job_id}")
def status(job_id: str):
    j = _job_or_404(job_id)
    return {k: j[k] for k in ("id", "filename", "status", "attempts", "progress", "error")}


@app.get("/v1/transcriptions/{job_id}/result")
def result(job_id: str, format: str = Query("json", pattern="^(json|srt|vtt|txt)$")):
    j = _job_or_404(job_id)
    if j["status"] != COMPLETED:
        raise HTTPException(409, f"job is {j['status']}")
    t = json.loads((config.DATA_DIR / "transcripts" / f"{job_id}.json").read_text())
    if format == "json":
        return t
    body = {"srt": formats.to_srt, "vtt": formats.to_vtt, "txt": formats.to_text}[format](t)
    return PlainTextResponse(body)


@app.post("/v1/transcriptions/{job_id}/retry", status_code=202)
def retry(job_id: str):
    j = _job_or_404(job_id)
    if j["status"] != FAILED:
        raise HTTPException(409, "only failed jobs can be retried")
    app.state.runner.retry(job_id)
    return {"id": job_id, "status": "queued"}
