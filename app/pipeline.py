"""Core: audio file -> normalised wav -> silence-aligned chunks -> transcribe -> merged timestamps.

Each finished chunk is checkpointed to disk, so a retry resumes instead of starting over.
"""
import json
import os
from pathlib import Path
from typing import Callable, Optional

from . import audio
from .transcriber import Transcriber


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)                        # never leaves a half-written checkpoint


def transcribe_file(src: Path, workdir: Path, transcriber: Transcriber, *,
                    language: Optional[str] = None, chunk_target: float = 300,
                    chunk_window: float = 30,
                    on_progress: Optional[Callable[[int, int], None]] = None) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)

    wav = workdir / "normalized.wav"
    if not wav.exists():
        audio.normalize(src, wav)
    duration = audio.probe_duration(wav)

    if duration <= chunk_target + chunk_window:
        chunks = [(0.0, duration)]               # short file: no splitting, no silence scan
    else:
        chunks = audio.plan_chunks(duration, audio.detect_silences(wav), chunk_target, chunk_window)

    segments, detected = [], language
    for i, (start, end) in enumerate(chunks):
        ckpt = workdir / f"chunk_{i:04d}.json"
        if ckpt.exists():                        # resume
            data = json.loads(ckpt.read_text())
        else:
            piece = wav if len(chunks) == 1 else workdir / f"chunk_{i:04d}.wav"
            if piece is not wav:
                audio.cut_chunk(wav, start, end, piece)
            segs, lang = transcriber.transcribe(piece, language)
            data = {"language": lang,
                    "segments": [{"start": round(s.start + start, 3),   # shift to global time
                                  "end": round(s.end + start, 3),
                                  "text": s.text} for s in segs]}
            _atomic_write(ckpt, data)
            if piece is not wav:
                piece.unlink(missing_ok=True)
        detected = detected or data["language"]
        segments.extend(data["segments"])
        if on_progress:
            on_progress(i + 1, len(chunks))

    for n, s in enumerate(segments):
        s["id"] = n
    return {"language": detected, "duration": round(duration, 3),
            "chunks": len(chunks), "segments": segments,
            "text": " ".join(s["text"] for s in segments)}
