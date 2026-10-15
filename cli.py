"""Usage: python cli.py audio.mp3 [--format json|srt|vtt|txt] [--language en] [--backend whisper|mock]"""
import argparse
import json
import tempfile
from pathlib import Path

from app import config, formats
from app.pipeline import transcribe_file
from app.transcriber import get_transcriber

p = argparse.ArgumentParser()
p.add_argument("audio", type=Path)
p.add_argument("--format", default="json", choices=["json", "srt", "vtt", "txt"])
p.add_argument("--language", default=None)
p.add_argument("--backend", default=config.STT_BACKEND)
p.add_argument("--model", default=config.WHISPER_MODEL)
p.add_argument("--workdir", type=Path, default=None, help="keep for resumable runs")
a = p.parse_args()

work = a.workdir or Path(tempfile.mkdtemp(prefix="stt_"))
t = transcribe_file(a.audio, work, get_transcriber(a.backend, a.model), language=a.language,
                    chunk_target=config.CHUNK_TARGET_S, chunk_window=config.CHUNK_SEARCH_S,
                    on_progress=lambda d, n: print(f"chunk {d}/{n}", flush=True) if n > 1 else None)
print({"json": lambda x: json.dumps(x, indent=2), "srt": formats.to_srt,
       "vtt": formats.to_vtt, "txt": formats.to_text}[a.format](t))
