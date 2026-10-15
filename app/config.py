import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
STT_BACKEND = os.getenv("STT_BACKEND", "whisper")      # whisper | mock
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
CHUNK_TARGET_S = float(os.getenv("CHUNK_TARGET_S", "300"))   # aim for ~5 min chunks
CHUNK_SEARCH_S = float(os.getenv("CHUNK_SEARCH_S", "30"))    # look +/- 30s for a silence to cut at
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
