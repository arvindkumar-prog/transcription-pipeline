"""STT backends behind one tiny interface so the engine is swappable."""
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol


@dataclass
class Segment:
    start: float
    end: float
    text: str


class Transcriber(Protocol):
    def transcribe(self, wav: Path, language: Optional[str] = None) -> tuple[list[Segment], str]:
        """Return (segments with times relative to this file, detected language)."""


class WhisperTranscriber:
    """faster-whisper: CTranslate2 Whisper, ~4x faster than openai-whisper, int8 on CPU."""

    def __init__(self, model_size: str = "base"):
        self._size = model_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:                       # load once even with several workers
            if self._model is None:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self._size, device="cpu", compute_type="float32")
        return self._model

    def transcribe(self, wav, language=None):
        segs, info = self._load().transcribe(str(wav), language=language, vad_filter=True)
        return [Segment(s.start, s.end, s.text.strip()) for s in segs], info.language


class MockTranscriber:
    """Deterministic fake: one segment per 5s. Lets the whole pipeline be tested without a model."""

    def transcribe(self, wav, language=None):
        with wave.open(str(wav)) as w:
            dur = w.getnframes() / w.getframerate()
        out, t, i = [], 0.0, 0
        while t < dur:
            end = min(t + 5.0, dur)
            out.append(Segment(round(t, 3), round(end, 3), f"mock segment {i} of this chunk"))
            t, i = end, i + 1
        return out, language or "en"


def get_transcriber(backend: str, model: str = "base") -> Transcriber:
    return MockTranscriber() if backend == "mock" else WhisperTranscriber(model)
