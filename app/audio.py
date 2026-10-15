"""ffmpeg helpers: normalise any input to 16 kHz mono WAV, probe, find silences, cut chunks."""
import re
import subprocess
from pathlib import Path

from .errors import InvalidAudioError


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def normalize(src: Path, dst: Path) -> None:
    """Decode anything ffmpeg understands (mp3, m4a, ogg, flac, mp4...) to 16k mono PCM WAV."""
    r = _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn",
              "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)])
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size < 100:
        dst.unlink(missing_ok=True)
        raise InvalidAudioError(f"Cannot decode audio: {r.stderr.strip()[:300]}")


def probe_duration(wav: Path) -> float:
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "default=nw=1:nk=1", str(wav)])
    try:
        return float(r.stdout.strip())
    except ValueError:
        raise InvalidAudioError("Cannot read audio duration")


def detect_silences(wav: Path, noise_db: int = -35, min_s: float = 0.4) -> list[tuple[float, float]]:
    r = _run(["ffmpeg", "-v", "info", "-i", str(wav),
              "-af", f"silencedetect=noise={noise_db}dB:d={min_s}", "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    return list(zip(starts, ends))


def plan_chunks(duration: float, silences: list[tuple[float, float]],
                target: float, window: float) -> list[tuple[float, float]]:
    """Greedy: cut near every `target` seconds, snapping to the closest silence midpoint
    inside +/- `window`. Cutting in silence avoids splitting words. Falls back to a hard cut."""
    mids = [(s + e) / 2 for s, e in silences]
    bounds, pos = [0.0], 0.0
    while duration - pos > target + window:
        ideal = pos + target
        cands = [m for m in mids if ideal - window <= m <= ideal + window]
        cut = min(cands, key=lambda m: abs(m - ideal)) if cands else ideal
        bounds.append(cut)
        pos = cut
    bounds.append(duration)
    return list(zip(bounds[:-1], bounds[1:]))


def cut_chunk(wav: Path, start: float, end: float, out: Path) -> None:
    r = _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
              "-i", str(wav), "-c", "copy", str(out)])
    if r.returncode != 0:
        raise RuntimeError(f"chunk cut failed: {r.stderr[:200]}")
