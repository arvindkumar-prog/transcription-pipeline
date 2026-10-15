import json
from pathlib import Path

import pytest

from app import audio
from app.errors import InvalidAudioError
from app.formats import to_srt, to_vtt
from app.pipeline import transcribe_file
from app.transcriber import MockTranscriber


def test_plan_chunks_snaps_to_silence():
    chunks = audio.plan_chunks(100, [(19.5, 20.5), (38, 39)], target=20, window=5)
    assert chunks[0] == (0.0, 20.0)                 # snapped to silence midpoint
    assert chunks[-1][1] == 100
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))   # contiguous, no gaps


def test_plan_chunks_hard_cut_without_silence():
    assert audio.plan_chunks(100, [], 40, 5)[0] == (0.0, 40.0)


@pytest.mark.parametrize("fmt", ["wav", "mp3"])
def test_formats_and_global_timestamps(audio_files, tmp_path, fmt):
    t = transcribe_file(audio_files[fmt], tmp_path, MockTranscriber(), chunk_target=20, chunk_window=5)
    assert t["chunks"] >= 3
    starts = [s["start"] for s in t["segments"]]
    assert starts == sorted(starts)                 # monotonic across chunks
    assert t["segments"][-1]["end"] == pytest.approx(70, abs=0.5)
    assert "00:00:00,000 -->" in to_srt(t) and to_vtt(t).startswith("WEBVTT")


def test_resume_skips_finished_chunks(audio_files, tmp_path):
    calls = []

    class Flaky(MockTranscriber):
        def transcribe(self, wav, language=None):
            calls.append(wav.name)
            if len(calls) == 3:
                raise RuntimeError("boom")
            return super().transcribe(wav, language)

    with pytest.raises(RuntimeError):
        transcribe_file(audio_files["wav"], tmp_path, Flaky(), chunk_target=20, chunk_window=5)
    calls.clear()
    t = transcribe_file(audio_files["wav"], tmp_path, Flaky(), chunk_target=20, chunk_window=5)
    assert "chunk_0000.wav" not in calls and "chunk_0001.wav" not in calls   # resumed from checkpoint
    assert t["chunks"] >= 3


def test_invalid_audio_is_permanent(tmp_path):
    bad = tmp_path / "x.mp3"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(InvalidAudioError):
        transcribe_file(bad, tmp_path / "w", MockTranscriber())
