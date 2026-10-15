import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "STT_BACKEND", "mock")
    monkeypatch.setattr(config, "CHUNK_TARGET_S", 20)
    monkeypatch.setattr(config, "CHUNK_SEARCH_S", 5)
    from app.api import app
    with TestClient(app) as c:
        app.state.runner.backoff_base = 0.01
        yield c


def wait(client, jid, want, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/v1/transcriptions/{jid}").json()
        if s["status"] in want:
            return s
        time.sleep(0.1)
    raise AssertionError(s)


def test_concurrent_uploads_and_formats(client, audio_files):
    ids = []
    for _ in range(4):                                  # more jobs than workers -> queued
        with open(audio_files["mp3"], "rb") as f:
            r = client.post("/v1/transcriptions", files={"file": ("a.mp3", f)})
        assert r.status_code == 202
        ids.append(r.json()["id"])
    for jid in ids:
        wait(client, jid, {"completed"})
    assert client.get(f"/v1/transcriptions/{ids[0]}/result").json()["segments"]
    assert "-->" in client.get(f"/v1/transcriptions/{ids[0]}/result?format=srt").text


def test_bad_file_fails_without_retries_and_result_409(client):
    r = client.post("/v1/transcriptions", files={"file": ("x.mp3", b"garbage")})
    jid = r.json()["id"]
    s = wait(client, jid, {"failed"})
    assert s["attempts"] == 1 and "decode" in s["error"].lower()
    assert client.get(f"/v1/transcriptions/{jid}/result").status_code == 409


def test_transient_failure_is_retried(client, audio_files, monkeypatch):
    t = client.app.state.runner.transcriber
    orig, n = t.transcribe, {"c": 0}

    def flaky(wav, language=None):
        n["c"] += 1
        if n["c"] == 1:
            raise RuntimeError("temporary")
        return orig(wav, language)

    monkeypatch.setattr(t, "transcribe", flaky)
    with open(audio_files["wav"], "rb") as f:
        jid = client.post("/v1/transcriptions", files={"file": ("a.wav", f)}).json()["id"]
    s = wait(client, jid, {"completed"})
    assert s["attempts"] == 2
