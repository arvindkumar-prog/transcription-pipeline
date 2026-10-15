import subprocess
import pytest


@pytest.fixture(scope="session")
def audio_files(tmp_path_factory):
    """70s of tone with a 1s silence every 10s, as both WAV and MP3 (exercises format handling)."""
    d = tmp_path_factory.mktemp("audio")
    expr = "aevalsrc='if(lt(mod(t,10),9), 0.3*sin(2*PI*440*t), 0)':d=70:s=22050"
    wav, mp3 = d / "a.wav", d / "a.mp3"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", expr, str(wav)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav), str(mp3)], check=True)
    return {"wav": wav, "mp3": mp3}
