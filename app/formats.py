"""Downstream-friendly renderings of a transcript dict."""


def _ts(sec: float, sep: str) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def to_srt(t: dict) -> str:
    return "\n".join(f"{s['id'] + 1}\n{_ts(s['start'], ',')} --> {_ts(s['end'], ',')}\n{s['text']}\n"
                     for s in t["segments"])


def to_vtt(t: dict) -> str:
    body = "\n".join(f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}\n{s['text']}\n"
                     for s in t["segments"])
    return "WEBVTT\n\n" + body


def to_text(t: dict) -> str:
    return t["text"]
