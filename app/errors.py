class PermanentError(Exception):
    """Retrying will not help (corrupt / non-audio input)."""


class InvalidAudioError(PermanentError):
    pass
