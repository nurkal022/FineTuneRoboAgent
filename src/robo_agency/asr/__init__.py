"""Речевой контур: дообучение Whisper под казахский язык."""

from .config import WhisperConfig, load_config
from .metrics import ErrorRate, character_error_rate, word_error_rate
from .normalize import characters, normalize, words

__all__ = [
    "ErrorRate",
    "WhisperConfig",
    "character_error_rate",
    "characters",
    "load_config",
    "normalize",
    "word_error_rate",
    "words",
]
