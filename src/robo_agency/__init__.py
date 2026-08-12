"""Обучение LLM проактивной агентности для управления вербальным роботом."""

from .schema import (
    Audio,
    DecisionType,
    Emotion,
    Gesture,
    Motion,
    Person,
    RobotDecision,
    Situation,
    Speech,
    Visual,
    decision_json_schema,
)

__all__ = [
    "Audio",
    "DecisionType",
    "Emotion",
    "Gesture",
    "Motion",
    "Person",
    "RobotDecision",
    "Situation",
    "Speech",
    "Visual",
    "decision_json_schema",
]

__version__ = "0.1.0"
