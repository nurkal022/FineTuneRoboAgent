"""Информационные модели ситуации и решения.

Это центральный контракт всей системы: конвертеры датасетов приводят
разнородные корпуса именно к `Situation` / `RobotDecision`, обучение идёт
на этих схемах, а грамматика декодера строится из `RobotDecision`.

Диапазоны серво заданы здесь один раз и попадают в грамматику автоматически,
поэтому модель физически не может выдать угол вне механических пределов.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, model_validator

# --- Механические пределы робота -------------------------------------------
PAN_MIN, PAN_MAX = -90, 90
TILT_MIN, TILT_MAX = -30, 30


class DecisionType(str, Enum):
    """Что робот делает в текущий момент времени.

    WAIT и OBSERVE — то, чего нет в реактивных системах: возможность
    осознанно не действовать.
    """

    ACT = "ACT"          # вмешаться: сказать и/или подвигаться
    WAIT = "WAIT"        # промолчать, ситуация не требует робота
    OBSERVE = "OBSERVE"  # не вмешиваться, но продолжать следить


class Gesture(str, Enum):
    NONE = "none"
    NOD = "nod"
    SHAKE = "shake"
    TURN_TO_SPEAKER = "turn_to_speaker"
    SCAN = "scan"


class Emotion(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISE = "surprise"
    FEAR = "fear"
    DISGUST = "disgust"
    UNKNOWN = "unknown"


# --- Информационная модель ситуации (вход) ---------------------------------


class Person(BaseModel):
    id: str | None = None
    name: str | None = None
    known: bool = False
    last_seen: str | None = None


class Visual(BaseModel):
    faces: int = 0
    distance: str | None = None          # "близко" | "средне" | "далеко"
    looking_at_robot: bool = False
    emotion: Emotion = Emotion.UNKNOWN


class Audio(BaseModel):
    speech: str | None = None            # распознанная реплика, None = тишина
    lang: str | None = None
    addressed_to_robot: bool | None = None


class Situation(BaseModel):
    """Единое представление потока восприятия.

    `observations` — свободный список недавних наблюдений. Именно он
    позволяет приводить к этой схеме корпуса из других доменов
    (ProactiveBench, When2Call), где восприятие не робототехническое.
    """

    time: str | None = None
    silence_sec: float = 0.0
    person: Person | None = None
    visual: Visual | None = None
    audio: Audio | None = None
    observations: list[str] = Field(default_factory=list)
    memory: list[str] = Field(default_factory=list)
    last_robot_action: str | None = None

    def to_prompt_json(self) -> dict[str, Any]:
        """Компактный JSON для промпта: пустые поля выкидываются."""
        return self.model_dump(exclude_none=True, exclude_defaults=False, mode="json")


# --- Информационная модель решения (выход) ---------------------------------


class Speech(BaseModel):
    text: str
    lang: str = "ru"


class Motion(BaseModel):
    pan: Annotated[int, Field(ge=PAN_MIN, le=PAN_MAX)] = 0
    tilt: Annotated[int, Field(ge=TILT_MIN, le=TILT_MAX)] = 0
    gesture: Gesture = Gesture.NONE


class RobotDecision(BaseModel):
    """Решение робота. Схема этого класса компилируется в грамматику декодера."""

    decision: DecisionType
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    speech: Speech | None = None
    motion: Motion | None = None
    memory_write: str | None = None

    @model_validator(mode="after")
    def _no_action_payload_when_not_acting(self) -> RobotDecision:
        """WAIT/OBSERVE не могут нести речь или движение.

        Без этой проверки модель быстро выучивает вырожденную стратегию:
        формально ответить WAIT, но всё равно приложить реплику.
        """
        if self.decision is not DecisionType.ACT:
            if self.speech is not None or self.motion is not None:
                raise ValueError(
                    f"decision={self.decision.value} не может содержать speech или motion"
                )
        return self

    @model_validator(mode="after")
    def _act_must_do_something(self) -> RobotDecision:
        if self.decision is DecisionType.ACT and self.speech is None and self.motion is None:
            raise ValueError("decision=ACT требует speech или motion")
        return self


def decision_json_schema() -> dict[str, Any]:
    """JSON Schema решения — для guided decoding в vLLM."""
    return RobotDecision.model_json_schema()
