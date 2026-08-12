"""Информационная модель неявной обратной связи.

Реакция человека на действие робота в окне 3–5 секунд превращается в скалярную
оценку, а оценки — в пары предпочтений для DPO. Ручная разметка не требуется.

Веса сигналов вынесены в конфиг сознательно: это исследуемый параметр, а не
константа. Валидация модели награды на HRI-корпусах (REACT, UE-HRI) — отдельный
этап, без неё веса остаются гипотезой.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..schema import Emotion

# Знак вклада эмоции: положительные эмоции поощряются, негативные штрафуются.
EMOTION_VALENCE: dict[Emotion, float] = {
    Emotion.HAPPY: 1.0,
    Emotion.SURPRISE: 0.2,
    Emotion.NEUTRAL: 0.0,
    Emotion.UNKNOWN: 0.0,
    Emotion.SAD: -0.6,
    Emotion.FEAR: -0.8,
    Emotion.DISGUST: -0.9,
    Emotion.ANGRY: -1.0,
}


@dataclass(slots=True)
class RewardWeights:
    emotion_shift: float = 1.0
    continued_dialogue: float = 1.0
    turned_away: float = -0.8
    left_frame: float = -1.2
    interrupted: float = -1.0
    repeated_question: float = -1.0
    engagement_seconds: float = 0.05


@dataclass(slots=True)
class ReactionWindow:
    """Наблюдение за человеком в течение 3–5 секунд после действия робота."""

    emotion_before: Emotion = Emotion.UNKNOWN
    emotion_after: Emotion = Emotion.UNKNOWN
    continued_dialogue: bool = False
    turned_away: bool = False
    left_frame: bool = False
    interrupted_robot: bool = False
    repeated_question: bool = False
    engagement_seconds: float = 0.0


@dataclass(slots=True)
class InteractionRecord:
    """Одно действие робота вместе с последовавшей реакцией."""

    situation_key: str
    messages: list[dict[str, str]]
    decision_json: str
    reaction: ReactionWindow
    weights: RewardWeights = field(default_factory=RewardWeights)

    @property
    def reward(self) -> float:
        return score_reaction(self.reaction, self.weights)


def score_reaction(reaction: ReactionWindow, weights: RewardWeights | None = None) -> float:
    """Скалярная оценка действия робота по реакции человека."""
    weights = weights or RewardWeights()

    valence_before = EMOTION_VALENCE.get(reaction.emotion_before, 0.0)
    valence_after = EMOTION_VALENCE.get(reaction.emotion_after, 0.0)

    total = weights.emotion_shift * (valence_after - valence_before)
    total += weights.engagement_seconds * reaction.engagement_seconds

    if reaction.continued_dialogue:
        total += weights.continued_dialogue
    if reaction.turned_away:
        total += weights.turned_away
    if reaction.left_frame:
        total += weights.left_frame
    if reaction.interrupted_robot:
        total += weights.interrupted
    if reaction.repeated_question:
        total += weights.repeated_question

    return total


def build_preference_pairs(
    records: Iterable[InteractionRecord],
    min_margin: float = 0.5,
) -> list[dict[str, Any]]:
    """Собирает пары chosen/rejected для DPO.

    Сравниваются только действия в одной и той же ситуации: сравнение наград
    между разными ситуациями бессмысленно, потому что сами ситуации различаются
    по тому, насколько на них вообще возможна хорошая реакция.

    `min_margin` отсекает пары, где разница в награде в пределах шума измерения
    эмоций — на таких парах DPO учит случайности.
    """
    grouped: dict[str, list[InteractionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.situation_key].append(record)

    pairs: list[dict[str, Any]] = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        ranked = sorted(group, key=lambda item: item.reward, reverse=True)
        best, worst = ranked[0], ranked[-1]
        if best.reward - worst.reward < min_margin:
            continue
        pairs.append(
            {
                "prompt": _prompt_messages(best.messages),
                "chosen": [{"role": "assistant", "content": best.decision_json}],
                "rejected": [{"role": "assistant", "content": worst.decision_json}],
                "margin": best.reward - worst.reward,
            }
        )
    return pairs


def _prompt_messages(messages: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Промпт без ответа ассистента."""
    return [dict(message) for message in messages if message.get("role") != "assistant"]
