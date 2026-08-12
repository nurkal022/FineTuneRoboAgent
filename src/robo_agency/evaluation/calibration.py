"""Этап 3: калибровка порога вмешательства.

Важное отличие от поля `confidence`, которое модель порождает текстом: оно
обучено на константах и калиброванным не является. Настоящая уверенность
берётся из логвероятности токена решения при декодировании и калибруется здесь.

Порог выбирается не по максимуму F1, а под ограничение на долю ложных
вмешательств: в эксплуатации навязчивый робот хуже, чем молчаливый, и это
инженерное решение должно быть явным, а не спрятанным в метрике.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(slots=True)
class ThresholdPoint:
    threshold: float
    timeliness: float
    false_intervention_rate: float
    f1: float


@dataclass(slots=True)
class CalibrationResult:
    threshold: float
    temperature: float
    curve: list[ThresholdPoint]
    satisfied_constraint: bool
    degenerate: bool

    def describe(self) -> str:
        status = "выполнено" if self.satisfied_constraint else "НЕ выполнено"
        lines = [
            f"Порог вмешательства: {self.threshold:.3f}",
            f"Температура: {self.temperature:.3f}",
            f"Ограничение на ложные вмешательства: {status}",
            f"Точек на кривой: {len(self.curve)}",
        ]
        if self.degenerate:
            lines.append(
                "ВНИМАНИЕ: вырожденная калибровка — при этом пороге робот не "
                "действует никогда. Ограничение формально выполнено ценой полного "
                "молчания. Ослабьте бюджет ложных вмешательств или улучшите модель."
            )
        return "\n".join(lines)


def apply_temperature(probability: float, temperature: float) -> float:
    """Температурное шкалирование вероятности.

    Работает через логит, поэтому монотонность сохраняется, а переуверенность
    модели (типичная после SFT) сглаживается.
    """
    probability = min(max(probability, 1e-6), 1 - 1e-6)
    logit = math.log(probability / (1 - probability))
    return 1.0 / (1.0 + math.exp(-logit / temperature))


def _point(scores: Sequence[float], labels: Sequence[bool], threshold: float) -> ThresholdPoint:
    true_positive = sum(1 for s, y in zip(scores, labels) if y and s >= threshold)
    false_negative = sum(1 for s, y in zip(scores, labels) if y and s < threshold)
    false_positive = sum(1 for s, y in zip(scores, labels) if not y and s >= threshold)
    negatives = sum(1 for y in labels if not y)

    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ThresholdPoint(
        threshold=threshold,
        timeliness=recall,
        false_intervention_rate=false_positive / negatives if negatives else 0.0,
        f1=f1,
    )


def calibrate(
    act_probabilities: Sequence[float],
    gold_is_act: Sequence[bool],
    max_false_intervention_rate: float = 0.10,
    temperature: float = 1.0,
    steps: int = 101,
) -> CalibrationResult:
    """Выбирает порог, максимизирующий своевременность при ограничении на ложные срабатывания."""
    if len(act_probabilities) != len(gold_is_act):
        raise ValueError("Длины вероятностей и меток не совпадают")
    if not act_probabilities:
        raise ValueError("Пустая валидационная выборка")

    scores = [apply_temperature(p, temperature) for p in act_probabilities]
    curve = [_point(scores, gold_is_act, i / (steps - 1)) for i in range(steps)]

    feasible = [p for p in curve if p.false_intervention_rate <= max_false_intervention_rate]
    if feasible:
        best = max(feasible, key=lambda p: (p.timeliness, -p.threshold))
        satisfied = True
    else:
        # Ограничение недостижимо — сообщаем об этом, а не подменяем цель молча.
        best = min(curve, key=lambda p: p.false_intervention_rate)
        satisfied = False

    return CalibrationResult(
        threshold=best.threshold,
        temperature=temperature,
        curve=curve,
        satisfied_constraint=satisfied,
        # Порог, при котором робот молчит всегда, формально удовлетворяет любому
        # бюджету ложных вмешательств. Это не решение, и молчать об этом нельзя.
        degenerate=best.timeliness == 0.0,
    )
