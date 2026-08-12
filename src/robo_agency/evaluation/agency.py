"""Метрики агентности.

Ключевая идея набора: одной точности мало. Модель, которая всегда отвечает ACT,
получит идеальную своевременность и при этом будет невыносимой в эксплуатации.
Поэтому своевременность всегда публикуется вместе с ложными вмешательствами
и навязчивостью.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..schema import DecisionType


@dataclass(slots=True)
class Prediction:
    gold: DecisionType
    predicted: DecisionType
    addressed_to_robot: bool = False
    duration_sec: float = 0.0


@dataclass(slots=True)
class AgencyMetrics:
    timeliness: float
    false_intervention_rate: float
    precision_act: float
    f1_act: float
    decision_accuracy: float
    intrusiveness_per_min: float
    support: int

    def describe(self) -> str:
        return (
            f"Своевременность (recall ACT):      {self.timeliness:.3f}\n"
            f"Ложные вмешательства:              {self.false_intervention_rate:.3f}\n"
            f"Точность ACT:                      {self.precision_act:.3f}\n"
            f"F1 по ACT:                         {self.f1_act:.3f}\n"
            f"Точность решения (3 класса):       {self.decision_accuracy:.3f}\n"
            f"Навязчивость (вмешательств/мин):   {self.intrusiveness_per_min:.3f}\n"
            f"Примеров:                          {self.support}"
        )


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute(predictions: Sequence[Prediction]) -> AgencyMetrics:
    if not predictions:
        raise ValueError("Пустой список предсказаний")

    true_positive = sum(
        1 for p in predictions if p.gold is DecisionType.ACT and p.predicted is DecisionType.ACT
    )
    false_negative = sum(
        1 for p in predictions if p.gold is DecisionType.ACT and p.predicted is not DecisionType.ACT
    )
    false_positive = sum(
        1 for p in predictions if p.gold is not DecisionType.ACT and p.predicted is DecisionType.ACT
    )
    gold_non_act = sum(1 for p in predictions if p.gold is not DecisionType.ACT)

    timeliness = _safe_div(true_positive, true_positive + false_negative)
    precision = _safe_div(true_positive, true_positive + false_positive)
    f1 = _safe_div(2 * precision * timeliness, precision + timeliness)

    accuracy = _safe_div(
        sum(1 for p in predictions if p.gold is p.predicted), len(predictions)
    )

    # Навязчивость считается только по фрагментам, где к роботу не обращались:
    # вмешательство в ответ на обращение навязчивостью не является.
    unaddressed = [p for p in predictions if not p.addressed_to_robot]
    unaddressed_acts = sum(1 for p in unaddressed if p.predicted is DecisionType.ACT)
    total_minutes = sum(p.duration_sec for p in unaddressed) / 60.0

    return AgencyMetrics(
        timeliness=timeliness,
        false_intervention_rate=_safe_div(false_positive, gold_non_act),
        precision_act=precision,
        f1_act=f1,
        decision_accuracy=accuracy,
        intrusiveness_per_min=_safe_div(unaddressed_acts, total_minutes),
        support=len(predictions),
    )
