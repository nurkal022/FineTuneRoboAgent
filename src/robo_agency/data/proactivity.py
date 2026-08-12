"""Конвертация корпусов проактивности в информационную модель ситуации.

Покрывает ProactiveBench (`thunlp/ProactiveAgent`) и When2Call, а также любой
корпус вида «поток наблюдений → предложенное действие → принято/отклонено».

Это и есть методика приведения разнородных корпусов к единой информационной
модели ситуации, заявленная в п.4 научной новизны: домены разные (десктопная
активность, диалог с инструментами, HRI), схема на выходе одна.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from ..prompts import build_example
from ..schema import Audio, DecisionType, RobotDecision, Situation, Speech
from .fields import as_observations, as_text, find_field, normalize_label

logger = logging.getLogger(__name__)

OBSERVATION_CANDIDATES = (
    "events", "event", "observations", "observation", "context", "history",
    "activity", "user_activity", "scenario", "state", "input", "conversation",
)
ACTION_CANDIDATES = (
    "task", "prediction", "proposed_task", "candidate_task", "task_description",
    "action", "response", "assistant", "output", "target",
)
LABEL_CANDIDATES = (
    "label", "accepted", "is_accepted", "judgement", "judgment", "human_label",
    "annotation", "verdict", "preference",
)


@dataclass(slots=True)
class ProactivityFieldMap:
    """Явное указание колонок. Любое поле None — определяется автоматически."""

    observation: str | None = None
    action: str | None = None
    label: str | None = None


@dataclass(slots=True)
class ProactivityConfig:
    """Параметры конвертации.

    `confidence_*` — вспомогательные константы. Обученное поле `confidence`
    не является калиброванным: калибровка выполняется на этапе 3 по логвероятностям
    токена решения, а не по этому числу (см. calibration.py).
    """

    confidence_accept: float = 0.85
    confidence_reject: float = 0.85
    reject_decision: DecisionType = DecisionType.WAIT
    max_observations: int = 12
    field_map: ProactivityFieldMap = field(default_factory=ProactivityFieldMap)


def _row_to_pair(
    row: dict[str, Any],
    columns: dict[str, str],
    config: ProactivityConfig,
) -> tuple[Situation, RobotDecision] | None:
    observation = as_observations(row.get(columns["observation"]), config.max_observations)
    action_text = as_text(row.get(columns["action"]))
    accepted = normalize_label(row.get(columns["label"]))

    if accepted is None or not observation:
        return None
    if accepted and not action_text:
        # Метка «принято» без текста действия обучать нечему.
        return None

    situation = Situation(observations=observation, audio=Audio(speech=None))

    if accepted:
        decision = RobotDecision(
            decision=DecisionType.ACT,
            confidence=config.confidence_accept,
            speech=Speech(text=action_text),
        )
    else:
        decision = RobotDecision(
            decision=config.reject_decision,
            confidence=config.confidence_reject,
        )
    return situation, decision


def convert(
    rows: Iterable[dict[str, Any]],
    columns_available: Iterable[str],
    config: ProactivityConfig | None = None,
) -> Iterator[dict[str, Any]]:
    """Превращает строки готового корпуса в обучающие примеры TRL."""
    config = config or ProactivityConfig()
    available = list(columns_available)

    columns = {
        "observation": find_field(
            available, OBSERVATION_CANDIDATES, "observation", config.field_map.observation
        ),
        "action": find_field(available, ACTION_CANDIDATES, "action", config.field_map.action),
        "label": find_field(available, LABEL_CANDIDATES, "label", config.field_map.label),
    }
    logger.info("Сопоставление полей проактивности: %s", columns)

    kept = skipped = 0
    for row in rows:
        pair = _row_to_pair(row, columns, config)
        if pair is None:
            skipped += 1
            continue
        kept += 1
        yield build_example(*pair)

    logger.info("Проактивность: оставлено %d, отброшено %d", kept, skipped)


def label_balance(examples: list[dict[str, Any]]) -> dict[str, float]:
    """Доли ACT / WAIT / OBSERVE в готовом наборе.

    Спека требует держать ACT/WAIT около 50/50: перекос в ACT превращает
    проактивность в навязчивость.
    """
    counts: dict[str, int] = {}
    for example in examples:
        assistant = example["messages"][-1]["content"]
        for name in ("ACT", "WAIT", "OBSERVE"):
            if f'"{name}"' in assistant:
                counts[name] = counts.get(name, 0) + 1
                break
    total = sum(counts.values()) or 1
    return {name: count / total for name, count in sorted(counts.items())}
