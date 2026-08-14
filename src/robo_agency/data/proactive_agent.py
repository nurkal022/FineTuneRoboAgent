"""Конвертер корпуса thunlp/ProactiveAgent в информационную модель ситуации.

Схема записи в репозитории:

    {
      "obs":         [{"time": "...", "event": "..."}, ...],
      "pred_task":   строка или null — что агент предложил сделать,
      "valid":       bool — удалось ли разобрать предсказание,
      "help_needed": bool — ТРЕБОВАЛОСЬ ли вмешательство (эталон),
      "annotation":  [bool, ...] — оценки нескольких разметчиков,
      "category":    "Correct-Detection (CD)" | "Missed-Need (MN)" |
                     "False-Alarm (FA)" | "Correct-Rejection (CR)"
    }

Эталонная метка берётся из `category`, а НЕ из `help_needed`. Проверка на
реальных данных показала, что `help_needed` отражает суждение самого агента,
а не истину: среди записей с `help_needed=True` встречаются 244 штуки с
категорией False-Alarm, то есть человек вмешательство отверг. `category` же
это уже сведённая матрица ошибок с человеческой разметкой.

Отображение категорий на решения:

  Correct-Detection (CD)  -> ACT с текстом pred_task: вмешался верно
  Correct-Rejection (CR)  -> WAIT: верно промолчал
  False-Alarm (FA)        -> WAIT: влез зря, и предложенный текст в ответ
                             не попадает — именно этого делать не следовало
  Missed-Need (MN)        -> пропуск: действовать было надо, но текста
                             действия в данных нет, а выдумывать нельзя

Пропуск MN и преобладание FA смещают баланс в сторону WAIT. Это известный
перекос корпуса, и выравнивается он при сборке микса, а не здесь: конвертер
обязан отдавать данные как есть.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from ..prompts import build_example
from ..schema import Audio, DecisionType, RobotDecision, Situation, Speech

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"obs", "help_needed"}

# Ключ — код категории в поле `category`; значение — целевое решение.
# None означает, что запись обучению не годится.
CATEGORY_TO_DECISION: dict[str, DecisionType | None] = {
    "CD": DecisionType.ACT,   # Correct-Detection
    "CR": DecisionType.WAIT,  # Correct-Rejection
    "FA": DecisionType.WAIT,  # False-Alarm
    "MN": None,               # Missed-Need: нужен ACT, но текста действия нет
}


@dataclass(slots=True)
class ProactiveAgentConfig:
    confidence_act: float = 0.85
    confidence_wait: float = 0.85
    max_observations: int = 12


def matches(rows: Iterable[dict[str, Any]]) -> bool:
    """Похожа ли выборка на корпус ProactiveAgent."""
    for row in rows:
        return isinstance(row, dict) and REQUIRED_KEYS.issubset(row.keys())
    return False


def _observations(raw: Any, limit: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    lines: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        time = str(item.get("time") or "").strip()
        event = str(item.get("event") or "").strip()
        if not event:
            continue
        lines.append(f"[{time}] {event}" if time else event)
    return lines[-limit:]


def category_code(category: Any) -> str | None:
    """Достаёт код из строки вида 'Correct-Detection (CD)'."""
    if not isinstance(category, str):
        return None
    for code in CATEGORY_TO_DECISION:
        if f"({code})" in category:
            return code
    return None


def _target_decision(row: dict[str, Any]) -> DecisionType | None:
    """Целевое решение по категории; при её отсутствии — по help_needed."""
    code = category_code(row.get("category"))
    if code is not None:
        return CATEGORY_TO_DECISION[code]

    help_needed = row.get("help_needed")
    if isinstance(help_needed, bool):
        return DecisionType.ACT if help_needed else DecisionType.WAIT
    return None


def convert(
    rows: Iterable[dict[str, Any]],
    config: ProactiveAgentConfig | None = None,
) -> Iterator[dict[str, Any]]:
    config = config or ProactiveAgentConfig()
    stats: Counter[str] = Counter()

    for row in rows:
        observations = _observations(row.get("obs"), config.max_observations)
        if not observations:
            stats["пропуск: нет наблюдений"] += 1
            continue

        target = _target_decision(row)
        if target is None:
            stats["пропуск: Missed-Need или нет метки"] += 1
            continue

        situation = Situation(observations=observations, audio=Audio(speech=None))

        if target is DecisionType.WAIT:
            # pred_task здесь намеренно отбрасывается: при False-Alarm это
            # ровно то действие, которое совершать не следовало.
            stats["WAIT"] += 1
            yield build_example(
                situation,
                RobotDecision(decision=DecisionType.WAIT, confidence=config.confidence_wait),
            )
            continue

        pred_task = row.get("pred_task")
        if not isinstance(pred_task, str) or not pred_task.strip():
            stats["пропуск: нужен ACT, но нет текста действия"] += 1
            continue

        stats["ACT"] += 1
        yield build_example(
            situation,
            RobotDecision(
                decision=DecisionType.ACT,
                confidence=config.confidence_act,
                speech=Speech(text=pred_task.strip()),
            ),
        )

    logger.info("ProactiveAgent: %s", dict(stats))
