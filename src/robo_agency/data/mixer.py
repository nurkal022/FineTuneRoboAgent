"""Сборка обучающего микса в заданных пропорциях.

Главная забота модуля — честность по объёму. Проактивные данные малы
(ProactiveBench ~6790 событий), и при доле 50% они жёстко ограничивают размер
всего корпуса. Смешивание молча не дублирует их, а сообщает предельный размер.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Source:
    name: str
    examples: list[dict[str, Any]]
    proportion: float


@dataclass(slots=True)
class MixReport:
    total: int
    per_source: dict[str, int]
    limiting_source: str
    max_possible: int
    truncated: dict[str, int]

    def describe(self) -> str:
        lines = [f"Итоговый размер микса: {self.total}"]
        for name, count in self.per_source.items():
            share = count / self.total if self.total else 0.0
            lines.append(f"  {name}: {count} ({share:.1%})")
        lines.append(f"Ограничивающий источник: {self.limiting_source} (предел {self.max_possible})")
        for name, dropped in self.truncated.items():
            lines.append(f"  недобрано из {name}: {dropped}")
        return "\n".join(lines)


def _max_total(sources: Sequence[Source]) -> tuple[int, str]:
    """Наибольший размер микса, при котором ни один источник не дублируется."""
    best_total = None
    limiting = ""
    for source in sources:
        if source.proportion <= 0:
            continue
        possible = int(len(source.examples) / source.proportion)
        if best_total is None or possible < best_total:
            best_total = possible
            limiting = source.name
    return (best_total or 0), limiting


def build_mix(
    sources: Sequence[Source],
    target_size: int | None = None,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], MixReport]:
    """Собирает микс без дублирования примеров.

    Если `target_size` больше достижимого, размер понижается до достижимого,
    а не добирается повторами: дублирование малого проактивного корпуса
    привело бы к переобучению на нём.
    """
    total_proportion = sum(source.proportion for source in sources)
    if abs(total_proportion - 1.0) > 1e-6:
        raise ValueError(f"Доли источников должны давать 1.0, получено {total_proportion:.3f}")

    max_possible, limiting = _max_total(sources)
    if target_size is None:
        total = max_possible
    elif target_size > max_possible:
        logger.warning(
            "Запрошено %d примеров, достижимо %d (ограничивает %s). Используем достижимое.",
            target_size, max_possible, limiting,
        )
        total = max_possible
    else:
        total = target_size

    rng = random.Random(seed)
    mixed: list[dict[str, Any]] = []
    per_source: dict[str, int] = {}
    truncated: dict[str, int] = {}

    for source in sources:
        wanted = int(round(total * source.proportion))
        available = len(source.examples)
        take = min(wanted, available)
        if take < wanted:
            truncated[source.name] = wanted - take

        pool = list(source.examples)
        rng.shuffle(pool)
        chunk = pool[:take]
        for example in chunk:
            example = dict(example)
            example["source"] = source.name
            mixed.append(example)
        per_source[source.name] = take

    rng.shuffle(mixed)
    report = MixReport(
        total=len(mixed),
        per_source=per_source,
        limiting_source=limiting,
        max_possible=max_possible,
        truncated=truncated,
    )
    return mixed, report


def train_val_split(
    examples: Sequence[dict[str, Any]],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio должен быть в интервале (0, 1)")
    pool = list(examples)
    random.Random(seed).shuffle(pool)
    cut = max(1, int(len(pool) * val_ratio))
    return pool[cut:], pool[:cut]
