"""Сборка обучающего микса в заданных пропорциях.

Главная забота модуля — честность по объёму. Проактивные данные малы
(ProactiveBench ~6790 событий), и при доле 50% они жёстко ограничивают размер
всего корпуса. Смешивание молча не дублирует их, а сообщает предельный размер.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

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


def drop_empty_and_renormalize(sources: Sequence[Source]) -> list[Source]:
    """Убирает пустые источники и перераспределяет их доли между остальными.

    Источник может оказаться пустым по внешней причине: датасет закрыт
    авторизацией, сети нет, формат не разобрался. Ронять из-за этого весь
    корпус неправильно, молча оставлять нулевую долю — тоже: build_mix тогда
    вернёт пустой микс, потому что предельный размер считается по минимуму.
    """
    alive = [source for source in sources if source.examples]
    dropped = [source.name for source in sources if not source.examples]

    if not alive:
        raise ValueError("Все источники данных пусты — собирать корпус не из чего")

    if dropped:
        total = sum(source.proportion for source in alive)
        logger.warning(
            "Источники без данных исключены из микса: %s. "
            "Доли остальных пересчитаны.", ", ".join(dropped),
        )
        alive = [
            Source(source.name, source.examples, source.proportion / total)
            for source in alive
        ]

    return alive


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


def downsample_to_balance(
    examples: Sequence[dict[str, Any]],
    key: Callable[[dict[str, Any]], str],
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Уравнивает классы, прореживая преобладающие.

    Нужно для проактивности: в готовых корпусах решений «промолчать» кратно
    больше, чем «вмешаться», и на несбалансированной выборке модель приходит
    к вырожденной стратегии — всегда WAIT. Такая модель показывает высокую
    точность и полную бесполезность.

    Прореживаем большинство, а не размножаем меньшинство: дублирование редких
    положительных примеров ведёт к переобучению на них.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        grouped.setdefault(key(example), []).append(example)

    if len(grouped) < 2:
        return list(examples), {name: len(items) for name, items in grouped.items()}

    target = min(len(items) for items in grouped.values())
    rng = random.Random(seed)

    balanced: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for name, items in sorted(grouped.items()):
        pool = list(items)
        rng.shuffle(pool)
        balanced.extend(pool[:target])
        counts[name] = target

    rng.shuffle(balanced)
    return balanced, counts


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
