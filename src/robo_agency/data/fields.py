"""Автоопределение полей в готовых датасетах.

Точные имена колонок в открытых корпусах меняются от версии к версии, поэтому
конвертеры не хардкодят их, а ищут по списку кандидатов. Если ничего не нашлось,
ошибка печатает реальный список колонок — правится одной строкой в конфиге.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


class FieldNotFoundError(LookupError):
    def __init__(self, role: str, candidates: Sequence[str], available: Sequence[str]) -> None:
        super().__init__(
            f"Не найдено поле для роли '{role}'.\n"
            f"  Искали: {list(candidates)}\n"
            f"  Есть в датасете: {list(available)}\n"
            f"  Укажите поле явно через field_map в конфиге."
        )
        self.role = role
        self.candidates = list(candidates)
        self.available = list(available)


def find_field(
    available: Iterable[str],
    candidates: Sequence[str],
    role: str,
    override: str | None = None,
    required: bool = True,
) -> str | None:
    """Ищет колонку: сначала явное указание, потом точное совпадение, потом подстроку."""
    available = list(available)

    if override is not None:
        if override not in available:
            raise FieldNotFoundError(role, [override], available)
        return override

    lowered = {name.lower(): name for name in available}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    for candidate in candidates:
        for name in available:
            if candidate.lower() in name.lower():
                return name

    if required:
        raise FieldNotFoundError(role, candidates, available)
    return None


_TRUE_TOKENS = {"accept", "accepted", "yes", "true", "1", "positive", "good", "act", "call"}
_FALSE_TOKENS = {"reject", "rejected", "no", "false", "0", "negative", "bad", "wait", "none"}


def normalize_label(value: Any) -> bool | None:
    """Приводит разнородные метки принятия к bool.

    Возвращает None, если метку распознать не удалось — такие примеры
    отбрасываются, а не угадываются.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        return None

    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return None


def as_text(value: Any, separator: str = "\n") -> str:
    """Разворачивает строку, список или словарь в плоский текст наблюдения."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return separator.join(as_text(item, separator) for item in value if item is not None).strip()
    if isinstance(value, dict):
        parts = [f"{key}: {as_text(val, separator)}" for key, val in value.items() if val is not None]
        return separator.join(parts).strip()
    return str(value).strip()


def as_observations(value: Any, limit: int = 12) -> list[str]:
    """Список последних наблюдений; хвост важнее головы, поэтому режем с начала."""
    text = as_text(value)
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-limit:] if lines else []
