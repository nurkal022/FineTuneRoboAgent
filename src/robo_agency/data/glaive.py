"""Конвертер glaiveai/glaive-function-calling-v2.

Схема датасета: `{system, chat}`, где `chat` — одна строка-стенограмма всего
диалога с маркерами ролей:

    USER: what's the weather in Paris?
    ASSISTANT: <functioncall> {"name": "get_weather", "arguments": ...} <|endoftext|>
    FUNCTION RESPONSE: {"temp": 18}
    ASSISTANT: It is 18 degrees in Paris. <|endoftext|>

Ни один из трёх форматов в `conversational.py` сюда не подходит, поэтому в
прогоне 001 датасет дал ноль примеров.

Для наших экспериментов он ценен прежде всего как ОТЛОЖЕННАЯ выборка: в
обучающем миксе его нет, значит на нём честно меряется, не сломался ли
вызов инструментов после дообучения.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"system", "chat"}

# Маркеры ролей в стенограмме. FUNCTION RESPONSE — это результат вызова,
# он приходит от среды, поэтому роль tool, а не assistant.
_ROLE_MARKERS = {
    "USER": "user",
    "ASSISTANT": "assistant",
    "FUNCTION RESPONSE": "tool",
}
_SPLIT_RE = re.compile(r"^(USER|ASSISTANT|FUNCTION RESPONSE):\s*", re.MULTILINE)
_END_TOKENS = ("<|endoftext|>",)


def matches(rows: Iterable[dict[str, Any]]) -> bool:
    for row in rows:
        return isinstance(row, dict) and REQUIRED_KEYS.issubset(row.keys())
    return False


def _clean(text: str) -> str:
    for token in _END_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def parse_chat(chat: str) -> list[dict[str, str]]:
    """Разбирает стенограмму в список сообщений."""
    if not isinstance(chat, str) or not chat.strip():
        return []

    parts = _SPLIT_RE.split(chat)
    # split с одной группой даёт [до первого маркера, маркер, текст, маркер, текст...]
    messages: list[dict[str, str]] = []
    for index in range(1, len(parts) - 1, 2):
        role = _ROLE_MARKERS.get(parts[index])
        content = _clean(parts[index + 1])
        if role and content:
            messages.append({"role": role, "content": content})
    return messages


def convert(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    kept = skipped = 0
    for row in rows:
        messages = parse_chat(row.get("chat", ""))

        # Диалог без ответа ассистента обучать нечему.
        if not any(message["role"] == "assistant" for message in messages):
            skipped += 1
            continue

        system = (row.get("system") or "").strip()
        if system:
            # Описание доступных инструментов — это системное сообщение.
            messages.insert(0, {"role": "system", "content": _clean(system)})

        if len(messages) < 2:
            skipped += 1
            continue

        kept += 1
        yield {"messages": messages}

    logger.info("glaive: оставлено %d, отброшено %d", kept, skipped)
