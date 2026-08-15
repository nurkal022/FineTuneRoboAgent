"""Приведение function-calling и replay-корпусов к диалоговому формату.

Эти корпуса намеренно НЕ конвертируются в схему решений робота: их задача —
сохранить у модели то, что она уже умеет (вызов функций, обычный диалог).
Перегонка их в `RobotDecision` уничтожила бы ровно тот навык, ради которого
они добавлены в микс.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)

_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "chatgpt": "assistant",
    "bot": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}

_PROMPT_KEYS = ("prompt", "instruction", "question", "query", "input")
_RESPONSE_KEYS = ("response", "answer", "answers", "output", "completion", "target")


class UnsupportedRowError(ValueError):
    pass


def _normalize_role(role: Any) -> str:
    return _ROLE_ALIASES.get(str(role).strip().lower(), "user")


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Разбирает строку датасета в список сообщений.

    Поддерживает три распространённых формата: готовый `messages`,
    ShareGPT-подобный `conversations` и плоскую пару «запрос / ответ»
    (xLAM, Glaive и подобные).
    """
    if isinstance(row.get("messages"), list) and row["messages"]:
        return [
            {"role": _normalize_role(m.get("role")), "content": _content_to_text(m.get("content", ""))}
            for m in row["messages"]
        ]

    if isinstance(row.get("conversations"), list) and row["conversations"]:
        return [
            {
                "role": _normalize_role(m.get("from") or m.get("role")),
                "content": _content_to_text(m.get("value") or m.get("content", "")),
            }
            for m in row["conversations"]
        ]

    # Стенограмма одной строкой (glaive): разбирается отдельным модулем.
    if isinstance(row.get("chat"), str) and row["chat"].strip():
        from . import glaive

        messages = glaive.parse_chat(row["chat"])
        if messages:
            system = (row.get("system") or "").strip()
            if system:
                messages.insert(0, {"role": "system", "content": system})
            return messages

    prompt_key = next((k for k in _PROMPT_KEYS if row.get(k)), None)
    response_key = next((k for k in _RESPONSE_KEYS if row.get(k)), None)
    if prompt_key and response_key:
        messages: list[dict[str, str]] = []
        # Описание доступных инструментов, если оно есть, идёт системным сообщением.
        tools = row.get("tools") or row.get("functions")
        if tools:
            messages.append({"role": "system", "content": _content_to_text(tools)})
        messages.append({"role": "user", "content": _content_to_text(row[prompt_key])})
        messages.append({"role": "assistant", "content": _content_to_text(row[response_key])})
        return messages

    raise UnsupportedRowError(
        f"Не удалось разобрать строку в диалог. Доступные поля: {sorted(row.keys())}"
    )


def convert(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    kept = skipped = 0
    for row in rows:
        try:
            messages = to_messages(row)
        except UnsupportedRowError:
            skipped += 1
            continue
        if len(messages) < 2:
            skipped += 1
            continue
        kept += 1
        yield {"messages": messages}

    if skipped:
        logger.warning("Диалоговые данные: оставлено %d, отброшено %d", kept, skipped)
