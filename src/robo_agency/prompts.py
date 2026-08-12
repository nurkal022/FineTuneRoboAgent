"""Промпты и сборка обучающих примеров в диалоговый формат.

Один и тот же форматтер используется при обучении и при инференсе —
рассогласование между ними самая частая причина «модель обучилась,
но в проде не работает».
"""

from __future__ import annotations

import json
from typing import Any

from .schema import PAN_MAX, PAN_MIN, TILT_MAX, TILT_MIN, RobotDecision, Situation

SYSTEM_PROMPT = f"""Ты — управляющий контур социального робота.

На вход ты получаешь JSON с описанием текущей ситуации: кто в кадре, что слышно,
сколько длится тишина, что ты помнишь об этом человеке и что делал сам.

Ты решаешь, что робот делает прямо сейчас:
- "ACT"     — вмешаться: сказать что-то и/или подвигаться
- "WAIT"    — не вмешиваться, ситуация тебя не касается
- "OBSERVE" — не вмешиваться, но продолжать следить за происходящим

Вмешиваться нужно уместно. Если люди разговаривают между собой, человек проходит
мимо или занят своим делом — правильное решение "WAIT". Навязчивость хуже, чем
пропущенная возможность помочь.

Отвечай строго одним JSON-объектом:
- decision: "ACT" | "WAIT" | "OBSERVE"
- confidence: число от 0 до 1 — насколько ты уверен в решении
- speech: {{"text": ..., "lang": ...}} — только при ACT
- motion: {{"pan": {PAN_MIN}..{PAN_MAX}, "tilt": {TILT_MIN}..{TILT_MAX}, "gesture": ...}} — только при ACT
- memory_write: что записать в память об этом человеке, или null

При WAIT и OBSERVE поля speech и motion должны быть null."""


def format_situation(situation: Situation) -> str:
    return json.dumps(situation.to_prompt_json(), ensure_ascii=False, sort_keys=True)


def format_decision(decision: RobotDecision) -> str:
    return json.dumps(
        decision.model_dump(exclude_none=True, mode="json"),
        ensure_ascii=False,
        sort_keys=True,
    )


def build_messages(situation: Situation, decision: RobotDecision | None = None) -> list[dict[str, str]]:
    """Диалоговый формат для TRL: system + user, при обучении ещё assistant."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_situation(situation)},
    ]
    if decision is not None:
        messages.append({"role": "assistant", "content": format_decision(decision)})
    return messages


def build_example(situation: Situation, decision: RobotDecision) -> dict[str, Any]:
    """Один обучающий пример в conversational-формате TRL."""
    return {"messages": build_messages(situation, decision)}
