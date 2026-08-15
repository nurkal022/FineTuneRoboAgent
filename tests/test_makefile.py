"""Проверки на запускающие цели Makefile.

Обучение идёт часами, и следить за лоссом надо ПО ХОДУ, а не после. Питон
буферизует stdout, когда тот уходит в конвейер, поэтому строки вида
`{'loss': ...}` тренера оседают в буфере, пока полоса прогресса (она пишется
в stderr) идёт в лог сразу. На первом прогоне из-за этого лосс не был виден
все полтора часа обучения.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

MAKEFILE = Path(__file__).resolve().parent.parent / "Makefile"


def recipe(target: str) -> str:
    """Строки рецепта цели: от заголовка до первой строки без отступа."""
    text = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(rf"^{target}:.*?$\n((?:\t.*\n|\n)*)", text, re.MULTILINE)
    assert match, f"В Makefile нет цели {target}"
    return match.group(1)


@pytest.mark.parametrize("target", ["sft", "dpo"])
def test_training_output_is_unbuffered(target):
    """Иначе лосс появится в логе только после конца обучения."""
    body = recipe(target)

    assert "| tee" in body, f"цель {target} должна писать лог через tee"
    assert "PYTHONUNBUFFERED=1" in body, (
        f"цель {target} пишет в конвейер, но не отключает буферизацию stdout: "
        "строки с лоссом осядут в буфере до конца прогона"
    )
