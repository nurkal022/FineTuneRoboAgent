#!/usr/bin/env python
"""Скачивание корпуса проактивности и подготовка его к конвертации.

ProactiveBench лежит в репозитории thunlp/ProactiveAgent, а не отдельным
датасетом на Hugging Face, и структура файлов там не документирована как
стабильная. Поэтому скрипт не угадывает путь, а клонирует репозиторий,
обходит все json/jsonl и проверяет каждый файл НАШИМ ЖЕ конвертером: годится
тот, из которого реально получаются обучающие примеры.

Такой подход переживёт переезд файлов внутри репозитория.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robo_agency.data import proactive_agent, proactivity  # noqa: E402

REPO_URL = "https://github.com/thunlp/ProactiveAgent"
GREEN, YELLOW, RED, BOLD, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[0m"


def clone(target: Path) -> Path:
    if target.exists():
        print(f"{YELLOW}Каталог {target} уже существует, пропускаю клонирование{RESET}")
        return target
    if shutil.which("git") is None:
        raise SystemExit("git не найден — установите git")
    print(f"Клонирую {REPO_URL} → {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(target)],
        check=True,
    )
    return target


def read_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Читает json или jsonl, отбрасывая всё, что не список словарей."""
    try:
        if path.suffix == ".jsonl":
            rows = []
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        rows.append(json.loads(line))
                    if limit and len(rows) >= limit:
                        break
            return rows

        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key in ("data", "examples", "events", "instances"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        return payload if isinstance(payload, list) else []
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return []


def usable_examples(rows: list[dict[str, Any]]) -> int:
    """Сколько обучающих примеров даёт файл нашими конвертерами."""
    if not rows or not isinstance(rows[0], dict):
        return 0

    if proactive_agent.matches(rows):
        return sum(1 for _ in proactive_agent.convert(rows))

    try:
        return sum(1 for _ in proactivity.convert(rows, rows[0].keys()))
    except Exception:  # noqa: BLE001 — неподходящий файл это норма, а не сбой
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Скачать и подготовить ProactiveBench")
    parser.add_argument("--repo-dir", default="data/raw/ProactiveAgent")
    parser.add_argument("--output", default="data/raw/proactive_bench.jsonl")
    parser.add_argument("--min-examples", type=int, default=50)
    args = parser.parse_args()

    repo = clone(Path(args.repo_dir))

    print(f"\n{BOLD}Ищу пригодные файлы{RESET}")
    candidates: list[tuple[Path, int, int]] = []
    for path in sorted(repo.rglob("*")):
        if path.suffix not in {".json", ".jsonl"} or not path.is_file():
            continue
        rows = read_rows(path)
        if not rows:
            continue
        count = usable_examples(rows)
        if count >= args.min_examples:
            candidates.append((path, len(rows), count))
            print(f"{GREEN}  +{RESET} {path.relative_to(repo)} — строк {len(rows)}, годных {count}")
        elif len(rows) > 100:
            print(f"    - {path.relative_to(repo)} — строк {len(rows)}, годных {count}")

    if not candidates:
        print(
            f"\n{RED}Не нашёл ни одного файла, из которого получаются примеры.{RESET}\n"
            "Структура репозитория могла измениться. Посмотрите файлы вручную:\n"
            f"  find {repo} -name '*.json*' | head -30\n"
            "и укажите путь и field_map в configs/data_mix.yaml."
        )
        return 1

    # Берём все пригодные файлы: чем больше проактивных данных, тем выше
    # потолок размера корпуса — он ограничен именно ими.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for path, _, _ in candidates:
            for row in read_rows(path):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    total_usable = sum(count for _, _, count in candidates)
    print(f"\n{BOLD}Записано{RESET} {written} строк в {output}")
    print(f"Из них конвертер извлечёт примерно {total_usable} обучающих примеров.")
    print(f"Потолок размера корпуса при доле 55%: около {int(total_usable / 0.55)} примеров.")
    print("\nДальше: make data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
