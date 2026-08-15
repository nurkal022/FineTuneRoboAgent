#!/usr/bin/env python
"""Обход всех чекпойнтов прогона с замером метрик агентности.

Зачем: в прогоне 001 чекпойнт с лучшим `eval_loss` оказался вырожденным
(«всегда ACT»), и по формальным метрикам он выигрывал у финального. Двух точек
мало, чтобы понять, что происходит между ними, — нужен замер решений на каждом
сохранённом шаге.

Скрипт не оценивает сам: он вызывает eval_decisions.py отдельным процессом на
каждый чекпойнт. Так каждый замер идёт с чистой загрузкой модели, без риска,
что состояние предыдущего адаптера подтекает в следующий.

Итог — таблица «шаг → вырожденность, ложные вмешательства, своевременность»,
то есть кривая выхода модели из вырожденного режима.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

BOLD, GREEN, RED, RESET = "\033[1m", "\033[32m", "\033[31m", "\033[0m"


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else 10**9  # финальный адаптер идёт последним


def find_checkpoints(run_dir: Path) -> list[Path]:
    """Все checkpoint-* плюс сам каталог адаптера как финальная точка."""
    points = sorted(
        (p for p in run_dir.glob("checkpoint-*") if p.is_dir()),
        key=checkpoint_step,
    )
    if (run_dir / "adapter_model.safetensors").exists():
        points.append(run_dir)
    return points


def eval_one(checkpoint: Path, args, out_dir: Path) -> dict | None:
    label = checkpoint.name
    json_path = out_dir / f"agency_{label}.json"

    command = [
        sys.executable, str(Path(__file__).parent / "eval_decisions.py"),
        "--adapter", str(checkpoint),
        "--val-file", args.val_file,
        "--limit", str(args.limit),
        "--show", "0",
        "--json", str(json_path),
        "--label", label,
    ]
    print(f"\n{BOLD}=== {label} ==={RESET}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"{RED}Замер не удался:{RESET}\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
        return None

    return json.loads(json_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Метрики агентности по всем чекпойнтам")
    parser.add_argument("--run-dir", default="outputs/adapter-decisions")
    parser.add_argument("--val-file", default="data/processed/val.jsonl")
    parser.add_argument("--limit", type=int, default=46)
    parser.add_argument("--out-dir", default="docs/experiments/sweeps")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoints = find_checkpoints(run_dir)
    if not checkpoints:
        print(f"{RED}В {run_dir} нет ни чекпойнтов, ни адаптера{RESET}")
        return 1

    out_dir = Path(args.out_dir) / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Чекпойнтов к замеру: {len(checkpoints)}")

    results = [r for r in (eval_one(cp, args, out_dir) for cp in checkpoints) if r]
    if not results:
        return 1

    header = f"\n{BOLD}Сводка по {run_dir}{RESET}"
    lines = [
        "| чекпойнт | ACT/WAIT | вырожден | своевременность | ложные | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for row in results:
        predicted = row["predicted"]
        metrics = row["metrics"] or {}
        mark = "ДА" if row["degenerate"] else "нет"
        lines.append(
            f"| {row['label']} "
            f"| {predicted.get('ACT', 0)}/{predicted.get('WAIT', 0)} "
            f"| {mark} "
            f"| {metrics.get('timeliness', 0):.3f} "
            f"| {metrics.get('false_intervention_rate', 0):.3f} "
            f"| {metrics.get('f1_act', 0):.3f} |"
        )

    table = "\n".join(lines)
    print(header)
    print(table)

    summary = out_dir / "summary.md"
    summary.write_text(f"# Обход чекпойнтов: {run_dir}\n\n{table}\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nСводка: {summary}")

    healthy = [r for r in results if not r["degenerate"]]
    if healthy:
        best = min(healthy, key=lambda r: r["metrics"]["false_intervention_rate"])
        print(
            f"{GREEN}Лучший невырожденный чекпойнт: {best['label']} "
            f"(ложные вмешательства {best['metrics']['false_intervention_rate']:.3f}){RESET}"
        )
    else:
        print(f"{RED}Все чекпойнты вырождены{RESET}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
