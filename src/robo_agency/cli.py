"""Командный интерфейс пайплайна."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Проактивная агентность вербального робота")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@app.command("inspect-dataset")
def inspect_dataset(
    path: str = typer.Argument(..., help="Путь на Hugging Face или локальный json/jsonl"),
    split: str = typer.Option("train", help="Сплит"),
    name: str | None = typer.Option(None, help="Подмножество (config name)"),
    samples: int = typer.Option(2, help="Сколько примеров показать"),
) -> None:
    """Печатает реальные колонки датасета.

    Нужно перед первым запуском конвертации: имена полей в открытых корпусах
    меняются между версиями, и сопоставление надо сверить, а не угадать.
    """
    from .data.build import DatasetSpec, _load_rows

    rows = _load_rows(DatasetSpec(path=path, split=split, name=name, limit=samples))
    if not rows:
        typer.echo("Датасет пуст")
        raise typer.Exit(code=1)

    typer.echo(f"Колонки: {sorted(rows[0].keys())}\n")
    for index, row in enumerate(rows):
        typer.echo(f"--- Пример {index + 1} ---")
        typer.echo(json.dumps(row, ensure_ascii=False, indent=2)[:2000])


@app.command("build-data")
def build_data(
    config: str = typer.Option("configs/data_mix.yaml", help="Конфиг микса"),
    output: str = typer.Option("data/processed", help="Куда писать train/val"),
) -> None:
    """Загружает готовые датасеты, конвертирует и собирает микс."""
    from .data.build import build_from_config

    build_from_config(config, output)


@app.command("train-sft")
def train_sft(config: str = typer.Option("configs/sft.yaml")) -> None:
    """Этап 2: SFT адаптера решений."""
    from .training.sft import load_config, train

    path = train(load_config(config))
    typer.echo(f"Адаптер: {path}")


@app.command("train-dpo")
def train_dpo(config: str = typer.Option("configs/dpo.yaml")) -> None:
    """Этап 5: DPO на неявной обратной связи."""
    from .training.dpo import load_config, train

    path = train(load_config(config))
    typer.echo(f"Адаптер после DPO: {path}")


@app.command("build-pairs")
def build_pairs(
    input_file: str = typer.Option(..., "--input", help="Лог взаимодействий (jsonl)"),
    output_file: str = typer.Option("data/processed/preference_pairs.jsonl", "--output"),
    min_margin: float = typer.Option(0.5, help="Минимальная разница наград в паре"),
) -> None:
    """Превращает лог взаимодействий в пары предпочтений для DPO."""
    from .reward.implicit import InteractionRecord, ReactionWindow, build_preference_pairs
    from .schema import Emotion

    records = []
    with Path(input_file).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            reaction_raw = dict(row["reaction"])
            for key in ("emotion_before", "emotion_after"):
                if key in reaction_raw:
                    reaction_raw[key] = Emotion(reaction_raw[key])
            records.append(
                InteractionRecord(
                    situation_key=row["situation_key"],
                    messages=row["messages"],
                    decision_json=row["decision_json"],
                    reaction=ReactionWindow(**reaction_raw),
                )
            )

    pairs = build_preference_pairs(records, min_margin=min_margin)
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    typer.echo(f"Записей в логе: {len(records)}, пар получено: {len(pairs)} → {out}")


@app.command("retention")
def retention(
    base_model: str = typer.Option("Qwen/Qwen3-8B", "--base"),
    adapter: str = typer.Option("outputs/adapter-decisions"),
    eval_file: str = typer.Option("data/processed/retention_eval.jsonl"),
    limit: int = typer.Option(300),
) -> None:
    """Этапы 0 и 6: замер сохранения речевых способностей."""
    from .evaluation.retention import compare

    report = compare(base_model, adapter, eval_file, limit=limit)
    typer.echo(report.describe())


@app.command("schema")
def show_schema() -> None:
    """Печатает JSON Schema решения — ту же, что уходит в грамматику декодера."""
    from .schema import decision_json_schema

    typer.echo(json.dumps(decision_json_schema(), ensure_ascii=False, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
