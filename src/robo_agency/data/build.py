"""Сборка обучающего корпуса из готовых датасетов по конфигу."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from . import conversational, proactive_agent, proactivity
from .mixer import Source, build_mix, downsample_to_balance, train_val_split

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DatasetSpec:
    path: str
    split: str = "train"
    name: str | None = None
    limit: int | None = None
    field_map: dict[str, str | None] | None = None
    # "auto" — определить по структуре записи; "proactive_agent" или
    # "generic" — задать явно, если автоопределение промахнулось.
    format: str = "auto"


def _load_rows(spec: DatasetSpec) -> list[dict[str, Any]]:
    """Грузит датасет с Hugging Face или локальный json/jsonl."""
    local = Path(spec.path)

    # Путь с расширением — это локальный файл. Если его нет, отправлять такую
    # строку в load_dataset бессмысленно: она утонет в стеке Hugging Face с
    # невнятной ошибкой вместо понятного «скачайте данные».
    if local.suffix in {".json", ".jsonl"} and not local.exists():
        raise FileNotFoundError(
            f"Не найден файл данных: {local}\n"
            f"  Скачайте корпус проактивности:  make fetch\n"
            f"  Либо укажите другой путь в конфиге микса."
        )

    if local.exists():
        if local.suffix == ".jsonl":
            with local.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        else:
            with local.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            rows = payload if isinstance(payload, list) else payload.get("data", [])
    else:
        from datasets import load_dataset  # импорт здесь: тесты не требуют datasets

        dataset = load_dataset(spec.path, spec.name, split=spec.split)
        rows = list(dataset)

    if spec.limit is not None:
        rows = rows[: spec.limit]
    logger.info("Загружено %d строк из %s", len(rows), spec.path)
    return rows


def _columns_of(rows: Iterable[dict[str, Any]]) -> list[str]:
    for row in rows:
        return list(row.keys())
    return []


def _decision_of(example: dict[str, Any]) -> str:
    """Тип решения в готовом примере — ключ для выравнивания классов."""
    assistant = example["messages"][-1]["content"]
    for name in ("ACT", "WAIT", "OBSERVE"):
        if f'"{name}"' in assistant:
            return name
    return "UNKNOWN"


def _resolve_format(spec: DatasetSpec, rows: list[dict[str, Any]]) -> str:
    if spec.format != "auto":
        return spec.format
    return "proactive_agent" if proactive_agent.matches(rows) else "generic"


def build_proactivity(specs: list[DatasetSpec]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for spec in specs:
        rows = _load_rows(spec)
        if not rows:
            continue

        fmt = _resolve_format(spec, rows)
        logger.info("Формат %s: %s", spec.path, fmt)

        if fmt == "proactive_agent":
            examples.extend(proactive_agent.convert(rows))
        else:
            field_map = proactivity.ProactivityFieldMap(**(spec.field_map or {}))
            config = proactivity.ProactivityConfig(field_map=field_map)
            examples.extend(proactivity.convert(rows, _columns_of(rows), config))

    logger.info("Проактивность: собрано %d примеров", len(examples))
    return examples


def build_conversational(specs: list[DatasetSpec]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for spec in specs:
        rows = _load_rows(spec)
        examples.extend(conversational.convert(rows))
    return examples


def _specs(raw: list[dict[str, Any]] | None) -> list[DatasetSpec]:
    return [DatasetSpec(**item) for item in (raw or [])]


def build_from_config(config_path: str | Path, output_dir: str | Path) -> None:
    """Полный цикл: загрузка → конвертация → микс → train/val на диск."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    proactive = build_proactivity(_specs(config.get("proactivity")))
    function_calling = build_conversational(_specs(config.get("function_calling")))
    replay = build_conversational(_specs(config.get("replay")))

    logger.info("Баланс до выравнивания: %s", proactivity.label_balance(proactive))

    if config.get("proactivity_balance", True):
        proactive, counts = downsample_to_balance(
            proactive, _decision_of, config.get("seed", 42)
        )
        logger.info("После выравнивания: %s, всего %d", counts, len(proactive))

    balance = proactivity.label_balance(proactive)
    act_share = balance.get("ACT", 0.0)
    if not 0.35 <= act_share <= 0.65:
        logger.warning(
            "Доля ACT равна %.1f%%, спека требует около 50%%. "
            "Перекос в WAIT ведёт к молчаливому роботу, перекос в ACT — к навязчивому.",
            act_share * 100,
        )

    proportions = config["proportions"]
    sources = [
        Source("proactivity", proactive, proportions["proactivity"]),
        Source("function_calling", function_calling, proportions["function_calling"]),
        Source("replay", replay, proportions["replay"]),
    ]
    mixed, report = build_mix(sources, config.get("target_size"), config.get("seed", 42))
    logger.info("\n%s", report.describe())

    train, val = train_val_split(mixed, config.get("val_ratio", 0.05), config.get("seed", 42))
    _write_jsonl(output / "train.jsonl", train)
    _write_jsonl(output / "val.jsonl", val)
    (output / "mix_report.txt").write_text(report.describe(), encoding="utf-8")
    logger.info("Записано: train=%d, val=%d в %s", len(train), len(val), output)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
