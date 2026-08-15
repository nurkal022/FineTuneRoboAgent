#!/usr/bin/env python
"""Сборка ОТЛОЖЕННЫХ выборок для замеров сохранения способностей.

Две выборки, обе не участвуют в обучении:

  retention_eval.jsonl — общий диалог. Срез ultrachat ЗА пределами того,
      что берётся в обучающий микс (там train_sft[:5000]).

  tools_eval.jsonl — вызов инструментов. Берётся glaive, которого в миксе
      нет вообще: в конфиге обучения стоит hermes. Полностью независимый
      источник, то есть замер честный.

Обе меряются одинаково — средним NLL базы против базы с адаптером
(`robo_agency.cli retention --eval-file ...`). Одна метрика, две способности:
рост NLL означает, что специализация начала вытеснять исходный навык.

NLL выбран вместо генерации с разбором сознательно: он не зависит от
декодера, от шаблона промпта и от парсера ответа, поэтому сравнение
«до/после» получается чистым.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robo_agency.env import ensure_writable_hf_cache  # noqa: E402

ensure_writable_hf_cache()

from robo_agency.data import conversational  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Обучение берёт ultrachat train_sft[:5000] — здесь начинаем строго после.
RETENTION_SOURCE = ("HuggingFaceH4/ultrachat_200k", None, "train_sft[5000:5600]")
# glaive в обучающем миксе не участвует: там hermes.
TOOLS_SOURCE = ("glaiveai/glaive-function-calling-v2", None, "train[:600]")


def build(path: str, name: str | None, split: str, limit: int, output: Path) -> int:
    from datasets import load_dataset

    logger.info("Загружаю %s (%s)", path, split)
    rows = list(load_dataset(path, name, split=split))
    examples = list(conversational.convert(rows))[:limit]

    if not examples:
        logger.error("Из %s не получилось ни одного примера", path)
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info("Записано %d примеров в %s", len(examples), output)
    return len(examples)


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать отложенные выборки для замеров")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    output = Path(args.output_dir)
    counts = {
        "retention_eval.jsonl": build(*RETENTION_SOURCE, args.limit, output / "retention_eval.jsonl"),
        "tools_eval.jsonl": build(*TOOLS_SOURCE, args.limit, output / "tools_eval.jsonl"),
    }

    print("\nИтог:")
    for name, count in counts.items():
        print(f"  {name}: {count}")
    return 0 if all(counts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
