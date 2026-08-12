"""Этапы 0 и 6: замер сохранения речевых способностей.

Метрика — средняя отрицательная логвероятность (NLL) на отложенных общих
диалогах, посчитанная дважды: базой без адаптера и базой с адаптером. Рост NLL
означает, что специализация начала вытеснять речь.

Выбор NLL вместо LLM-судьи сознателен: судья вносит собственную предвзятость и
делает результат невоспроизводимым, а для главы про компромисс нужен именно
воспроизводимый и монотонный показатель. Полноценные BFCL и tau-bench
запускаются отдельно своими штатными харнессами.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetentionReport:
    base_nll: float
    adapter_nll: float
    examples: int

    @property
    def delta(self) -> float:
        return self.adapter_nll - self.base_nll

    @property
    def relative_degradation(self) -> float:
        return self.delta / self.base_nll if self.base_nll else 0.0

    def describe(self) -> str:
        verdict = "деградация" if self.delta > 0 else "улучшение"
        return (
            f"NLL базы:            {self.base_nll:.4f}\n"
            f"NLL с адаптером:     {self.adapter_nll:.4f}\n"
            f"Дельта:              {self.delta:+.4f} ({verdict})\n"
            f"Относительно:        {self.relative_degradation:+.2%}\n"
            f"Примеров:            {self.examples}"
        )


def _load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _mean_nll(model, tokenizer, rows: list[dict[str, Any]], max_length: int) -> float:
    import torch

    total_nll = 0.0
    counted = 0
    model.eval()
    with torch.no_grad():
        for row in rows:
            text = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
            encoded = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            ).to(model.device)
            outputs = model(**encoded, labels=encoded["input_ids"])
            total_nll += float(outputs.loss)
            counted += 1
    return total_nll / counted if counted else float("nan")


def compare(
    base_model: str,
    adapter_path: str,
    eval_file: str | Path,
    limit: int | None = 300,
    max_length: int = 2048,
) -> RetentionReport:
    """Считает NLL базы и базы с адаптером на одних и тех же примерах."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = _load_jsonl(eval_file, limit)
    if not rows:
        raise ValueError(f"Нет примеров в {eval_file}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    logger.info("Замер базы без адаптера на %d примерах", len(rows))
    base_nll = _mean_nll(model, tokenizer, rows, max_length)

    logger.info("Замер базы с адаптером")
    model = PeftModel.from_pretrained(model, adapter_path)
    adapter_nll = _mean_nll(model, tokenizer, rows, max_length)

    report = RetentionReport(base_nll=base_nll, adapter_nll=adapter_nll, examples=len(rows))
    logger.info("\n%s", report.describe())
    return report
