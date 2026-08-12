"""Этап 5: DPO на парах из неявной обратной связи.

Дообучается тот же адаптер решений, что и на этапе SFT. Базовые веса
по-прежнему заморожены. Бэкенд переключается полем `engine`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .backends import load_for_dpo, validate_engine

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DpoLoraSpec:
    """Присутствует ради единого интерфейса загрузчика.

    На этапе DPO адаптер уже существует и заново не создаётся, поэтому эти
    значения используются только как метаданные.
    """

    r: int = 32
    alpha: int = 64
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass(slots=True)
class DpoConfigSpec:
    base_model: str = "Qwen/Qwen3-8B"
    engine: str = "unsloth"
    adapter_path: str = "outputs/adapter-decisions"
    pairs_file: str = "data/processed/preference_pairs.jsonl"
    output_dir: str = "outputs/adapter-decisions-dpo"
    # Низкий lr: DPO на нескольких тысячах пар легко ломает результат SFT.
    learning_rate: float = 5e-6
    beta: float = 0.1
    num_epochs: float = 1.0
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 4096
    max_prompt_length: int = 3072
    logging_steps: int = 10
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    load_in_4bit: bool = False
    lora: DpoLoraSpec = field(default_factory=DpoLoraSpec)

    def __post_init__(self) -> None:
        validate_engine(self.engine)


def load_config(path: str | Path) -> DpoConfigSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    lora_raw = raw.pop("lora", {}) or {}
    return DpoConfigSpec(lora=DpoLoraSpec(**lora_raw), **raw)


def _load_pairs(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            # margin нужен при отборе пар, тренеру он не передаётся.
            row.pop("margin", None)
            rows.append(row)
    return rows


def train(config: DpoConfigSpec) -> str:
    engine = validate_engine(config.engine)

    pairs = _load_pairs(config.pairs_file)
    if not pairs:
        raise ValueError(f"Нет пар предпочтений в {config.pairs_file}")
    logger.info("Пар предпочтений: %d", len(pairs))

    # Загрузка до импорта trl: см. комментарий в sft.py.
    loaded = load_for_dpo(config)

    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    training_args = DPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        beta=config.beta,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        logging_steps=config.logging_steps,
        lr_scheduler_type="cosine",
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing and engine == "hf",
        seed=config.seed,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=loaded.model,
        args=training_args,
        train_dataset=Dataset.from_list(pairs),
        processing_class=loaded.tokenizer,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    loaded.tokenizer.save_pretrained(config.output_dir)
    logger.info("Адаптер после DPO сохранён в %s (бэкенд %s)", config.output_dir, engine)
    return config.output_dir
