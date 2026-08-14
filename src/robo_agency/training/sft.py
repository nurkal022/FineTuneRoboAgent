"""Этап 2: SFT LoRA-адаптера решений.

Базовые веса заморожены — обучается только адаптер. Это архитектурная гарантия
против катастрофического забывания: речевой контур это та же база с отключённым
адаптером, и он не участвует в обучении вообще.

Бэкенд обучения переключается полем `engine`: "hf" или "unsloth".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .backends import apply_assistant_only_loss, load_for_sft, validate_engine

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoraConfigSpec:
    r: int = 32
    alpha: int = 64
    # 0.0 по умолчанию: Unsloth оптимизирован именно под этот случай,
    # а на корпусе в 12-15 тысяч примеров вклад dropout невелик.
    dropout: float = 0.0
    # Все линейные слои: сужение до q/v заметно снижает качество структурного вывода.
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass(slots=True)
class SftConfigSpec:
    base_model: str = "Qwen/Qwen3-8B"
    engine: str = "unsloth"
    train_file: str = "data/processed/train.jsonl"
    val_file: str = "data/processed/val.jsonl"
    output_dir: str = "outputs/adapter-decisions"
    learning_rate: float = 1e-4
    num_epochs: float = 3.0
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 4096
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    load_in_4bit: bool = False
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)

    def __post_init__(self) -> None:
        validate_engine(self.engine)


def load_config(path: str | Path) -> SftConfigSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    lora_raw = raw.pop("lora", {}) or {}
    return SftConfigSpec(lora=LoraConfigSpec(**lora_raw), **raw)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Не найден обучающий корпус: {path}\n"
            f"  Соберите его:  make fetch && make data"
        )

    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                # `source` нужен только для отчёта о миксе, обучению он мешает.
                row.pop("source", None)
                rows.append(row)
    return rows


def train(config: SftConfigSpec) -> str:
    """Запускает SFT и возвращает путь к сохранённому адаптеру."""
    engine = validate_engine(config.engine)

    train_rows = _load_jsonl(config.train_file)
    val_rows = _load_jsonl(config.val_file)
    logger.info("Обучающих примеров: %d, валидационных: %d", len(train_rows), len(val_rows))

    # Загрузка идёт ДО импорта trl: unsloth патчит trl при своём импорте,
    # и обратный порядок молча лишил бы нас экономии памяти.
    loaded = load_for_sft(config)

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    training_args = SFTConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        eval_strategy="steps" if val_rows else "no",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        bf16=config.bf16,
        # У Unsloth чекпойнтинг включается при навешивании LoRA, повторно не надо.
        gradient_checkpointing=config.gradient_checkpointing and engine == "hf",
        seed=config.seed,
        report_to=[],
        # Учим только на ответе ассистента. В unsloth-ветке за это отвечает
        # apply_assistant_only_loss ниже.
        assistant_only_loss=(engine == "hf"),
    )

    trainer = SFTTrainer(
        model=loaded.model,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(val_rows) if val_rows else None,
        processing_class=loaded.tokenizer,
        peft_config=loaded.peft_config,
    )
    trainer = apply_assistant_only_loss(trainer, engine)

    trainer.train()
    trainer.save_model(config.output_dir)
    loaded.tokenizer.save_pretrained(config.output_dir)
    logger.info("Адаптер сохранён в %s (бэкенд %s)", config.output_dir, engine)
    return config.output_dir
