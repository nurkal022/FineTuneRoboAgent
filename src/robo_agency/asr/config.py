"""Конфигурация дообучения Whisper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .data import DatasetSpec

# Размеры, между которыми имеет смысл выбирать на карте с 16 ГБ.
KNOWN_SIZES = {
    "openai/whisper-tiny": 39,
    "openai/whisper-base": 74,
    "openai/whisper-small": 244,
    "openai/whisper-medium": 769,
    "openai/whisper-large-v3": 1550,
}


@dataclass(slots=True)
class WhisperConfig:
    # small — компромисс: medium даёт лучше качество, но обучается в разы дольше,
    # а base заметно хуже держит казахскую фонетику.
    model: str = "openai/whisper-small"
    # Казахский есть в списке языков Whisper, поэтому задаём его явно: без этого
    # модель угадывает язык и на коротких записях регулярно ошибается.
    language: str = "kazakh"
    task: str = "transcribe"

    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    output_dir: str = "outputs/whisper-kk"

    learning_rate: float = 1e-5
    warmup_steps: int = 50
    # Шаги, а не эпохи: в потоковом режиме размер датасета заранее неизвестен.
    max_steps: int = 1500
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    eval_steps: int = 250
    save_steps: int = 500
    save_total_limit: int = 2
    logging_steps: int = 25
    eval_examples: int = 200
    generation_max_length: int = 225
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps должен быть положительным")
        if self.per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size должен быть положительным")

    @property
    def effective_batch(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps

    @property
    def params_millions(self) -> int | None:
        return KNOWN_SIZES.get(self.model)


def load_config(path: str | Path) -> WhisperConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    dataset_raw = raw.pop("dataset", {}) or {}
    return WhisperConfig(dataset=DatasetSpec(**dataset_raw), **raw)
