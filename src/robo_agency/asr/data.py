"""Данные для дообучения Whisper на казахском.

Выбор корпуса продиктован диском: на машине около 10 ГБ свободно, а KSC2
(заявленные 1000+ часов) — это сотни гигабайт, он туда не поместится. Поэтому
по умолчанию берётся FLEURS kk_kz: порядка десяти часов, открытый, и его
достаточно, чтобы сдвинуть WER с 77% в разумную область.

Датасет грузится потоком (streaming): аудио не оседает на диске целиком, а
проходит через препроцессор и выбрасывается. Иначе даже небольшой корпус в
распакованном виде съел бы весь остаток места.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Whisper обучен на 16 кГц и другой частоты не принимает.
SAMPLE_RATE = 16_000


@dataclass(slots=True)
class DatasetSpec:
    path: str = "google/fleurs"
    name: str | None = "kk_kz"
    train_split: str = "train"
    eval_split: str = "validation"
    # Поле с расшифровкой. У FLEURS это `transcription`, у Common Voice `sentence`.
    text_column: str | None = None
    audio_column: str = "audio"
    streaming: bool = True


_TEXT_CANDIDATES = ("transcription", "sentence", "text", "raw_transcription", "normalized_text")


def resolve_text_column(columns: list[str], override: str | None = None) -> str:
    """Ищет колонку с расшифровкой: имена различаются между корпусами."""
    if override:
        if override not in columns:
            raise KeyError(f"Колонки {override!r} нет. Доступные: {columns}")
        return override

    for candidate in _TEXT_CANDIDATES:
        if candidate in columns:
            return candidate

    raise KeyError(
        f"Не нашёл колонку с расшифровкой среди {columns}. "
        f"Укажите её явно через text_column в конфиге."
    )


def load_split(spec: DatasetSpec, split: str):
    """Грузит сплит, приводя аудио к 16 кГц."""
    from datasets import Audio, load_dataset

    dataset = load_dataset(spec.path, spec.name, split=split, streaming=spec.streaming)
    return dataset.cast_column(spec.audio_column, Audio(sampling_rate=SAMPLE_RATE))


def build_preprocessor(processor, spec: DatasetSpec, text_column: str):
    """Аудио -> лог-мел признаки, текст -> идентификаторы токенов."""

    def prepare(batch: dict[str, Any]) -> dict[str, Any]:
        audio = batch[spec.audio_column]
        features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        labels = processor.tokenizer(batch[text_column]).input_ids
        return {"input_features": features, "labels": labels}

    return prepare


@dataclass
class SpeechCollator:
    """Складывает батч: признаки паддятся отдельно от меток.

    Паддинг в метках заменяется на -100, иначе модель будет учиться
    предсказывать padding-токены как настоящий текст.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        audio_inputs = [{"input_features": item["input_features"]} for item in features]
        batch = self.processor.feature_extractor.pad(audio_inputs, return_tensors="pt")

        label_inputs = [{"input_ids": item["labels"]} for item in features]
        labels_batch = self.processor.tokenizer.pad(label_inputs, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Токен начала декодирования модель добавляет сама; если он уже есть
        # в метках, при обучении получится сдвиг на один токен.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


@dataclass(slots=True)
class SplitStats:
    examples: int
    empty_text: int
    text_column: str
    columns: list[str] = field(default_factory=list)


def peek(spec: DatasetSpec, split: str, limit: int = 50) -> SplitStats:
    """Смотрит на сплит, не скачивая его целиком.

    Нужно до запуска обучения: имена колонок в корпусах различаются, и
    выяснять это после загрузки модели — потерянные минуты.
    """
    dataset = load_split(spec, split)
    rows: list[dict[str, Any]] = []
    for row in dataset:
        rows.append(row)
        if len(rows) >= limit:
            break

    if not rows:
        raise ValueError(f"Сплит {split} пуст")

    columns = list(rows[0].keys())
    text_column = resolve_text_column(columns, spec.text_column)
    empty = sum(1 for row in rows if not str(row.get(text_column) or "").strip())
    return SplitStats(
        examples=len(rows), empty_text=empty, text_column=text_column, columns=columns
    )


def iter_reference_texts(spec: DatasetSpec, split: str, limit: int) -> Iterator[str]:
    dataset = load_split(spec, split)
    text_column = None
    for index, row in enumerate(dataset):
        if text_column is None:
            text_column = resolve_text_column(list(row.keys()), spec.text_column)
        yield row[text_column]
        if index + 1 >= limit:
            return
