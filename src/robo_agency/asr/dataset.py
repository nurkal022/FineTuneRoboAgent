"""Потоковый датасет поверх прямого чтения parquet.

Обучение не может ходить через `datasets` по той же причине, что и замер:
в версии 4.x распаковка аудио требует `torchcodec`, привязанного к версии
torch, а трогать torch на этой машине опасно — установка связанных с ним
пакетов дважды подменяла CUDA-сборку на CPU.

Датасет итерируемый: длина заранее неизвестна, и обучение задаётся шагами,
а не эпохами.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator

from .parquet_reader import iter_examples

logger = logging.getLogger(__name__)


def build_preprocessor(processor, audio_column: str, text_column: str) -> Callable:
    """Аудио -> лог-мел признаки, расшифровка -> идентификаторы токенов."""

    def prepare(row: dict[str, Any]) -> dict[str, Any]:
        audio = row[audio_column]
        features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        labels = processor.tokenizer(row[text_column]).input_ids
        return {"input_features": features, "labels": labels}

    return prepare


class ParquetSpeechDataset:
    """Итерируемый набор записей корпуса, уже приведённых к входу модели.

    Наследование от torch.utils.data.IterableDataset делается лениво, чтобы
    модуль импортировался в окружении без torch (например, в тестах разбора).
    """

    def __init__(
        self,
        repo: str,
        split: str,
        prepare: Callable[[dict[str, Any]], dict[str, Any]],
        config: str | None = None,
        limit: int | None = None,
        audio_column: str = "audio",
        repeat: bool = False,
    ) -> None:
        self.repo = repo
        self.split = split
        self.prepare = prepare
        self.config = config
        self.limit = limit
        self.audio_column = audio_column
        # Обучение задаётся шагами: когда корпус кончится, начинаем сначала.
        self.repeat = repeat

    def __iter__(self) -> Iterator[dict[str, Any]]:
        passes = 0
        while True:
            produced = 0
            for row in iter_examples(
                self.repo, self.split, self.config, self.limit, self.audio_column
            ):
                produced += 1
                yield self.prepare(row)

            passes += 1
            if not self.repeat:
                return
            if produced == 0:
                raise RuntimeError(f"Сплит {self.split} пуст, повторять нечего")
            logger.info("Проход %d по корпусу завершён (%d записей)", passes, produced)


def as_torch_dataset(dataset: ParquetSpeechDataset):
    """Оборачивает в torch.utils.data.IterableDataset для тренера."""
    from torch.utils.data import IterableDataset

    class _Wrapped(IterableDataset):
        def __iter__(self):
            return iter(dataset)

    return _Wrapped()


def first_row(repo: str, split: str, config: str | None, audio_column: str = "audio") -> dict:
    """Одна запись — чтобы определить имя колонки с расшифровкой."""
    for row in iter_examples(repo, split, config, limit=1, audio_column=audio_column):
        return row
    raise ValueError(f"Сплит {split} пуст")
