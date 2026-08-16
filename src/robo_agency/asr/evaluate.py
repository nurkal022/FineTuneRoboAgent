"""Замер WER и CER для Whisper на казахском.

Одним и тем же кодом снимается базовая точка (модель без дообучения) и
результат после него. Это принципиально: сравнивать замеры, сделанные разными
скриптами с разной нормализацией, бессмысленно.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import WhisperConfig
from .data import SAMPLE_RATE, resolve_text_column
from .parquet_reader import iter_examples
from .metrics import ErrorRate, character_error_rate, word_error_rate

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AsrReport:
    model: str
    examples: int
    wer: ErrorRate
    cer: ErrorRate
    samples: list[tuple[str, str]]

    def describe(self) -> str:
        lines = [
            f"Модель: {self.model}",
            f"Примеров: {self.examples}",
            "",
            self.wer.describe("WER"),
            "",
            self.cer.describe("CER"),
        ]
        if self.samples:
            lines.append("\nПримеры распознавания:")
            for reference, hypothesis in self.samples:
                lines.append(f"  эталон:  {reference}")
                lines.append(f"  модель:  {hypothesis}")
                lines.append("")
        return "\n".join(lines)


def transcribe(
    model_path: str,
    config: WhisperConfig,
    split: str | None = None,
    limit: int | None = None,
    show: int = 5,
) -> AsrReport:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    spec = config.dataset
    split = split or spec.eval_split
    limit = limit or config.eval_examples

    processor = WhisperProcessor.from_pretrained(
        model_path, language=config.language, task=config.task
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Читаем parquet напрямую: datasets 4.x требует torchcodec для распаковки
    # аудио, а тянуть ещё одну привязанную к torch зависимость на этой машине
    # рискованно — установка пакетов тут дважды подменяла CUDA-сборку на CPU.
    dataset = iter_examples(
        spec.path, split, config=spec.name, limit=limit, audio_column=spec.audio_column
    )

    references: list[str] = []
    hypotheses: list[str] = []
    samples: list[tuple[str, str]] = []
    text_column: str | None = None

    with torch.no_grad():
        for index, row in enumerate(dataset):
            if index >= limit:
                break
            if text_column is None:
                text_column = resolve_text_column(list(row.keys()), spec.text_column)

            audio = row[spec.audio_column]
            features = processor.feature_extractor(
                audio["array"], sampling_rate=audio.get("sampling_rate", SAMPLE_RATE),
                return_tensors="pt",
            ).input_features.to(device)

            predicted = model.generate(
                features,
                max_length=config.generation_max_length,
                language=config.language,
                task=config.task,
            )
            hypothesis = processor.batch_decode(predicted, skip_special_tokens=True)[0]

            references.append(row[text_column])
            hypotheses.append(hypothesis)
            if len(samples) < show:
                samples.append((row[text_column], hypothesis))

            if (index + 1) % 25 == 0:
                logger.info("Распознано %d из %d", index + 1, limit)

    if not references:
        raise ValueError("Не удалось получить ни одного примера")

    return AsrReport(
        model=model_path,
        examples=len(references),
        wer=word_error_rate(references, hypotheses),
        cer=character_error_rate(references, hypotheses),
        samples=samples,
    )
