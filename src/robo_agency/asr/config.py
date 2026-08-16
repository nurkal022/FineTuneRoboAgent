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
    "openai/whisper-large-v3-turbo": 809,
    "openai/whisper-large-v3": 1550,
}

# Доля параметров в энкодере. У turbo декодер урезан до 4 слоёв вместо 32,
# поэтому энкодер занимает почти всё: заморозив его, мы обучаем около
# 174 млн параметров вместо 809 млн.
ENCODER_SHARE = {
    "openai/whisper-large-v3-turbo": 0.79,
    "openai/whisper-large-v3": 0.41,
    "openai/whisper-medium": 0.40,
    "openai/whisper-small": 0.39,
}


@dataclass(slots=True)
class WhisperConfig:
    # turbo: 809 млн, но декодер всего 4 слоя вместо 32, поэтому он в разы
    # быстрее large-v3 при близком качестве.
    model: str = "openai/whisper-large-v3-turbo"
    # Казахский есть в списке языков Whisper, поэтому задаём его явно: без этого
    # модель угадывает язык и на коротких записях регулярно ошибается.
    language: str = "kazakh"
    # turbo обучен только транскрибировать; перевода он не умеет.
    task: str = "transcribe"

    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    output_dir: str = "outputs/whisper-kk"

    # Заморозка энкодера. Энкодер отвечает за акустику и уже силён на многих
    # языках; переучивать под казахский надо орфографию, то есть декодер.
    # Для turbo это ещё и вопрос памяти: без заморозки состояния оптимизатора
    # на 809 млн параметров не помещаются в 16 ГБ.
    freeze_encoder: bool = True
    # Восьмибитный Adam: состояния занимают 2 байта на параметр вместо 8.
    optim: str = "adamw_bnb_8bit"

    # Декодеру turbo нужен более осторожный шаг: слоёв мало, и на 1e-5 он
    # разъезжается быстрее, чем успевает выучить орфографию.
    learning_rate: float = 6e-6
    warmup_steps: int = 50
    # Шаги, а не эпохи: в потоковом режиме размер датасета заранее неизвестен.
    max_steps: int = 1500
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
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
        """Размер модели по имени.

        Дообучения выкладывают под произвольными именами
        (`shyngys879/kazakh-whisper-large-v3-turbo`), поэтому кроме точного
        совпадения проверяется вхождение известной архитектуры в имя.
        Порядок проверки от длинных имён к коротким: иначе
        `whisper-large-v3-turbo` совпал бы с `whisper-large-v3`.
        """
        if self.model in KNOWN_SIZES:
            return KNOWN_SIZES[self.model]

        name = self.model.lower()
        for known in sorted(KNOWN_SIZES, key=len, reverse=True):
            architecture = known.split("/")[-1]
            if architecture in name:
                return KNOWN_SIZES[known]
        return None

    @property
    def trainable_millions(self) -> int | None:
        """Сколько параметров реально обучается с учётом заморозки энкодера."""
        total = self.params_millions
        if total is None:
            return None
        if not self.freeze_encoder:
            return total
        share = ENCODER_SHARE.get(self.model)
        if share is None:
            name = self.model.lower()
            for known in sorted(ENCODER_SHARE, key=len, reverse=True):
                if known.split("/")[-1] in name:
                    share = ENCODER_SHARE[known]
                    break
        return total if share is None else round(total * (1 - share))

    def estimate_vram_gb(self) -> float | None:
        """Грубая оценка пика VRAM.

        Считает по обучаемым параметрам: веса и градиенты в bf16 по 2 байта,
        состояния оптимизатора 2 байта при восьмибитном Adam и 12 при обычном
        (мастер-веса в fp32 плюс два момента). Замороженный энкодер занимает
        память только весами.
        """
        total = self.params_millions
        trainable = self.trainable_millions
        if total is None or trainable is None:
            return None

        optimizer_bytes = 2 if "8bit" in self.optim else 12
        weights = total * 2 / 1024
        grads = trainable * 2 / 1024
        states = trainable * optimizer_bytes / 1024

        # Активации. Энкодер Whisper всегда обрабатывает окно в 30 секунд
        # (1500 позиций), поэтому от длины записи они не зависят — зато растут
        # и с батчем, и с размером модели: у large-v3 это 32 слоя по 1280
        # измерений против 12 по 768 у small. Плоская константа занижала бы
        # оценку для больших моделей в разы.
        per_sample = (total / 1000) * (0.35 if self.gradient_checkpointing else 1.5)
        activations = per_sample * self.per_device_batch_size

        return round(weights + grads + states + activations + 1.2, 2)


def load_config(path: str | Path) -> WhisperConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    dataset_raw = raw.pop("dataset", {}) or {}
    return WhisperConfig(dataset=DatasetSpec(**dataset_raw), **raw)
