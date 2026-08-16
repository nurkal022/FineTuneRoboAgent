"""Дообучение Whisper на казахском.

В отличие от LLM-части проекта, здесь обучается вся модель, а не адаптер:
whisper-small это 244 млн параметров, инвариант «замороженная база» относится
к языковой модели и речевого контура не касается.
"""

from __future__ import annotations

import logging

from .config import WhisperConfig
from .data import SpeechCollator, resolve_text_column
from .dataset import ParquetSpeechDataset, as_torch_dataset, build_preprocessor, first_row

logger = logging.getLogger(__name__)


def train(config: WhisperConfig) -> str:
    import torch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    processor = WhisperProcessor.from_pretrained(
        config.model, language=config.language, task=config.task
    )
    model = WhisperForConditionalGeneration.from_pretrained(config.model)

    # Язык и задача прибиваются к модели: иначе на инференсе Whisper снова
    # начнёт определять язык сам и на казахском будет срываться на русский.
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    # Подавление токенов мешает редким казахским буквам, отключаем.
    model.config.suppress_tokens = []

    if config.gradient_checkpointing:
        model.config.use_cache = False

    if config.freeze_encoder:
        # У turbo энкодер это почти четыре пятых модели. Заморозка оставляет
        # обучаемым только декодер — тот, что и отвечает за письменность.
        if hasattr(model, "freeze_encoder"):
            model.freeze_encoder()
        else:
            for parameter in model.model.encoder.parameters():
                parameter.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Обучаемых параметров: %.1f млн из %.1f млн (%.0f%%)",
        trainable / 1e6, total / 1e6, 100 * trainable / total,
    )

    spec = config.dataset
    probe = first_row(spec.path, spec.train_split, spec.name, spec.audio_column)
    text_column = resolve_text_column(list(probe.keys()), spec.text_column)
    logger.info("Колонка с расшифровкой: %s", text_column)

    prepare = build_preprocessor(processor, spec.audio_column, text_column)

    # repeat=True: обучение задаётся шагами, и корпус проходится столько раз,
    # сколько нужно, чтобы их набрать.
    train_dataset = as_torch_dataset(ParquetSpeechDataset(
        spec.path, spec.train_split, prepare, spec.name,
        audio_column=spec.audio_column, repeat=True,
    ))
    eval_dataset = as_torch_dataset(ParquetSpeechDataset(
        spec.path, spec.eval_split, prepare, spec.name,
        limit=config.eval_examples, audio_column=spec.audio_column,
    ))

    collator = SpeechCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    arguments = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16 and torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        per_device_eval_batch_size=max(1, config.per_device_batch_size // 2),
        predict_with_generate=True,
        generation_max_length=config.generation_max_length,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        optim=config.optim,
        report_to=[],
        seed=config.seed,
        remove_unused_columns=False,
        # Ноль работников намеренно: у итерируемого датасета несколько
        # процессов читали бы один и тот же поток и дублировали записи.
        dataloader_num_workers=0,
    )

    trainer = Seq2SeqTrainer(
        args=arguments,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    logger.info("Модель сохранена в %s", config.output_dir)
    return config.output_dir
