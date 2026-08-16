"""Дообучение Whisper на казахском.

В отличие от LLM-части проекта, здесь обучается вся модель, а не адаптер:
whisper-small это 244 млн параметров, инвариант «замороженная база» относится
к языковой модели и речевого контура не касается.
"""

from __future__ import annotations

import logging

from .config import WhisperConfig
from .data import SpeechCollator, build_preprocessor, load_split, resolve_text_column

logger = logging.getLogger(__name__)


def _first_columns(dataset) -> list[str]:
    for row in dataset:
        return list(row.keys())
    raise ValueError("Датасет пуст")


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
    train_raw = load_split(spec, spec.train_split)
    eval_raw = load_split(spec, spec.eval_split)

    text_column = resolve_text_column(_first_columns(train_raw), spec.text_column)
    logger.info("Колонка с расшифровкой: %s", text_column)

    prepare = build_preprocessor(processor, spec, text_column)
    drop = [c for c in _first_columns(train_raw)]
    train_dataset = train_raw.map(prepare, remove_columns=drop)
    eval_dataset = eval_raw.map(prepare, remove_columns=drop)

    if spec.streaming:
        # В потоковом режиме перемешивание идёт по буферу, а не по всему корпусу.
        train_dataset = train_dataset.shuffle(seed=config.seed, buffer_size=500)
        eval_dataset = eval_dataset.take(config.eval_examples)

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
        # В потоковом режиме длина неизвестна, и тренер не должен её спрашивать.
        dataloader_num_workers=2,
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
