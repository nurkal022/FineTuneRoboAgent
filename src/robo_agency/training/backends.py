"""Загрузка модели: бэкенд HF/peft или Unsloth.

Unsloth даёт примерно двукратную экономию VRAM и ускорение на LoRA, но платит
за это жёсткой привязкой к версиям transformers/trl, которые он патчит при
импорте. Для диссертации это риск воспроизводимости через год-два, поэтому
бэкенд именно переключаемый: обе ветки рабочие и дают один и тот же адаптер.

ВАЖНО ПРО ПОРЯДОК ИМПОРТА: unsloth обязан импортироваться раньше transformers
и trl, иначе его патчи не применятся и экономии не будет. Поэтому загрузка
модели вызывается до импорта тренера, а не наоборот.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

Engine = Literal["hf", "unsloth"]
SUPPORTED_ENGINES: tuple[str, ...] = ("hf", "unsloth")

# Разметка ролей в шаблоне ChatML (Qwen3). Нужна Unsloth-ветке, чтобы считать
# лосс только по ответу ассистента.
QWEN_INSTRUCTION_PART = "<|im_start|>user\n"
QWEN_RESPONSE_PART = "<|im_start|>assistant\n"


@dataclass(slots=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    # None означает, что модель уже обёрнута в LoRA (случай Unsloth) и
    # передавать peft_config тренеру не нужно.
    peft_config: Any | None


def validate_engine(engine: str) -> Engine:
    if engine not in SUPPORTED_ENGINES:
        raise ValueError(f"engine должен быть одним из {SUPPORTED_ENGINES}, получено {engine!r}")
    return engine  # type: ignore[return-value]


def _target_modules(config: Any) -> list[str]:
    return list(config.lora.target_modules)


def load_unsloth(config: Any, for_training: bool = True) -> LoadedModel:
    """Загружает базу и сразу навешивает LoRA средствами Unsloth."""
    from unsloth import FastLanguageModel

    if config.lora.dropout != 0.0:
        logger.warning(
            "Unsloth оптимизирован под lora_dropout=0, задано %.3f. "
            "Обучение сработает, но быстрого пути не будет.",
            config.lora.dropout,
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_length,
        dtype=None,  # Unsloth сам выберет bf16/fp16 по возможностям карты
        load_in_4bit=config.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=_target_modules(config),
        bias="none",
        # "unsloth" — их вариант чекпойнтинга, экономит заметно больше обычного
        use_gradient_checkpointing="unsloth" if config.gradient_checkpointing else False,
        random_state=config.seed,
    )
    logger.info("Бэкенд Unsloth: модель загружена, LoRA навешана")
    return LoadedModel(model=model, tokenizer=tokenizer, peft_config=None)


def load_hf(config: Any) -> LoadedModel:
    """Стандартная загрузка transformers + peft."""
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if config.bf16 else torch.float16,
        "trust_remote_code": True,
    }
    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=_target_modules(config),
        bias="none",
        task_type="CAUSAL_LM",
    )
    logger.info("Бэкенд HF: модель загружена, LoRA будет навешана тренером")
    return LoadedModel(model=model, tokenizer=tokenizer, peft_config=peft_config)


def load_for_sft(config: Any) -> LoadedModel:
    engine = validate_engine(config.engine)
    return load_unsloth(config) if engine == "unsloth" else load_hf(config)


def load_for_dpo(config: Any) -> LoadedModel:
    """Загружает адаптер после SFT для продолжения обучения через DPO."""
    engine = validate_engine(config.engine)

    if engine == "unsloth":
        from unsloth import FastLanguageModel, PatchDPOTrainer

        # Патч обязан быть применён до создания DPOTrainer.
        PatchDPOTrainer()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.adapter_path,  # Unsloth сам подхватит базу из adapter_config
            max_seq_length=config.max_length,
            dtype=None,
            load_in_4bit=config.load_in_4bit,
        )
        FastLanguageModel.for_training(model)
        logger.info("Бэкенд Unsloth: адаптер после SFT загружен для DPO")
        return LoadedModel(model=model, tokenizer=tokenizer, peft_config=None)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, config.adapter_path, is_trainable=True)
    model.config.use_cache = False
    logger.info("Бэкенд HF: адаптер после SFT загружен для DPO")
    return LoadedModel(model=model, tokenizer=tokenizer, peft_config=None)


def prepare_dataset_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    engine: Engine,
) -> list[dict[str, Any]]:
    """Приводит корпус к виду, который понимает выбранный бэкенд.

    TRL разбирает диалоговый формат сам, и HF-ветке отдаём `messages` как есть:
    на этом же разборе держится assistant_only_loss.

    Unsloth подменяет `_prepare_dataset` своей версией, а та принимает только
    готовый `text`, пару `prompt`/`completion` или уже токенизированный вход.
    Колонка `messages` до неё не доживает: тренер падает с требованием
    formatting_func — причём уже ПОСЛЕ загрузки восьмимиллиардной модели.
    Поэтому шаблон чата разворачиваем здесь, до создания тренера.

    Разворачивает ровно тот же шаблон, что потом применяется на инференсе,
    поэтому рассогласования между обучением и работой не возникает.
    """
    if validate_engine(engine) == "hf":
        return rows

    rendered: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not messages:
            raise ValueError(
                f"Строка {index} без поля `messages`, развернуть шаблон чата не из чего: "
                f"доступные поля {sorted(row)}"
            )
        rendered.append(
            {
                "text": tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    # Ответ ассистента должен остаться в тексте: именно его
                    # модель и учится порождать.
                    add_generation_prompt=False,
                )
            }
        )
    return rendered


def apply_assistant_only_loss(trainer: Any, engine: Engine) -> Any:
    """Считать лосс только по ответу ассистента.

    В HF-ветке это делает сам SFTConfig(assistant_only_loss=True). В Unsloth
    механизм другой, и без него модель будет учиться порождать собственный вход,
    то есть описание ситуации.
    """
    if engine != "unsloth":
        return trainer

    from unsloth.chat_templates import train_on_responses_only

    return train_on_responses_only(
        trainer,
        instruction_part=QWEN_INSTRUCTION_PART,
        response_part=QWEN_RESPONSE_PART,
    )
