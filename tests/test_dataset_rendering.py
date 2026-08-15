"""Подготовка датасета под конкретный бэкенд обучения.

Ветка Unsloth не умеет разбирать диалоговый формат сама: её `_prepare_dataset`
принимает только готовый `text`, пару `prompt`/`completion` или уже
токенизированный вход. Колонка `messages` до этого падала с
`RuntimeError: Unsloth: You must specify a formatting_func` — уже после
загрузки восьмимиллиардной модели, то есть через несколько минут ожидания.
"""

from __future__ import annotations

import pytest

from robo_agency.training.backends import QWEN_RESPONSE_PART, prepare_dataset_rows


class FakeTokenizer:
    """Мимикрия под шаблон ChatML: важен лишь маркер ответа ассистента."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        self.calls.append({"messages": messages, "tokenize": tokenize, **kwargs})
        parts = []
        for message in messages:
            parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
        return "".join(parts)


ROWS = [
    {
        "messages": [
            {"role": "system", "content": "система"},
            {"role": "user", "content": "ситуация"},
            {"role": "assistant", "content": '{"decision": "WAIT"}'},
        ]
    }
]


def test_hf_engine_keeps_conversational_format():
    """У TRL свой разбор `messages`, и assistant_only_loss держится на нём."""
    rows = prepare_dataset_rows(ROWS, FakeTokenizer(), "hf")

    assert rows == ROWS


def test_unsloth_engine_renders_text_column():
    rows = prepare_dataset_rows(ROWS, FakeTokenizer(), "unsloth")

    assert list(rows[0]) == ["text"]
    assert '{"decision": "WAIT"}' in rows[0]["text"]


def test_rendered_text_keeps_marker_for_assistant_only_loss():
    """Без этого маркера train_on_responses_only не найдёт границу ответа.

    Тогда лосс считается по всему тексту, и модель учится порождать ещё и
    описание ситуации — то есть собственный вход.
    """
    rows = prepare_dataset_rows(ROWS, FakeTokenizer(), "unsloth")

    assert QWEN_RESPONSE_PART in rows[0]["text"]


def test_rendering_does_not_add_generation_prompt():
    """add_generation_prompt обрезал бы ответ ассистента — учиться было бы нечему."""
    tokenizer = FakeTokenizer()
    prepare_dataset_rows(ROWS, tokenizer, "unsloth")

    assert tokenizer.calls[0]["tokenize"] is False
    assert tokenizer.calls[0].get("add_generation_prompt", False) is False


def test_rows_without_messages_are_rejected_early():
    """Лучше упасть на подготовке, чем через несколько минут внутри тренера."""
    with pytest.raises(ValueError, match="messages"):
        prepare_dataset_rows([{"text": "уже готово"}], FakeTokenizer(), "unsloth")
