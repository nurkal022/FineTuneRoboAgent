"""Тесты по фактической схеме репозитория thunlp/ProactiveAgent."""

import json

import pytest

from robo_agency.data import proactive_agent
from robo_agency.data.build import DatasetSpec, _resolve_format

OBS = [{"time": "10:00", "event": "открыл редактор"}, {"time": "10:01", "event": "правит файл"}]


def row(**overrides):
    base = {
        "obs": OBS,
        "pred_task": "предложить рефакторинг",
        "valid": True,
        "help_needed": True,
        "annotation": [True, True, False],
        "category": "Correct-Detection (CD)",
    }
    base.update(overrides)
    return base


def decisions(rows):
    return [json.loads(ex["messages"][-1]["content"]) for ex in proactive_agent.convert(rows)]


def test_correct_detection_becomes_act():
    result = decisions([row()])
    assert len(result) == 1
    assert result[0]["decision"] == "ACT"
    assert result[0]["speech"]["text"] == "предложить рефакторинг"


def test_correct_rejection_becomes_wait():
    result = decisions([row(pred_task=None, category="Correct-Rejection (CR)")])
    assert result[0]["decision"] == "WAIT"
    assert "speech" not in result[0]


def test_false_alarm_becomes_wait_and_drops_the_wrong_action():
    """Агент влез зря: предложенный текст — ровно то, чего делать не следовало."""
    result = decisions([row(category="False-Alarm (FA)")])
    assert result[0]["decision"] == "WAIT"
    assert "предложить рефакторинг" not in json.dumps(result[0], ensure_ascii=False)


def test_false_alarm_wins_over_help_needed_flag():
    """На реальных данных help_needed=True встречается у False-Alarm.

    Эталон — category с человеческой разметкой, а не суждение агента.
    """
    result = decisions([row(help_needed=True, category="False-Alarm (FA)")])
    assert result[0]["decision"] == "WAIT"


def test_missed_need_is_skipped_not_invented():
    """Действовать было надо, но текста действия в данных нет — выдумывать нельзя."""
    assert decisions([row(pred_task=None, category="Missed-Need (MN)")]) == []


def test_valid_flag_does_not_drop_correct_detections():
    """Флаг valid относится к разбору генерации, а не к качеству метки."""
    result = decisions([row(valid=False)])
    assert result[0]["decision"] == "ACT"


def test_falls_back_to_help_needed_when_category_missing():
    result = decisions([row(category=None, help_needed=False, pred_task=None)])
    assert result[0]["decision"] == "WAIT"


@pytest.mark.parametrize(
    "category,expected",
    [
        ("Correct-Detection (CD)", "CD"),
        ("False-Alarm (FA)", "FA"),
        ("что-то другое", None),
        (None, None),
    ],
)
def test_category_code_extraction(category, expected):
    assert proactive_agent.category_code(category) == expected


def test_observations_carry_time_and_event():
    example = next(iter(proactive_agent.convert([row()])))
    user_message = example["messages"][1]["content"]
    assert "открыл редактор" in user_message
    assert "10:00" in user_message


def test_rows_without_observations_skipped():
    assert decisions([row(obs=[])]) == []


def test_category_wins_over_missing_help_needed():
    """Категория есть — help_needed не смотрим вовсе."""
    result = decisions([row(help_needed=None, category="Correct-Detection (CD)")])
    assert result[0]["decision"] == "ACT"


def test_skipped_when_neither_category_nor_help_needed_usable():
    assert decisions([row(help_needed=None, category=None)]) == []


def test_observations_truncated_to_limit():
    many = [{"time": f"10:{i:02d}", "event": f"событие {i}"} for i in range(40)]
    config = proactive_agent.ProactiveAgentConfig(max_observations=5)
    example = next(iter(proactive_agent.convert([row(obs=many)], config)))
    user_message = example["messages"][1]["content"]
    # Хвост важнее головы: сохраняются последние наблюдения.
    assert "событие 39" in user_message
    assert "событие 0\"" not in user_message


@pytest.mark.parametrize(
    "rows,expected",
    [
        ([row()], True),
        ([{"events": "x", "task": "y", "label": "accepted"}], False),
        ([], False),
    ],
)
def test_format_detection(rows, expected):
    assert proactive_agent.matches(rows) is expected


def test_build_resolves_format_automatically():
    spec = DatasetSpec(path="whatever.jsonl")
    assert _resolve_format(spec, [row()]) == "proactive_agent"
    assert _resolve_format(spec, [{"events": "x", "task": "y", "label": 1}]) == "generic"


def test_explicit_format_overrides_detection():
    spec = DatasetSpec(path="whatever.jsonl", format="generic")
    assert _resolve_format(spec, [row()]) == "generic"
