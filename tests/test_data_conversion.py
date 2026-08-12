import pytest

from robo_agency.data import conversational, proactivity
from robo_agency.data.fields import FieldNotFoundError, normalize_label


def test_accepted_row_becomes_act():
    rows = [{"events": "человек подошёл", "task": "поздороваться", "label": "accepted"}]
    examples = list(proactivity.convert(rows, rows[0].keys()))

    assert len(examples) == 1
    assistant = examples[0]["messages"][-1]["content"]
    assert '"ACT"' in assistant
    assert "поздороваться" in assistant


def test_rejected_row_becomes_wait_without_speech():
    rows = [{"events": "люди говорят между собой", "task": "вмешаться", "label": "rejected"}]
    examples = list(proactivity.convert(rows, rows[0].keys()))

    assistant = examples[0]["messages"][-1]["content"]
    assert '"WAIT"' in assistant
    assert "вмешаться" not in assistant, "отклонённое действие не должно утекать в ответ"


def test_unrecognised_label_is_dropped_not_guessed():
    rows = [{"events": "что-то", "task": "что-то", "label": "maybe"}]
    assert list(proactivity.convert(rows, rows[0].keys())) == []


def test_missing_field_error_lists_available_columns():
    rows = [{"foo": 1, "bar": 2}]
    with pytest.raises(FieldNotFoundError) as excinfo:
        list(proactivity.convert(rows, rows[0].keys()))
    assert "foo" in str(excinfo.value)


def test_explicit_field_map_overrides_autodetection():
    rows = [{"ctx": "человек рядом", "suggestion": "спросить", "verdict": 1}]
    config = proactivity.ProactivityConfig(
        field_map=proactivity.ProactivityFieldMap(
            observation="ctx", action="suggestion", label="verdict"
        )
    )
    examples = list(proactivity.convert(rows, rows[0].keys(), config))
    assert '"ACT"' in examples[0]["messages"][-1]["content"]


@pytest.mark.parametrize(
    "value,expected",
    [("accepted", True), ("Rejected", False), (1, True), (0, False), (True, True), ("maybe", None)],
)
def test_label_normalisation(value, expected):
    assert normalize_label(value) == expected


def test_label_balance_reports_act_share():
    rows = [
        {"events": "a", "task": "t", "label": "accepted"},
        {"events": "b", "task": "t", "label": "rejected"},
    ]
    examples = list(proactivity.convert(rows, rows[0].keys()))
    balance = proactivity.label_balance(examples)
    assert balance["ACT"] == pytest.approx(0.5)
    assert balance["WAIT"] == pytest.approx(0.5)


def test_sharegpt_format_normalised():
    rows = [{"conversations": [{"from": "human", "value": "привет"}, {"from": "gpt", "value": "ага"}]}]
    result = list(conversational.convert(rows))
    assert result[0]["messages"][0]["role"] == "user"
    assert result[0]["messages"][1]["role"] == "assistant"


def test_query_answer_format_with_tools():
    rows = [{"query": "погода?", "answers": '[{"name": "get_weather"}]', "tools": "[...]"}]
    messages = list(conversational.convert(rows))[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_unparseable_row_skipped():
    assert list(conversational.convert([{"nothing": "useful"}])) == []


def test_existing_messages_passed_through():
    rows = [{"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}]
    assert list(conversational.convert(rows))[0]["messages"][0]["content"] == "a"
