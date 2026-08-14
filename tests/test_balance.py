import pytest

from robo_agency.data.build import _decision_of
from robo_agency.data.mixer import downsample_to_balance


def example(decision: str, tag: int):
    return {"messages": [
        {"role": "user", "content": f"ситуация {tag}"},
        {"role": "assistant", "content": f'{{"decision": "{decision}", "confidence": 0.8}}'},
    ]}


def test_majority_class_is_downsampled():
    """Перекос в WAIT учит робота молчать всегда — это выглядит как высокая
    точность и полная бесполезность."""
    data = [example("WAIT", i) for i in range(100)] + [example("ACT", i) for i in range(20)]

    balanced, counts = downsample_to_balance(data, _decision_of, seed=1)

    assert counts == {"ACT": 20, "WAIT": 20}
    assert len(balanced) == 40


def test_minority_class_is_not_duplicated():
    """Размножение редких положительных примеров ведёт к переобучению на них."""
    data = [example("WAIT", i) for i in range(50)] + [example("ACT", i) for i in range(5)]

    balanced, _ = downsample_to_balance(data, _decision_of, seed=1)

    acts = [e["messages"][0]["content"] for e in balanced if _decision_of(e) == "ACT"]
    assert len(acts) == len(set(acts))


def test_single_class_passed_through_unchanged():
    data = [example("WAIT", i) for i in range(10)]
    balanced, counts = downsample_to_balance(data, _decision_of, seed=1)

    assert len(balanced) == 10
    assert counts == {"WAIT": 10}


def test_balancing_is_deterministic():
    data = [example("WAIT", i) for i in range(50)] + [example("ACT", i) for i in range(10)]

    first, _ = downsample_to_balance(data, _decision_of, seed=7)
    second, _ = downsample_to_balance(data, _decision_of, seed=7)

    assert [e["messages"][0]["content"] for e in first] == [
        e["messages"][0]["content"] for e in second
    ]


@pytest.mark.parametrize("decision", ["ACT", "WAIT", "OBSERVE"])
def test_decision_extracted_from_example(decision):
    assert _decision_of(example(decision, 1)) == decision


def test_unknown_decision_labelled_explicitly():
    broken = {"messages": [{"role": "assistant", "content": "{}"}]}
    assert _decision_of(broken) == "UNKNOWN"
