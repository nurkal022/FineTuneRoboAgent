import pytest

from robo_agency.data.mixer import Source, build_mix, drop_empty_and_renormalize


def make(count: int, tag: str):
    return [{"messages": [{"role": "user", "content": f"{tag}-{i}"}]} for i in range(count)]


def test_empty_source_is_dropped_and_shares_renormalized():
    """Датасет закрыт авторизацией — корпус должен собраться из остальных."""
    sources = [
        Source("proactivity", make(1000, "p"), 0.55),
        Source("function_calling", [], 0.20),
        Source("replay", make(1000, "r"), 0.25),
    ]

    alive = drop_empty_and_renormalize(sources)

    assert [s.name for s in alive] == ["proactivity", "replay"]
    assert sum(s.proportion for s in alive) == pytest.approx(1.0)
    assert alive[0].proportion == pytest.approx(0.55 / 0.80)


def test_all_sources_empty_is_an_error():
    sources = [Source("a", [], 0.5), Source("b", [], 0.5)]

    with pytest.raises(ValueError, match="пусты"):
        drop_empty_and_renormalize(sources)


def test_nothing_changes_when_all_sources_present():
    sources = [Source("a", make(10, "a"), 0.5), Source("b", make(10, "b"), 0.5)]

    alive = drop_empty_and_renormalize(sources)

    assert [s.proportion for s in alive] == [0.5, 0.5]


def test_mix_builds_after_a_source_is_lost():
    """Раньше нулевой источник давал пустой корпус: предел считается по минимуму."""
    sources = [
        Source("proactivity", make(1000, "p"), 0.55),
        Source("function_calling", [], 0.20),
        Source("replay", make(1000, "r"), 0.25),
    ]

    mixed, report = build_mix(drop_empty_and_renormalize(sources), target_size=400, seed=1)

    assert len(mixed) == 400
    assert "function_calling" not in report.per_source
