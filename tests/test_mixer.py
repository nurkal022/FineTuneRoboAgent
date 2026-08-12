import pytest

from robo_agency.data.mixer import Source, build_mix, train_val_split


def make(count: int, tag: str):
    return [{"messages": [{"role": "user", "content": f"{tag}-{i}"}]} for i in range(count)]


def test_proportions_respected():
    sources = [
        Source("proactivity", make(1000, "p"), 0.55),
        Source("function_calling", make(1000, "f"), 0.20),
        Source("replay", make(1000, "r"), 0.25),
    ]
    mixed, report = build_mix(sources, target_size=500, seed=1)

    assert report.total == len(mixed)
    assert report.per_source["proactivity"] == pytest.approx(275, abs=1)
    assert report.per_source["function_calling"] == pytest.approx(100, abs=1)
    assert report.per_source["replay"] == pytest.approx(125, abs=1)


def test_small_source_limits_total_size():
    """ProactiveBench мал — он и должен ограничивать весь корпус."""
    sources = [
        Source("proactivity", make(6790, "p"), 0.55),
        Source("function_calling", make(60000, "f"), 0.20),
        Source("replay", make(200000, "r"), 0.25),
    ]
    _, report = build_mix(sources, target_size=None, seed=1)

    assert report.limiting_source == "proactivity"
    assert 12000 <= report.max_possible <= 13000


def test_oversized_request_is_capped_not_duplicated():
    sources = [
        Source("small", make(100, "s"), 0.5),
        Source("big", make(10000, "b"), 0.5),
    ]
    mixed, report = build_mix(sources, target_size=100000, seed=1)

    assert report.total == 200
    contents = [m["messages"][0]["content"] for m in mixed]
    assert len(contents) == len(set(contents)), "примеры дублироваться не должны"


def test_proportions_must_sum_to_one():
    with pytest.raises(ValueError, match="1.0"):
        build_mix([Source("a", make(10, "a"), 0.4), Source("b", make(10, "b"), 0.4)])


def test_source_tag_attached():
    sources = [Source("proactivity", make(10, "p"), 1.0)]
    mixed, _ = build_mix(sources, target_size=10, seed=1)
    assert all(example["source"] == "proactivity" for example in mixed)


def test_mix_is_deterministic_for_same_seed():
    def run():
        sources = [
            Source("a", make(200, "a"), 0.5),
            Source("b", make(200, "b"), 0.5),
        ]
        mixed, _ = build_mix(sources, target_size=100, seed=7)
        return [m["messages"][0]["content"] for m in mixed]

    assert run() == run()


def test_train_val_split_is_disjoint():
    data = make(100, "x")
    train, val = train_val_split(data, val_ratio=0.1, seed=3)

    assert len(val) == 10
    assert len(train) == 90
    train_ids = {m["messages"][0]["content"] for m in train}
    val_ids = {m["messages"][0]["content"] for m in val}
    assert not (train_ids & val_ids)


def test_train_val_split_rejects_bad_ratio():
    with pytest.raises(ValueError):
        train_val_split(make(10, "x"), val_ratio=0.0)
