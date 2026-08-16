"""Тесты потокового датасета поверх прямого чтения parquet."""

import pytest

from robo_agency.asr.dataset import ParquetSpeechDataset, build_preprocessor


class _Features:
    """Двойник feature_extractor: важен только факт вызова и форма ответа."""

    def __call__(self, array, sampling_rate=None):
        return type("R", (), {"input_features": [[float(len(array))]]})()


class _Tokenizer:
    def __call__(self, text):
        return type("R", (), {"input_ids": [len(text)]})()


class FakeProcessor:
    def __init__(self):
        self.feature_extractor = _Features()
        self.tokenizer = _Tokenizer()


def fake_rows(count: int):
    return [
        {"audio": {"array": [0.0] * (i + 1), "sampling_rate": 16000}, "transcription": "сәлем"}
        for i in range(count)
    ]


@pytest.fixture
def patched(monkeypatch):
    def fake_iter(repo, split, config=None, limit=None, audio_column="audio"):
        rows = fake_rows(3)
        yield from (rows[:limit] if limit else rows)

    monkeypatch.setattr("robo_agency.asr.dataset.iter_examples", fake_iter)


def test_preprocessor_produces_model_inputs():
    prepare = build_preprocessor(FakeProcessor(), "audio", "transcription")
    result = prepare({"audio": {"array": [0.0, 0.0], "sampling_rate": 16000}, "transcription": "аб"})

    assert "input_features" in result
    assert "labels" in result


def test_single_pass_stops_at_end(patched):
    prepare = build_preprocessor(FakeProcessor(), "audio", "transcription")
    dataset = ParquetSpeechDataset("repo", "train", prepare, repeat=False)

    assert len(list(dataset)) == 3


def test_repeat_cycles_the_corpus(patched):
    """Обучение задаётся шагами: корпус должен проходиться повторно."""
    prepare = build_preprocessor(FakeProcessor(), "audio", "transcription")
    dataset = ParquetSpeechDataset("repo", "train", prepare, repeat=True)

    produced = []
    for item in dataset:
        produced.append(item)
        if len(produced) >= 7:
            break

    assert len(produced) == 7


def test_limit_respected(patched):
    prepare = build_preprocessor(FakeProcessor(), "audio", "transcription")
    dataset = ParquetSpeechDataset("repo", "validation", prepare, limit=2)

    assert len(list(dataset)) == 2


def test_empty_corpus_with_repeat_raises(monkeypatch):
    """Бесконечный цикл по пустому корпусу должен падать, а не висеть."""
    monkeypatch.setattr(
        "robo_agency.asr.dataset.iter_examples",
        lambda *a, **k: iter(()),
    )
    prepare = build_preprocessor(FakeProcessor(), "audio", "transcription")
    dataset = ParquetSpeechDataset("repo", "train", prepare, repeat=True)

    with pytest.raises(RuntimeError, match="пуст"):
        next(iter(dataset))
