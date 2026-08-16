import pytest

from robo_agency.asr.metrics import character_error_rate, word_error_rate


def test_perfect_match_is_zero():
    assert word_error_rate(["сәлем әлем"], ["сәлем әлем"]).rate == 0.0


def test_case_and_punctuation_do_not_count_as_errors():
    """Нормализация встроена в метрику: иначе меряется пунктуация, а не речь."""
    assert word_error_rate(["Сәлем, әлем!"], ["сәлем әлем"]).rate == 0.0


def test_single_substitution():
    result = word_error_rate(["екі сөз бар"], ["екі сөз жоқ"])
    assert result.substitutions == 1
    assert result.rate == pytest.approx(1 / 3)


def test_deletion_counted():
    result = word_error_rate(["екі сөз бар"], ["екі сөз"])
    assert result.deletions == 1
    assert result.insertions == 0


def test_insertion_counted():
    result = word_error_rate(["екі сөз"], ["екі сөз бар"])
    assert result.insertions == 1
    assert result.deletions == 0


def test_error_types_separated():
    """Разложение важнее числа: вставки и пропуски — разные болезни модели."""
    result = word_error_rate(["бір екі үш төрт"], ["бір өзге үш төрт бес"])
    assert result.substitutions == 1
    assert result.insertions == 1
    assert result.deletions == 0


def test_empty_hypothesis_is_total_loss():
    result = word_error_rate(["бір екі үш"], [""])
    assert result.deletions == 3
    assert result.rate == 1.0


def test_rate_can_exceed_one_on_hallucination():
    """Модель, дописавшая лишнее, даёт WER выше единицы — это норма метрики."""
    result = word_error_rate(["бір"], ["бір екі үш төрт"])
    assert result.rate > 1.0


def test_cer_lower_than_wer_on_ending_error():
    """Одна ошибка в окончании роняет целое слово по WER, но не по CER.

    В казахском это происходит постоянно из-за агглютинации, поэтому CER
    показательнее.
    """
    reference = ["кітапты оқыдым"]
    hypothesis = ["кітапты оқыдык"]

    wer = word_error_rate(reference, hypothesis).rate
    cer = character_error_rate(reference, hypothesis).rate
    assert cer < wer


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="не совпадает"):
        word_error_rate(["бір", "екі"], ["бір"])


def test_multiple_utterances_aggregated():
    result = word_error_rate(["бір екі", "үш төрт"], ["бір екі", "үш бес"])
    assert result.substitutions == 1
    assert result.reference_length == 4
    assert result.rate == pytest.approx(0.25)


def test_describe_mentions_breakdown():
    text = word_error_rate(["бір екі"], ["бір"]).describe()
    assert "пропуски" in text
    assert "WER" in text
