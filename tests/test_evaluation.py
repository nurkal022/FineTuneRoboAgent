import pytest

from robo_agency.evaluation.agency import Prediction, compute
from robo_agency.evaluation.calibration import apply_temperature, calibrate
from robo_agency.schema import DecisionType

ACT = DecisionType.ACT
WAIT = DecisionType.WAIT


def test_always_act_gets_perfect_timeliness_but_bad_false_rate():
    """Главная ловушка метрик агентности: одна своевременность ничего не значит."""
    predictions = [Prediction(gold=ACT, predicted=ACT) for _ in range(10)]
    predictions += [Prediction(gold=WAIT, predicted=ACT) for _ in range(10)]

    metrics = compute(predictions)
    assert metrics.timeliness == 1.0
    assert metrics.false_intervention_rate == 1.0
    assert metrics.decision_accuracy == 0.5


def test_always_wait_has_zero_false_interventions():
    predictions = [Prediction(gold=ACT, predicted=WAIT) for _ in range(10)]
    predictions += [Prediction(gold=WAIT, predicted=WAIT) for _ in range(10)]

    metrics = compute(predictions)
    assert metrics.timeliness == 0.0
    assert metrics.false_intervention_rate == 0.0


def test_intrusiveness_counts_only_unaddressed_segments():
    predictions = [
        Prediction(gold=ACT, predicted=ACT, addressed_to_robot=True, duration_sec=60),
        Prediction(gold=WAIT, predicted=ACT, addressed_to_robot=False, duration_sec=60),
    ]
    metrics = compute(predictions)
    assert metrics.intrusiveness_per_min == pytest.approx(1.0)


def test_empty_predictions_rejected():
    with pytest.raises(ValueError):
        compute([])


def test_temperature_above_one_softens_confidence():
    assert apply_temperature(0.99, 2.0) < 0.99
    assert apply_temperature(0.5, 2.0) == pytest.approx(0.5)


def test_calibration_respects_false_intervention_budget():
    probabilities = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2]
    labels = [True, True, True, False, False, False]

    result = calibrate(probabilities, labels, max_false_intervention_rate=0.0)
    assert result.satisfied_constraint
    assert result.threshold > 0.4


def test_calibration_flags_degenerate_silent_robot():
    """Неразделимые классы: бюджет выполним только полным молчанием робота.

    Формально ограничение соблюдено, практически это немой робот — результат
    обязан быть помечен как вырожденный, а не выдан за успех.
    """
    probabilities = [0.9, 0.9]
    labels = [True, False]

    result = calibrate(probabilities, labels, max_false_intervention_rate=0.0, steps=11)
    assert result.degenerate
    assert "ВНИМАНИЕ" in result.describe()


def test_good_calibration_is_not_degenerate():
    probabilities = [0.9, 0.8, 0.2, 0.1]
    labels = [True, True, False, False]

    result = calibrate(probabilities, labels, max_false_intervention_rate=0.0)
    assert result.satisfied_constraint
    assert not result.degenerate


def test_calibration_length_mismatch_rejected():
    with pytest.raises(ValueError):
        calibrate([0.5], [True, False])
