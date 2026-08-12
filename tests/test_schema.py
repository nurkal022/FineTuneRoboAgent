import pytest
from pydantic import ValidationError

from robo_agency.schema import (
    DecisionType,
    Gesture,
    Motion,
    RobotDecision,
    Situation,
    Speech,
    decision_json_schema,
)


def test_act_with_speech_is_valid():
    decision = RobotDecision(
        decision=DecisionType.ACT,
        confidence=0.9,
        speech=Speech(text="Привет"),
    )
    assert decision.decision is DecisionType.ACT


def test_wait_cannot_carry_speech():
    """Вырожденная стратегия «формально WAIT, но реплику всё равно приложу»."""
    with pytest.raises(ValidationError):
        RobotDecision(
            decision=DecisionType.WAIT,
            confidence=0.7,
            speech=Speech(text="Привет"),
        )


def test_wait_cannot_carry_motion():
    with pytest.raises(ValidationError):
        RobotDecision(
            decision=DecisionType.OBSERVE,
            confidence=0.7,
            motion=Motion(pan=10),
        )


def test_act_must_do_something():
    with pytest.raises(ValidationError):
        RobotDecision(decision=DecisionType.ACT, confidence=0.9)


def test_wait_without_payload_is_valid():
    decision = RobotDecision(decision=DecisionType.WAIT, confidence=0.7)
    assert decision.speech is None
    assert decision.motion is None


@pytest.mark.parametrize("pan", [-91, 91, 180])
def test_pan_outside_servo_range_rejected(pan):
    with pytest.raises(ValidationError):
        Motion(pan=pan)


@pytest.mark.parametrize("tilt", [-31, 31])
def test_tilt_outside_servo_range_rejected(tilt):
    with pytest.raises(ValidationError):
        Motion(tilt=tilt)


def test_pan_at_boundary_accepted():
    assert Motion(pan=-90).pan == -90
    assert Motion(pan=90).pan == 90


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_confidence_outside_unit_interval_rejected(confidence):
    with pytest.raises(ValidationError):
        RobotDecision(decision=DecisionType.WAIT, confidence=confidence)


def test_gesture_enum_is_closed():
    with pytest.raises(ValidationError):
        Motion(gesture="dance")


def test_json_schema_carries_servo_limits():
    """Грамматика декодера строится из этой схемы — пределы обязаны в неё попасть."""
    schema = decision_json_schema()
    motion = schema["$defs"]["Motion"]["properties"]
    assert motion["pan"]["minimum"] == -90
    assert motion["pan"]["maximum"] == 90
    assert motion["tilt"]["minimum"] == -30
    assert motion["tilt"]["maximum"] == 30


def test_situation_prompt_json_drops_empty_fields():
    situation = Situation(observations=["человек вошёл"], silence_sec=3.0)
    payload = situation.to_prompt_json()
    assert payload["observations"] == ["человек вошёл"]
    assert "person" not in payload


def test_decision_roundtrip_through_json():
    original = RobotDecision(
        decision=DecisionType.ACT,
        confidence=0.8,
        speech=Speech(text="Привет", lang="ru"),
        motion=Motion(pan=15, gesture=Gesture.NOD),
        memory_write="поздоровался",
    )
    restored = RobotDecision.model_validate_json(original.model_dump_json())
    assert restored == original
