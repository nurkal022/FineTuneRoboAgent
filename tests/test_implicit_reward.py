from robo_agency.reward.implicit import (
    InteractionRecord,
    ReactionWindow,
    build_preference_pairs,
    score_reaction,
)
from robo_agency.schema import Emotion


def test_smile_scores_higher_than_frown():
    positive = ReactionWindow(emotion_before=Emotion.NEUTRAL, emotion_after=Emotion.HAPPY)
    negative = ReactionWindow(emotion_before=Emotion.NEUTRAL, emotion_after=Emotion.ANGRY)
    assert score_reaction(positive) > score_reaction(negative)


def test_leaving_frame_is_penalised():
    assert score_reaction(ReactionWindow(left_frame=True)) < 0


def test_continued_dialogue_is_rewarded():
    assert score_reaction(ReactionWindow(continued_dialogue=True)) > 0


def test_repeated_question_is_penalised():
    """Переспросил то же самое — робот не был понят."""
    assert score_reaction(ReactionWindow(repeated_question=True)) < 0


def test_emotion_shift_uses_delta_not_absolute():
    """Человек пришёл злым и остался злым — это не вина робота."""
    stayed_angry = ReactionWindow(emotion_before=Emotion.ANGRY, emotion_after=Emotion.ANGRY)
    became_angry = ReactionWindow(emotion_before=Emotion.NEUTRAL, emotion_after=Emotion.ANGRY)
    assert score_reaction(stayed_angry) > score_reaction(became_angry)


def make_record(key: str, reaction: ReactionWindow, text: str) -> InteractionRecord:
    return InteractionRecord(
        situation_key=key,
        messages=[{"role": "user", "content": key}, {"role": "assistant", "content": text}],
        decision_json=text,
        reaction=reaction,
    )


def test_pairs_built_within_same_situation():
    records = [
        make_record("s1", ReactionWindow(continued_dialogue=True), "good"),
        make_record("s1", ReactionWindow(left_frame=True), "bad"),
    ]
    pairs = build_preference_pairs(records)

    assert len(pairs) == 1
    assert pairs[0]["chosen"][0]["content"] == "good"
    assert pairs[0]["rejected"][0]["content"] == "bad"


def test_pairs_not_built_across_situations():
    """Награды из разных ситуаций несравнимы."""
    records = [
        make_record("s1", ReactionWindow(continued_dialogue=True), "a"),
        make_record("s2", ReactionWindow(left_frame=True), "b"),
    ]
    assert build_preference_pairs(records) == []


def test_small_margin_pairs_dropped():
    """Разница в пределах шума измерения эмоций учит случайности."""
    records = [
        make_record("s1", ReactionWindow(engagement_seconds=1.0), "a"),
        make_record("s1", ReactionWindow(engagement_seconds=2.0), "b"),
    ]
    assert build_preference_pairs(records, min_margin=0.5) == []


def test_prompt_excludes_assistant_turn():
    records = [
        make_record("s1", ReactionWindow(continued_dialogue=True), "good"),
        make_record("s1", ReactionWindow(left_frame=True), "bad"),
    ]
    prompt = build_preference_pairs(records)[0]["prompt"]
    assert all(message["role"] != "assistant" for message in prompt)
