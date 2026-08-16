"""Проверки настроек под whisper-large-v3-turbo.

У turbo декодер урезан до 4 слоёв, поэтому энкодер занимает почти всю модель.
Отсюда и заморозка энкодера, и восьмибитный оптимизатор: без них 809 млн
параметров не помещаются в 16 ГБ карты.
"""

import pytest

from robo_agency.asr.config import WhisperConfig, load_config

CARD_VRAM_GB = 15.5


def test_shipped_config_uses_turbo():
    config = load_config("configs/whisper_kk.yaml")

    assert config.model == "openai/whisper-large-v3-turbo"
    assert config.freeze_encoder
    assert "8bit" in config.optim
    # turbo не умеет переводить, только транскрибировать.
    assert config.task == "transcribe"


def test_freezing_encoder_leaves_only_decoder():
    """У turbo энкодер около 79% параметров: обучаемым остаётся ~174 млн."""
    config = WhisperConfig(model="openai/whisper-large-v3-turbo", freeze_encoder=True)

    assert config.params_millions == 809
    assert 150 <= config.trainable_millions <= 200


def test_without_freezing_everything_is_trainable():
    config = WhisperConfig(model="openai/whisper-large-v3-turbo", freeze_encoder=False)
    assert config.trainable_millions == config.params_millions


def test_full_finetune_of_turbo_has_no_headroom():
    """Полное дообучение 809 млн обычным Adam упирается в потолок карты.

    Утверждать «точно не влезет» оценщик не вправе: он грубый. Но запаса нет,
    и именно поэтому по умолчанию энкодер заморожен.
    """
    config = WhisperConfig(
        model="openai/whisper-large-v3-turbo",
        freeze_encoder=False,
        optim="adamw_torch",
    )
    assert config.estimate_vram_gb() > CARD_VRAM_GB * 0.9


def test_freezing_gives_real_headroom():
    common = dict(model="openai/whisper-large-v3-turbo", optim="adamw_bnb_8bit")
    frozen = WhisperConfig(**common, freeze_encoder=True).estimate_vram_gb()
    full = WhisperConfig(**common, freeze_encoder=False).estimate_vram_gb()

    # Заморозка убирает градиенты и состояния для 79% параметров.
    assert frozen < full
    assert frozen < CARD_VRAM_GB * 0.75


def test_activations_scale_with_model_size():
    """Вклад активаций должен расти с размером модели, а не быть константой.

    Сравниваются не полные оценки (там доминируют веса), а прирост от
    увеличения батча — он и есть чистый вклад активаций.
    """
    def growth(model: str) -> float:
        common = dict(model=model, freeze_encoder=True)
        big = WhisperConfig(**common, per_device_batch_size=8).estimate_vram_gb()
        small = WhisperConfig(**common, per_device_batch_size=4).estimate_vram_gb()
        return big - small

    # Энкодер turbo больше энкодера small примерно втрое по числу параметров.
    assert growth("openai/whisper-large-v3-turbo") > growth("openai/whisper-small") * 2.5


def test_shipped_config_fits_the_card():
    config = load_config("configs/whisper_kk.yaml")
    estimate = config.estimate_vram_gb()

    assert estimate is not None
    assert estimate < CARD_VRAM_GB * 0.9


def test_eight_bit_optimizer_saves_memory():
    common = dict(model="openai/whisper-large-v3-turbo", freeze_encoder=False)
    eight = WhisperConfig(**common, optim="adamw_bnb_8bit").estimate_vram_gb()
    full = WhisperConfig(**common, optim="adamw_torch").estimate_vram_gb()

    assert eight < full


def test_gradient_checkpointing_reduces_estimate():
    common = dict(model="openai/whisper-large-v3-turbo")
    with_cp = WhisperConfig(**common, gradient_checkpointing=True).estimate_vram_gb()
    without = WhisperConfig(**common, gradient_checkpointing=False).estimate_vram_gb()

    assert with_cp < without


def test_unknown_model_has_no_estimate():
    """Для неизвестной модели лучше не оценивать вовсе, чем угадывать."""
    config = WhisperConfig(model="своя/модель")
    assert config.estimate_vram_gb() is None
    assert config.trainable_millions is None


def test_turbo_learning_rate_is_conservative():
    """4 слоя декодера разъезжаются на обычном для Whisper 1e-5."""
    assert load_config("configs/whisper_kk.yaml").learning_rate <= 1e-5


@pytest.mark.parametrize(
    "model,expected_smaller_than",
    [("openai/whisper-small", 244), ("openai/whisper-large-v3", 1550)],
)
def test_freezing_helps_every_size(model, expected_smaller_than):
    config = WhisperConfig(model=model, freeze_encoder=True)
    assert config.trainable_millions < expected_smaller_than
