import pytest

from robo_agency.training.backends import SUPPORTED_ENGINES, validate_engine
from robo_agency.training.dpo import DpoConfigSpec
from robo_agency.training.dpo import load_config as load_dpo_config
from robo_agency.training.sft import SftConfigSpec
from robo_agency.training.sft import load_config as load_sft_config


def test_supported_engines():
    assert set(SUPPORTED_ENGINES) == {"hf", "unsloth"}


@pytest.mark.parametrize("engine", ["hf", "unsloth"])
def test_valid_engine_accepted(engine):
    assert validate_engine(engine) == engine


def test_unknown_engine_rejected_early():
    """Опечатка в конфиге должна падать сразу, а не через час обучения."""
    with pytest.raises(ValueError, match="engine"):
        SftConfigSpec(engine="unslot")


def test_dpo_rejects_unknown_engine():
    with pytest.raises(ValueError, match="engine"):
        DpoConfigSpec(engine="unslot")


def test_sft_config_roundtrip_from_yaml(tmp_path):
    path = tmp_path / "sft.yaml"
    path.write_text(
        """
base_model: Qwen/Qwen3-4B
engine: hf
learning_rate: 2.0e-4
lora:
  r: 16
  alpha: 32
  dropout: 0.05
""",
        encoding="utf-8",
    )
    config = load_sft_config(path)

    assert config.base_model == "Qwen/Qwen3-4B"
    assert config.engine == "hf"
    assert config.learning_rate == pytest.approx(2e-4)
    assert config.lora.r == 16
    assert config.lora.alpha == 32
    assert config.lora.dropout == pytest.approx(0.05)


def test_sft_defaults_match_spec():
    config = SftConfigSpec()
    assert config.lora.r == 32
    assert config.lora.alpha == 64
    assert config.max_length == 4096
    # Все линейные слои: сужение до q/v роняет качество структурного вывода.
    assert len(config.lora.target_modules) == 7


def test_dpo_config_roundtrip_from_yaml(tmp_path):
    path = tmp_path / "dpo.yaml"
    path.write_text("engine: hf\nbeta: 0.2\nlearning_rate: 1.0e-6\n", encoding="utf-8")
    config = load_dpo_config(path)

    assert config.engine == "hf"
    assert config.beta == pytest.approx(0.2)
    assert config.learning_rate == pytest.approx(1e-6)


def test_dpo_learning_rate_stays_low_by_default():
    """Высокий lr на DPO разрушает результат SFT — дефолт обязан быть низким."""
    assert DpoConfigSpec().learning_rate <= 1e-5


def test_shipped_configs_are_valid():
    """Конфиги в репозитории должны грузиться без правок."""
    sft = load_sft_config("configs/sft.yaml")
    dpo = load_dpo_config("configs/dpo.yaml")

    assert sft.engine in SUPPORTED_ENGINES
    assert dpo.engine in SUPPORTED_ENGINES
    # Адаптер, обученный одним бэкендом, дообучается им же.
    assert sft.engine == dpo.engine
    assert dpo.adapter_path == sft.output_dir
