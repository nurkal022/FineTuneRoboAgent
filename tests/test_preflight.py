import importlib.util
from pathlib import Path

import pytest

from robo_agency.training.dpo import load_config as load_dpo_config
from robo_agency.training.sft import load_config as load_sft_config

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "preflight.py"

spec = importlib.util.spec_from_file_location("preflight", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)

RTX_5080_VRAM_GB = 16.0


@pytest.mark.parametrize(
    "model,expected",
    [
        ("Qwen/Qwen3-8B", 8.0),
        ("Qwen/Qwen3-4B", 4.0),
        # Версия в имени (3.1) не должна путаться с размером модели (8B).
        ("meta-llama/Llama-3.1-8B-Instruct", 8.0),
        ("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
    ],
)
def test_params_parsed_from_model_name(model, expected):
    assert preflight.params_billions(model) == expected


def test_unknown_model_falls_back_to_conservative_guess():
    assert preflight.params_billions("some/custom-model") == 8.0


def test_bf16_8b_does_not_fit_into_16gb():
    """Базовый конфиг на 5080 невозможен — preflight обязан это поймать."""
    config = load_sft_config("configs/sft.yaml")
    estimate, _ = preflight.estimate_vram_gb(config, 8.0)
    assert estimate > RTX_5080_VRAM_GB


def test_rtx5080_profile_fits_into_16gb_with_headroom():
    """Не просто «влезает», а с запасом: без него OOM ловится на длинных примерах."""
    config = load_sft_config("configs/sft_rtx5080.yaml")
    estimate, _ = preflight.estimate_vram_gb(config, 8.0)
    assert estimate < RTX_5080_VRAM_GB * 0.75


def test_estimator_matches_known_qlora_footprint():
    """Калибровка: 8B QLoRA, батч 1, длина 2048 — это порядка 7-9 ГБ.

    Если оценщик уедет от реальности, preflight начнёт либо пугать зря,
    либо пропускать конфиги, которые упадут в OOM.
    """
    config = load_sft_config("configs/sft_rtx5080.yaml")
    estimate, _ = preflight.estimate_vram_gb(config, 8.0)
    assert 6.0 < estimate < 10.0


def test_four_bit_weights_cheaper_than_bf16():
    config = load_sft_config("configs/sft_rtx5080.yaml")
    _, quantised = preflight.estimate_vram_gb(config, 8.0)

    config.load_in_4bit = False
    _, full = preflight.estimate_vram_gb(config, 8.0)

    assert quantised["веса"] < full["веса"]


def test_longer_sequences_cost_more_memory():
    config = load_sft_config("configs/sft_rtx5080.yaml")
    short, _ = preflight.estimate_vram_gb(config, 8.0)

    config.max_length = 4096
    long, _ = preflight.estimate_vram_gb(config, 8.0)

    assert long > short


def test_gradient_checkpointing_reduces_activations():
    config = load_sft_config("configs/sft_rtx5080.yaml")
    _, with_checkpointing = preflight.estimate_vram_gb(config, 8.0)

    config.gradient_checkpointing = False
    _, without = preflight.estimate_vram_gb(config, 8.0)

    assert with_checkpointing["активации"] < without["активации"]


def test_rtx5080_configs_are_consistent():
    sft = load_sft_config("configs/sft_rtx5080.yaml")
    dpo = load_dpo_config("configs/dpo_rtx5080.yaml")

    # На 16 ГБ 4 бита обязательны для обеих стадий.
    assert sft.load_in_4bit
    assert dpo.load_in_4bit
    assert sft.engine == dpo.engine
    assert dpo.adapter_path == sft.output_dir
    # DPO держит и политику, и опорную модель — промпт должен быть короче.
    assert dpo.max_prompt_length < dpo.max_length


def test_rtx5080_effective_batch_matches_base_profile():
    """Уменьшаем пик памяти, но не эффективный размер батча."""
    base = load_sft_config("configs/sft.yaml")
    tuned = load_sft_config("configs/sft_rtx5080.yaml")

    base_effective = base.per_device_batch_size * base.gradient_accumulation_steps
    tuned_effective = tuned.per_device_batch_size * tuned.gradient_accumulation_steps

    assert tuned_effective == base_effective
    assert tuned.per_device_batch_size < base.per_device_batch_size
