#!/usr/bin/env python
"""Проверка окружения и оценка памяти ДО запуска обучения.

Смысл: OOM на 8B модели случается не сразу, а через несколько минут разогрева,
и после каждой правки конфига цикл повторяется. Дешевле посчитать заранее.

Оценка грубая и намеренно консервативная — она не заменяет запуск, но ловит
заведомо невозможные конфигурации (например, 8B в bf16 на 16 ГБ).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

GREEN, YELLOW, RED, BOLD, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[0m"


def ok(message: str) -> None:
    print(f"{GREEN}  OK{RESET}  {message}")


def warn(message: str) -> None:
    print(f"{YELLOW}  ??{RESET}  {message}")


def fail(message: str) -> None:
    print(f"{RED}  !!{RESET}  {message}")


def section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")


def params_billions(model_name: str) -> float:
    """Достаёт размер модели из имени: Qwen3-8B -> 8.0."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb](?:\b|-)", model_name)
    return float(match.group(1)) if match else 8.0


def estimate_vram_gb(config, params_b: float) -> tuple[float, dict[str, float]]:
    """Грубая оценка пикового потребления VRAM при LoRA-обучении.

    Коэффициенты подобраны по фактическим замерам QLoRA на моделях 7-8B:
    при батче 1 и длине 2048 такая конфигурация укладывается примерно в 7-8 ГБ.
    Оценка консервативная, но не паническая — цель отсечь заведомо невозможное,
    а не запретить всё подряд.
    """
    # Веса: 4 бита ~0.6 ГБ на миллиард с учётом накладных квантизации, bf16 — 2 ГБ.
    weights = params_b * (0.6 if config.load_in_4bit else 2.0)

    # Сам адаптер мал, но состояния AdamW к нему втрое-вчетверо тяжелее.
    adapter = params_b * 0.08

    # Активации: растут линейно по числу токенов в микробатче. Чекпойнтинг
    # снижает их примерно вчетверо ценой повторного прямого прохода.
    tokens = config.per_device_batch_size * config.max_length
    per_1k_tokens = 0.25 if config.gradient_checkpointing else 1.0
    activations = (tokens / 1000) * (params_b / 8) * per_1k_tokens

    overhead = 1.5  # аллокатор CUDA, фрагментация, буферы, ядра

    breakdown = {
        "веса": weights,
        "адаптер и оптимизатор": adapter,
        "активации": activations,
        "накладные": overhead,
    }
    return sum(breakdown.values()), breakdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight перед обучением")
    parser.add_argument("--config", default="configs/sft_rtx5080.yaml")
    args = parser.parse_args()

    problems = 0

    section("Окружение")
    try:
        import torch
    except ImportError:
        fail("torch не установлен — запустите scripts/setup_server.sh")
        return 1

    ok(f"torch {torch.__version__}, собран под CUDA {torch.version.cuda}")

    if not torch.cuda.is_available():
        fail("torch не видит CUDA. Сборка torch не совпадает с драйвером.")
        return 1

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    ok(f"{name}, sm_{major}{minor}, {total_gb:.1f} ГБ")

    arch_list = torch.cuda.get_arch_list()
    if f"sm_{major}{minor}" in arch_list:
        ok(f"сборка torch содержит sm_{major}{minor}")
    else:
        fail(
            f"сборка torch НЕ содержит sm_{major}{minor}. Доступно: {arch_list}. "
            "Нужна сборка под CUDA 12.8."
        )
        problems += 1

    if torch.cuda.is_bf16_supported():
        ok("bf16 поддерживается")
    else:
        warn("bf16 не поддерживается — переключите bf16: false в конфиге")

    section("Бэкенд обучения")
    try:
        import unsloth  # noqa: F401

        ok("unsloth установлен")
        unsloth_available = True
    except Exception as error:  # noqa: BLE001
        warn(f"unsloth недоступен ({type(error).__name__}) — используйте engine: hf")
        unsloth_available = False

    section(f"Конфиг: {args.config}")
    from robo_agency.training.sft import load_config

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        fail(f"файл {args.config} не найден")
        return 1

    ok(f"модель {config.base_model}, engine={config.engine}")
    if config.engine == "unsloth" and not unsloth_available:
        fail("в конфиге engine: unsloth, но пакет не установлен")
        problems += 1

    params_b = params_billions(config.base_model)
    estimate, breakdown = estimate_vram_gb(config, params_b)

    section("Оценка VRAM")
    for label, value in breakdown.items():
        print(f"      {label:<24} {value:6.2f} ГБ")
    print(f"      {'ИТОГО (оценка)':<24} {estimate:6.2f} ГБ из {total_gb:.1f} ГБ")

    effective_batch = config.per_device_batch_size * config.gradient_accumulation_steps
    print(
        f"\n      батч {config.per_device_batch_size} x накопление "
        f"{config.gradient_accumulation_steps} = эффективный {effective_batch}, "
        f"длина {config.max_length}"
    )

    if estimate > total_gb:
        fail(
            f"оценка превышает объём карты. Варианты: load_in_4bit: true, "
            f"уменьшить max_length или per_device_batch_size, взять модель меньше."
        )
        problems += 1
    elif estimate > total_gb * 0.9:
        warn("оценка близка к пределу — вероятен OOM на длинных примерах")
    else:
        ok("оценка укладывается в память карты")

    section("Данные")
    for path_str in (config.train_file, config.val_file):
        path = Path(path_str)
        if path.exists():
            lines = sum(1 for line in path.open(encoding="utf-8") if line.strip())
            ok(f"{path} — {lines} примеров")
        else:
            warn(f"{path} отсутствует — запустите: make data")

    print()
    if problems:
        print(f"{RED}{BOLD}Проблем: {problems}. Запускать обучение рано.{RESET}")
        return 1
    print(f"{GREEN}{BOLD}Всё готово к обучению.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
