#!/usr/bin/env python
"""Проверка, что torch собран под нужную архитектуру и видит карту.

Вызывается после КАЖДОГО шага установки, который может тронуть torch.
Причина: unsloth ограничивает версию torch сверху, и при понижении версии
пакетный менеджер уходит в дефолтный индекс и ставит CPU-сборку. Внешне всё
успешно, а обучение потом идёт на процессоре сутками вместо часов.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch не установлен", file=sys.stderr)
        return 1

    build = torch.version.cuda
    print(f"torch      : {torch.__version__}")
    print(f"CUDA build : {build}")

    if build is None or "+cpu" in torch.__version__:
        print(
            "\nЭто CPU-сборка torch. Обучение на ней технически запустится,\n"
            "но будет идти в десятки раз медленнее.",
            file=sys.stderr,
        )
        return 1

    if not torch.cuda.is_available():
        print("\ntorch не видит CUDA: сборка не совпадает с драйвером.", file=sys.stderr)
        return 1

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()

    print(f"GPU        : {name} ({arch})")
    print(f"Архитектуры: {arch_list}")

    if arch not in arch_list:
        print(
            f"\nСборка torch не содержит {arch}. Ядра для этой карты отсутствуют.",
            file=sys.stderr,
        )
        return 1

    print("torch собран под эту карту")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
