"""Подготовка окружения Hugging Face до импорта его библиотек.

Кеш HF часто выносят на внешний диск, а тот бывает смонтирован только на
чтение. `datasets` и `huggingface_hub` читают путь из переменных окружения и
падают с PermissionError уже в середине загрузки. Unsloth умеет подменять
такой путь себе, но только себе — на `datasets` это не распространяется.

Порядок вызова важен: huggingface_hub вычисляет пути к кешу в момент импорта,
поэтому переменные надо править ДО первого импорта его или datasets.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Qwen3-8B в bf16 весит около 16 ГБ, плюс датасеты и промежуточные файлы.
RECOMMENDED_FREE_GB = 30.0


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path):
            return True
    except (OSError, PermissionError):
        return False


def _candidates() -> list[Path]:
    """Пути кеша по убыванию приоритета: сначала то, что задал пользователь."""
    found: list[Path] = []
    if cache := os.environ.get("HF_HUB_CACHE"):
        found.append(Path(cache))
    if home := os.environ.get("HF_HOME"):
        found.append(Path(home) / "hub")
    found.append(Path.home() / ".cache" / "huggingface" / "hub")
    return found


def free_space_gb(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1024**3


def ensure_writable_hf_cache() -> Path:
    """Выбирает доступный на запись кеш HF и прописывает его в окружение."""
    candidates = _candidates()

    chosen: Path | None = None
    for candidate in candidates:
        if _is_writable(candidate):
            chosen = candidate
            break
        logger.warning("Кеш Hugging Face недоступен на запись: %s", candidate)

    if chosen is None:
        raise RuntimeError(
            "Ни один путь кеша Hugging Face не доступен на запись:\n"
            + "\n".join(f"  {path}" for path in candidates)
            + "\nЗадайте писуемый путь: export HF_HOME=/путь/с/правами"
        )

    if chosen != candidates[0]:
        logger.warning(
            "Кеш Hugging Face перенаправлен в %s.\n"
            "  Если это не то, что нужно, смонтируйте нужный диск на запись "
            "или задайте HF_HOME явно.",
            chosen,
        )

    os.environ["HF_HUB_CACHE"] = str(chosen)
    os.environ.setdefault("HF_HOME", str(chosen.parent))

    free = free_space_gb(chosen)
    if free < RECOMMENDED_FREE_GB:
        logger.warning(
            "На разделе с кешем свободно %.1f ГБ, рекомендуется от %.0f ГБ: "
            "одна только базовая модель весит около 16 ГБ.",
            free, RECOMMENDED_FREE_GB,
        )
    else:
        logger.info("Кеш Hugging Face: %s (свободно %.1f ГБ)", chosen, free)

    return chosen
