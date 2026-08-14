#!/usr/bin/env bash
# Установка окружения на Ubuntu с картой RTX 50xx (Blackwell, sm_120).
#
# Главная ловушка этой установки: unsloth ограничивает версию torch сверху.
# Если поставить сначала torch под CUDA, а потом unsloth отдельной командой,
# менеджер пакетов ПОНИЗИТ версию torch и при этом уйдёт в дефолтный индекс,
# то есть заменит рабочую CUDA-сборку на CPU-сборку. Внешне установка успешна,
# а обучение потом идёт на процессоре.
#
# Поэтому здесь два правила:
#   1. torch указывается явно в той же команде, что и unsloth — одна резолюция.
#   2. после каждого шага, способного тронуть torch, идёт проверка.

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-.venv}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[33mВНИМАНИЕ: %s\033[0m\n' "$1"; }
die() { printf '\033[31mОШИБКА: %s\033[0m\n' "$1" >&2; exit 1; }

verify_torch() {
    local stage="$1"
    if python scripts/verify_torch.py; then
        return 0
    fi
    warn "после шага «${stage}» torch оказался нерабочим — восстанавливаю"
    return 1
}

repair_torch() {
    # unsloth диктует верхнюю границу версии torch. Берём ту версию,
    # которую он выбрал, и переставляем её же, но из CUDA-индекса.
    local pinned
    pinned="$(python -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || echo "")"
    if [ -n "$pinned" ]; then
        say "Переустановка torch==${pinned} из CUDA-индекса"
        uv pip install --reinstall "torch==${pinned}" torchvision --torch-backend=auto
    else
        say "Переустановка torch из CUDA-индекса"
        uv pip install --reinstall torch torchvision --torch-backend=auto
    fi
}

say "Проверка драйвера NVIDIA"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi не найден. Установите драйвер NVIDIA (>= 570 для Blackwell)."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
if [ "${DRIVER_MAJOR:-0}" -lt 570 ]; then
    warn "Драйвер ${DRIVER_MAJOR}.x старее 570 — для Blackwell нужен 570+."
fi

say "Установка uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

say "Создание окружения (Python ${PYTHON_VERSION})"
uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

say "Установка Unsloth вместе с torch (одна резолюция)"
# torch и torchvision указаны ЯВНО: без этого uv понизит torch до CPU-сборки.
uv pip install unsloth unsloth_zoo torch torchvision --torch-backend=auto

say "Проверка torch после установки Unsloth"
if ! verify_torch "установка Unsloth"; then
    repair_torch
    python scripts/verify_torch.py || die "не удалось получить CUDA-сборку torch, совместимую с unsloth.
Обходной путь: переключите engine: hf в configs/*.yaml и поставьте стек без unsloth:
  uv pip install torch torchvision --torch-backend=auto
  uv pip install -e '.[train]'"
fi

say "Установка проекта"
uv pip install -e ".[train,dev]"

say "Проверка torch после установки проекта"
if ! verify_torch "установка проекта"; then
    repair_torch
    python scripts/verify_torch.py || die "torch снова сломан после установки проекта"
fi

say "Прогон тестов"
python -m pytest tests -q

say "Preflight по конфигу RTX 5080"
python scripts/preflight.py --config configs/sft_rtx5080.yaml || true

cat <<'DONE'

Готово. Дальше:

  source .venv/bin/activate
  make fetch       # скачать ProactiveBench
  make data        # собрать корпус
  make sft         # обучить адаптер
DONE
