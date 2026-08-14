#!/usr/bin/env bash
# Установка окружения на Ubuntu с картой RTX 50xx (Blackwell, sm_120).
#
# Blackwell требует, чтобы ВЕСЬ стек был собран под CUDA 12.8. Обычный
# `pip install torch` ставит сборку под старую CUDA, которая на sm_120 либо
# падает, либо молча работает на CPU. Поэтому torch ставится отдельно и с
# явным индексом, а unsloth — через uv, который сам подберёт совместимые версии.

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-.venv}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[33mВНИМАНИЕ: %s\033[0m\n' "$1"; }
die() { printf '\033[31mОШИБКА: %s\033[0m\n' "$1" >&2; exit 1; }

say "Проверка драйвера NVIDIA"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi не найден. Установите драйвер NVIDIA (>= 570 для Blackwell)."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
if [ "${DRIVER_MAJOR:-0}" -lt 570 ]; then
    warn "Драйвер ${DRIVER_MAJOR}.x старее 570 — для Blackwell нужен 570+. Обновите драйвер, иначе CUDA 12.8 не заработает."
fi

say "Установка uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

say "Создание окружения (Python ${PYTHON_VERSION})"
uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

say "Установка PyTorch под CUDA 12.8"
# --torch-backend=auto просит uv определить CUDA и взять правильную сборку.
uv pip install torch torchvision --torch-backend=auto

say "Проверка, что torch видит карту"
python - <<'PY'
import sys
import torch

print(f"torch      : {torch.__version__}")
print(f"CUDA build : {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("torch не видит CUDA — сборка не совпала с драйвером")

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print(f"GPU        : {name} (sm_{major}{minor})")

supported = torch.cuda.get_arch_list()
print(f"Архитектуры: {supported}")
if f"sm_{major}{minor}" not in supported:
    sys.exit(
        f"Сборка torch не содержит sm_{major}{minor}. Нужна сборка под CUDA 12.8:\n"
        "  uv pip install torch --torch-backend=auto\n"
        "или nightly: uv pip install --pre torch --index-url "
        "https://download.pytorch.org/whl/nightly/cu128"
    )
print("torch и карта совместимы")
PY

say "Установка Unsloth"
uv pip install unsloth unsloth_zoo --torch-backend=auto

say "Установка проекта"
uv pip install -e ".[train,dev]"

say "Прогон тестов"
python -m pytest tests -q

say "Preflight по конфигу RTX 5080"
python scripts/preflight.py --config configs/sft_rtx5080.yaml || true

cat <<'DONE'

Готово. Дальше:

  source .venv/bin/activate
  make data        # собрать корпус из готовых датасетов
  make sft         # обучить адаптер решений
DONE
