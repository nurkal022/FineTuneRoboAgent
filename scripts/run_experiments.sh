#!/usr/bin/env bash
# Ночная серия экспериментов. Запускается без присмотра, поэтому падение
# одного шага не должно уносить остальные: set -e здесь намеренно НЕ включён,
# каждый шаг оборачивается в run_step и его отказ только помечается в сводке.
#
# Порядок шагов — по убыванию научной ценности: если ночи не хватит, недостающим
# окажется наименее важное.
#
#   A. 4 эпохи, чекпойнт каждые 50 шагов -> кривая выхода из вырожденности
#   B. без выравнивания классов          -> проверка инварианта 4
#   C. без replay                        -> нужен ли replay на самом деле
#
# Замеры после каждого обучения: агентность на валидации, NLL на отложенных
# выборках общего диалога и вызова инструментов.

cd "$(dirname "$0")/.." || exit 1

PY="${PY:-.venv/bin/python}"
RESULTS="docs/experiments/nightly"
LOGS="logs/nightly"
mkdir -p "$RESULTS" "$LOGS"

SUMMARY="$RESULTS/progress.md"
STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"

export PYTHONUNBUFFERED=1

say() { printf '\n\033[1m==> %s\033[0m  [%s]\n' "$1" "$(date '+%H:%M:%S')"; }

free_gb() { df -BG --output=avail ~ | tail -1 | tr -dc '0-9'; }

# Чекпойнт весит ~514 МБ, а свободного места на этой машине около 11 ГБ.
# После обхода метрики уже сохранены в JSON, сами веса больше не нужны:
# финальный адаптер остаётся, промежуточные удаляются.
drop_checkpoints() {
    local run_dir="$1"
    local before after
    before=$(free_gb)
    rm -rf "${run_dir}"/checkpoint-*
    after=$(free_gb)
    printf '  чекпойнты %s удалены, свободно было %s ГБ, стало %s ГБ\n' \
        "$run_dir" "$before" "$after"
    note "| очистка $run_dir | ${before}->${after} ГБ | | |"
}

note() { printf '%s\n' "$1" >> "$SUMMARY"; }

run_step() {
    local name="$1"; shift
    local log="$LOGS/${name}.log"
    local began ended elapsed

    say "$name"
    began=$(date +%s)
    if "$@" > "$log" 2>&1; then
        ended=$(date +%s); elapsed=$(( (ended - began) / 60 ))
        printf '\033[32m  готово за %d мин\033[0m -> %s\n' "$elapsed" "$log"
        note "| $name | готово | ${elapsed} мин | \`$log\` |"
        return 0
    fi
    ended=$(date +%s); elapsed=$(( (ended - began) / 60 ))
    printf '\033[31m  ОТКАЗ через %d мин\033[0m -> %s\n' "$elapsed" "$log"
    printf '  последние строки:\n'; tail -5 "$log" | sed 's/^/    /'
    note "| $name | ОТКАЗ | ${elapsed} мин | \`$log\` |"
    return 1
}

cat > "$SUMMARY" <<EOF
# Ночная серия

Начата: $STARTED_AT
Свободно на диске при старте: $(free_gb) ГБ

| шаг | статус | время | лог |
|---|---|---|---|
EOF

# --- Подготовка ------------------------------------------------------------

run_step "00-doctor" $PY scripts/verify_torch.py
run_step "01-tests" $PY -m pytest tests -q
run_step "02-fetch" $PY scripts/fetch_data.py
run_step "03-data-main" $PY -m robo_agency.cli build-data \
    --config configs/data_mix.yaml --output data/processed
run_step "04-eval-sets" $PY scripts/build_eval_sets.py

# Базовые замеры: без них не с чем сравнивать адаптеры.
run_step "05-retention-base" $PY -m robo_agency.cli retention \
    --base unsloth/Qwen3-8B-unsloth-bnb-4bit \
    --adapter outputs/adapter-decisions \
    --eval-file data/processed/retention_eval.jsonl --limit 200
run_step "06-tools-base" $PY -m robo_agency.cli retention \
    --base unsloth/Qwen3-8B-unsloth-bnb-4bit \
    --adapter outputs/adapter-decisions \
    --eval-file data/processed/tools_eval.jsonl --limit 200

# --- A: кривая по чекпойнтам ----------------------------------------------

run_step "10-train-A-4epochs" $PY -m robo_agency.cli train-sft \
    --config configs/exp_a_4epochs.yaml
run_step "11-sweep-A" $PY scripts/sweep_checkpoints.py \
    --run-dir outputs/exp-a-4epochs --out-dir "$RESULTS/sweeps"
drop_checkpoints outputs/exp-a-4epochs

run_step "12-retention-A" $PY -m robo_agency.cli retention \
    --base unsloth/Qwen3-8B-unsloth-bnb-4bit \
    --adapter outputs/exp-a-4epochs \
    --eval-file data/processed/retention_eval.jsonl --limit 200
run_step "13-tools-A" $PY -m robo_agency.cli retention \
    --base unsloth/Qwen3-8B-unsloth-bnb-4bit \
    --adapter outputs/exp-a-4epochs \
    --eval-file data/processed/tools_eval.jsonl --limit 200

# --- B: без выравнивания классов ------------------------------------------

run_step "20-data-B" $PY -m robo_agency.cli build-data \
    --config configs/data_mix_nobalance.yaml --output data/processed_nobalance
run_step "21-train-B" $PY -m robo_agency.cli train-sft \
    --config configs/exp_b_nobalance.yaml
# Меряем на ОСНОВНОЙ валидации: сравнивать надо на одной выборке.
run_step "22-eval-B" $PY scripts/eval_decisions.py \
    --adapter outputs/exp-b-nobalance --val-file data/processed/val.jsonl \
    --limit 46 --show 0 --json "$RESULTS/agency_B_nobalance.json" --label B-nobalance

drop_checkpoints outputs/exp-b-nobalance

# --- C: без replay ---------------------------------------------------------

run_step "30-data-C" $PY -m robo_agency.cli build-data \
    --config configs/data_mix_noreplay.yaml --output data/processed_noreplay
run_step "31-train-C" $PY -m robo_agency.cli train-sft \
    --config configs/exp_c_noreplay.yaml
run_step "32-eval-C" $PY scripts/eval_decisions.py \
    --adapter outputs/exp-c-noreplay --val-file data/processed/val.jsonl \
    --limit 46 --show 0 --json "$RESULTS/agency_C_noreplay.json" --label C-noreplay
run_step "33-retention-C" $PY -m robo_agency.cli retention \
    --base unsloth/Qwen3-8B-unsloth-bnb-4bit \
    --adapter outputs/exp-c-noreplay \
    --eval-file data/processed/retention_eval.jsonl --limit 200
run_step "34-tools-C" $PY -m robo_agency.cli retention \
    --base unsloth/Qwen3-8B-unsloth-bnb-4bit \
    --adapter outputs/exp-c-noreplay \
    --eval-file data/processed/tools_eval.jsonl --limit 200

drop_checkpoints outputs/exp-c-noreplay

note ""
note "Завершена: $(date '+%Y-%m-%d %H:%M:%S'), свободно $(free_gb) ГБ"

say "Серия завершена"
cat "$SUMMARY"
