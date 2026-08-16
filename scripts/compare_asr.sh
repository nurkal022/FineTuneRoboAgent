#!/usr/bin/env bash
# Сравнение готовых казахских моделей с базовым turbo на одной выборке.
#
# Смысл: прежде чем тратить часы GPU на дообучение, надо узнать, чего
# достигают уже опубликованные модели. Они обучены на сотнях часов (KSC2 и
# другие корпуса), тогда как доступный нам FLEURS kk_kz — около десяти.
#
# Все замеры одним кодом и на одной выборке: цифры из разных скриптов с
# разной нормализацией несравнимы.

cd "$(dirname "$0")/.." || exit 1
PY="${PY:-.venv/bin/python}"
LIMIT="${LIMIT:-100}"
OUT="docs/experiments/asr"
mkdir -p "$OUT" logs

MODELS=(
    "openai/whisper-large-v3-turbo|base-turbo"
    "abilmansplus/whisper-turbo-ksc2|ksc2"
    "shyngys879/kazakh-whisper-large-v3-turbo|multi-corpus"
)

for entry in "${MODELS[@]}"; do
    model="${entry%%|*}"
    label="${entry##*|}"
    printf '\n\033[1m==> %s (%s)\033[0m  [%s]\n' "$label" "$model" "$(date '+%H:%M:%S')"
    $PY scripts/asr.py --no-streaming eval \
        --model "$model" --split validation --limit "$LIMIT" --show 3 \
        --json "$OUT/asr_${label}.json" > "logs/asr_${label}.log" 2>&1 \
        && echo "  готово" \
        || { echo "  ОТКАЗ:"; tail -5 "logs/asr_${label}.log" | sed 's/^/    /'; }
done

printf '\n\033[1mСводка\033[0m\n'
$PY - <<'PYEOF'
import json, pathlib
rows = []
for path in sorted(pathlib.Path("docs/experiments/asr").glob("asr_*.json")):
    d = json.loads(path.read_text(encoding="utf-8"))
    rows.append((path.stem.replace("asr_", ""), d["wer"], d["cer"], d["examples"]))
if rows:
    print(f"{'модель':16} {'WER':>8} {'CER':>8} {'примеров':>10}")
    for name, wer, cer, n in sorted(rows, key=lambda r: r[1]):
        print(f"{name:16} {wer:8.4f} {cer:8.4f} {n:10}")
PYEOF
