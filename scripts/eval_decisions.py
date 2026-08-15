#!/usr/bin/env python
"""Проверка обученного адаптера на валидационных ситуациях.

Главное, что проверяет скрипт, — не выродилось ли обучение. Модель, которая
на любую ситуацию отвечает WAIT, покажет приличную формальную точность на
несбалансированной выборке и будет полностью бесполезна, поэтому распределение
решений печатается всегда и рядом с метриками.

Валидность JSON здесь проверяется БЕЗ грамматики декодера: в бою её обеспечивает
guided decoding, а нам нужно увидеть, что выучила сама модель. Поэтому доля
разобравшихся ответов — это диагностика обучения, а не характеристика системы.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robo_agency.env import ensure_writable_hf_cache  # noqa: E402

ensure_writable_hf_cache()

from robo_agency.evaluation.agency import Prediction, compute  # noqa: E402
from robo_agency.prompts import SYSTEM_PROMPT  # noqa: E402
from robo_agency.schema import DecisionType, RobotDecision  # noqa: E402

BOLD, GREEN, YELLOW, RED, RESET = "\033[1m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def load_decision_rows(path: Path, limit: int) -> list[dict]:
    """Берёт из вала только проактивные примеры.

    В вал попадают ещё function-calling и replay: они в родном диалоговом
    формате и к схеме решений отношения не имеют.
    """
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            if len(messages) < 3 or messages[0].get("content") != SYSTEM_PROMPT:
                continue
            try:
                gold = RobotDecision.model_validate_json(messages[-1]["content"])
            except Exception:  # noqa: BLE001 — не проактивный пример, просто пропускаем
                continue
            rows.append({"situation": messages[1]["content"], "gold": gold})
            if len(rows) >= limit:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Прогнать адаптер по валидации")
    parser.add_argument("--adapter", default="outputs/adapter-decisions")
    parser.add_argument("--val-file", default="data/processed/val.jsonl")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--show", type=int, default=5, help="Сколько ответов напечатать целиком")
    parser.add_argument(
        "--base",
        action="store_true",
        help="Отключить адаптер: та же база без обучения, для сравнения",
    )
    parser.add_argument("--json", dest="json_out", help="Куда записать метрики машиночитаемо")
    parser.add_argument("--label", default="", help="Метка прогона для сводной таблицы")
    args = parser.parse_args()

    rows = load_decision_rows(Path(args.val_file), args.limit)
    if not rows:
        print(f"{RED}В {args.val_file} нет проактивных примеров{RESET}")
        return 1
    print(f"Проактивных примеров в выборке: {len(rows)}")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Речевой контур — это та же база с ВЫКЛЮЧЕННЫМ адаптером. Сравнение с ним
    # показывает, что изменилось именно обучение, а не промпт.
    if args.base:
        model.disable_adapter_layers()
        print(f"{YELLOW}Адаптер выключен: отвечает голая база{RESET}")

    predictions: list[Prediction] = []
    predicted_counts: Counter[str] = Counter()
    gold_counts: Counter[str] = Counter()
    invalid = 0
    shown = 0

    for index, row in enumerate(rows):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["situation"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # Qwen3 по умолчанию открывает блок рассуждений, а обучали мы
            # отвечать сразу JSON. Рассогласование промпта между обучением и
            # инференсом — самая частая причина «обучилось, но не работает».
            enable_thinking=False,
        )
        if index == 0:
            print(f"\n{BOLD}Хвост промпта:{RESET} {prompt[-80:]!r}\n")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,  # жадный декод: нужна воспроизводимость, а не разнообразие
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        gold: RobotDecision = row["gold"]
        gold_counts[gold.decision.value] += 1

        try:
            parsed = RobotDecision.model_validate_json(answer)
            predicted = parsed.decision
            predicted_counts[predicted.value] += 1
        except Exception:  # noqa: BLE001 — невалидный ответ это результат замера
            invalid += 1
            predicted_counts["INVALID"] += 1
            predicted = None

        if predicted is not None:
            predictions.append(Prediction(gold=gold.decision, predicted=predicted))

        if shown < args.show:
            shown += 1
            print(f"{BOLD}--- Пример {index + 1} ---{RESET}")
            print(f"  ситуация: {row['situation'][:300]}")
            print(f"  эталон:   {gold.decision.value}")
            mark = GREEN if predicted is gold.decision else RED
            print(f"  ответ:    {mark}{answer[:300]}{RESET}")

    print(f"\n{BOLD}Распределение решений{RESET}")
    print(f"  эталон:  {dict(gold_counts)}")
    print(f"  модель:  {dict(predicted_counts)}")
    print(f"  невалидный JSON: {invalid} из {len(rows)}")

    # Вырожденность важнее любых метрик: одинаковый ответ на всё означает, что
    # обучение не удалось, какой бы ни была формальная точность.
    distinct = {name for name in predicted_counts if name != "INVALID"}
    if len(distinct) <= 1:
        print(f"\n{RED}{BOLD}ВЫРОЖДЕНО: модель отвечает одинаково на все ситуации{RESET}")
    else:
        print(f"\n{GREEN}Не вырождено: модель выбирает между {sorted(distinct)}{RESET}")

    metrics = compute(predictions) if predictions else None
    if metrics is not None:
        print(f"\n{BOLD}Метрики агентности{RESET}")
        print(metrics.describe())

    if args.json_out:
        payload = {
            "label": args.label or args.adapter,
            "adapter": args.adapter,
            "base_only": args.base,
            "examples": len(rows),
            "gold": dict(gold_counts),
            "predicted": dict(predicted_counts),
            "invalid_json": invalid,
            # Вырожденность выносится отдельным полем: по ней отбирается
            # чекпойнт, и она важнее любой из метрик ниже.
            "degenerate": len(distinct) <= 1,
            "metrics": {
                "timeliness": metrics.timeliness,
                "false_intervention_rate": metrics.false_intervention_rate,
                "precision_act": metrics.precision_act,
                "f1_act": metrics.f1_act,
                "decision_accuracy": metrics.decision_accuracy,
            } if metrics is not None else None,
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nМетрики записаны в {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
