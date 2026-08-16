#!/usr/bin/env python
"""Речевой контур: осмотр данных, замер и дообучение Whisper.

Три команды в порядке применения:

    peek      посмотреть на корпус, не скачивая его целиком
    baseline  замерить модель БЕЗ дообучения — точка отсчёта
    train     дообучить
    eval      замерить, что получилось

Базовый замер обязателен до обучения: без него утверждение «мы улучшили
распознавание» ничем не подкреплено.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robo_agency.env import ensure_writable_hf_cache  # noqa: E402

ensure_writable_hf_cache()

from robo_agency.asr import data as asr_data  # noqa: E402
from robo_agency.asr.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BOLD, GREEN, YELLOW, RESET = "\033[1m", "\033[32m", "\033[33m", "\033[0m"


def cmd_peek(args) -> int:
    config = _apply_overrides(load_config(args.config), args)
    for split in (config.dataset.train_split, config.dataset.eval_split):
        stats = asr_data.peek(config.dataset, split, args.limit)
        print(f"\n{BOLD}{split}{RESET}")
        print(f"  колонки: {stats.columns}")
        print(f"  расшифровка в поле: {stats.text_column}")
        print(f"  пустых расшифровок: {stats.empty_text} из {stats.examples}")

    print(f"\n{BOLD}Модель{RESET}: {config.model}")
    if config.params_millions:
        trainable = config.trainable_millions
        frozen = " (энкодер заморожен)" if config.freeze_encoder else ""
        print(f"  параметров: {config.params_millions} млн, обучается {trainable} млн{frozen}")
    print(f"  оптимизатор: {config.optim}")
    print(f"  эффективный батч: {config.effective_batch}, шагов: {config.max_steps}")

    estimate = config.estimate_vram_gb()
    if estimate is not None:
        print(f"\n{BOLD}Оценка VRAM{RESET}: {estimate} ГБ")
        try:
            import torch

            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                verdict = GREEN + "укладывается" if estimate < total * 0.9 else YELLOW + "впритык"
                print(f"  доступно: {total:.1f} ГБ — {verdict}{RESET}")
        except ImportError:
            pass
    return 0


def _report(args, model_path: str, title: str) -> int:
    from robo_agency.asr.evaluate import transcribe

    config = _apply_overrides(load_config(args.config), args)
    report = transcribe(
        model_path, config, split=args.split, limit=args.limit, show=args.show
    )

    print(f"\n{BOLD}{title}{RESET}")
    print(report.describe())

    if args.json_out:
        payload = {
            "title": title,
            "model": report.model,
            "examples": report.examples,
            "wer": report.wer.rate,
            "cer": report.cer.rate,
            "wer_detail": {
                "substitutions": report.wer.substitutions,
                "deletions": report.wer.deletions,
                "insertions": report.wer.insertions,
                "reference_length": report.wer.reference_length,
            },
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nЗаписано: {out}")
    return 0


def cmd_baseline(args) -> int:
    config = _apply_overrides(load_config(args.config), args)
    print(f"{YELLOW}Замер БЕЗ дообучения: {config.model}{RESET}")
    return _report(args, config.model, "База без дообучения")


def cmd_eval(args) -> int:
    config = _apply_overrides(load_config(args.config), args)
    model_path = args.model or config.output_dir
    # Путь без слеша или существующий каталог — локальная модель; иначе это
    # идентификатор на хабе, и проверять его наличие на диске бессмысленно.
    looks_like_repo_id = "/" in model_path and not Path(model_path).is_absolute()
    if not Path(model_path).exists() and not looks_like_repo_id:
        print(f"Модель {model_path} не найдена — сначала обучите: make asr-train")
        return 1
    return _report(args, model_path, "После дообучения")


def cmd_train(args) -> int:
    from robo_agency.asr.train import train

    config = _apply_overrides(load_config(args.config), args)
    print(f"{BOLD}Дообучение {config.model}{RESET}, шагов: {config.max_steps}")
    path = train(config)
    print(f"{GREEN}Готово: {path}{RESET}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Whisper для казахского")
    parser.add_argument("--config", default="configs/whisper_kk.yaml")
    # Поток экономит диск, но упирается в сеть: на медленном канале чтение
    # parquet с встроенным аудио срывается по таймауту. Скачанный один раз
    # сплит читается локально и надёжнее — если он помещается на диск.
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Скачать сплит целиком вместо потокового чтения",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    peek = subparsers.add_parser("peek", help="Осмотреть корпус")
    peek.add_argument("--limit", type=int, default=30)
    peek.set_defaults(func=cmd_peek)

    for name, handler, help_text in (
        ("baseline", cmd_baseline, "Замерить модель без дообучения"),
        ("eval", cmd_eval, "Замерить дообученную модель"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--split", default=None)
        sub.add_argument("--limit", type=int, default=None)
        sub.add_argument("--show", type=int, default=5)
        sub.add_argument("--json", dest="json_out", default=None)
        if name == "eval":
            sub.add_argument("--model", default=None)
        sub.set_defaults(func=handler)

    train_parser = subparsers.add_parser("train", help="Дообучить")
    train_parser.set_defaults(func=cmd_train)

    args = parser.parse_args()
    return args.func(args)


def _apply_overrides(config, args):
    if getattr(args, "no_streaming", False):
        config.dataset.streaming = False
        logger.info("Потоковое чтение отключено: сплит будет скачан целиком")
    return config


if __name__ == "__main__":
    raise SystemExit(main())
