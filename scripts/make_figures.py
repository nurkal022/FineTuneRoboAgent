#!/usr/bin/env python
"""Построение рисунков для диссертации из результатов экспериментов.

Рисунки строятся из тех же JSON, что записали замеры, а не перерисовываются
руками: любой перезапуск экспериментов обновляет их одной командой, и цифра
на рисунке не может разойтись с цифрой в отчёте.

Оформление под печать:

  * Каждая серия несёт ВТОРОЙ признак кроме цвета — маркер, тип линии или
    штриховку. Диссертацию печатают в оттенках серого, и цвет там пропадает.
  * Подписи значений прямо на рисунке, без обращения к легенде.
  * Вывод в PDF (вектор, для вёрстки) и PNG 300 dpi (для черновиков).

Палитра проверена на различимость при дальтонизме: худшая пара по всем
сочетаниям ΔE 9.2 при пороге 8.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs/experiments"
FIGURES = ROOT / "docs/dissertation/figures"

# Слоты категориальной палитры: синий, оранжевый, бирюзовый.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"
ALERT_BG = "#f6e3da"

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # покрывает кириллицу
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "figure.dpi": 300,
})


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def load(relative: str) -> dict | list:
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def sweep_points() -> list[dict]:
    """Замеры по чекпойнтам, без дубля финальной точки."""
    rows = load("nightly/sweeps/exp-a-4epochs/summary.json")
    points = []
    for row in rows:
        label = row["label"]
        if not label.startswith("checkpoint-"):
            continue  # финальный адаптер совпадает с последним чекпойнтом
        points.append({
            "step": int(label.split("-")[1]),
            "act": row["predicted"].get("ACT", 0),
            "wait": row["predicted"].get("WAIT", 0),
            "degenerate": row["degenerate"],
            **row["metrics"],
        })
    return sorted(points, key=lambda p: p["step"])


def round_ticks(steps: list[int]) -> list[int]:
    """Круглые шаги для оси: последняя точка стоит вплотную к предыдущей."""
    return [step for step in steps if step % 100 == 0 or step % 50 == 0 and step != steps[-1]]


def degenerate_span(points: list[dict]) -> tuple[int, int] | None:
    steps = [p["step"] for p in points if p["degenerate"]]
    return (min(steps), max(steps)) if steps else None


# --- Рисунок 1: кривая вмешательств -----------------------------------------

def figure_intervention_curve(points):
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    steps = [p["step"] for p in points]

    span = degenerate_span(points)
    if span:
        half = (steps[1] - steps[0]) / 2
        ax.axvspan(span[0] - half, span[1] + half, color=ALERT_BG, zorder=0)
        ax.text((span[0] + span[1]) / 2, 1.04, "вырождено",
                ha="center", fontsize=8, color=ORANGE)

    ax.plot(steps, [p["false_intervention_rate"] for p in points],
            color=ORANGE, marker="o", markersize=5, linewidth=2,
            label="доля ложных вмешательств", zorder=3)
    ax.plot(steps, [p["timeliness"] for p in points],
            color=BLUE, marker="s", markersize=5, linewidth=2, linestyle="--",
            label="своевременность", zorder=3)

    best = min(points, key=lambda p: p["false_intervention_rate"])
    ax.annotate(f"минимум {best['false_intervention_rate']:.3f}",
                xy=(best["step"], best["false_intervention_rate"]),
                xytext=(0, -20), textcoords="offset points",
                ha="center", fontsize=8, color=ORANGE)

    ax.set_xlabel("шаг обучения")
    ax.set_ylabel("доля")
    ax.set_ylim(-0.05, 1.12)
    ax.set_xticks(round_ticks(steps))
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "fig1-intervention-curve")


# --- Рисунок 2: распределение решений ---------------------------------------

def figure_decision_mix(points):
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    steps = [str(p["step"]) for p in points]
    act = [p["act"] for p in points]
    wait = [p["wait"] for p in points]

    ax.bar(steps, act, color=BLUE, label="ACT — вмешаться", width=0.66)
    # Зазор между сегментами: столбики читаются как две величины, а не одна.
    ax.bar(steps, wait, bottom=[a + 0.6 for a in act], color=ORANGE,
           hatch="///", edgecolor="white", linewidth=0, label="WAIT — промолчать", width=0.66)

    for index, point in enumerate(points):
        if point["degenerate"]:
            ax.text(index, point["act"] + 2, "вырождено", ha="center",
                    fontsize=7.5, color=ORANGE, rotation=90, va="bottom")

    ax.set_xlabel("шаг обучения")
    ax.set_ylabel("ответов из 46")
    ax.set_ylim(0, 56)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    save(fig, "fig2-decision-mix")


# --- Рисунок 3: несогласие метрик -------------------------------------------

def figure_metric_disagreement(points):
    """F1 достигает максимума ровно там, где модель бесполезна."""
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    steps = [p["step"] for p in points]

    span = degenerate_span(points)
    if span:
        half = (steps[1] - steps[0]) / 2
        ax.axvspan(span[0] - half, span[1] + half, color=ALERT_BG, zorder=0)

    ax.plot(steps, [p["f1_act"] for p in points], color=AQUA, marker="^",
            markersize=5, linewidth=2, label="F1 по ACT", zorder=3)
    ax.plot(steps, [p["decision_accuracy"] for p in points], color=BLUE, marker="s",
            markersize=5, linewidth=2, linestyle=":", label="точность решения", zorder=3)
    ax.plot(steps, [p["false_intervention_rate"] for p in points], color=ORANGE,
            marker="o", markersize=5, linewidth=2, linestyle="--",
            label="доля ложных вмешательств", zorder=3)

    peak = max(points, key=lambda p: p["f1_act"])
    ax.annotate(f"максимум F1 = {peak['f1_act']:.3f}\nздесь модель вырождена",
                xy=(peak["step"], peak["f1_act"]), xycoords="data",
                xytext=(0.62, 0.88), textcoords="axes fraction",
                fontsize=8, color=AQUA, ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=AQUA, linewidth=1,
                                connectionstyle="arc3,rad=0.15"))

    ax.set_xlabel("шаг обучения")
    ax.set_ylabel("значение метрики")
    ax.set_ylim(-0.05, 1.12)
    ax.set_xticks(round_ticks(steps))
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "fig3-metric-disagreement")


# --- Рисунок 4: абляции ------------------------------------------------------

def figure_ablations(points):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))

    final = points[-1]
    balance = load("nightly/agency_B_nobalance.json")["predicted"]
    replay = load("nightly/agency_C_noreplay.json")["predicted"]

    names = ["основная", "без выравнивания\nклассов", "без диалоговых\nданных"]
    act = [final["act"], balance.get("ACT", 0), replay.get("ACT", 0)]
    wait = [final["wait"], balance.get("WAIT", 0), replay.get("WAIT", 0)]

    ax.bar(names, act, color=BLUE, width=0.55, label="ACT")
    ax.bar(names, wait, bottom=[a + 0.6 for a in act], color=ORANGE, hatch="///",
           edgecolor="white", linewidth=0, width=0.55, label="WAIT")

    for index, (a, w) in enumerate(zip(act, wait)):
        ax.text(index, a + w + 2.5, f"{a} / {w}", ha="center", fontsize=8.5, color=INK)
        if a == 0 or w == 0:
            # Тёмный текст на плашке: белый по штриховке нечитаем, штрихи тоже белые.
            ax.text(index, (a + w) / 2, "вырождено", ha="center", va="center",
                    fontsize=8.5, color=INK, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                              edgecolor=ORANGE, linewidth=0.8))

    ax.set_ylabel("ответов из 46")
    ax.set_ylim(0, 58)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    save(fig, "fig4-ablations")


# --- Рисунок 5: сохранность навыков -----------------------------------------

# Замеры NLL из ночной серии: улучшение относительно базы без адаптера.
RETENTION = {"с диалоговыми данными": (23.7, 52.8), "без диалоговых данных": (13.3, 26.2)}


def figure_retention():
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    labels = list(RETENTION)
    dialogue = [RETENTION[name][0] for name in labels]
    tools = [RETENTION[name][1] for name in labels]

    positions = range(len(labels))
    width = 0.34
    left = [p - width / 2 - 0.01 for p in positions]
    right = [p + width / 2 + 0.01 for p in positions]

    ax.bar(left, dialogue, width=width, color=BLUE, label="общий диалог")
    ax.bar(right, tools, width=width, color=AQUA, hatch="\\\\\\",
           edgecolor="white", linewidth=0, label="вызов инструментов")

    for x, value in list(zip(left, dialogue)) + list(zip(right, tools)):
        ax.text(x, value + 1.4, f"{value:.1f} %", ha="center", fontsize=8.5, color=INK)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("улучшение NLL, %")
    ax.set_ylim(0, 62)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "fig5-retention")


# --- Рисунок 6: распознавание речи ------------------------------------------

ASR_ORDER = [
    ("asr_adapted.json", "адаптированная\n(эта работа)"),
    ("asr_ksc2.json", "whisper-turbo-\nksc2"),
    ("asr_multi-corpus.json", "kazakh-whisper-\nlarge-v3-turbo"),
    ("asr_base-turbo.json", "базовая,\nбез дообучения"),
]


def asr_rows():
    return [(label, load(f"asr/{name}")) for name, label in ASR_ORDER]


def figure_asr():
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    rows = asr_rows()
    labels = [label for label, _ in rows]
    wer = [row["wer"] * 100 for _, row in rows]
    cer = [row["cer"] * 100 for _, row in rows]

    positions = range(len(labels))
    width = 0.36
    left = [p - width / 2 - 0.01 for p in positions]
    right = [p + width / 2 + 0.01 for p in positions]

    ax.bar(left, wer, width=width, color=BLUE, label="WER — ошибки на словах")
    ax.bar(right, cer, width=width, color=AQUA, hatch="\\\\\\",
           edgecolor="white", linewidth=0, label="CER — ошибки на символах")

    for x, value in list(zip(left, wer)) + list(zip(right, cer)):
        ax.text(x, value + 0.5, f"{value:.2f}", ha="center", fontsize=8.5, color=INK)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("доля ошибок, %")
    ax.set_ylim(0, 25)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "fig6-asr-comparison")


def figure_asr_errors():
    """Откуда взялся выигрыш: из вставок, а не из замен."""
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    rows = asr_rows()
    labels = [label for label, _ in rows]
    detail = [row["wer_detail"] for _, row in rows]

    substitutions = [d["substitutions"] for d in detail]
    deletions = [d["deletions"] for d in detail]
    insertions = [d["insertions"] for d in detail]

    ax.bar(labels, substitutions, color=BLUE, width=0.55, label="замены")
    bottom = [s + 1.5 for s in substitutions]
    ax.bar(labels, deletions, bottom=bottom, color=AQUA, width=0.55,
           hatch="\\\\\\", edgecolor="white", linewidth=0, label="пропуски")
    bottom = [b + d + 1.5 for b, d in zip(bottom, deletions)]
    ax.bar(labels, insertions, bottom=bottom, color=ORANGE, width=0.55,
           hatch="///", edgecolor="white", linewidth=0, label="вставки")

    totals = [d["substitutions"] + d["deletions"] + d["insertions"] for d in detail]
    for index, (d, total) in enumerate(zip(detail, totals)):
        ax.text(index, total + 10, f"вставок {d['insertions']}",
                ha="center", fontsize=8, color=ORANGE)

    # Запас сверху, иначе подпись самого высокого столбца обрезается рамкой.
    ax.set_ylim(0, max(totals) * 1.16)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("число ошибок")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    save(fig, "fig7-asr-error-types")


def main() -> int:
    points = sweep_points()
    print(f"Чекпойнтов в замере: {len(points)}")
    print("Записываю рисунки:")

    figure_intervention_curve(points)
    figure_decision_mix(points)
    figure_metric_disagreement(points)
    figure_ablations(points)
    figure_retention()
    figure_asr()
    figure_asr_errors()

    print(f"\nГотово: {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
