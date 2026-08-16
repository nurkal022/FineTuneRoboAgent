#!/usr/bin/env python
"""Построение схем для диссертации.

Схемы рисуются тем же инструментом и той же палитрой, что и графики
(`make_figures.py`): в одной работе шрифты, толщины и цвета должны совпадать,
иначе иллюстрации выглядят собранными из разных источников.

Соглашения обозначений, единые для всех схем:

  * СПЛОШНАЯ рамка — существующий, работающий элемент;
    ПУНКТИРНАЯ — запланированный, экспериментально не проверенный.
  * Серая заливка — замороженные веса; синяя — обучаемое.
  * Сплошная стрелка — поток данных при работе;
    пунктирная — поток при обучении или ещё не реализованный.

Вывод в PDF (вектор, для вёрстки) и PNG 300 dpi.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "docs/dissertation/figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#1a1a1a", "#5c5c5c"
FROZEN_FILL, FROZEN_EDGE = "#eceff2", "#9aa5b1"
TRAINED_FILL = "#dbe8f8"
PLANNED_FILL = "#faece5"
NEUTRAL_FILL = "#ffffff"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 300})


def canvas(width: float, height: float, limits: tuple[float, float]):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, limits[0])
    ax.set_ylim(0, limits[1])
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    return fig, ax


def box(ax, x, y, w, h, title, subtitle=None, *, fill=NEUTRAL_FILL, edge=MUTED,
        dashed=False, title_size=8.5, sub_size=7.2, title_color=INK, lw=1.1):
    """Прямоугольник со скруглением; координаты — левый нижний угол."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.8",
        facecolor=fill, edgecolor=edge, linewidth=lw,
        linestyle=(0, (4, 2.5)) if dashed else "solid",
    ))
    if subtitle:
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                fontsize=title_size, color=title_color, weight="bold")
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
                fontsize=sub_size, color=MUTED)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, color=title_color, weight="bold")


def arrow(ax, start, end, *, color=MUTED, dashed=False, lw=1.2, rad=0.0, label=None,
          label_offset=(0, 1.4), label_color=None):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=11,
        color=color, linewidth=lw,
        linestyle=(0, (4, 2.5)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1, shrinkB=1,
    ))
    if label:
        mid = ((start[0] + end[0]) / 2 + label_offset[0],
               (start[1] + end[1]) / 2 + label_offset[1])
        ax.text(*mid, label, ha="center", va="center", fontsize=6.8,
                color=label_color or color)


def elbow(ax, points, *, color=MUTED, dashed=False, lw=1.2):
    """Ломаная со стрелкой на конце.

    Нужна для обратных связей: дуга через всю схему пересекает чужие блоки,
    а обвод по свободному краю читается однозначно.
    """
    style = (0, (4, 2.5)) if dashed else "solid"
    xs = [p[0] for p in points[:-1]]
    ys = [p[1] for p in points[:-1]]
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=style,
            solid_capstyle="round", zorder=1)
    arrow(ax, points[-2], points[-1], color=color, dashed=dashed, lw=lw)


def caption(ax, x, y, text, size=7, color=MUTED, ha="left"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color)


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


# --- Схема 1: общая архитектура ---------------------------------------------

def diagram_system():
    """Общая схема: восприятие, представление, решение, действие."""
    fig, ax = canvas(10.2, 4.1, (204, 82))

    box(ax, 2, 48, 26, 13, "Микрофон", title_size=8)
    box(ax, 2, 28, 26, 13, "Камера", title_size=8)

    box(ax, 34, 48, 40, 13, "Распознавание речи", "Whisper, казахский",
        title_size=7.8, sub_size=6.8)
    box(ax, 34, 28, 40, 13, "Распознавание лица", "личность собеседника",
        title_size=7.8, sub_size=6.8)
    box(ax, 34, 8, 40, 13, "Память", "эпизоды по личности",
        title_size=7.8, sub_size=6.8)

    box(ax, 82, 28, 34, 33, "Модель\nситуации", fill=TRAINED_FILL, edge=BLUE,
        title_size=8.2)

    box(ax, 124, 28, 40, 33, "Контур решений",
        "база + адаптер\nграмматика декодера", fill=TRAINED_FILL, edge=BLUE,
        title_size=8.2, sub_size=6.8)

    box(ax, 172, 48, 30, 13, "Речь", "синтез", title_size=8, sub_size=6.8)
    box(ax, 172, 28, 30, 13, "Движение", "сервоприводы", title_size=8, sub_size=6.8)

    box(ax, 124, 3, 40, 16, "Модель неявной\nнаграды", "реакция человека",
        fill=PLANNED_FILL, edge=ORANGE, dashed=True, title_size=7.4, sub_size=6.6)

    arrow(ax, (28, 54), (34, 54))
    arrow(ax, (28, 34), (34, 34))
    arrow(ax, (74, 54), (82, 50))
    arrow(ax, (74, 34), (82, 42))
    arrow(ax, (74, 14), (82, 32))
    arrow(ax, (116, 44), (124, 44), color=BLUE, lw=1.5)
    arrow(ax, (164, 50), (172, 54), color=BLUE, lw=1.5)
    arrow(ax, (164, 38), (172, 34), color=BLUE, lw=1.5)

    # Замкнутая петля: действие наблюдается, оценка возвращается в обучение.
    arrow(ax, (187, 28), (164, 11), color=ORANGE, dashed=True, rad=-0.3,
          label="реакция", label_offset=(12, -6), label_color=ORANGE)
    arrow(ax, (124, 11), (99, 28), color=ORANGE, dashed=True, rad=-0.3,
          label="дообучение", label_offset=(-4, -6), label_color=ORANGE)

    for x, name in ((2, "Восприятие"), (82, "Представление"),
                    (124, "Принятие решения"), (172, "Действие")):
        caption(ax, x, 70, name, size=8, color=INK)

    caption(ax, 2, 1, "пунктиром — запланированное, экспериментально не проверенное")

    save(fig, "sch1-system")


# --- Схема 2: двухконтурная архитектура -------------------------------------

def diagram_two_contours():
    fig, ax = canvas(7.4, 4.2, (148, 84))

    box(ax, 34, 8, 80, 18, "Базовая языковая модель",
        "веса заморожены на всё время работы",
        fill=FROZEN_FILL, edge=FROZEN_EDGE, title_size=9.5, lw=1.4)

    box(ax, 84, 36, 30, 14, "Адаптер решений",
        "обучается", fill=TRAINED_FILL, edge=BLUE, lw=1.4)

    box(ax, 8, 58, 56, 18, "Речевой контур",
        "адаптер ВЫКЛЮЧЕН\nразговор с человеком",
        fill=NEUTRAL_FILL, edge=MUTED)
    box(ax, 84, 58, 56, 18, "Контур решений",
        "адаптер ВКЛЮЧЁН\nACT / WAIT / OBSERVE",
        fill=TRAINED_FILL, edge=BLUE, lw=1.4)

    arrow(ax, (50, 26), (36, 58), color=MUTED, rad=0.12,
          label="адаптер\nне применяется", label_offset=(-11, 0))
    arrow(ax, (99, 26), (99, 36), color=BLUE, lw=1.4)
    arrow(ax, (99, 50), (105, 58), color=BLUE, lw=1.4)

    caption(ax, 74, 32, "одна и та же модель, два режима", size=7.5, color=INK)
    caption(ax, 8, 2,
            "Деградация речи исключена архитектурно: речевой контур в обучении не участвует")

    save(fig, "sch2-two-contours")


# --- Схема 3: информационные модели -----------------------------------------

def diagram_information_models():
    fig, ax = canvas(8.6, 4.6, (172, 92))

    box(ax, 4, 40, 46, 44, "Модель ситуации", fill=NEUTRAL_FILL, edge=BLUE)
    situation = [
        "время, длительность тишины",
        "личность: кто, знаком ли",
        "визуальный контекст",
        "речь и её язык",
        "память об этом человеке",
        "предыдущее действие робота",
    ]
    for index, line in enumerate(situation):
        ax.text(8, 74 - index * 5.4, f"·  {line}", fontsize=6.8, color=MUTED, va="center")

    box(ax, 62, 52, 44, 20, "Контур решений", "база + адаптер",
        fill=TRAINED_FILL, edge=BLUE, lw=1.4)

    box(ax, 62, 24, 44, 16, "Грамматика декодера",
        "схема решения → ограничение", fill=NEUTRAL_FILL, edge=AQUA, lw=1.4)

    box(ax, 118, 40, 50, 44, "Модель решения", fill=NEUTRAL_FILL, edge=BLUE)
    decision = [
        "решение: ACT / WAIT / OBSERVE",
        "уверенность",
        "речь — только при ACT",
        "движение — только при ACT",
        "что записать в память",
    ]
    for index, line in enumerate(decision):
        ax.text(122, 74 - index * 5.4, f"·  {line}", fontsize=6.8, color=MUTED, va="center")

    arrow(ax, (50, 62), (62, 62), color=BLUE, lw=1.4)
    arrow(ax, (106, 62), (118, 62), color=BLUE, lw=1.4)
    arrow(ax, (84, 40), (84, 52), color=AQUA, lw=1.4)

    box(ax, 4, 4, 164, 14,
        "Инвариант схемы: решения WAIT и OBSERVE не могут нести речь или движение",
        "иначе выучивается вырожденная стратегия «формально промолчал, но реплику приложил»",
        fill=PLANNED_FILL, edge=ORANGE, title_size=8, sub_size=7)

    save(fig, "sch3-information-models")


# --- Схема 4: формирование корпуса ------------------------------------------

def diagram_corpus():
    fig, ax = canvas(8.8, 4.0, (176, 78))

    box(ax, 4, 54, 46, 16, "Корпус проактивности", "1749 записей",
        fill=NEUTRAL_FILL, title_size=7.8)
    box(ax, 4, 32, 46, 16, "Вызов инструментов", "открытые корпуса",
        fill=NEUTRAL_FILL, title_size=7.8)
    box(ax, 4, 10, 46, 16, "Диалоговые данные", "общий диалог",
        fill=NEUTRAL_FILL, title_size=7.8)

    box(ax, 58, 54, 44, 16, "Приведение к схеме",
        "эталон — категория разметки", fill=TRAINED_FILL, edge=BLUE, title_size=8)
    box(ax, 58, 21, 44, 16, "Родной формат",
        "к схеме НЕ приводятся", fill=NEUTRAL_FILL, edge=MUTED, title_size=8)

    box(ax, 110, 54, 30, 16, "Выравнивание", "504 / 504",
        fill=TRAINED_FILL, edge=BLUE, title_size=8)

    box(ax, 110, 21, 30, 16, "Микс", "55 / 20 / 25 %",
        fill=TRAINED_FILL, edge=BLUE, title_size=8)
    box(ax, 148, 21, 24, 16, "1832", "train / val", fill=NEUTRAL_FILL, edge=BLUE)

    arrow(ax, (50, 62), (58, 62))
    arrow(ax, (50, 40), (58, 32), rad=0.1)
    arrow(ax, (50, 18), (58, 26), rad=-0.1)
    arrow(ax, (102, 62), (110, 62), color=BLUE)
    arrow(ax, (125, 54), (125, 37), color=BLUE)
    arrow(ax, (102, 29), (110, 29))
    arrow(ax, (140, 29), (148, 29), color=BLUE)

    caption(ax, 58, 46,
            "1626 примеров, доля ACT 31 % — прореживается до равновесия", size=6.8)
    caption(ax, 4, 3,
            "Function-calling и диалог остаются в родном формате: их задача — сохранить "
            "имеющиеся навыки, и перегонка в схему решений их уничтожает")

    save(fig, "sch4-corpus")


# --- Схема 5: модель неявной награды ----------------------------------------

def diagram_implicit_reward():
    """Замкнутая петля: действие — реакция — оценка — дообучение."""
    fig, ax = canvas(8.8, 4.7, (176, 94))

    box(ax, 4, 58, 32, 17, "Действие робота", "речь и движение",
        fill=TRAINED_FILL, edge=BLUE, title_size=8)
    box(ax, 44, 58, 34, 17, "Окно наблюдения", "3–5 секунд",
        fill=PLANNED_FILL, edge=ORANGE, dashed=True, title_size=8)

    box(ax, 86, 30, 48, 45, "", fill=NEUTRAL_FILL, edge=ORANGE, dashed=True)
    ax.text(110, 69, "Признаки реакции", ha="center", va="center",
            fontsize=8, color=INK, weight="bold")
    signals = [
        "смена выражения лица",
        "продолжил ли диалог",
        "отвернулся, вышел из кадра",
        "перебил на полуслове",
        "переспросил то же самое",
    ]
    for index, text in enumerate(signals):
        ax.text(90, 60 - index * 7, f"·  {text}", fontsize=6.8, color=MUTED, va="center")

    box(ax, 142, 58, 30, 17, "Оценка", "скалярная",
        fill=PLANNED_FILL, edge=ORANGE, dashed=True, title_size=8)
    box(ax, 142, 30, 30, 19, "Пары\nпредпочтений", "для DPO",
        fill=PLANNED_FILL, edge=ORANGE, dashed=True, title_size=7.6)

    arrow(ax, (36, 66), (44, 66), color=BLUE, lw=1.4)
    arrow(ax, (78, 66), (86, 62), color=ORANGE, dashed=True)
    arrow(ax, (134, 62), (142, 66), color=ORANGE, dashed=True)
    arrow(ax, (157, 58), (157, 49), color=ORANGE, dashed=True)

    # Возврат в обучение идёт понизу, в обход блоков.
    elbow(ax, [(157, 30), (157, 14), (20, 14), (20, 58)],
          color=ORANGE, dashed=True)
    ax.text(88, 17, "дообучение адаптера", ha="center", va="center",
            fontsize=7, color=ORANGE)

    caption(ax, 4, 6, "Ручная разметка не требуется: оценку даёт сама реакция человека",
            size=7.5, color=INK)
    caption(ax, 4, 1, "Пунктиром — запланированное; экспериментально не проверялось")

    save(fig, "sch5-implicit-reward")


def main() -> int:
    print("Записываю схемы:")
    diagram_system()
    diagram_two_contours()
    diagram_information_models()
    diagram_corpus()
    diagram_implicit_reward()
    print(f"\nГотово: {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
