"""WER и CER собственной реализацией.

Зависимость на jiwer/evaluate сознательно не берётся: на этой машине установка
любого пакета уже дважды ломала стек (unsloth понижает torch до CPU-сборки),
а расстояние Левенштейна — это двадцать строк, которые можно покрыть тестами
и больше к ним не возвращаться.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .normalize import characters, words


@dataclass(slots=True)
class ErrorRate:
    """Доля ошибок вместе с разложением на типы.

    Разложение важнее самого числа: WER 30% из одних вставок и WER 30% из одних
    пропусков — это разные болезни. Первое обычно означает галлюцинации модели
    на тишине, второе — что она проглатывает окончания.
    """

    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return self.errors / self.reference_length if self.reference_length else 0.0

    def describe(self, name: str = "WER") -> str:
        return (
            f"{name}: {self.rate:.4f} ({self.rate * 100:.2f}%)\n"
            f"  замены:  {self.substitutions}\n"
            f"  пропуски: {self.deletions}\n"
            f"  вставки:  {self.insertions}\n"
            f"  длина эталона: {self.reference_length}"
        )


def _align(reference: Sequence[str], hypothesis: Sequence[str]) -> tuple[int, int, int]:
    """Расстояние Левенштейна с разбором на замены, пропуски и вставки."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1

    # distance[i][j] — цена приведения первых i эталонных к первым j гипотезы.
    distance = [[0] * cols for _ in range(rows)]
    # Вместе с ценой тянем тройку счётчиков, иначе после подсчёта пришлось бы
    # восстанавливать путь обратным проходом.
    counts = [[(0, 0, 0)] * cols for _ in range(rows)]

    for i in range(1, rows):
        distance[i][0] = i
        counts[i][0] = (0, i, 0)  # всё удалено
    for j in range(1, cols):
        distance[0][j] = j
        counts[0][j] = (0, 0, j)  # всё вставлено

    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
                counts[i][j] = counts[i - 1][j - 1]
                continue

            substitute = distance[i - 1][j - 1] + 1
            delete = distance[i - 1][j] + 1
            insert = distance[i][j - 1] + 1
            best = min(substitute, delete, insert)
            distance[i][j] = best

            if best == substitute:
                s, d, ins = counts[i - 1][j - 1]
                counts[i][j] = (s + 1, d, ins)
            elif best == delete:
                s, d, ins = counts[i - 1][j]
                counts[i][j] = (s, d + 1, ins)
            else:
                s, d, ins = counts[i][j - 1]
                counts[i][j] = (s, d, ins + 1)

    return counts[-1][-1]


def _rate(references: Sequence[str], hypotheses: Sequence[str], tokenize) -> ErrorRate:
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Число эталонов ({len(references)}) и гипотез ({len(hypotheses)}) не совпадает"
        )

    total = [0, 0, 0]
    length = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_tokens = tokenize(reference)
        hyp_tokens = tokenize(hypothesis)
        substitutions, deletions, insertions = _align(ref_tokens, hyp_tokens)
        total[0] += substitutions
        total[1] += deletions
        total[2] += insertions
        length += len(ref_tokens)

    return ErrorRate(total[0], total[1], total[2], length)


def word_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> ErrorRate:
    return _rate(references, hypotheses, words)


def character_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> ErrorRate:
    """CER устойчивее WER на агглютинативных языках.

    В казахском одно слово несёт то, что в русском выражается несколькими, и
    ошибка в одном окончании роняет WER на целое слово. Поэтому по CER судить
    о качестве распознавания надёжнее, и в статьях по казахскому ASR приводят
    обе метрики.
    """
    return _rate(references, hypotheses, characters)
