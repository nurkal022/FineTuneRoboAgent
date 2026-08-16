"""Чтение аудио-корпусов прямо из parquet, минуя библиотеку datasets.

Причина обходного пути: начиная с datasets 4.x декодирование аудио требует
пакета torchcodec, а он жёстко привязан к версии torch. На этой машине
установка пакетов уже дважды подменяла CUDA-сборку torch на CPU, поэтому
тянуть ещё одну связанную с torch зависимость ради распаковки wav неразумно.

Второй выигрыш — диск. `datasets` при загрузке разворачивает parquet в arrow,
и для FLEURS kk_kz это 3.7 ГБ поверх 3.6 ГБ самих parquet. Здесь читаются
сразу parquet, никакой второй копии не создаётся.

Аудио в этих корпусах лежит структурой {bytes, path}, где bytes — обычный wav
или flac. Его распаковывает soundfile, а частоту приводит librosa: оба уже
установлены.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


def find_parquet_files(repo: str, split: str, config: str | None = None) -> list[str]:
    """Ищет файлы сплита в репозитории датасета.

    Раскладка отличается между корпусами (`parquet-data/kk_kz/validation-*`,
    `data/validation-*`, `kk_kz/validation-*`), поэтому имена не угадываются,
    а берутся из перечня файлов репозитория.
    """
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
    if config:
        scoped = [f for f in files if f"/{config}/" in f or f.startswith(f"{config}/")]
        if scoped:
            files = scoped

    matched = [f for f in files if split in f.rsplit("/", 1)[-1]]
    if not matched:
        raise FileNotFoundError(
            f"В {repo} не нашёл parquet для сплита {split!r}"
            + (f" и конфигурации {config!r}" if config else "")
            + f". Всего parquet-файлов: {len(files)}"
        )
    return sorted(matched)


def decode_audio(payload: Any) -> tuple[Any, int]:
    """Распаковывает поле аудио в массив и частоту дискретизации."""
    import numpy as np
    import soundfile

    if isinstance(payload, dict):
        if payload.get("array") is not None:
            return np.asarray(payload["array"]), int(payload.get("sampling_rate") or SAMPLE_RATE)
        raw = payload.get("bytes")
        if raw is None:
            raise ValueError(f"В поле аудио нет ни массива, ни байтов: ключи {list(payload)}")
    else:
        raw = payload

    array, rate = soundfile.read(io.BytesIO(raw), dtype="float32")
    # Whisper работает с моно: лишние каналы усредняем.
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array, int(rate)


def resample(array: Any, source_rate: int, target_rate: int = SAMPLE_RATE) -> Any:
    if source_rate == target_rate:
        return array
    import librosa

    return librosa.resample(array, orig_sr=source_rate, target_sr=target_rate)


def iter_examples(
    repo: str,
    split: str,
    config: str | None = None,
    limit: int | None = None,
    audio_column: str = "audio",
) -> Iterator[dict[str, Any]]:
    """Выдаёт записи корпуса с уже распакованным аудио на 16 кГц.

    Читает по группам строк, поэтому весь файл в память не поднимается.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    produced = 0
    for remote in find_parquet_files(repo, split, config):
        local = hf_hub_download(repo, remote, repo_type="dataset")
        parquet = pq.ParquetFile(local)
        logger.info("Читаю %s: строк %d", remote, parquet.metadata.num_rows)

        for batch in parquet.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                array, rate = decode_audio(row[audio_column])
                row[audio_column] = {
                    "array": resample(array, rate),
                    "sampling_rate": SAMPLE_RATE,
                }
                yield row
                produced += 1
                if limit is not None and produced >= limit:
                    return
