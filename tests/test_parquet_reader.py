"""Тесты чтения аудио-корпусов в обход datasets.

Обход появился потому, что datasets 4.x требует torchcodec для распаковки
аудио, а он привязан к версии torch — на этой машине лишняя связанная с torch
зависимость уже дважды оборачивалась подменой CUDA-сборки на CPU.
"""

import io

import pytest

from robo_agency.asr.parquet_reader import SAMPLE_RATE, decode_audio, find_parquet_files, resample

numpy = pytest.importorskip("numpy")


def make_wav(duration_sec: float = 0.1, rate: int = SAMPLE_RATE, channels: int = 1) -> bytes:
    soundfile = pytest.importorskip("soundfile")
    samples = numpy.zeros((int(rate * duration_sec), channels), dtype="float32")
    samples[:, 0] = numpy.linspace(-0.5, 0.5, samples.shape[0])
    buffer = io.BytesIO()
    soundfile.write(buffer, samples.squeeze(), rate, format="WAV")
    return buffer.getvalue()


def test_decode_from_bytes():
    array, rate = decode_audio({"bytes": make_wav(), "path": "a.wav"})
    assert rate == SAMPLE_RATE
    assert array.ndim == 1
    assert len(array) > 0


def test_decode_prefers_ready_array():
    payload = {"array": [0.1, 0.2, 0.3], "sampling_rate": 8000}
    array, rate = decode_audio(payload)
    assert rate == 8000
    assert len(array) == 3


def test_stereo_is_mixed_to_mono():
    """Whisper принимает моно; лишние каналы надо свести, а не уронить."""
    array, _ = decode_audio({"bytes": make_wav(channels=2)})
    assert array.ndim == 1


def test_raw_bytes_accepted():
    array, rate = decode_audio(make_wav())
    assert rate == SAMPLE_RATE
    assert len(array) > 0


def test_missing_payload_reports_keys():
    with pytest.raises(ValueError, match="ключи"):
        decode_audio({"path": "a.wav"})


def test_resample_changes_length():
    # librosa нужен только когда частота не совпадает с 16 кГц; в наших
    # корпусах она совпадает, поэтому зависимость мягкая.
    pytest.importorskip("librosa")
    array = numpy.zeros(8000, dtype="float32")
    result = resample(array, 8000, 16000)
    assert len(result) == pytest.approx(16000, rel=0.01)


def test_resample_is_noop_at_same_rate():
    array = numpy.zeros(100, dtype="float32")
    assert resample(array, SAMPLE_RATE, SAMPLE_RATE) is array


def test_split_files_selected_by_name(monkeypatch):
    files = [
        "parquet-data/kk_kz/train-00000-of-00001.parquet",
        "parquet-data/kk_kz/validation-00000-of-00001.parquet",
        "parquet-data/ru_ru/validation-00000-of-00001.parquet",
        "README.md",
    ]
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        type("M", (), {"list_repo_files": staticmethod(lambda *a, **k: files)}),
    )

    found = find_parquet_files("google/fleurs", "validation", "kk_kz")
    assert found == ["parquet-data/kk_kz/validation-00000-of-00001.parquet"]


def test_missing_split_error_is_informative(monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        type("M", (), {"list_repo_files": staticmethod(lambda *a, **k: ["a/train.parquet"])}),
    )
    with pytest.raises(FileNotFoundError, match="validation"):
        find_parquet_files("some/repo", "validation", None)
