import pytest

from robo_agency.data.build import DatasetSpec, _load_rows


def test_missing_local_file_gives_actionable_error(tmp_path):
    """Отсутствующий локальный файл не должен уходить в стек Hugging Face.

    Раньше такой путь передавался в load_dataset и всплывал как невнятная
    ошибка библиотеки вместо подсказки «скачайте данные».
    """
    spec = DatasetSpec(path=str(tmp_path / "proactive_bench.jsonl"))

    with pytest.raises(FileNotFoundError) as excinfo:
        _load_rows(spec)

    message = str(excinfo.value)
    assert "make fetch" in message
    assert "proactive_bench.jsonl" in message


def test_existing_local_jsonl_is_read(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")

    rows = _load_rows(DatasetSpec(path=str(path)))
    assert rows == [{"a": 1}, {"a": 2}]


def test_limit_truncates_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(f'{{"a": {i}}}' for i in range(10)), encoding="utf-8")

    rows = _load_rows(DatasetSpec(path=str(path), limit=3))
    assert len(rows) == 3


def test_hub_style_path_not_treated_as_missing_file():
    """Имя датасета на хабе не имеет расширения и не должно падать как файл."""
    spec = DatasetSpec(path="Salesforce/xlam-function-calling-60k")
    with pytest.raises(Exception) as excinfo:
        _load_rows(spec)
    # Ошибка может быть любой (нет сети, нет пакета), но не нашей FileNotFoundError
    assert "make fetch" not in str(excinfo.value)
