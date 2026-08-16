import pytest

from robo_agency.asr.config import WhisperConfig, load_config
from robo_agency.asr.data import DatasetSpec, SpeechCollator, resolve_text_column


def test_shipped_config_loads():
    config = load_config("configs/whisper_kk.yaml")

    # После эксперимента 003 конфиг указывает на готовое казахское
    # дообучение, а не на базовый turbo: своё обучение отменено.
    assert config.model == "shyngys879/kazakh-whisper-large-v3-turbo"
    assert config.language == "kazakh"
    assert config.dataset.name == "kk_kz"
    # Поток обязателен: корпус целиком на диск не влезет.
    assert config.dataset.streaming


def test_effective_batch():
    config = WhisperConfig(per_device_batch_size=8, gradient_accumulation_steps=2)
    assert config.effective_batch == 16


def test_known_model_size():
    assert WhisperConfig(model="openai/whisper-small").params_millions == 244
    assert WhisperConfig(model="какая-то/своя").params_millions is None


@pytest.mark.parametrize("field,value", [("max_steps", 0), ("per_device_batch_size", 0)])
def test_invalid_values_rejected_early(field, value):
    with pytest.raises(ValueError):
        WhisperConfig(**{field: value})


def test_config_roundtrip(tmp_path):
    path = tmp_path / "w.yaml"
    path.write_text(
        "model: openai/whisper-base\nmax_steps: 100\n"
        "dataset:\n  path: google/fleurs\n  name: kk_kz\n  streaming: false\n",
        encoding="utf-8",
    )
    config = load_config(path)

    assert config.model == "openai/whisper-base"
    assert config.max_steps == 100
    assert config.dataset.streaming is False


@pytest.mark.parametrize(
    "columns,expected",
    [
        (["audio", "transcription"], "transcription"),   # FLEURS
        (["audio", "sentence"], "sentence"),             # Common Voice
        (["audio", "text"], "text"),
    ],
)
def test_text_column_autodetected(columns, expected):
    assert resolve_text_column(columns) == expected


def test_text_column_override():
    assert resolve_text_column(["audio", "raw"], "raw") == "raw"


def test_missing_text_column_lists_available():
    with pytest.raises(KeyError) as excinfo:
        resolve_text_column(["audio", "id"])
    assert "id" in str(excinfo.value)


def test_unknown_override_rejected():
    with pytest.raises(KeyError):
        resolve_text_column(["audio", "text"], "нет_такой")


def test_collator_masks_padding_in_labels():
    """Паддинг в метках должен стать -100.

    Иначе модель учится предсказывать padding-токены как настоящий текст,
    и распознавание начинает обрываться на середине фразы.
    """
    torch = pytest.importorskip("torch")

    class FakePad:
        def pad(self, items, return_tensors=None):
            key = "input_features" if "input_features" in items[0] else "input_ids"
            longest = max(len(item[key]) for item in items)
            padded, mask = [], []
            for item in items:
                values = list(item[key])
                gap = longest - len(values)
                padded.append(values + [0] * gap)
                mask.append([1] * len(values) + [0] * gap)
            result = {key: torch.tensor(padded)}
            if key == "input_ids":
                result["attention_mask"] = torch.tensor(mask)

            class Batch(dict):
                pass

            batch = Batch(result)
            batch.attention_mask = torch.tensor(mask)
            return batch

    class FakeProcessor:
        feature_extractor = FakePad()
        tokenizer = FakePad()

    collator = SpeechCollator(processor=FakeProcessor(), decoder_start_token_id=50258)
    batch = collator([
        {"input_features": [1.0, 2.0], "labels": [5, 6, 7]},
        {"input_features": [3.0, 4.0], "labels": [8]},
    ])

    labels = batch["labels"]
    assert (labels[1][1:] == -100).all()
    assert labels[0].tolist()[:3] == [5, 6, 7]


def test_dataset_spec_defaults_to_fleurs():
    spec = DatasetSpec()
    assert spec.path == "google/fleurs"
    assert spec.name == "kk_kz"
