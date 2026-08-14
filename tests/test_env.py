import os
from pathlib import Path

import pytest

from robo_agency import env


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)


def test_writable_cache_from_env_is_kept(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))

    chosen = env.ensure_writable_hf_cache()

    assert chosen == cache
    assert os.environ["HF_HUB_CACHE"] == str(cache)


def test_unwritable_cache_falls_back_to_home(tmp_path, monkeypatch):
    """Кеш на внешнем диске, смонтированном только на чтение.

    Именно этот случай ронял `make data` с PermissionError: unsloth
    перенаправлял себя, а datasets читал переменную напрямую.
    """
    blocked = tmp_path / "readonly" / "hub"
    fallback = tmp_path / "home" / ".cache" / "huggingface" / "hub"

    monkeypatch.setenv("HF_HUB_CACHE", str(blocked))
    monkeypatch.setattr(env, "_is_writable", lambda path: path != blocked)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    chosen = env.ensure_writable_hf_cache()

    assert chosen == fallback
    assert os.environ["HF_HUB_CACHE"] == str(fallback)


def test_hf_home_derived_when_only_home_set(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    chosen = env.ensure_writable_hf_cache()

    assert chosen == tmp_path / "hf" / "hub"


def test_error_lists_all_tried_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "a"))
    monkeypatch.setattr(env, "_is_writable", lambda path: False)

    with pytest.raises(RuntimeError) as excinfo:
        env.ensure_writable_hf_cache()

    message = str(excinfo.value)
    assert "HF_HOME" in message
    assert str(tmp_path / "a") in message


def test_low_disk_space_warns(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(env, "free_space_gb", lambda path: 3.0)

    with caplog.at_level("WARNING"):
        env.ensure_writable_hf_cache()

    assert "свободно 3.0" in caplog.text


def test_sufficient_disk_space_does_not_warn(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(env, "free_space_gb", lambda path: 500.0)

    with caplog.at_level("WARNING"):
        env.ensure_writable_hf_cache()

    assert "свободно" not in caplog.text


def test_free_space_probes_nearest_existing_parent(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    assert env.free_space_gb(deep) > 0
