"""Smoke test: config builds a DB URL from a synthetic env."""
from __future__ import annotations

import os


def test_settings_db_url(monkeypatch):
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "56432")
    monkeypatch.setenv("DB_NAME", "rag")
    monkeypatch.setenv("DB_USER", "rag")
    monkeypatch.setenv("DB_PASSWORD", "secret")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text-v2-moe")

    # Drop the lazy-loaded singleton
    import importlib
    import src.config as cfg_mod
    importlib.reload(cfg_mod)
    s = cfg_mod.Settings()
    assert s.db_url == "postgresql://rag:secret@localhost:56432/rag"


def test_settings_rerank_use_softmax_parses_true_and_false(monkeypatch):
    import importlib
    import src.config as cfg_mod

    monkeypatch.setenv("RERANK_USE_SOFTMAX", "true")
    importlib.reload(cfg_mod)
    assert cfg_mod.Settings().rerank_use_softmax is True

    monkeypatch.setenv("RERANK_USE_SOFTMAX", "false")
    importlib.reload(cfg_mod)
    assert cfg_mod.Settings().rerank_use_softmax is False


def test_settings_rerank_use_softmax_defaults_disabled(monkeypatch):
    import importlib
    import src.config as cfg_mod
    from pathlib import Path

    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: False if path.name == ".env" else original_exists(path))

    monkeypatch.delenv("RERANK_USE_SOFTMAX", raising=False)
    importlib.reload(cfg_mod)
    assert cfg_mod.Settings().rerank_use_softmax is False


def test_settings_retrieval_limits_are_configurable(monkeypatch):
    import importlib
    import src.config as cfg_mod

    monkeypatch.setenv("DENSE_TOP_K", "7")
    monkeypatch.setenv("SPARSE_TOP_K", "9")
    monkeypatch.setenv("RRF_TOP_N", "4")
    monkeypatch.setenv("RERANK_TOP_N", "2")
    importlib.reload(cfg_mod)

    configured = cfg_mod.Settings()
    assert configured.dense_top_k == 7
    assert configured.sparse_top_k == 9
    assert configured.rrf_top_n == 4
    assert configured.rerank_top_n == 2
    assert not hasattr(configured, "rerank_candidates")


def test_jev_settings_validate_numeric_ranges():
    import pytest
    from src.config import Settings
    for threshold in (-1, 2, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Settings(jev_block_threshold=threshold)
    for timeout in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Settings(jev_timeout_seconds=timeout)
    assert not hasattr(Settings(), "guard_model")
    assert not hasattr(Settings(), "guard_enabled")
