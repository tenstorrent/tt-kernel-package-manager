# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Backward-compat: the tt-kernel -> tt-model rename must not break old installs
(legacy TT_KERNEL_* env vars, legacy ~/.cache|.config/tt-kernel dirs, legacy command name)."""

from pathlib import Path

from tt_kernel import compat, runtime, localdb


def test_env_prefers_new_then_falls_back_to_legacy(monkeypatch):
    monkeypatch.delenv("TT_MODEL_MODELS_DIR", raising=False)
    monkeypatch.delenv("TT_KERNEL_MODELS_DIR", raising=False)
    assert compat.env("TT_MODEL_MODELS_DIR") is None
    monkeypatch.setenv("TT_KERNEL_MODELS_DIR", "/legacy")      # only old var set
    assert compat.env("TT_MODEL_MODELS_DIR") == "/legacy"       # honored
    monkeypatch.setenv("TT_MODEL_MODELS_DIR", "/new")          # new var wins
    assert compat.env("TT_MODEL_MODELS_DIR") == "/new"


def test_data_dir_prefers_legacy_only_when_new_absent(tmp_path):
    base = tmp_path
    # neither exists -> new default
    assert compat.data_dir(base) == base / "tt-model"
    # only legacy exists -> legacy (a pre-rename install keeps its data)
    (base / "tt-kernel").mkdir()
    assert compat.data_dir(base) == base / "tt-kernel"
    # new exists too -> new wins
    (base / "tt-model").mkdir()
    assert compat.data_dir(base) == base / "tt-model"


def test_resolve_models_dir_honors_legacy_env(monkeypatch, tmp_path):
    monkeypatch.delenv("TT_MODEL_MODELS_DIR", raising=False)
    monkeypatch.setenv("TT_KERNEL_MODELS_DIR", str(tmp_path / "old"))
    got = runtime.resolve_models_dir(None, "org/name")
    assert got == tmp_path / "old" / "org" / "name"


def test_localdb_uses_legacy_cache_dir_if_present(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "tt-kernel").mkdir()          # a pre-rename install's index dir
    assert localdb._index_path() == tmp_path / "tt-kernel" / "installed.json"


def test_invoked_as_legacy(monkeypatch):
    monkeypatch.setattr("sys.argv", ["/usr/bin/tt-kernel", "serve", "x"])
    assert compat.invoked_as_legacy() is True
    monkeypatch.setattr("sys.argv", ["/usr/bin/tt-model", "serve", "x"])
    assert compat.invoked_as_legacy() is False
