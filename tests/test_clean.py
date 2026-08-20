# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for `tt-model clean` — clearing cache subtrees to force a clean state (#2)."""

from typer.testing import CliRunner

from tt_kernel import cache, cli

runner = CliRunner()


def _mk_cache(tmp, *keys):
    """Create fake build_key subtrees under a cache root, return the out_root."""
    out = cache.resolve_out_root(str(tmp))
    for k in keys:
        (cache.build_key_path(out, k) / "kernels").mkdir(parents=True)
    return out


def test_clean_one_build_key(tmp_path):
    out = _mk_cache(tmp_path, 111, 222)
    res = runner.invoke(cli.app, ["clean", "--cache-dir", str(tmp_path), "--build-key", "111"])
    assert res.exit_code == 0, res.output
    assert cache.list_build_keys(out) == [222]


def test_clean_all_with_yes(tmp_path):
    out = _mk_cache(tmp_path, 111, 222)
    res = runner.invoke(cli.app, ["clean", "--cache-dir", str(tmp_path), "--all", "--yes"])
    assert res.exit_code == 0, res.output
    assert cache.list_build_keys(out) == []


def test_clean_all_abort_keeps_everything(tmp_path):
    out = _mk_cache(tmp_path, 111)
    res = runner.invoke(cli.app, ["clean", "--cache-dir", str(tmp_path), "--all"], input="n\n")
    assert res.exit_code != 0                 # confirmation aborted
    assert cache.list_build_keys(out) == [111]  # nothing removed


def test_clean_requires_a_target(tmp_path):
    res = runner.invoke(cli.app, ["clean", "--cache-dir", str(tmp_path)])
    assert res.exit_code != 0


def test_clean_build_key_and_all_conflict(tmp_path):
    res = runner.invoke(cli.app, ["clean", "--cache-dir", str(tmp_path), "--all", "--build-key", "1"])
    assert res.exit_code != 0


def test_clean_missing_build_key_is_noop(tmp_path):
    out = _mk_cache(tmp_path, 111)
    res = runner.invoke(cli.app, ["clean", "--cache-dir", str(tmp_path), "--build-key", "999"])
    assert res.exit_code == 0, res.output
    assert "not present" in res.output
    assert cache.list_build_keys(out) == [111]
