# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""resolve_version() must not trust frozen metadata over a live source tree.

The bug this pins: tt-metal was upgraded in place from 0.65.1 to v0.77.0 in a
tree installed editable. importlib.metadata still reported the version recorded
when `pip install -e` ran, because an editable install writes its metadata once
and never revisits it. tt-kernel believed it and printed

    ! tt-metal: version 0.65.1rc17.dev6200 is older than required 0.72.0 — upgrade

telling the operator to perform an upgrade they had just performed. A version
probe that is wrong in the stale direction is worse than one that returns None:
None is visibly missing, stale looks authoritative.
"""
import subprocess

from tt_kernel import metal


def test_git_describe_wins_over_stale_editable_metadata(monkeypatch, tmp_path):
    """A git work tree is the truth when one exists; metadata is the fallback."""
    monkeypatch.setattr(metal, "_tt_metal_home", lambda: str(tmp_path))
    monkeypatch.setattr(metal.shutil, "which", lambda _: "/usr/bin/git")

    def fake_run(argv, **kwargs):
        if "rev-parse" in argv and "--is-inside-work-tree" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="true\n", stderr="")
        if "describe" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="v0.77.0\n", stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    monkeypatch.setattr(metal.subprocess, "run", fake_run)
    # Metadata is present AND stale -- exactly the editable-install situation.
    monkeypatch.setattr(metal, "_version_from_metadata", lambda: "0.65.1rc17.dev6200")

    assert metal.resolve_version() == "v0.77.0"


def test_metadata_is_used_when_there_is_no_source_tree(monkeypatch):
    """A wheel install has no git tree, and then metadata is the only truth."""
    monkeypatch.setattr(metal, "_tt_metal_home", lambda: None)
    monkeypatch.setattr(metal, "_version_from_metadata", lambda: "0.77.0")
    assert metal.resolve_version() == "0.77.0"


def test_metadata_is_used_when_the_tree_is_not_a_git_checkout(monkeypatch, tmp_path):
    """A source tree exported without .git must not silently resolve to None."""
    monkeypatch.setattr(metal, "_tt_metal_home", lambda: str(tmp_path))
    monkeypatch.setattr(metal.shutil, "which", lambda _: "/usr/bin/git")
    monkeypatch.setattr(
        metal.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 128, stdout="", stderr="not a git repo"),
    )
    monkeypatch.setattr(metal, "_version_from_metadata", lambda: "0.74.0")
    assert metal.resolve_version() == "0.74.0"
