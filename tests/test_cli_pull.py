# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Integration tests for the v2 `pull` pipeline (kernels + runner + weights).

The HF download, pip install, and weights download are all monkeypatched; a synthetic
v2 bundle is built on disk so the real verify/install/partition logic runs for real.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tt_kernel import cache, cli, hub, localdb, metal, runtime
from tt_kernel.manifest import (
    FileEntry,
    Manifest,
    Producer,
    RunnerPayload,
    WeightsRef,
)
from tt_kernel.metal import LocalEnv

runner = CliRunner()
BUILD_KEY = 12345
WHEEL_NAME = "fake_runner-0.1-py3-none-any.whl"


def _populate_bundle(dest: Path, *, with_runner=True, with_weights=True,
                     version="v1.0.0-abc", corrupt_wheel=False,
                     reference_runner=False) -> Manifest:
    """Lay out a synthetic bundle under ``dest`` and return its matching manifest.

    ``reference_runner=True`` records a runner spec with no shipped wheel (reference mode);
    ``with_runner=True`` (default) ships a packaged wheel.
    """
    # kernel subtree under <build_key>/
    sub = dest / str(BUILD_KEY)
    (sub / "kernels" / "reader" / "tensix").mkdir(parents=True)
    (sub / "kernels" / "reader" / "tensix" / "reader.elf").write_bytes(b"reader-binary")
    (sub / "firmware").mkdir()
    (sub / "firmware" / "fw.elf").write_bytes(b"fw")
    files = cache.index_subtree(sub)

    runner_block = None
    if reference_runner:
        runner_block = RunnerPayload(spec="pkg.mod:Runner", entry_point="qwen36",
                                     source="pip:fake-runner")
    elif with_runner:
        (dest / "python").mkdir()
        wheel = dest / "python" / WHEEL_NAME
        wheel.write_bytes(b"PK\x03\x04 fake wheel bytes")
        files = files + [FileEntry(path=f"python/{WHEEL_NAME}",
                                   sha256=cache.sha256_file(wheel),
                                   size=wheel.stat().st_size)]
        if corrupt_wheel:
            wheel.write_bytes(b"DIFFERENT BYTES - sha will not match")
        runner_block = RunnerPayload(spec="pkg.mod:Runner", wheels=[WHEEL_NAME],
                                     entry_point="qwen36")

    weights_block = WeightsRef(repo_id="org/model") if with_weights else None

    m = Manifest(
        name="demo", tt_metal_version=version, arch="blackhole",
        device_count=1, build_key=BUILD_KEY, kernel_count=cache.count_kernels(sub),
        files=files, producer=Producer(tt_kernel_version="0.1.0", created_at="now"),
        runner=runner_block, weights=weights_block,
    )
    (dest / "tt_kernel_manifest.json").write_text(m.to_json())
    return m


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Monkeypatch HF/pip/weights and env; expose call recorders and a cache dir."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))  # localdb lands here
    monkeypatch.setattr(metal, "local_env",
                        lambda **kw: LocalEnv(tt_metal_version="v1.0.0-abc", arch="blackhole",
                                              device_count=1, harvesting_mask=0, build_key=None))
    calls = {"pip": [], "weights": []}
    monkeypatch.setattr(runtime, "pip_install_wheels",
                        lambda wheels, **kw: calls["pip"].append([Path(w).name for w in wheels]))
    monkeypatch.setattr(runtime, "download_weights",
                        lambda w, dest: calls["weights"].append(w.repo_id) or dest)
    monkeypatch.setattr(runtime, "ttnn_importable", lambda python=None: True)
    monkeypatch.setattr(runtime, "runner_spec_importable", lambda spec, python=None: True)
    monkeypatch.setattr(runtime, "dispatch_available", lambda: True)

    state = {"opts": {}}

    def fake_download(repo_id, revision, dest=None):
        d = Path(dest)
        _populate_bundle(d, **state["opts"])
        return d

    monkeypatch.setattr(hub, "download_bundle", fake_download)
    cache_dir = tmp_path / "cache"
    return calls, state, cache_dir


def _invoke(cache_dir, *args):
    return runner.invoke(cli.app, ["pull", "org/demo-bh", "--cache-dir", str(cache_dir), *args])


def test_pull_full_installs_everything(wired):
    calls, state, cache_dir = wired
    state["opts"] = {}
    res = _invoke(cache_dir)
    assert res.exit_code == 0, res.output
    assert calls["pip"] == [[WHEEL_NAME]]
    assert calls["weights"] == ["org/model"]
    # ready-to-run command printed
    assert "serve --unsafe --runner pkg.mod:Runner" in res.output
    rec = localdb.get("org/demo-bh")
    assert rec["python_installed"] is True
    assert rec["weights_installed"] is True
    assert rec["runner_spec"] == "pkg.mod:Runner"


def test_pull_kernels_only(wired):
    calls, state, cache_dir = wired
    state["opts"] = {}
    res = _invoke(cache_dir, "--kernels-only")
    assert res.exit_code == 0, res.output
    assert calls["pip"] == []
    assert calls["weights"] == []
    rec = localdb.get("org/demo-bh")
    assert rec["python_installed"] is False
    assert rec["weights_installed"] is False


def test_pull_no_python(wired):
    calls, state, cache_dir = wired
    state["opts"] = {}
    res = _invoke(cache_dir, "--no-python")
    assert res.exit_code == 0, res.output
    assert calls["pip"] == []
    assert calls["weights"] == ["org/model"]


def test_pull_no_weights(wired):
    calls, state, cache_dir = wired
    state["opts"] = {}
    res = _invoke(cache_dir, "--no-weights")
    assert res.exit_code == 0, res.output
    assert calls["pip"] == [[WHEEL_NAME]]
    assert calls["weights"] == []


def test_pull_wheel_integrity_failure_aborts_before_pip(wired):
    calls, state, cache_dir = wired
    state["opts"] = {"corrupt_wheel": True}
    res = _invoke(cache_dir)
    assert res.exit_code == 1
    assert "integrity check failed" in res.output.lower()
    assert calls["pip"] == []  # never reached pip


def test_pull_version_mismatch_warns_but_installs_with_force(wired):
    calls, state, cache_dir = wired
    state["opts"] = {"version": "v9.9.9-other"}  # mismatches local v1.0.0-abc
    # Without --force the kernel gate blocks:
    blocked = _invoke(cache_dir)
    assert blocked.exit_code == 1
    assert calls["pip"] == []
    # With --force: kernels forced, runner/weights install anyway with an advisory warning.
    res = _invoke(cache_dir, "--force")
    assert res.exit_code == 0, res.output
    assert calls["pip"] == [[WHEEL_NAME]]
    assert calls["weights"] == ["org/model"]
    assert "will NOT run until" in res.output


def test_pull_kernels_only_bundle(wired):
    # A bundle with neither runner nor weights — a pure warm compile-cache share.
    calls, state, cache_dir = wired
    state["opts"] = {"with_runner": False, "with_weights": False}
    res = _invoke(cache_dir)
    assert res.exit_code == 0, res.output
    assert calls["pip"] == []
    assert calls["weights"] == []
    assert "✓ Installed org/demo-bh" in res.output


def test_pull_reference_runner_resolves_without_pip(wired):
    # Reference runner: nothing is pip-installed, but a resolvable spec is "ready",
    # so the serve command still prints and weights still download.
    calls, state, cache_dir = wired
    state["opts"] = {"reference_runner": True}
    res = _invoke(cache_dir)
    assert res.exit_code == 0, res.output
    assert calls["pip"] == []  # reference => nothing shipped to install
    assert calls["weights"] == ["org/model"]
    assert "reference; not shipped" in res.output
    assert "serve --unsafe --runner pkg.mod:Runner" in res.output
    rec = localdb.get("org/demo-bh")
    assert rec["python_installed"] is False
    assert rec["runner_spec"] == "pkg.mod:Runner"


def test_pull_reference_runner_unresolvable_is_pending(monkeypatch, wired):
    # If the reference runner isn't importable, it's reported pending (not installed),
    # with its source — but the pull still succeeds (kernels + weights land).
    calls, state, cache_dir = wired
    monkeypatch.setattr(runtime, "runner_spec_importable", lambda spec, python=None: False)
    state["opts"] = {"reference_runner": True}
    res = _invoke(cache_dir)
    assert res.exit_code == 0, res.output
    assert calls["pip"] == []
    assert "pip:fake-runner" in res.output
    assert "pending" in res.output.lower()
