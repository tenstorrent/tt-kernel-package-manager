# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for the detect-only resolver and the ``run`` orchestrator (WS1).

No hardware, no network: localdb is redirected to a tmp dir via XDG_CACHE_HOME, the
Hub manifest fetch is monkeypatched, and ``run`` is exercised with ``--print`` so it
never launches a server.
"""

from typer.testing import CliRunner

from tt_kernel import cli, hub, localdb, runtime
from tt_kernel.manifest import Manifest, Producer, RunnerPayload, WeightsRef
from tt_kernel.resolve import resolve

runner = CliRunner()


def _seed_installed(repo_id, *, runner_spec=None, weights_path=None, weights_repo=None,
                    build_key=42, python_installed=False):
    """Write an installed-bundle record like ``_record_pull`` would."""
    localdb.record(repo_id, {
        "name": repo_id.split("/")[-1],
        "build_key": build_key,
        "arch": "blackhole",
        "tt_metal_version": "0.72.0",
        "out_root": "/tmp/cache/",
        "schema_version": "2",
        "runner_spec": runner_spec,
        "entry_point": None,
        "weights_repo": weights_repo,
        "weights_path": weights_path,
        "python_installed": python_installed,
        "weights_installed": weights_path is not None,
        "installed_at": "now",
    })


def _published_manifest(*, with_runner):
    return Manifest(
        name="demo",
        tt_metal_version="0.72.0",
        arch="blackhole",
        build_key=99,
        producer=Producer(tt_kernel_version="0.1.0", created_at="now"),
        runner=RunnerPayload(spec="pkg.mod:R", wheels=["r.whl"]) if with_runner else None,
        weights=WeightsRef(repo_id="org/weights"),
    )


# ----------------------------------------------------------------- resolve() unit
def test_resolve_no_bundle_local_only(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    r = resolve("nobody/nothing", local_only=True)
    assert not r.exists and r.tier == 3 and not r.has_runner


def test_resolve_installed_runner_is_tier1(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _seed_installed("org/tuned", runner_spec="pkg:R", weights_path="/w/org/tuned",
                    python_installed=True)
    r = resolve("org/tuned", local_only=True)
    assert r.installed and r.has_runner and r.tier == 1
    assert r.runner_spec == "pkg:R" and r.serve_target == "/w/org/tuned"


def test_resolve_installed_kernels_only_is_tier2(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _seed_installed("org/cache", weights_repo="org/weights")
    r = resolve("org/cache", local_only=True)
    assert r.installed and not r.has_runner and r.tier == 2
    assert r.serve_target == "org/weights"


def test_resolve_published_not_installed_uses_hub(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(hub, "fetch_manifest", lambda rid, rev: _published_manifest(with_runner=True))
    r = resolve("org/pub")
    assert r.exists and not r.installed and r.source == "hub"
    assert r.has_runner and r.runner_spec == "pkg.mod:R" and r.weights_repo == "org/weights"


def test_resolve_local_only_skips_hub(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    called = []
    monkeypatch.setattr(hub, "fetch_manifest", lambda rid, rev: called.append(rid))
    r = resolve("org/pub", local_only=True)
    assert not r.exists and called == []  # never touched the network


def test_resolve_vllm_backend_from_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    m = Manifest(
        name="v", tt_metal_version="0.72.0", arch="blackhole", build_key=None,
        producer=Producer(tt_kernel_version="0.1.0", created_at="now"),
        runner=RunnerPayload(backend="vllm", bundle_dir="vllm_bundle"),
        weights=WeightsRef(repo_id="org/w"),
    )
    monkeypatch.setattr(hub, "fetch_manifest", lambda rid, rev: m)
    r = resolve("org/vllm")
    assert r.exists and r.is_vllm and r.has_runner and r.backend == "vllm"


def test_resolve_vllm_backend_from_localdb(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    localdb.record("org/vllm", {
        "name": "vllm", "backend": "vllm", "build_key": None, "arch": "blackhole",
        "bundle_path": "/bundles/org__vllm", "installed_at": "now",
    })
    r = resolve("org/vllm", local_only=True)
    assert r.installed and r.is_vllm and r.has_runner and r.bundle_path == "/bundles/org__vllm"


# -------------------------------------------------------------------- run routing
def _run(args, monkeypatch):
    monkeypatch.setattr(runtime, "legacy_serve_available", lambda: True)
    return runner.invoke(cli.app, ["run", *args, "--print", "--local-only"])


def test_run_installed_legacy_runner_uses_legacy_serve(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _seed_installed("org/tuned", runner_spec="pkg:R", weights_path="/w/t", python_installed=True)
    res = _run(["org/tuned"], monkeypatch)
    assert res.exit_code == 0, res.output
    # legacy-runner server command, not the retired tt_api dispatch path
    assert "tt_kernel.legacy_serve" in res.output
    assert "--runner pkg:R" in res.output and "--model /w/t" in res.output
    assert "tt_api" not in res.output


def test_run_kernels_only_is_not_servable(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _seed_installed("org/cache", weights_repo="org/weights")  # no runner
    res = _run(["org/cache"], monkeypatch)
    assert res.exit_code == 1
    assert "Nothing to serve" in res.output
    assert "tt_api" not in res.output


def test_run_bare_repo_is_not_servable(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    res = _run(["some/hf-model"], monkeypatch)
    assert res.exit_code == 1
    assert "Nothing to serve" in res.output


def test_run_published_legacy_runner_says_pull_first(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(hub, "fetch_manifest", lambda rid, rev: _published_manifest(with_runner=True))
    res = runner.invoke(cli.app, ["run", "org/pub", "--print"])  # allow Hub lookup
    assert res.exit_code == 0, res.output
    assert "legacy runner" in res.output
    assert "tt-model pull org/pub" in res.output
    assert "tt_api" not in res.output                  # no dead dispatch command emitted
