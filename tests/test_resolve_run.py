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


# -------------------------------------------------------------------- run routing
def _run(args, monkeypatch):
    monkeypatch.setattr(runtime, "dispatch_available", lambda: True)
    return runner.invoke(cli.app, ["run", *args, "--print", "--local-only"])


def test_run_tier1_installed_uses_custom_runner(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _seed_installed("org/tuned", runner_spec="pkg:R", weights_path="/w/t", python_installed=True)
    res = _run(["org/tuned"], monkeypatch)
    assert res.exit_code == 0, res.output
    assert "--runner pkg:R" in res.output and "/w/t" in res.output
    assert "custom runner" in res.output


def test_run_tier2_kernels_only_is_dynamic_on_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _seed_installed("org/cache", weights_repo="org/weights")
    res = _run(["org/cache"], monkeypatch)
    assert res.exit_code == 0, res.output
    assert "--runner" not in res.output                # no custom runner
    assert "serve --unsafe org/weights" in res.output  # dynamic on the weights repo
    assert "Precompiled kernels present" in res.output


def test_run_tier3_no_bundle_is_dynamic_on_bare_id(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    res = _run(["some/hf-model"], monkeypatch)
    assert res.exit_code == 0, res.output
    assert "--runner" not in res.output
    assert "serve --unsafe some/hf-model" in res.output
    assert "no bundle" in res.output


def test_run_tier1_published_notifies_then_runs_dynamic(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    # local_only is forced by _run, so seed the "published" path via a non-local resolve:
    monkeypatch.setattr(runtime, "dispatch_available", lambda: True)
    monkeypatch.setattr(hub, "fetch_manifest", lambda rid, rev: _published_manifest(with_runner=True))
    res = runner.invoke(cli.app, ["run", "org/pub", "--print"])  # allow Hub lookup
    assert res.exit_code == 0, res.output
    assert "tuned tt-kernel bundle exists" in res.output
    assert "tt-kernel pull org/pub" in res.output
    assert "--runner" not in res.output                # ran dynamic, not the bundle's runner
    assert "serve --unsafe org/weights" in res.output  # dynamic on the bundle's weights repo
