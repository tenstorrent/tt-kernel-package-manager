# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for the v4 unified "model + manifest" packaging model: schema back-compat,
range-aware resolution, launch rendering, and the push->pull->serve CLI round-trip that
renders vllm_metadata.json from the manifest. No hardware, no network.
"""

import json

import pytest
from typer.testing import CliRunner

from tt_kernel import bundles, cli, hub, localdb, metal, runtime, toolchain
from tt_kernel.device import DeviceInfo
from tt_kernel.manifest import (
    Capabilities,
    Entrypoint,
    Manifest,
    Platform,
    Producer,
    Resources,
    Runtime,
    WeightsRef,
    compare,
)

runner = CliRunner()


# --------------------------------------------------------------------- fixtures
def _v4_manifest(**over):
    """A minimal valid v4 Manifest; override any field."""
    base = dict(
        schema_version="4",
        name="Laguna",
        tt_metal_version="0.73.0",
        arch="blackhole",
        device_count=4,
        producer=Producer(tt_kernel_version="0", created_at="t"),
        entrypoint=Entrypoint(**{"class": "ttl.gen:LagunaForCausalLM", "arch_name": "LagunaForCausalLM"}),
        weights=WeightsRef(repo="poolside/Laguna-XS-2.1"),
    )
    base.update(over)
    return Manifest(**base)


def _env(**over):
    base = dict(arch="blackhole", device_count=4, tt_metal_version="0.73.0", vllm_version="0.24.1")
    base.update(over)
    return metal.LocalEnv(**base)


# --------------------------------------------------------------------- schema
def test_v4_manifest_parses():
    m = _v4_manifest(platform=Platform(ttnn=">=0.72,<0.76"))
    assert m.is_v4 is True
    j = m.to_json()
    m2 = Manifest.from_json(j)
    assert m2.is_v4 and m2.entrypoint.cls == "ttl.gen:LagunaForCausalLM"
    assert m2.weights.repo_id == "poolside/Laguna-XS-2.1"


def test_v4_json_aliases_accepted():
    """An authored manifest uses the natural `class` / `repo` aliases."""
    text = json.dumps({
        "schema_version": "4", "name": "L", "tt_metal_version": "0.73.0", "arch": "blackhole",
        "producer": {"tt_kernel_version": "0", "created_at": "t"},
        "entrypoint": {"class": "a:B", "arch_name": "B"},
        "weights": {"repo": "x/y"},
    })
    m = Manifest.from_json(text)
    assert m.entrypoint.cls == "a:B" and m.weights.repo_id == "x/y"


def test_v3_manifest_still_parses():
    text = json.dumps({
        "schema_version": "3", "name": "legacy", "tt_metal_version": "0.72.0", "arch": "blackhole",
        "build_key": 4242, "producer": {"tt_kernel_version": "0", "created_at": "t"},
    })
    m = Manifest.from_json(text)
    assert m.schema_version == "3" and m.is_v4 is False and m.build_key == 4242


def test_unknown_schema_rejected():
    text = json.dumps({
        "schema_version": "99", "name": "x", "tt_metal_version": "0", "arch": "blackhole",
        "producer": {"tt_kernel_version": "0", "created_at": "t"},
    })
    with pytest.raises(ValueError, match="Unsupported bundle schema_version"):
        Manifest.from_json(text)


# --------------------------------------------------------------------- version ranges
@pytest.mark.parametrize("installed,spec,expected", [
    ("0.73.0", ">=0.72,<0.76", True),
    ("0.80.0", ">=0.72,<0.76", False),
    ("0.72.0-5-gabc", ">=0.72,<0.76", True),   # git-describe decoration tolerated
    ("deadbeef", ">=0.72,<0.76", None),        # bare sha -> assume OK
    (None, ">=0.72", None),                    # unresolved -> assume OK
    ("0.73.0", "not-a-spec!!", None),          # malformed spec -> None, never raises
])
def test_version_satisfies(installed, spec, expected):
    assert toolchain.version_satisfies(installed, spec) is expected


def test_compare_v4_in_range_compatible():
    m = _v4_manifest(platform=Platform(ttnn=">=0.72,<0.76"), runtime=Runtime(version=">=0.24"))
    r = compare(m, _env())
    assert r.compatible and not r.issues


def test_compare_v4_out_of_range_is_forceable_not_fatal():
    m = _v4_manifest(platform=Platform(ttnn=">=0.72,<0.76"), runtime=Runtime(version=">=0.24"))
    r = compare(m, _env(tt_metal_version="0.80.0", vllm_version="0.20.0"))
    assert not r.compatible and r.forceable and not r.has_fatal
    fields = {i.field for i in r.issues}
    assert "platform.ttnn" in fields and "runtime.vllm" in fields


def test_compare_v4_arch_still_fatal():
    m = _v4_manifest(platform=Platform(ttnn=">=0.72,<0.76"))
    r = compare(m, _env(arch="wormhole_b0"))
    assert r.has_fatal and any(i.field == "arch" and i.fatal for i in r.issues)


def test_compare_v4_dev_checkout_not_blocked():
    """A bare git sha for ttnn/vllm must not be reported out-of-range."""
    m = _v4_manifest(platform=Platform(ttnn=">=0.72,<0.76"), runtime=Runtime(version=">=0.24"))
    r = compare(m, _env(tt_metal_version="deadbeef", vllm_version=None))
    assert r.compatible and not r.issues


# --------------------------------------------------------------------- render
def test_render_composes_launch_command():
    m = _v4_manifest(
        resources=Resources(max_model_len=131072, max_num_seqs=8, block_size=64,
                             trace_region_bytes=1500000000),
        capabilities=Capabilities(tool_parser="poolside_v1", reasoning_parser="poolside_v1"),
        env={"MESH_DEVICE": "P150"},
    )
    md = bundles.render_vllm_metadata(m)
    assert md["arch"] == "LagunaForCausalLM"
    assert md["main_class"] == "ttl.gen:LagunaForCausalLM"
    assert md["hf_weights"] == "poolside/Laguna-XS-2.1"
    cmd = md["launch"]["default"]["command"]
    assert cmd[:4] == ["python3", "server_example_tt.py", "--model", "poolside/Laguna-XS-2.1"]
    for flag, val in [("--max_model_len", "131072"), ("--max_num_seqs", "8"),
                      ("--block_size", "64"), ("--trace_region_size", "1500000000"),
                      ("--tool_parser", "poolside_v1"), ("--reasoning_parser", "poolside_v1")]:
        assert cmd[cmd.index(flag) + 1] == val
    env = md["launch"]["default"]["env"]
    assert env["VLLM_USE_V1"] == "1" and env["MESH_DEVICE"] == "P150"


def test_render_extra_args_appended_and_override_replaces():
    m = _v4_manifest(resources=Resources(
        max_num_seqs=8, extra_args=["--enable-prefix-caching"],
        command_override={"blackhole-4card": ["python3", "bh.py", "--fast"]},
    ))
    md = bundles.render_vllm_metadata(m)
    assert md["launch"]["default"]["command"][-1] == "--enable-prefix-caching"
    assert md["launch"]["blackhole-4card"]["command"] == ["python3", "bh.py", "--fast"]


def test_render_requires_entrypoint():
    m = _v4_manifest()
    m.entrypoint = None
    with pytest.raises(ValueError, match="entrypoint"):
        bundles.render_vllm_metadata(m)


# --------------------------------------------------------------------- CLI round-trip
def _fake_hub(monkeypatch, *, tt_metal="0.73.0", vllm="0.24.1"):
    """Redirect hub I/O to a local remote dir; deterministic device + versions."""
    import shutil
    remotes = {}

    def push_folder(repo_id, staged, commit_message=""):
        dst = remotes[repo_id] = staged.parent / f"remote__{bundles.model_key(repo_id)}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(staged, dst)

    def download_bundle(repo_id, revision, dest):
        shutil.copytree(remotes[repo_id], dest, dirs_exist_ok=True)
        from pathlib import Path
        return Path(dest)

    monkeypatch.setattr(hub, "create_repo", lambda *a, **k: None)
    monkeypatch.setattr(hub, "set_visibility", lambda *a, **k: None)
    monkeypatch.setattr(hub, "tag_repo", lambda *a, **k: None)
    monkeypatch.setattr(hub, "push_folder", push_folder)
    monkeypatch.setattr(hub, "download_bundle", download_bundle)
    monkeypatch.setattr(metal, "detect_device",
                        lambda arch_override=None: DeviceInfo(arch="blackhole", device_count=4, source="test"))
    monkeypatch.setattr(metal, "resolve_version", lambda: tt_metal)
    monkeypatch.setattr(metal, "_vllm_version", lambda: vllm)
    # Silence the toolchain warning path (no real tt-metal/vLLM in the test env).
    monkeypatch.setattr(toolchain, "check_toolchain", lambda: toolchain.ToolchainReport(components=[]))
    return remotes  # {repo_id: published-folder Path}


def _write_v4_manifest_file(tmp_path, **extra):
    doc = {
        "platform": {"ttnn": ">=0.72,<0.76"},
        "runtime": {"kind": "vllm", "version": ">=0.24"},
        "target": "p150x4",
        "mesh": {"devices": 4, "topology": "1x4", "fabric": "FABRIC_1D_RING"},
        "entrypoint": {"class": "ttl.gen:LagunaForCausalLM", "arch_name": "LagunaForCausalLM"},
        "weights": {"repo": "poolside/Laguna-XS-2.1"},
        "resources": {"max_num_seqs": 8},
        "capabilities": {"tool_parser": "poolside_v1"},
        "env": {"MESH_DEVICE": "P150"},
    }
    doc.update(extra)
    p = tmp_path / "laguna.json"
    p.write_text(json.dumps(doc))
    return p


def test_v4_push_pull_serve_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    remotes = _fake_hub(monkeypatch)

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "generator_vllm.py").write_text("# adapter code\n")
    mp = _write_v4_manifest_file(tmp_path)

    push = runner.invoke(cli.app, ["push", "acme/laguna", "--private", "--backend", "vllm",
                                   "--manifest", str(mp), "--bundle-dir", str(adapter)])
    assert push.exit_code == 0, push.output

    # The published bundle ships NO vllm_metadata.json — it's rendered on pull.
    import pathlib
    remote = remotes["acme/laguna"]
    assert not (remote / "vllm_bundle" / bundles.VLLM_METADATA_NAME).exists()
    assert (remote / "vllm_bundle" / "generator_vllm.py").exists()  # adapter code shipped
    assert json.loads((remote / "tt_kernel_manifest.json").read_text())["schema_version"] == "4"

    pull = runner.invoke(cli.app, ["pull", "acme/laguna", "--arch", "blackhole"])
    assert pull.exit_code == 0, pull.output
    assert "rendered vllm_metadata.json" in pull.output

    rendered = pathlib.Path(localdb.get("acme/laguna")["bundle_path"]) / bundles.VLLM_METADATA_NAME
    assert rendered.exists()
    md = json.loads(rendered.read_text())
    assert md["main_class"] == "ttl.gen:LagunaForCausalLM"

    serve = runner.invoke(cli.app, ["serve", "acme/laguna", "--print", "--local-only"])
    assert serve.exit_code == 0, serve.output
    assert "server_example_tt.py" in serve.output
    assert "--max_num_seqs 8" in serve.output and "--tool_parser poolside_v1" in serve.output
    assert "MESH_DEVICE=P150" in serve.output


def test_v4_push_builtin_entrypoint_no_bundle_dir(monkeypatch, tmp_path):
    """An entrypoint that references a tt-metal built-in needs no shipped code folder."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    _fake_hub(monkeypatch)
    mp = _write_v4_manifest_file(tmp_path)

    push = runner.invoke(cli.app, ["push", "acme/builtin", "--private", "--backend", "vllm",
                                   "--manifest", str(mp)])
    assert push.exit_code == 0, push.output
    pull = runner.invoke(cli.app, ["pull", "acme/builtin", "--arch", "blackhole"])
    assert pull.exit_code == 0, pull.output
    import pathlib
    rendered = pathlib.Path(localdb.get("acme/builtin")["bundle_path"]) / bundles.VLLM_METADATA_NAME
    assert rendered.exists()


def test_v4_pull_out_of_range_blocks_then_forces(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    _fake_hub(monkeypatch, tt_metal="0.80.0", vllm="0.20.0")  # both out of the declared range
    mp = _write_v4_manifest_file(tmp_path)
    runner.invoke(cli.app, ["push", "acme/oor", "--private", "--backend", "vllm", "--manifest", str(mp)])

    blocked = runner.invoke(cli.app, ["pull", "acme/oor", "--arch", "blackhole"])
    assert blocked.exit_code == 1
    assert "--force" in blocked.output

    forced = runner.invoke(cli.app, ["pull", "acme/oor", "--arch", "blackhole", "--force"])
    assert forced.exit_code == 0, forced.output


def test_search_target_and_arch_tags(monkeypatch):
    captured = {}

    class _M:
        def __init__(self, i):
            self.id, self.private, self.downloads, self.last_modified = i, False, 1, ""

    class _Api:
        @staticmethod
        def list_models(filter=None, search=None, limit=50):
            captured["filter"] = filter
            return [_M("acme/laguna")]

    monkeypatch.setattr(hub, "_api", lambda: _Api())
    res = runner.invoke(cli.app, ["search", "laguna", "--arch", "Blackhole", "--target", "P150x4"])
    assert res.exit_code == 0, res.output
    assert captured["filter"] == ["tt-kernel-cache", "blackhole", "p150x4"]  # ANDed, lowercased


def test_doctor_reports_bundle_ranges(monkeypatch):
    m = _v4_manifest(platform=Platform(ttnn=">=0.72,<0.76"), runtime=Runtime(version=">=0.24"),
                     target="p150x4")
    monkeypatch.setattr(hub, "fetch_manifest", lambda rid, rev: m)
    monkeypatch.setattr(metal, "detect_device",
                        lambda arch_override=None: DeviceInfo(arch="blackhole", device_count=4, source="test"))
    monkeypatch.setattr(metal, "resolve_version", lambda: "0.80.0")  # out of range
    monkeypatch.setattr(metal, "_vllm_version", lambda: "0.24.1")    # in range
    monkeypatch.setattr(toolchain, "check_toolchain", lambda: toolchain.ToolchainReport(components=[]))

    res = runner.invoke(cli.app, ["doctor", "acme/laguna", "--arch", "blackhole"])
    assert res.exit_code == 0, res.output
    assert "Bundle requirements" in res.output
    assert "require >=0.72,<0.76, installed 0.80.0" in res.output  # flagged out of range
    assert "target: p150x4" in res.output
