# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for the tt-metal instance registry + selection (instances.py) and its wiring into
pull (pin) / serve (replay). No hardware, no network, no real subprocess probes.
"""

import json

import pytest
from typer.testing import CliRunner

from tt_kernel import bundles, cli, hub, instances, localdb, metal, runtime, toolchain
from tt_kernel.device import DeviceInfo
from tt_kernel.instances import Instance, InstanceVersions

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch, tmp_path):
    """Redirect the registry file to a tmp XDG_CONFIG_HOME for every test."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


# --------------------------------------------------------------------- registry file
def test_registry_add_remove_roundtrip():
    instances.add_instance("m073", "/opt/tt/073/bin/python", tt_metal_home="/opt/tt/073",
                           env={"LD_LIBRARY_PATH": "/l"})
    instances.add_instance("m080", "/opt/tt/080/bin/python", tt_metal_home="/opt/tt/080")
    names = {i.name for i in instances.registry_instances()}
    assert names == {"m073", "m080"}
    got = next(i for i in instances.registry_instances() if i.name == "m073")
    assert got.tt_metal_home == "/opt/tt/073" and got.env == {"LD_LIBRARY_PATH": "/l"}
    assert got.activation_env()["TT_METAL_HOME"] == "/opt/tt/073"

    assert instances.remove_instance("m073") is True
    assert {i.name for i in instances.registry_instances()} == {"m080"}
    assert instances.remove_instance("m073") is False  # already gone


def test_add_replaces_same_name():
    instances.add_instance("x", "/a/python")
    instances.add_instance("x", "/b/python")
    regs = [i for i in instances.registry_instances() if i.name == "x"]
    assert len(regs) == 1 and regs[0].python == "/b/python"


# --------------------------------------------------------------------- scan
def _make_checkout(root, name, with_python=True):
    co = root / name
    (co / "tt_metal" / "hw" / "inc").mkdir(parents=True)
    if with_python:
        bindir = co / "build" / "python_env" / "bin"
        bindir.mkdir(parents=True)
        (bindir / "python").write_text("#!/bin/sh\n")
    return co


def test_scan_finds_checkouts_and_flags_unlaunchable(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    _make_checkout(root, "metal-good", with_python=True)
    _make_checkout(root, "metal-nobuild", with_python=False)
    pairs = dict((p[0].split("/")[-1], p[1]) for p in
                 ((str(h), py) for h, py in instances.scan_checkouts([str(root)])))
    assert pairs["metal-good"] is not None
    assert pairs["metal-nobuild"] is None
    # Only the launchable one becomes a selectable instance.
    insts = instances.scan_instances([str(root)])
    assert [i.name for i in insts] == ["scan:metal-good"]
    assert insts[0].tt_metal_home.endswith("metal-good")


# --------------------------------------------------------------------- dedup / precedence
def test_all_instances_dedup_and_precedence(monkeypatch):
    active = Instance(name="active", python="/venv/bin/python", source="active")
    monkeypatch.setattr(instances, "active_instance", lambda: active)
    # A registry entry and a scan entry that resolve to the SAME python -> registry wins.
    monkeypatch.setattr(instances, "registry_instances",
                        lambda: [Instance(name="reg", python="/shared/python", source="registry")])
    monkeypatch.setattr(instances, "scan_instances",
                        lambda roots=None: [Instance(name="scan:x", python="/shared/python", source="scan")])
    alls = instances.all_instances(roots=[])
    assert [i.name for i in alls] == ["active", "reg"]  # scan dupe dropped, active always kept


# --------------------------------------------------------------------- select
def _insts(monkeypatch, table):
    """table: {name: (python, ttnn, vllm, plugin)} -> wire all_instances + probe_versions."""
    objs = [Instance(name=n, python=p, source="registry") for n, (p, *_v) in table.items()]
    monkeypatch.setattr(instances, "all_instances", lambda roots=None: objs)
    vers = {n: InstanceVersions(t, vl, pl) for n, (_p, t, vl, pl) in table.items()}
    monkeypatch.setattr(instances, "probe_versions",
                        lambda inst, use_cache=True: vers[inst.name])


def test_select_newest_satisfying(monkeypatch):
    _insts(monkeypatch, {
        "old": ("/a", "0.71.0", "0.24.1", "0.3.0"),   # ttnn too old
        "good": ("/b", "0.73.0", "0.24.1", "0.3.2"),  # satisfies
        "newer_good": ("/c", "0.75.0", "0.26.0", "0.3.9"),  # satisfies, newer
        "toonew": ("/d", "0.80.0", "0.26.0", "0.3.9"),  # ttnn out of upper bound
    })
    res = instances.select(ttnn=">=0.72,<0.76", vllm=">=0.24", plugin=">=0.3,<0.4")
    assert res.chosen.name == "newer_good"


def test_select_excludes_on_vllm_or_plugin(monkeypatch):
    _insts(monkeypatch, {
        "bad_vllm": ("/a", "0.73.0", "0.20.0", "0.3.2"),    # vllm too old
        "bad_plugin": ("/b", "0.73.0", "0.24.1", "0.5.0"),  # plugin out of range
    })
    res = instances.select(ttnn=">=0.72,<0.76", vllm=">=0.24", plugin=">=0.3,<0.4")
    assert res.chosen is None
    assert all(not c.satisfies for c in res.candidates)


def test_select_gitsha_assumed_ok(monkeypatch):
    _insts(monkeypatch, {"dev": ("/a", "deadbeef", None, None)})
    res = instances.select(ttnn=">=0.72,<0.76", vllm=">=0.24", plugin=">=0.3,<0.4")
    assert res.chosen.name == "dev"  # unparseable/None versions ⇒ assume OK


# --------------------------------------------------------------------- CLI: instances
def test_cli_instances_add_list_remove(monkeypatch):
    monkeypatch.setattr(instances, "scan_instances", lambda roots=None: [])
    monkeypatch.setattr(instances, "scan_checkouts", lambda roots=None: [])
    monkeypatch.setattr(instances, "probe_versions",
                        lambda inst, use_cache=True: InstanceVersions("0.73.0", "0.24.1", "0.3.2"))
    add = runner.invoke(cli.app, ["instances", "add", "--name", "m073",
                                  "--python", "/opt/tt/073/bin/python", "--env", "LD_LIBRARY_PATH=/l"])
    assert add.exit_code == 0, add.output
    lst = runner.invoke(cli.app, ["instances", "list"])
    assert lst.exit_code == 0 and "m073" in lst.output and "ttnn=0.73.0" in lst.output
    rm = runner.invoke(cli.app, ["instances", "remove", "m073"])
    assert rm.exit_code == 0 and "Removed" in rm.output


# --------------------------------------------------------------------- pull pins / serve replays
def _v4_manifest_file(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({
        "platform": {"ttnn": ">=0.72,<0.76"},
        "runtime": {"kind": "vllm", "version": ">=0.24", "plugin_version": ">=0.3,<0.4"},
        "entrypoint": {"class": "ttl:LagunaForCausalLM", "arch_name": "LagunaForCausalLM"},
        "weights": {"repo": "p/L"}, "resources": {"max_num_seqs": 8}, "env": {"MESH_DEVICE": "P150"},
    }))
    return p


def _fake_hub(monkeypatch, tmp_path):
    import shutil
    remote = tmp_path / "remote"

    def push_folder(rid, staged, commit_message=""):
        if remote.exists():
            shutil.rmtree(remote)
        shutil.copytree(staged, remote)

    def download_bundle(rid, revision, dest):
        shutil.copytree(remote, dest, dirs_exist_ok=True)
        from pathlib import Path
        return Path(dest)

    monkeypatch.setattr(hub, "create_repo", lambda *a, **k: None)
    monkeypatch.setattr(hub, "set_visibility", lambda *a, **k: None)
    monkeypatch.setattr(hub, "tag_repo", lambda *a, **k: None)
    monkeypatch.setattr(hub, "push_folder", push_folder)
    monkeypatch.setattr(hub, "download_bundle", download_bundle)
    monkeypatch.setattr(metal, "detect_device",
                        lambda arch_override=None: DeviceInfo(arch="blackhole", device_count=4, source="test"))
    monkeypatch.setattr(toolchain, "check_toolchain", lambda: toolchain.ToolchainReport(components=[]))


def test_pull_pins_newest_satisfying_then_serve_replays(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    _fake_hub(monkeypatch, tmp_path)
    # Two candidates; m073 satisfies, m080 is too new. Give m073 an existing python so serve
    # uses the pin directly (no re-resolve).
    py073 = tmp_path / "py073"
    py073.write_text("#!/bin/sh\n")
    table = {
        "m073": Instance(name="m073", python=str(py073), tt_metal_home="/opt/073",
                         env={"LD_LIBRARY_PATH": "/l73"}, source="registry"),
        "m080": Instance(name="m080", python="/opt/080/python", source="registry"),
    }
    vers = {"m073": InstanceVersions("0.73.0", "0.24.1", "0.3.2"),
            "m080": InstanceVersions("0.80.0", "0.26.0", "0.3.9")}
    monkeypatch.setattr(instances, "all_instances", lambda roots=None: list(table.values()))
    monkeypatch.setattr(instances, "probe_versions", lambda inst, use_cache=True: vers[inst.name])

    mp = _v4_manifest_file(tmp_path)
    assert runner.invoke(cli.app, ["push", "acme/l", "--private", "--backend", "vllm",
                                   "--manifest", str(mp)]).exit_code == 0
    pull = runner.invoke(cli.app, ["pull", "acme/l", "--arch", "blackhole"])
    assert pull.exit_code == 0, pull.output
    assert "instance: m073" in pull.output

    entry = localdb.get("acme/l")
    assert entry["instance_name"] == "m073" and entry["instance_python"] == str(py073)
    assert entry["platform_ttnn"] == ">=0.72,<0.76"
    assert entry["runtime_plugin_version"] == ">=0.3,<0.4"

    serve = runner.invoke(cli.app, ["serve", "acme/l", "--print", "--local-only"])
    assert serve.exit_code == 0, serve.output
    assert serve.output.count(str(py073)) >= 1          # launched under the pinned interpreter
    assert "TT_METAL_HOME=/opt/073" in serve.output      # activation env threaded in
    assert "LD_LIBRARY_PATH=/l73" in serve.output
    assert "MESH_DEVICE=P150" in serve.output            # bundle launch env still applied


def test_pull_blocks_when_none_in_range_then_force(monkeypatch, tmp_path):
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    _fake_hub(monkeypatch, tmp_path)
    active = Instance(name="active", python="/venv/bin/python", source="active")
    monkeypatch.setattr(instances, "all_instances", lambda roots=None: [active])
    monkeypatch.setattr(instances, "active_instance", lambda: active)
    monkeypatch.setattr(instances, "probe_versions",
                        lambda inst, use_cache=True: InstanceVersions("0.80.0", "0.20.0", None))

    mp = _v4_manifest_file(tmp_path)
    runner.invoke(cli.app, ["push", "acme/oor", "--private", "--backend", "vllm", "--manifest", str(mp)])
    blocked = runner.invoke(cli.app, ["pull", "acme/oor", "--arch", "blackhole"])
    assert blocked.exit_code == 1 and "--force" in blocked.output
    forced = runner.invoke(cli.app, ["pull", "acme/oor", "--arch", "blackhole", "--force"])
    assert forced.exit_code == 0, forced.output
    assert localdb.get("acme/oor")["instance_name"] == "active"  # forced fallback to active


def test_serve_reresolves_when_pin_missing(monkeypatch, tmp_path):
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    _fake_hub(monkeypatch, tmp_path)
    # Pin an instance whose python does not exist -> serve must re-resolve from stored ranges.
    good_py = tmp_path / "good"
    good_py.write_text("#!/bin/sh\n")
    replacement = Instance(name="m074", python=str(good_py), tt_metal_home="/opt/074", source="registry")
    monkeypatch.setattr(instances, "all_instances", lambda roots=None: [replacement])
    monkeypatch.setattr(instances, "probe_versions",
                        lambda inst, use_cache=True: InstanceVersions("0.74.0", "0.24.1", "0.3.5"))

    # Seed a localdb entry with a dead pin + the ranges.
    dest = tmp_path / "bundles" / "acme__x"
    dest.mkdir(parents=True)
    bundles.write_vllm_metadata(dest, {
        "arch": "B", "main_class": "m:C", "hf_weights": "p/w",
        "launch": {"default": {"command": ["python3", "server_example_tt.py", "--model", "p/w"], "env": {}}},
    })
    localdb.record("acme/x", {
        "name": "x", "backend": "vllm", "build_key": None, "bundle_path": str(dest),
        "instance_name": "gone", "instance_python": "/does/not/exist/python",
        "platform_ttnn": ">=0.72,<0.76", "runtime_version": ">=0.24",
        "runtime_plugin_version": ">=0.3,<0.4", "installed_at": "now",
    })
    serve = runner.invoke(cli.app, ["serve", "acme/x", "--print", "--local-only"])
    assert serve.exit_code == 0, serve.output
    assert "re-resolving" in serve.output and "m074" in serve.output
    assert str(good_py) in serve.output and "TT_METAL_HOME=/opt/074" in serve.output


def test_v3_bundle_no_selection(monkeypatch, tmp_path):
    """A bundle with no platform/runtime ranges must not trigger instance selection."""
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    _fake_hub(monkeypatch, tmp_path)
    monkeypatch.setattr(metal, "resolve_version", lambda: "0.73.0")
    monkeypatch.setattr(metal, "_vllm_version", lambda: "0.24.1")
    monkeypatch.setattr(metal, "_vllm_plugin_version", lambda: None)
    # If selection were invoked, this would blow up (all_instances not patched to anything real).
    called = {"n": 0}
    monkeypatch.setattr(instances, "select",
                        lambda **k: called.__setitem__("n", called["n"] + 1) or instances.SelectionResult(None, [], "x"))

    # Author a legacy verbatim vLLM bundle (no v4 blocks) via --bundle-dir.
    folder = tmp_path / "b"
    folder.mkdir()
    bundles.write_vllm_metadata(folder, {
        "arch": "B", "main_class": "m:C", "hf_weights": "p/w",
        "launch": {"default": {"command": ["python3", "server_example_tt.py", "--model", "p/w"], "env": {}}},
    })
    runner.invoke(cli.app, ["push", "acme/legacy", "--private", "--backend", "vllm", "--bundle-dir", str(folder)])
    pull = runner.invoke(cli.app, ["pull", "acme/legacy", "--arch", "blackhole"])
    assert pull.exit_code == 0, pull.output
    assert called["n"] == 0  # no instance selection for a range-less bundle
    assert localdb.get("acme/legacy")["instance_name"] is None
