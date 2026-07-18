# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for the toolchain version-check (WS2). No hardware; all detection is faked."""

from typer.testing import CliRunner

from tt_kernel import cli, metal, toolchain
from tt_kernel.toolchain import _meets, _parse_version, check_toolchain

runner = CliRunner()


def test_parse_version_variants():
    assert _parse_version("v0.72.0-5-gabc") == (0, 72, 0)   # git-describe suffix dropped
    assert _parse_version("0.72.1.dev3") == (0, 72, 1)      # prerelease tail dropped
    assert _parse_version("1.1.3.dev0+light") == (1, 1, 3)  # build metadata dropped
    assert _parse_version("1.1.3") == (1, 1, 3)
    assert _parse_version("deadbeef") is None               # bare sha => uncomparable
    assert _parse_version(None) is None


def test_meets():
    assert _meets("0.72.0", "0.72.0") is True
    assert _meets("0.72.1.dev3", "0.72.0") is True
    assert _meets("0.71.9", "0.72.0") is False
    assert _meets("1.2", "1.1.3") is True
    assert _meets("gitsha", "0.1.0") is None  # unparseable -> unknown


def _fake_detection(monkeypatch, *, tt_metal):
    monkeypatch.setattr(metal, "resolve_version", lambda: tt_metal)
    monkeypatch.setattr(toolchain, "_dist_version", lambda dists: None)
    monkeypatch.setattr(toolchain, "_spec_present", lambda *names: True)


def test_all_adequate(monkeypatch):
    _fake_detection(monkeypatch, tt_metal="0.72.0")
    report = check_toolchain()
    assert report.ok and not report.problems


def test_prototype_deps_not_checked(monkeypatch):
    # tt-api and tt-lang are earlier-prototype leftovers: never reported as required.
    _fake_detection(monkeypatch, tt_metal="0.72.0")
    names = {c.name for c in check_toolchain().components}
    assert "tt-api" not in names and "tt-lang" not in names
    assert "tt-api" not in toolchain.LOCK and "tt-lang" not in toolchain.LOCK
    # The default serving stack is exactly tt-metal + vLLM.
    assert names == {"tt-metal", "vllm"}


def test_old_version_flagged_not_fatal(monkeypatch):
    _fake_detection(monkeypatch, tt_metal="0.71.9")  # below the 0.72.0 floor
    report = check_toolchain()
    assert not report.ok
    probs = {c.name for c in report.problems}
    assert probs == {"tt-metal"}
    assert "older than required" in next(c for c in report.problems).message


def test_missing_component(monkeypatch):
    monkeypatch.setattr(metal, "resolve_version", lambda: None)
    monkeypatch.setattr(toolchain, "_dist_version", lambda dists: None)
    # ttnn absent => tt-metal not found; vllm + plugin present.
    monkeypatch.setattr(toolchain, "_spec_present", lambda *names: "ttnn" not in names)
    report = check_toolchain()
    tt_metal = next(c for c in report.components if c.name == "tt-metal")
    assert not tt_metal.found and not tt_metal.adequate and "not found" in tt_metal.message


class _Dev:
    arch = "blackhole"
    device_count = 1
    source = "tt-smi"


def test_doctor_exit_nonzero_when_inadequate(monkeypatch):
    _fake_detection(monkeypatch, tt_metal="0.71.9")
    monkeypatch.setattr(metal, "detect_device", lambda *a, **k: _Dev())
    res = runner.invoke(cli.app, ["doctor"])
    assert res.exit_code == 1, res.output
    assert "tt-metal" in res.output


def test_doctor_ok(monkeypatch):
    _fake_detection(monkeypatch, tt_metal="0.72.0")
    monkeypatch.setattr(metal, "detect_device", lambda *a, **k: _Dev())
    res = runner.invoke(cli.app, ["doctor"])
    assert res.exit_code == 0, res.output
    assert "toolchain adequate" in res.output
