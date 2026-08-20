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


# ------------------------------------------- environment coherence (pip check)
_PIP_CHECK_REAL = (
    'opencv-python-headless 5.0.0.93 has requirement numpy>=2; python_version >= "3.9", '
    "but you have numpy 1.26.4.\n"
)


def _fake_pip(monkeypatch, stdout, returncode=1):
    import subprocess as sp

    class R:
        pass

    def run(*a, **k):
        r = R()
        r.returncode, r.stdout, r.stderr = returncode, stdout, ""
        return r

    monkeypatch.setattr(sp, "run", run)


def test_parses_the_real_pip_check_wording(monkeypatch):
    """Captured from the venv in the bug report. Note pip says "has requirement", not
    "requires" — matching only the latter silently found zero conflicts on an environment
    pip had just called broken."""
    _fake_pip(monkeypatch, _PIP_CHECK_REAL)
    conflicts = toolchain.check_environment()
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.package == "opencv-python-headless"
    assert c.requirement == 'numpy>=2; python_version >= "3.9"'
    assert c.installed == "numpy 1.26.4"
    assert "numpy>=2" in c.message


def test_parses_the_other_pip_wording(monkeypatch):
    _fake_pip(monkeypatch, "foo 1.0 requires bar>=2, but you have bar 1.0.\n")
    assert toolchain.check_environment()[0].package == "foo"


def test_requirement_containing_a_comma_is_not_truncated(monkeypatch):
    """"numpy>=1.24,<2" is one requirement. Splitting on the first comma would report it
    as "numpy>=1.24" and quietly understate the constraint."""
    _fake_pip(monkeypatch, "foo 1.0 has requirement numpy>=1.24,<2, but you have numpy 2.1.\n")
    c = toolchain.check_environment()[0]
    assert c.requirement == "numpy>=1.24,<2"


def test_missing_dependency_records_no_installed_version(monkeypatch):
    _fake_pip(monkeypatch, "foo 1.0 requires bar, but you have bar not installed.\n")
    c = toolchain.check_environment()[0]
    assert c.installed is None
    assert "not installed" in c.message


def test_clean_environment_reports_nothing(monkeypatch):
    _fake_pip(monkeypatch, "", returncode=0)
    assert toolchain.check_environment() == []


def test_unparseable_output_does_not_invent_conflicts(monkeypatch):
    _fake_pip(monkeypatch, "something entirely unexpected\n")
    assert toolchain.check_environment() == []


def test_pip_unavailable_is_not_a_conflict(monkeypatch):
    """An unavailable check is not a passing check, but it is also not a conflict we can
    name — so report nothing rather than a fabricated problem."""
    import subprocess as sp

    def boom(*a, **k):
        raise OSError("no pip")

    monkeypatch.setattr(sp, "run", boom)
    assert toolchain.check_environment() == []


def test_doctor_surfaces_a_conflict_without_claiming_plain_adequacy(monkeypatch):
    """The bug: pip printed a hard ERROR about numpy and `doctor` printed
    "✓ toolchain adequate" immediately after. Advisory, so still exit 0 — but never
    an unqualified claim of adequacy."""
    from tt_kernel import cli

    monkeypatch.setattr(toolchain, "check_environment", lambda *a, **k: [
        toolchain.EnvConflict("opencv-python-headless", "numpy>=2", "numpy 1.26.4")])
    res = runner.invoke(cli.app, ["doctor"])
    assert "opencv-python-headless" in res.output
    assert "environment conflict" in res.output
    if res.exit_code == 0:
        assert "✓ toolchain adequate —" in res.output
