# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Compatibility comparison matrix — the correctness core."""

from tt_kernel.manifest import (
    BuildKeyInputs,
    Manifest,
    Producer,
    RunnerPayload,
    WeightsRef,
    compare,
    runner_version_advisory,
)
from tt_kernel.metal import LocalEnv


def _manifest(**overrides) -> Manifest:
    base = dict(
        name="m",
        tt_metal_version="v1.0.0-abc",
        arch="blackhole",
        device_count=1,
        build_key=12345,
        build_key_inputs=BuildKeyInputs(),
        producer=Producer(tt_kernel_version="0.1.0", created_at="now"),
    )
    bki = overrides.pop("build_key_inputs", None)
    base.update(overrides)
    if bki is not None:
        base["build_key_inputs"] = bki
    return Manifest(**base)


def _env(**overrides) -> LocalEnv:
    base = dict(
        tt_metal_version="v1.0.0-abc",
        arch="blackhole",
        device_count=1,
        harvesting_mask=0,
        build_key=None,
    )
    base.update(overrides)
    return LocalEnv(**base)


def test_fully_compatible():
    report = compare(_manifest(), _env())
    assert report.compatible
    assert not report.issues


def test_arch_mismatch_is_fatal():
    report = compare(_manifest(arch="blackhole"), _env(arch="wormhole_b0"))
    assert not report.compatible
    assert report.has_fatal
    assert not report.forceable
    assert report.issues[0].field == "arch"


# --------------------------------------------------------- kernels-less (vLLM) bundles
def test_kernels_less_skips_cache_gates():
    # build_key=None + a tt_metal_version mismatch that WOULD block a kernel bundle.
    m = _manifest(build_key=None, tt_metal_version="v9.9.9")
    report = compare(m, _env(tt_metal_version="v1.0.0-abc"))
    # No cache => tt_metal_version is not a gate here; only arch/device_count are.
    assert report.compatible and not report.issues


def test_kernels_less_arch_still_fatal():
    report = compare(_manifest(build_key=None), _env(arch="wormhole_b0"))
    assert report.has_fatal


def test_kernels_less_device_count_warns():
    report = compare(_manifest(build_key=None, device_count=2), _env(device_count=1))
    assert report.forceable and not report.has_fatal


def test_version_mismatch_is_forceable():
    report = compare(_manifest(tt_metal_version="v1"), _env(tt_metal_version="v2"))
    assert not report.compatible
    assert not report.has_fatal
    assert report.forceable
    assert any(i.field == "tt_metal_version" for i in report.issues)


def test_device_count_mismatch_is_forceable():
    report = compare(_manifest(device_count=1), _env(device_count=2))
    assert report.forceable
    assert any(i.field == "device_count" for i in report.issues)


def test_harvesting_ignored_when_virtualization_enabled():
    m = _manifest(build_key_inputs=BuildKeyInputs(coordinate_virtualization_enabled=True, harvesting_mask=1))
    report = compare(m, _env(harvesting_mask=99))
    assert report.compatible  # harvesting excluded from key


def test_harvesting_checked_when_virtualization_disabled():
    m = _manifest(build_key_inputs=BuildKeyInputs(coordinate_virtualization_enabled=False, harvesting_mask=1))
    report = compare(m, _env(harvesting_mask=99))
    assert report.forceable
    assert any(i.field == "harvesting_mask" for i in report.issues)


def test_probed_build_key_mismatch_is_forceable():
    report = compare(_manifest(build_key=111), _env(build_key=222))
    assert report.forceable
    assert any(i.field == "build_key" for i in report.issues)


def test_manifest_json_roundtrip():
    m = _manifest()
    assert Manifest.from_json(m.to_json()) == m


def test_unknown_local_fields_do_not_block():
    # When detection fails (arch/version unknown), we don't fabricate mismatches.
    report = compare(_manifest(), _env(arch=None, tt_metal_version=None, device_count=0))
    assert report.compatible


# --------------------------------------------------------------------------- schema gate + runner modes

import pytest

_LEGACY_JSON = """
{
  "schema_version": "1",
  "name": "legacy",
  "tt_metal_version": "v1.0.0-abc",
  "arch": "blackhole",
  "device_count": 1,
  "build_key": 12345,
  "kernel_count": 0,
  "files": [],
  "producer": {"tt_kernel_version": "0.1.0", "created_at": "now"}
}
"""


def test_legacy_schema_is_rejected():
    # tt-kernel is pre-release: exactly one supported schema, no silent half-reads.
    with pytest.raises(ValueError, match="schema_version"):
        Manifest.from_json(_LEGACY_JSON)


def test_manifest_roundtrip_with_payload():
    m = _manifest(
        runner=RunnerPayload(spec="pkg.mod:Runner", wheels=["r-0.1-py3-none-any.whl"],
                             entry_point="qwen36"),
        weights=WeightsRef(repo_id="org/model", revision="main"),
    )
    assert Manifest.from_json(m.to_json()) == m


def test_packaged_runner_is_packaged():
    assert RunnerPayload(spec="pkg.mod:Runner", wheels=["r.whl"]).is_packaged is True


def test_reference_runner_is_not_packaged_and_roundtrips():
    r = RunnerPayload(spec="pkg.mod:Runner", source="pip:fake-runner")
    assert r.is_packaged is False
    m = _manifest(runner=r)
    assert Manifest.from_json(m.to_json()) == m


def test_compare_verdict_ignores_runtime_payload():
    # The kernel verdict must be byte-identical whether or not runner/weights are present.
    plain = compare(_manifest(), _env())
    with_payload = compare(
        _manifest(runner=RunnerPayload(spec="pkg:Runner"), weights=WeightsRef(repo_id="org/m")),
        _env(),
    )
    assert plain == with_payload


def test_advisory_none_without_payload():
    assert runner_version_advisory(_manifest(), _env(tt_metal_version="v2")) is None


def test_advisory_none_when_versions_match():
    m = _manifest(runner=RunnerPayload(spec="pkg:Runner"))
    assert runner_version_advisory(m, _env()) is None


def test_advisory_nonfatal_on_mismatch():
    m = _manifest(tt_metal_version="v1", weights=WeightsRef(repo_id="org/m"))
    adv = runner_version_advisory(m, _env(tt_metal_version="v2"))
    assert adv is not None
    assert adv.fatal is False
    assert adv.field == "runner_tt_metal_version"
