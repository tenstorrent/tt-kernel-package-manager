"""Compatibility comparison matrix — the correctness core."""

from tt_kernel.manifest import BuildKeyInputs, Manifest, Producer, compare
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
