"""Parsing of tt-smi snapshot JSON and arch normalisation."""

from tt_kernel import device


def test_normalize_arch_aliases():
    assert device.normalize_arch("Blackhole") == "blackhole"
    assert device.normalize_arch("WH") == "wormhole_b0"
    assert device.normalize_arch("wormhole_b0") == "wormhole_b0"
    assert device.normalize_arch(None) is None


def test_parse_snapshot_basic():
    snap = {
        "device_info": [
            {"board_info": {"board_type": "blackhole", "harvesting": "0x0"}},
            {"board_info": {"board_type": "blackhole", "harvesting": "0x0"}},
        ]
    }
    info = device._parse_snapshot(snap)
    assert info.arch == "blackhole"
    assert info.device_count == 2
    assert info.harvesting_mask == 0
    assert info.source == "tt-smi"


def test_parse_snapshot_integer_harvesting():
    snap = {"device_info": [{"board_type": "wormhole", "harvesting_mask": 6}]}
    info = device._parse_snapshot(snap)
    assert info.arch == "wormhole_b0"
    assert info.harvesting_mask == 6


def test_parse_snapshot_nested_mask():
    snap = {"devices": [{"arch": "blackhole", "harvesting": {"mask": "0x2"}}]}
    info = device._parse_snapshot(snap)
    assert info.arch == "blackhole"
    assert info.harvesting_mask == 2


def test_parse_snapshot_empty():
    assert device._parse_snapshot({}).arch is None


def test_detect_env_fallback(monkeypatch):
    monkeypatch.setattr(device, "_run_tt_smi_snapshot", lambda: None)
    monkeypatch.setenv("ARCH_NAME", "wormhole_b0")
    info = device.detect()
    assert info.arch == "wormhole_b0"
    assert info.source == "env"


def test_detect_flag_fallback(monkeypatch):
    monkeypatch.setattr(device, "_run_tt_smi_snapshot", lambda: None)
    monkeypatch.delenv("ARCH_NAME", raising=False)
    info = device.detect(arch_override="blackhole")
    assert info.arch == "blackhole"
    assert info.source == "flag"
