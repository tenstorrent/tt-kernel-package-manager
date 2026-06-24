# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2024 Tenstorrent USA, Inc.

"""build_key input assembly and compile-hash fingerprinting."""

from tt_kernel import metal


def test_compile_hash_empty_is_production_default(monkeypatch):
    for var in metal._COMPILE_HASH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    assert metal.compile_hash_fingerprint() == ""


def test_compile_hash_changes_with_debug_env(monkeypatch):
    for var in metal._COMPILE_HASH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TT_METAL_WATCHER", "1")
    fp = metal.compile_hash_fingerprint()
    assert fp != ""
    # Deterministic for the same env.
    assert fp == metal.compile_hash_fingerprint()


def test_build_key_inputs_defaults(monkeypatch):
    for var in metal._COMPILE_HASH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    inp = metal.build_key_inputs()
    assert inp.dispatch_core_type == "WORKER"
    assert inp.dispatch_core_axis == "ROW"
    assert inp.num_hw_cqs == 1
    assert inp.coordinate_virtualization_enabled is True
    assert inp.compile_hash_string == ""


def test_build_key_inputs_overrides():
    inp = metal.build_key_inputs(num_hw_cqs=2, harvesting_mask=4)
    assert inp.num_hw_cqs == 2
    assert inp.harvesting_mask == 4
