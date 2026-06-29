# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for cross-host tt-metal tree-dep relocation (#3 follow-up).

A .dephash lists in-cache generated sources (relocated by relocate_dephashes) AND
tt-metal tree headers under the producer's tt-metal root. Pulled onto a host with a
different tt-metal path, the tree-dep prefix must be rewritten too or every kernel
recompiles. detect_cache_tt_metal_root records the producer root; relocate_tt_metal_tree
rewrites it to the consumer's.
"""

from tt_kernel import cache, metal


def _write_dephash(target, bk, producer_metal):
    dephash = target / "kernels" / "ln" / "abc" / "trisc0" / "trisck.o.dephash"
    dephash.parent.mkdir(parents=True)
    dephash.write_text(
        f'"{target}/kernels/ln/abc/trisc0/../chlkc_descriptors.h"\t111\n'
        f'"{producer_metal}/tt_metal/hw/inc/firmware_common.h"\t222\n'
        f'"{producer_metal}/runtime/sfpi/include/sfpi.h"\t333\n'
    )
    return dephash


def test_detect_cache_tt_metal_root_finds_external_prefix(tmp_path):
    bk = 5052757421748276575
    target = cache.build_key_path(cache.resolve_out_root(str(tmp_path)), bk)
    _write_dephash(target, bk, "/home/alice/dispatch/tt-metal")
    # In-cache paths are excluded; the external deps share the tt-metal root.
    assert cache.detect_cache_tt_metal_root(target) == "/home/alice/dispatch/tt-metal"


def test_relocate_tree_rewrites_producer_root(tmp_path):
    bk = 42
    target = cache.build_key_path(cache.resolve_out_root(str(tmp_path)), bk)
    dephash = _write_dephash(target, bk, "/home/alice/dispatch/tt-metal")

    changed = cache.relocate_tt_metal_tree(target, "/home/alice/dispatch/tt-metal", "/opt/tt-metal")
    assert changed == 1

    text = dephash.read_text()
    assert "/opt/tt-metal/tt_metal/hw/inc/firmware_common.h" in text
    assert "/opt/tt-metal/runtime/sfpi/include/sfpi.h" in text
    assert "/home/alice/dispatch/tt-metal" not in text   # producer prefix gone
    assert str(target) in text                            # in-cache path untouched
    assert "\t222" in text and "\t333" in text            # content hashes preserved


def test_relocate_tree_noop_when_roots_equal(tmp_path):
    bk = 7
    target = cache.build_key_path(cache.resolve_out_root(str(tmp_path)), bk)
    _write_dephash(target, bk, "/opt/tt-metal")
    assert cache.relocate_tt_metal_tree(target, "/opt/tt-metal", "/opt/tt-metal") == 0


def test_detect_returns_none_without_external_deps(tmp_path):
    bk = 9
    target = cache.build_key_path(cache.resolve_out_root(str(tmp_path)), bk)
    dephash = target / "kernels" / "k" / "v" / "trisc0" / "trisck.o.dephash"
    dephash.parent.mkdir(parents=True)
    dephash.write_text(f'"{target}/kernels/k/v/trisc0/../chlkc.h"\t9\n')  # only in-cache
    assert cache.detect_cache_tt_metal_root(target) is None


def test_detect_tt_metal_home_prefers_env(tmp_path, monkeypatch):
    metal_root = tmp_path / "my-tt-metal"
    metal_root.mkdir()
    monkeypatch.setenv("TT_METAL_HOME", str(metal_root))
    assert metal.detect_tt_metal_home() == str(metal_root)
