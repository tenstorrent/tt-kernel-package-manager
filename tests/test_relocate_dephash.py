# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for relocate_dephashes — make a pulled cache produce JIT hits at the new path.

tt-metal records absolute dependency paths in .dephash and recompiles if a listed path
can't be read; the kernel's generated sources live inside the build_key subtree, so their
paths must be rewritten from the producer's cache dir to the install dir on pull.
"""

from tt_kernel import cache


def test_rewrites_in_cache_paths_keeps_tree_paths(tmp_path):
    bk = 5052757421748276575
    out_root = cache.resolve_out_root(str(tmp_path))     # tmp/tt-metal-cache (no trailing slash)
    target = cache.build_key_path(out_root, bk)          # tmp/tt-metal-cache<bk>

    # a kernel variant's dephash, as written by the producer at a DIFFERENT cache dir
    old_root = "/home/someone/.cache/tt-metal-cache/%d" % bk
    dephash = target / "kernels" / "layernorm" / "abc" / "trisc0" / "trisck.o.dephash"
    dephash.parent.mkdir(parents=True)
    dephash.write_text(
        f'"{old_root}/kernels/layernorm/abc/trisc0/../chlkc_descriptors.h"\t111\n'
        f'"/home/someone/dispatch/tt-metal/tt_metal/hw/inc/firmware_common.h"\t222\n'
    )

    changed = cache.relocate_dephashes(target, bk)
    assert changed == 1

    text = dephash.read_text()
    assert str(target) in text                       # in-cache path rewritten to install dir
    assert old_root not in text                       # producer prefix gone
    assert "/dispatch/tt-metal/tt_metal/hw/inc/firmware_common.h" in text  # tree dep untouched
    assert "\t111" in text and "\t222" in text        # hashes preserved


def test_noop_when_already_local(tmp_path):
    bk = 123
    out_root = cache.resolve_out_root(str(tmp_path))
    target = cache.build_key_path(out_root, bk)
    dephash = target / "kernels" / "k" / "v" / "trisc0" / "trisck.o.dephash"
    dephash.parent.mkdir(parents=True)
    # already points at the install dir -> nothing to change
    dephash.write_text(f'"{target}/kernels/k/v/trisc0/../chlkc.h"\t9\n')
    assert cache.relocate_dephashes(target, bk) == 0


def test_handles_no_dephashes(tmp_path):
    bk = 7
    target = cache.build_key_path(cache.resolve_out_root(str(tmp_path)), bk)
    (target / "kernels").mkdir(parents=True)
    assert cache.relocate_dephashes(target, bk) == 0
