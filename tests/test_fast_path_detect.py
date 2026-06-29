# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for fast-path bundle completeness detection (#6).

A baseline (trace-disabled) warmup omits the traced-decode / on-device-lm_head kernels a
DISPATCH_TRACE/DISPATCH_ONDEVICE_LMHEAD consumer needs. detect_fast_path_kernels flags
their presence so the manifest records it and pull can warn.
"""

from tt_kernel import cache


def _mk_subtree(tmp, kernel_groups):
    sub = tmp / "tt-metal-cache123" / "5"  # any build_key subtree shape
    for g in kernel_groups:
        (sub / "kernels" / g).mkdir(parents=True)
    return sub


def test_baseline_cache_has_no_fast_path_kernels(tmp_path):
    sub = _mk_subtree(tmp_path, ["layernorm", "bmm_large_block", "embeddings"])
    assert cache.detect_fast_path_kernels(sub) is False


def test_fast_path_cache_detected(tmp_path):
    sub = _mk_subtree(tmp_path, ["layernorm", "sdpa_flash_decode", "reader_decode_all"])
    assert cache.detect_fast_path_kernels(sub) is True


def test_argmax_lmhead_marker_alone_is_enough(tmp_path):
    sub = _mk_subtree(tmp_path, ["embeddings", "reader_argmax_tile_layout"])
    assert cache.detect_fast_path_kernels(sub) is True


def test_no_kernels_dir_is_false(tmp_path):
    sub = tmp_path / "tt-metal-cache123" / "5"
    sub.mkdir(parents=True)
    assert cache.detect_fast_path_kernels(sub) is False
