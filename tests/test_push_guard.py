# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for the pre-push isolation guard (#2, proposal 3) in `cache.publish_warnings`."""

from tt_kernel import cache


def _mk_cache(tmp, *keys):
    """Create fake build_key subtrees (with a kernels/ dir) and return the out_root."""
    out = cache.resolve_out_root(str(tmp))
    for k in keys:
        (cache.build_key_path(out, k) / "kernels").mkdir(parents=True)
    return out


def test_no_warnings_for_isolated_explicit_cache(tmp_path):
    # One build_key, an explicit (non-default) cache dir => looks model-specific.
    out = _mk_cache(tmp_path, 111)
    assert cache.publish_warnings(out, 111, default_cache=False) == []


def test_warns_when_sibling_build_keys_present(tmp_path):
    out = _mk_cache(tmp_path, 111, 222)
    warnings = cache.publish_warnings(out, 111, default_cache=False)
    assert len(warnings) == 1
    assert "222" in warnings[0]          # the sibling key is named
    assert "111" in warnings[0]          # and the one actually being published


def test_warns_when_publishing_from_default_shared_cache(tmp_path):
    out = _mk_cache(tmp_path, 111)
    warnings = cache.publish_warnings(out, 111, default_cache=True)
    assert len(warnings) == 1
    assert "default shared cache" in warnings[0]


def test_both_warnings_when_default_and_multiple_keys(tmp_path):
    out = _mk_cache(tmp_path, 111, 222, 333)
    warnings = cache.publish_warnings(out, 111, default_cache=True)
    assert len(warnings) == 2
