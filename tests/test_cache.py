# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Cache-root resolution, build_key selection, and packaging round-trip."""

import os

import pytest

from tt_kernel import cache


def _make_subtree(root, build_key):
    """Create a realistic <build_key>/{kernels,firmware}/... subtree."""
    base = root / str(build_key)
    (base / "kernels" / "reader" / "tensix").mkdir(parents=True)
    (base / "kernels" / "reader" / "tensix" / "reader.elf").write_bytes(b"reader-binary")
    (base / "kernels" / "writer" / "tensix").mkdir(parents=True)
    (base / "kernels" / "writer" / "tensix" / "writer.elf").write_bytes(b"writer-binary")
    (base / "firmware" / "tensix").mkdir(parents=True)
    (base / "firmware" / "tensix" / "fw.elf").write_bytes(b"firmware-binary")
    return base


def test_resolve_out_root_env_glues_prefix(monkeypatch, tmp_path):
    # TT_METAL_CACHE=X -> out_root X/tt-metal-cache (no trailing slash); build_key glued.
    monkeypatch.setenv("TT_METAL_CACHE", str(tmp_path))
    out_root = cache.resolve_out_root()
    assert out_root == str(tmp_path / "tt-metal-cache")
    assert not out_root.endswith("/")
    # The build dir is a SIBLING named tt-metal-cache<key>, not a child.
    assert cache.build_key_path(out_root, 42) == tmp_path / "tt-metal-cache42"


def test_resolve_out_root_home_default_is_child(monkeypatch, tmp_path):
    monkeypatch.delenv("TT_METAL_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    out_root = cache.resolve_out_root()
    assert out_root == str(tmp_path / ".cache" / "tt-metal-cache") + "/"
    # Trailing slash -> build_key is a clean child dir.
    assert cache.build_key_path(out_root, 42) == tmp_path / ".cache" / "tt-metal-cache" / "42"


def test_glued_layout_enumeration(tmp_path):
    # Simulate the real TT_METAL_CACHE on-disk layout: X/tt-metal-cache<key>/...
    out_root = str(tmp_path / "tt-metal-cache")
    for key in (111, 222):
        base = tmp_path / f"tt-metal-cache{key}"
        (base / "kernels" / "reader" / "tensix").mkdir(parents=True)
        (base / "kernels" / "reader" / "tensix" / "reader.elf").write_bytes(b"x")
    assert cache.list_build_keys(out_root) == [111, 222]


def _out_root_child(tmp_path) -> str:
    # Default-style out_root with trailing slash (build_key dirs are children).
    return str(tmp_path) + "/"


def test_select_build_key_single(tmp_path):
    _make_subtree(tmp_path, 777)
    assert cache.select_build_key(_out_root_child(tmp_path), None) == 777


def test_select_build_key_ambiguous(tmp_path):
    _make_subtree(tmp_path, 1)
    _make_subtree(tmp_path, 2)
    with pytest.raises(ValueError):
        cache.select_build_key(_out_root_child(tmp_path), None)


def test_select_build_key_explicit_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        cache.select_build_key(_out_root_child(tmp_path), 999)


def test_index_and_count(tmp_path):
    sub = _make_subtree(tmp_path, 42)
    entries = cache.index_subtree(sub)
    assert len(entries) == 3
    assert cache.count_kernels(sub) == 2
    # Paths are relative to the subtree root.
    assert {e.path for e in entries} == {
        os.path.join("kernels", "reader", "tensix", "reader.elf"),
        os.path.join("kernels", "writer", "tensix", "writer.elf"),
        os.path.join("firmware", "tensix", "fw.elf"),
    }


def test_packaging_roundtrip_and_integrity(tmp_path):
    src_root = tmp_path / "src"
    src_root.mkdir()
    sub = _make_subtree(src_root, 5)
    entries = cache.index_subtree(sub)

    # Install into a fresh cache root and verify integrity (child-dir out_root).
    dest_root = tmp_path / "dest"
    dest_root.mkdir()
    out_root = str(dest_root) + "/"
    target = cache.install_subtree(sub, out_root, 5)
    assert target == dest_root / "5"
    assert cache.verify_files(target, entries) == []


def test_install_glued_layout(tmp_path):
    # Install under a TT_METAL_CACHE-style glued out_root.
    src = tmp_path / "src"
    src.mkdir()
    sub = _make_subtree(src, 7)
    out_root = str(tmp_path / "cache" / "tt-metal-cache")
    target = cache.install_subtree(sub, out_root, 7)
    assert target == tmp_path / "cache" / "tt-metal-cache7"
    assert cache.list_build_keys(out_root) == [7]


def test_verify_detects_corruption(tmp_path):
    sub = _make_subtree(tmp_path, 9)
    entries = cache.index_subtree(sub)
    (sub / "firmware" / "tensix" / "fw.elf").write_bytes(b"tampered-different-size")
    problems = cache.verify_files(sub, entries)
    assert problems
    assert any("fw.elf" in p for p in problems)


def test_remove_subtree(tmp_path):
    _make_subtree(tmp_path, 3)
    out_root = str(tmp_path) + "/"
    assert cache.remove_subtree(out_root, 3) is True
    assert cache.remove_subtree(out_root, 3) is False
