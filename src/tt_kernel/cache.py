"""Locate the tt-metal kernel cache, enumerate build_key subtrees, and package/install.

Mirrors the verified tt-metal source:
  - Cache root: ``TT_METAL_CACHE`` env, else ``$HOME/.cache/tt-metal-cache/``, else
    ``/tmp/tt-metal-cache/`` (rtoptions.cpp:421 + build.cpp:90-109).
  - Layout: ``<cache_root>/<build_key>/{kernels,firmware}/...``
    (jit_compile_server.cpp:109-120).
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import List, Optional

from .manifest import FileEntry

_CACHE_SUBDIR = "tt-metal-cache"


def resolve_cache_root(cache_dir: Optional[str] = None) -> Path:
    """Resolve the kernel cache root, matching tt-metal's resolution order.

    An explicit ``cache_dir`` (from ``--cache-dir``) wins. Otherwise: ``TT_METAL_CACHE``
    -> ``$HOME/.cache/tt-metal-cache`` -> ``/tmp/tt-metal-cache``.
    """
    if cache_dir:
        return Path(cache_dir).expanduser()

    env = os.environ.get("TT_METAL_CACHE")
    if env:
        # tt-metal normalises by appending the cache subdir name if absent.
        p = Path(env).expanduser()
        return p if p.name == _CACHE_SUBDIR else p / _CACHE_SUBDIR

    home = os.environ.get("HOME")
    if home and Path(home).exists():
        return Path(home) / ".cache" / _CACHE_SUBDIR
    return Path("/tmp") / _CACHE_SUBDIR


def list_build_keys(cache_root: Path) -> List[int]:
    """Return the numeric build_key directories present under the cache root."""
    if not cache_root.is_dir():
        return []
    keys: List[int] = []
    for child in cache_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            keys.append(int(child.name))
    return sorted(keys)


def select_build_key(cache_root: Path, explicit: Optional[int]) -> int:
    """Pick the build_key to package, or raise with guidance if ambiguous."""
    if explicit is not None:
        path = cache_root / str(explicit)
        if not path.is_dir():
            raise FileNotFoundError(
                f"No build_key directory {explicit} under {cache_root}. "
                f"Available: {list_build_keys(cache_root) or 'none'}"
            )
        return explicit

    keys = list_build_keys(cache_root)
    if not keys:
        raise FileNotFoundError(
            f"No build_key directories found under {cache_root}. "
            "Run your model once to populate the cache, or pass --cache-dir."
        )
    if len(keys) > 1:
        raise ValueError(
            f"Multiple build_key directories under {cache_root}: {keys}. "
            "Pass --build-key N to choose which to publish."
        )
    return keys[0]


def build_key_path(cache_root: Path, build_key: int) -> Path:
    return cache_root / str(build_key)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def index_subtree(subtree: Path) -> List[FileEntry]:
    """Walk a ``<build_key>/`` subtree, returning a sha256 index of every file.

    Paths are stored relative to the subtree root so they can be re-rooted on install.
    """
    entries: List[FileEntry] = []
    for path in sorted(subtree.rglob("*")):
        if path.is_file():
            entries.append(
                FileEntry(
                    path=str(path.relative_to(subtree)),
                    sha256=_sha256_file(path),
                    size=path.stat().st_size,
                )
            )
    return entries


def count_kernels(subtree: Path) -> int:
    """Count entries under ``kernels/`` (informational ``kernel_count``)."""
    kdir = subtree / "kernels"
    if not kdir.is_dir():
        return 0
    return sum(1 for c in kdir.iterdir() if c.is_dir())


def verify_files(root: Path, entries: List[FileEntry]) -> List[str]:
    """Verify a set of files under ``root`` against their manifest entries.

    Returns a list of human-readable problems (empty == all good).
    """
    problems: List[str] = []
    for entry in entries:
        fpath = root / entry.path
        if not fpath.is_file():
            problems.append(f"missing: {entry.path}")
            continue
        actual = fpath.stat().st_size
        if actual != entry.size:
            problems.append(f"size mismatch: {entry.path} ({actual} != {entry.size})")
            continue
        if _sha256_file(fpath) != entry.sha256:
            problems.append(f"sha256 mismatch: {entry.path}")
    return problems


def install_subtree(staged: Path, cache_root: Path, build_key: int) -> Path:
    """Merge a downloaded ``<build_key>/`` subtree into the local cache root.

    ``staged`` is the directory holding the bundle's build_key subtree contents.
    Existing files are overwritten; the target is created if absent.
    """
    target = build_key_path(cache_root, build_key)
    target.mkdir(parents=True, exist_ok=True)
    for src in staged.rglob("*"):
        if src.is_file():
            rel = src.relative_to(staged)
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return target


def remove_subtree(cache_root: Path, build_key: int) -> bool:
    """Remove a locally installed build_key subtree. Returns True if it existed."""
    target = build_key_path(cache_root, build_key)
    if target.is_dir():
        shutil.rmtree(target)
        return True
    return False
