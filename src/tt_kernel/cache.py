"""Locate the tt-metal kernel cache, enumerate build_key subtrees, and package/install.

Mirrors the verified tt-metal source. tt-metal builds the per-build directory as the
**string concatenation** ``f"{out_root}{build_key}"`` (build.cpp:354-355), where
``out_root`` is resolved as (rtoptions.cpp:260-266, 423 + build.cpp:90-109):

  - ``TT_METAL_CACHE=X`` set  -> out_root ``X/tt-metal-cache`` (NO trailing slash), so
    the build dirs are siblings named ``tt-metal-cache<build_key>`` directly under ``X``.
  - unset                     -> out_root ``$HOME/.cache/tt-metal-cache/`` (trailing
    slash) or ``/tmp/tt-metal-cache/``, so build dirs are clean children ``<build_key>``.

We therefore model ``out_root`` as a prefix string and decompose it into a parent
directory plus a filename prefix. Layout under each build dir is
``{kernels,firmware}/...`` (jit_compile_server.cpp:109-120).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from .manifest import FileEntry

_CACHE_SUBDIR = "tt-metal-cache"


def resolve_out_root(cache_dir: Optional[str] = None) -> str:
    """Resolve tt-metal's ``out_root`` prefix string, byte-for-byte as tt-metal does.

    ``cache_dir`` (from ``--cache-dir``) is treated exactly like a ``TT_METAL_CACHE``
    value: ``normalize_path(value, "tt-metal-cache")`` == ``value/tt-metal-cache`` with
    no trailing separator. With nothing set, the default has a trailing separator.
    """
    explicit = cache_dir if cache_dir is not None else os.environ.get("TT_METAL_CACHE")
    if explicit:
        p = Path(explicit).expanduser()
        if p.name != _CACHE_SUBDIR:
            p = p / _CACHE_SUBDIR
        return os.path.normpath(str(p))  # no trailing slash -> build_key is glued on

    home = os.environ.get("HOME")
    base = (
        Path(home) / ".cache" / _CACHE_SUBDIR
        if home and Path(home).exists()
        else Path("/tmp") / _CACHE_SUBDIR
    )
    return str(base) + os.sep  # trailing slash -> build_key is a child dir


def _parent_and_prefix(out_root: str) -> Tuple[Path, str]:
    """Split an ``out_root`` prefix into (parent dir to scan, filename prefix)."""
    if out_root.endswith(os.sep):
        return Path(out_root.rstrip(os.sep)), ""
    return Path(os.path.dirname(out_root)), os.path.basename(out_root)


def list_build_keys(out_root: str) -> List[int]:
    """Return the numeric build_keys present under ``out_root`` (handles glued prefix)."""
    parent, prefix = _parent_and_prefix(out_root)
    if not parent.is_dir():
        return []
    pat = re.compile("^" + re.escape(prefix) + r"(\d+)$")
    keys: List[int] = []
    for child in parent.iterdir():
        if child.is_dir():
            m = pat.match(child.name)
            if m:
                keys.append(int(m.group(1)))
    return sorted(keys)


def build_key_path(out_root: str, build_key: int) -> Path:
    """The on-disk directory for a build_key, matching tt-metal's string concat."""
    parent, prefix = _parent_and_prefix(out_root)
    return parent / f"{prefix}{build_key}"


def select_build_key(out_root: str, explicit: Optional[int]) -> int:
    """Pick the build_key to package, or raise with guidance if ambiguous."""
    if explicit is not None:
        path = build_key_path(out_root, explicit)
        if not path.is_dir():
            raise FileNotFoundError(
                f"No build_key directory for {explicit} at {path}. "
                f"Available: {list_build_keys(out_root) or 'none'}"
            )
        return explicit

    keys = list_build_keys(out_root)
    if not keys:
        raise FileNotFoundError(
            f"No build_key directories found for out_root {out_root!r}. "
            "Run your model once to populate the cache, or pass --cache-dir."
        )
    if len(keys) > 1:
        raise ValueError(
            f"Multiple build_key directories for out_root {out_root!r}: {keys}. "
            "Pass --build-key N to choose which to publish."
        )
    return keys[0]


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    """Public alias of the streaming sha256 helper (used to index shipped wheels)."""
    return _sha256_file(path)


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


def install_subtree(staged: Path, out_root: str, build_key: int) -> Path:
    """Merge a downloaded build_key subtree into the local cache at ``out_root``.

    ``staged`` is the directory holding the bundle's build_key subtree contents.
    Existing files are overwritten; the target is created if absent.
    """
    target = build_key_path(out_root, build_key)
    target.mkdir(parents=True, exist_ok=True)
    for src in staged.rglob("*"):
        if src.is_file():
            rel = src.relative_to(staged)
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return target


def remove_subtree(out_root: str, build_key: int) -> bool:
    """Remove a locally installed build_key subtree. Returns True if it existed."""
    target = build_key_path(out_root, build_key)
    if target.is_dir():
        shutil.rmtree(target)
        return True
    return False
