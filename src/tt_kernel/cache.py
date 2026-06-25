# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

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


# Signature kernel groups that only a fast-path warmup (DISPATCH_TRACE +
# DISPATCH_ONDEVICE_LMHEAD) produces: the paged flash-decode, the traced decode
# reader/writer, and the on-device-argmax lm_head reader. A baseline (trace-disabled)
# cache lacks them, so a fast-path consumer would re-JIT ~half its kernels.
_FAST_PATH_KERNEL_MARKERS = (
    "sdpa_flash_decode",
    "reader_decode_all",
    "writer_decode_all",
    "reader_argmax_tile_layout",
)


def detect_fast_path_kernels(subtree: Path) -> bool:
    """Whether a build_key subtree carries the traced-decode / on-device-lm_head kernels a
    ``DISPATCH_TRACE``/``DISPATCH_ONDEVICE_LMHEAD`` consumer needs (#6 bundle completeness).

    Heuristic by signature kernel-group name under ``kernels/``. Recorded in the manifest
    so ``pull`` can warn when a baseline-only bundle would re-JIT on the fast path.
    """
    kdir = subtree / "kernels"
    if not kdir.is_dir():
        return False
    names = {c.name for c in kdir.iterdir() if c.is_dir()}
    return any(marker in names for marker in _FAST_PATH_KERNEL_MARKERS)


def publish_warnings(out_root: str, build_key: int, *, default_cache: bool) -> List[str]:
    """Pre-push guard (#2, proposal 3): warn when the cache being published does not look
    isolated to a single model, so a producer does not accidentally ship the shared
    all-models cache.

    tt-metal's JIT cache is keyed by build_key and **not** tagged by model: every model
    compiled on one build shares that build_key's ``kernels/`` dir. We cannot tell models
    apart within a build_key, but we can flag the two detectable risk signals — sibling
    build_keys under the same root, and publishing straight from the default shared cache.
    Returns human-readable warnings (empty == looks isolated). Warn-only by design: the
    automated producer (``release.py``) publishes from a fresh ``--cache-dir`` and so never
    trips these, while an interactive user pushing from their shared cache gets told.
    """
    warnings: List[str] = []
    others = [k for k in list_build_keys(out_root) if k != build_key]
    if others:
        warnings.append(
            f"cache root also holds build_key(s) {others}; publishing only {build_key}. "
            "If this is a shared cache, the subtree may mix kernels from multiple models. "
            "Warm a fresh --cache-dir for a model-specific bundle (see `tt-kernel clean`)."
        )
    if default_cache:
        warnings.append(
            f"publishing from the default shared cache ({out_root}); every model compiled "
            "on this build shares this build_key. Use a fresh --cache-dir (or run "
            "`tt-kernel clean` first) for a model-specific bundle."
        )
    return warnings


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


def relocate_dephashes(target: Path, build_key: int) -> int:
    """Rewrite in-cache absolute paths inside ``.dephash`` files to the install location.

    tt-metal's JIT cache validity check (``dependencies_up_to_date`` in
    ``jit_build/depend.cpp``) reads each dependency by its stored **absolute** path and
    recompiles if that path can't be read. A kernel's generated sources
    (``chlkc_*.h``/``.cpp``, ``defines_generated.h``) live *inside* the build_key subtree,
    so their stored paths embed the **producer's** cache location. Pulled into a different
    cache dir, those paths no longer exist and every such kernel recompiles — defeating the
    point of the bundle.

    This rewrites the producer's build_key-dir prefix (detected from the dephash content,
    which always ends in ``<build_key>/kernels/``) to ``target`` (the new build_key dir).
    The tt-metal *tree* deps (absolute, outside the cache) are left untouched — correct on
    the same host; cross-host relocation would also need the tree prefix rewritten.
    Returns the number of dephash files changed. Best-effort: never raises.
    """
    new_prefix = str(target)
    pat = re.compile(r'([^\s"]*' + re.escape(str(build_key)) + r')/kernels/')
    changed = 0
    for dephash in target.rglob("*.dephash"):
        try:
            text = dephash.read_text()
        except OSError:
            continue
        m = pat.search(text)
        if not m:
            continue
        old_prefix = m.group(1)
        if old_prefix == new_prefix:
            continue
        new_text = text.replace(old_prefix, new_prefix)
        if new_text != text:
            try:
                dephash.write_text(new_text)
                changed += 1
            except OSError:
                continue
    return changed


def detect_cache_tt_metal_root(subtree: Path) -> Optional[str]:
    """Detect the producer's tt-metal source root embedded in a build_key subtree.

    Every ``.dephash`` lists two kinds of dependency path: generated sources *inside* the
    subtree (handled by ``relocate_dephashes``) and tt-metal tree headers/compiler files
    *outside* it. The latter all share the producer's tt-metal root as a common prefix, so
    we return ``os.path.commonpath`` of the external deps. Recorded in the manifest so a
    cross-host consumer can rewrite that prefix on install. Returns None if none are found.
    """
    subtree_prefix = os.path.normpath(str(subtree)) + os.sep
    externals: List[str] = []
    for dephash in subtree.rglob("*.dephash"):
        try:
            text = dephash.read_text()
        except OSError:
            continue
        for m in re.finditer(r'"([^"]+)"', text):
            path = m.group(1)
            if path.startswith("/") and not os.path.normpath(path).startswith(subtree_prefix):
                externals.append(path)
        if len(externals) >= 256:  # a stable common prefix needs only a sample
            break
    if not externals:
        return None
    try:
        root = os.path.commonpath(externals)
    except ValueError:
        return None
    # Guard against a degenerate prefix (a stray non-tt-metal dep would shrink it to /).
    return root if root and root.count(os.sep) >= 2 else None


def relocate_tt_metal_tree(target: Path, producer_root: str, consumer_root: str) -> int:
    """Rewrite the producer's tt-metal source-root prefix to the consumer's in every
    ``.dephash`` under ``target`` — the cross-host complement to ``relocate_dephashes``.

    No-op when the roots are equal. The dependency *content* hashes in the dephash are
    left intact: the compat check already requires a matching ``tt_metal_version``, so the
    headers at the rewritten path hash-match and the kernel hits. Returns the number of
    files changed. Best-effort: never raises.
    """
    producer_root = os.path.normpath(producer_root)
    consumer_root = os.path.normpath(consumer_root)
    if producer_root == consumer_root:
        return 0
    changed = 0
    for dephash in target.rglob("*.dephash"):
        try:
            text = dephash.read_text()
        except OSError:
            continue
        if producer_root not in text:
            continue
        new_text = text.replace(producer_root, consumer_root)
        if new_text != text:
            try:
                dephash.write_text(new_text)
                changed += 1
            except OSError:
                continue
    return changed


def install_subtree(staged: Path, out_root: str, build_key: int) -> Path:
    """Merge a downloaded build_key subtree into the local cache at ``out_root``.

    ``staged`` is the directory holding the bundle's build_key subtree contents.
    Existing files are overwritten; the target is created if absent. After copying, the
    ``.dephash`` validity files are relocated so the cache produces JIT hits at this path
    rather than recompiling (see ``relocate_dephashes``).
    """
    target = build_key_path(out_root, build_key)
    target.mkdir(parents=True, exist_ok=True)
    for src in staged.rglob("*"):
        if src.is_file():
            rel = src.relative_to(staged)
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    relocate_dephashes(target, build_key)
    return target


def remove_subtree(out_root: str, build_key: int) -> bool:
    """Remove a locally installed build_key subtree. Returns True if it existed."""
    target = build_key_path(out_root, build_key)
    if target.is_dir():
        shutil.rmtree(target)
        return True
    return False
