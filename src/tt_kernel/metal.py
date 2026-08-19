# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Resolve the local tt-metal version and build_key inputs without opening a device.

``build_key`` itself is C++-only (``DeviceBuildEnv::build_key()``) and not exposed to
Python, so the authoritative integer comes from the cache directory name on ``push``
and, optionally, from a ``--probe`` device open on ``pull``. Everything else here is
best-effort detection from the installed package, git, and ``TT_METAL_*`` env vars.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from .device import detect as detect_device
from .manifest import BuildKeyInputs

# rtoptions debug/feature env vars that feed get_compile_hash_string() in tt-metal.
# We fingerprint these so a debug-enabled producer cache is flagged against a plain
# consumer (and vice versa). This is best-effort: the exact C++ string is internal.
_COMPILE_HASH_ENV_VARS = (
    "TT_METAL_WATCHER",
    "TT_METAL_DPRINT_CORES",
    "TT_METAL_DPRINT_CHIPS",
    "TT_METAL_DPRINT_RISCVS",
    "TT_METAL_DPRINT_FILE",
    "TT_METAL_RISCV_DEBUG_INFO",
    "TT_METAL_KERNELS_EARLY_RETURN",
    "TT_METAL_ENABLE_ERISC_IRAM",
    "TT_METAL_GDB_SERVER",
)


@dataclass
class LocalEnv:
    """Everything ``manifest.compare`` needs about the local environment."""

    tt_metal_version: Optional[str] = None
    arch: Optional[str] = None
    device_count: int = 0
    harvesting_mask: Optional[int] = None
    build_key: Optional[int] = None  # only set via --probe
    # Installed serving-runtime version (vLLM), for a v4 manifest's ``runtime.version`` range.
    # ``tt_metal_version`` doubles as the installed ttnn version for ``platform.ttnn`` (both
    # come from ``resolve_version``, which probes the ttnn dist first). None => not resolvable.
    vllm_version: Optional[str] = None
    # Installed Tenstorrent vLLM plugin version (vllm_tt_plugin), for ``runtime.plugin_version``.
    # A distinct package from vLLM core. None => not resolvable (older/presence-only install).
    vllm_plugin_version: Optional[str] = None


def _tt_metal_home() -> Optional[str]:
    return os.environ.get("TT_METAL_HOME") or os.environ.get("TT_METAL_RUNTIME_ROOT")


def detect_tt_metal_home() -> Optional[str]:
    """Best-effort absolute path of the local tt-metal source root — the prefix embedded
    in ``.dephash`` dependency paths — resolved WITHOUT importing ttnn or opening a device.

    Order: ``TT_METAL_HOME``/``TT_METAL_RUNTIME_ROOT`` env -> the ancestor of the ttnn
    package (located via ``find_spec``, which does not execute the module) that contains
    ``tt_metal/hw/inc``. Returns None if neither resolves; callers then skip the cross-host
    tree-dep rewrite (in-cache relocation alone is correct on the same host).
    """
    import importlib.util
    from pathlib import Path

    home = _tt_metal_home()
    if home:
        return os.path.normpath(str(Path(home).expanduser()))
    try:
        spec = importlib.util.find_spec("ttnn")
    except (ImportError, ValueError):
        spec = None
    if spec and spec.origin:
        for parent in Path(spec.origin).resolve().parents:
            if (parent / "tt_metal" / "hw" / "inc").is_dir():
                return str(parent)
    return None


def _version_from_metadata() -> Optional[str]:
    """First version string among the tt-metal distributions, or None."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        for dist in ("ttnn", "tt-metal", "tt_metal", "metal-libs"):
            try:
                return version(dist)
            except PackageNotFoundError:
                continue
    except Exception:
        pass
    return None


def _version_from_git(home: str) -> Optional[str]:
    """``git describe`` in *home*, or None if it is not a git work tree."""
    if not shutil.which("git"):
        return None
    try:
        inside = subprocess.run(
            ["git", "-C", home, "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=10,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return None
    except (subprocess.SubprocessError, OSError):
        return None

    for argv in (
        ["git", "-C", home, "describe", "--tags", "--always", "--dirty"],
        ["git", "-C", home, "rev-parse", "HEAD"],
    ):
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=True)
            val = out.stdout.strip()
            if val:
                return val
        except (subprocess.SubprocessError, OSError):
            continue
    return None


def resolve_version() -> Optional[str]:
    """Resolve a tt-metal version string.

    Order: ``git describe`` in ``TT_METAL_HOME`` when that is a git work tree ->
    installed package metadata -> None.

    The git tree comes FIRST, and that ordering is the whole point of this
    function. tt-metal is very often installed editable from a source checkout,
    and an editable install writes its metadata exactly once -- at ``pip install
    -e`` time -- and never revisits it. Upgrading the checkout in place therefore
    leaves ``importlib.metadata`` reporting whatever the tree happened to be
    months ago, with no indication that it is stale.

    That is not hypothetical. Upgrading tt-metal to v0.77.0 in an editable tree
    left the metadata reading 0.65.1rc17.dev6200, and this function believed it,
    so ``tt-kernel serve`` told the operator:

        ! tt-metal: version 0.65.1rc17.dev6200 is older than required 0.72.0 —
          upgrade

    instructing them to perform the upgrade they had just performed. A probe that
    is wrong in the *stale* direction is worse than one that returns None: None
    is visibly missing, whereas stale looks authoritative and sends people to fix
    the wrong thing.

    Metadata remains correct for a wheel install, which has no git tree, so it
    stays as the fallback rather than being removed.
    """
    home = _tt_metal_home()
    if home:
        from_git = _version_from_git(home)
        if from_git:
            return from_git
    return _version_from_metadata()


def compile_hash_fingerprint() -> str:
    """Stable fingerprint of the compile-affecting env vars (best-effort).

    Empty (all-unset) is the production default and hashes to a fixed sentinel so two
    plain environments compare equal regardless of unrelated env.
    """
    present = {k: os.environ[k] for k in _COMPILE_HASH_ENV_VARS if k in os.environ}
    if not present:
        return ""  # production default: no debug features
    blob = "\n".join(f"{k}={present[k]}" for k in sorted(present))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_key_inputs(
    *,
    num_hw_cqs: Optional[int] = None,
    dispatch_core_type: Optional[str] = None,
    dispatch_core_axis: Optional[str] = None,
    coordinate_virtualization_enabled: Optional[bool] = None,
    harvesting_mask: Optional[int] = None,
) -> BuildKeyInputs:
    """Assemble build_key inputs, defaulting to the production case.

    Explicit kwargs (from CLI flags) win; otherwise fields default to WORKER / ROW / 1
    with coordinate virtualization enabled — the common production configuration.
    """
    return BuildKeyInputs(
        dispatch_core_type=dispatch_core_type or "WORKER",
        dispatch_core_axis=dispatch_core_axis or "ROW",
        num_hw_cqs=num_hw_cqs if num_hw_cqs is not None else 1,
        coordinate_virtualization_enabled=(
            True if coordinate_virtualization_enabled is None else coordinate_virtualization_enabled
        ),
        harvesting_mask=harvesting_mask if harvesting_mask is not None else 0,
        compile_hash_string=compile_hash_fingerprint(),
    )


def probe_build_key() -> Optional[int]:
    """Open a device via ttnn to read the true local build_key (used by --probe).

    Returns None if ttnn isn't importable or no device is present. Opening a device is
    not free, so this is opt-in.
    """
    try:
        import ttnn  # type: ignore
    except Exception:
        return None
    device = None
    try:
        device = ttnn.open_device(device_id=0)
        for attr in ("build_key", "build_id"):
            getter = getattr(device, attr, None)
            if callable(getter):
                val = getter()
                if isinstance(val, int):
                    return val
            elif isinstance(getter, int):
                return getter
    except Exception:
        return None
    finally:
        if device is not None:
            try:
                ttnn.close_device(device)
            except Exception:
                pass
    return None


def _dist_version_any(dists: tuple) -> Optional[str]:
    """First resolvable version among ``dists``, or None. Never raises (metadata lookup is
    best-effort — a missing/broken dist must not break a compatibility resolve)."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        for dist in dists:
            try:
                return version(dist)
            except PackageNotFoundError:
                continue
    except Exception:  # noqa: BLE001
        return None
    return None


def _vllm_version() -> Optional[str]:
    """Installed vLLM core distribution version, or None."""
    return _dist_version_any(("vllm",))


def _vllm_plugin_version() -> Optional[str]:
    """Installed Tenstorrent vLLM plugin version (vllm_tt_plugin), or None."""
    return _dist_version_any(("vllm_tt_plugin", "vllm-tt-plugin"))


def local_env(arch_override: Optional[str] = None, probe: bool = False) -> LocalEnv:
    """Gather the full local environment for compatibility comparison."""
    dev = detect_device(arch_override=arch_override)
    return LocalEnv(
        tt_metal_version=resolve_version(),
        arch=dev.arch,
        device_count=dev.device_count,
        harvesting_mask=dev.harvesting_mask,
        build_key=probe_build_key() if probe else None,
        vllm_version=_vllm_version(),
        vllm_plugin_version=_vllm_plugin_version(),
    )
