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


def resolve_version() -> Optional[str]:
    """Resolve a tt-metal version string.

    Order: installed package metadata (ttnn / tt-metal / tt_metal) -> ``git describe``
    in ``TT_METAL_HOME`` -> raw git sha. Returns None if nothing resolves.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        for dist in ("ttnn", "tt-metal", "tt_metal", "metal-libs"):
            try:
                return version(dist)
            except PackageNotFoundError:
                continue
    except Exception:
        pass

    home = _tt_metal_home()
    if home and shutil.which("git"):
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


def local_env(arch_override: Optional[str] = None, probe: bool = False) -> LocalEnv:
    """Gather the full local environment for compatibility comparison."""
    dev = detect_device(arch_override=arch_override)
    return LocalEnv(
        tt_metal_version=resolve_version(),
        arch=dev.arch,
        device_count=dev.device_count,
        harvesting_mask=dev.harvesting_mask,
        build_key=probe_build_key() if probe else None,
    )
