# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Materialize and read vLLM *bundle folders* — the serving-layer half of a vLLM bundle.

A vLLM bundle ships a self-contained per-model folder that the Tenstorrent vLLM plugin
discovers via ``EXTRA_MODELS_DIR``. The folder holds a plugin-owned ``vllm_metadata.json``
(arch name, main-class path, per-machine launch command, HF weights ref) plus the
``VllmGeneratorAdapter`` class and its dependencies. tt-kernel lays the folder into a local
``bundles_dir`` on ``pull`` and points the plugin at that dir on ``serve`` — it never invents
the serving contract, it only *reads* the arch name and the launch command for this machine.

The layout on disk::

    <bundles_dir>/<model_key>/
        vllm_metadata.json
        <adapter class + deps ...>

``<bundles_dir>`` is what ``serve`` exports as ``EXTRA_MODELS_DIR``; the plugin scans every
``<model_key>/`` under it and registers the model found there.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from . import device

ENV_BUNDLES_DIR = "TT_KERNEL_BUNDLES_DIR"
ENV_MACHINE = "TT_KERNEL_MACHINE"
# The plugin-owned metadata file at the root of a bundle folder. tt-kernel ships it verbatim
# and reads only ``arch`` + the per-machine launch command from it.
VLLM_METADATA_NAME = "vllm_metadata.json"


def resolve_bundles_dir(bundles_dir: Optional[str] = None) -> Path:
    """Where vLLM bundle folders are laid down / read from (== ``EXTRA_MODELS_DIR``).

    Resolution (flag > env > default, mirroring ``runtime.resolve_models_dir``):
    ``--bundles-dir`` > ``TT_KERNEL_BUNDLES_DIR`` > ``~/.cache/tt-kernel/bundles``.
    """
    explicit = bundles_dir if bundles_dir is not None else os.environ.get(ENV_BUNDLES_DIR)
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HOME")
    return (Path(home) / ".cache" if home else Path("/tmp")) / "tt-kernel" / "bundles"


def model_key(repo_id: str) -> str:
    """A single flat folder name for a model under ``bundles_dir``.

    ``EXTRA_MODELS_DIR`` holds one folder per model, so a nested ``org/name`` is flattened
    to ``org__name`` (a stable, filesystem-safe, collision-free key that round-trips for
    ``rm``). A revision suffix (``@rev``) is dropped by the caller before this is reached.
    """
    return repo_id.replace("/", "__")


def install_bundle(staged: Path, bundles_dir: Path, key: str) -> Path:
    """Lay a pulled bundle folder into ``<bundles_dir>/<key>/`` (idempotent overwrite).

    ``staged`` is the folder (inside a bundle snapshot) that contains ``vllm_metadata.json``
    and the adapter code. Any existing install for the same key is replaced so a re-pull is
    clean.
    """
    dest = bundles_dir / key
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staged, dest)
    return dest


def remove_bundle(bundles_dir: Path, key: str) -> bool:
    """Remove an installed bundle folder. Returns True if something was removed."""
    dest = bundles_dir / key
    if dest.is_dir():
        shutil.rmtree(dest)
        return True
    return False


@dataclass
class LaunchSpec:
    """The launch command + env for one machine, read from ``vllm_metadata.json``."""

    command: List[str]
    env: dict


@dataclass
class VllmMetadata:
    """A thin read-only view over the plugin-owned ``vllm_metadata.json``.

    tt-kernel does not own this schema, so we keep the raw dict and expose only the fields
    the serving orchestration needs: ``arch`` (HF architecture name — the plugin prepends
    ``TT`` when registering), ``main_class`` (``"module:Class"``), an optional ``hf_weights``
    ref, and a per-machine ``launch`` map.
    """

    raw: dict

    @property
    def arch(self) -> Optional[str]:
        return self.raw.get("arch")

    @property
    def main_class(self) -> Optional[str]:
        return self.raw.get("main_class")

    @property
    def hf_weights(self) -> Optional[str]:
        return self.raw.get("hf_weights")

    @property
    def launch(self) -> dict:
        val = self.raw.get("launch")
        return val if isinstance(val, dict) else {}


def read_vllm_metadata(bundle_folder: Path) -> VllmMetadata:
    """Parse ``<bundle_folder>/vllm_metadata.json``. Raises FileNotFoundError/ValueError."""
    path = bundle_folder / VLLM_METADATA_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (not a vLLM bundle folder).")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object.")
    return VllmMetadata(raw=data)


def machine_candidates(arch_override: Optional[str] = None) -> List[str]:
    """Ordered launch-map keys to try for the local machine, most specific first.

    ``TT_KERNEL_MACHINE`` (explicit override) > ``<arch>-<n>card`` > ``<arch>`` > ``default``.
    The bundle author picks whichever granularity they key their ``launch`` map on; we probe
    from most to least specific and fall back to ``default``.
    """
    cands: List[str] = []
    override = os.environ.get(ENV_MACHINE)
    if override:
        cands.append(override)
    info = device.detect(arch_override=arch_override)
    if info.arch:
        if info.device_count:
            cands.append(f"{info.arch}-{info.device_count}card")
        cands.append(info.arch)
    cands.append("default")
    # de-dupe, preserve order
    seen: set = set()
    return [c for c in cands if not (c in seen or seen.add(c))]


def select_launch(
    metadata: VllmMetadata, arch_override: Optional[str] = None
) -> Tuple[Optional[str], Optional[LaunchSpec]]:
    """Pick the launch spec for the local machine from a bundle's ``launch`` map.

    Returns ``(matched_key, LaunchSpec)`` or ``(None, None)`` when the map has no entry for
    any candidate key (caller should warn — the bundle wasn't built for this machine).
    """
    launch = metadata.launch
    if not launch:
        return None, None
    for key in machine_candidates(arch_override):
        entry = launch.get(key)
        if isinstance(entry, dict):
            command = entry.get("command") or []
            env = entry.get("env") or {}
            if isinstance(command, list) and command:
                return key, LaunchSpec(command=[str(c) for c in command], env=dict(env))
    return None, None


__all__ = [
    "ENV_BUNDLES_DIR",
    "ENV_MACHINE",
    "VLLM_METADATA_NAME",
    "resolve_bundles_dir",
    "model_key",
    "install_bundle",
    "remove_bundle",
    "read_vllm_metadata",
    "machine_candidates",
    "select_launch",
    "LaunchSpec",
    "VllmMetadata",
]
