# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Materialize and read vLLM *bundle folders* — the serving-layer half of a vLLM bundle.

A vLLM bundle ships a self-contained per-model folder that the Tenstorrent vLLM plugin
discovers via ``EXTRA_MODELS_DIR``. The folder holds a plugin-owned ``vllm_metadata.json``
(arch name, main-class path, per-machine launch command, HF weights ref) plus the
``VllmGeneratorAdapter`` class and its dependencies. tt-model lays the folder into a local
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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from . import device

if TYPE_CHECKING:  # avoid import at runtime (manifest doesn't import bundles; keep it one-way)
    from .manifest import Manifest

ENV_BUNDLES_DIR = "TT_MODEL_BUNDLES_DIR"
ENV_MACHINE = "TT_MODEL_MACHINE"
# The plugin-owned metadata file at the root of a bundle folder. tt-model ships it verbatim
# and reads only ``arch`` + the per-machine launch command from it.
VLLM_METADATA_NAME = "vllm_metadata.json"


def resolve_bundles_dir(bundles_dir: Optional[str] = None) -> Path:
    """Where vLLM bundle folders are laid down / read from (== ``EXTRA_MODELS_DIR``).

    Resolution (flag > env > default, mirroring ``runtime.resolve_models_dir``):
    ``--bundles-dir`` > ``TT_MODEL_BUNDLES_DIR`` > ``~/.cache/tt-model/bundles``.
    """
    explicit = bundles_dir if bundles_dir is not None else os.environ.get(ENV_BUNDLES_DIR)
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HOME")
    return (Path(home) / ".cache" if home else Path("/tmp")) / "tt-model" / "bundles"


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

    tt-model does not own this schema, so we keep the raw dict and expose only the fields
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


# The serving entrypoint script inside the TT vLLM fork. The composed launch command invokes
# it; matches the convention documented in docs/authoring_runners.md and asserted in tests.
_VLLM_SERVER_SCRIPT = "server_example_tt.py"


def _compose_launch_command(manifest: "Manifest", weights: Optional[str]) -> List[str]:
    """Compose the vLLM launch argv from a v4 manifest's structured resources/capabilities.

    Mirrors the ``server_example_tt.py --model ... --max_num_seqs N`` convention (underscore
    flags) used throughout this repo. This is the ONLY place the opaque argv is synthesized;
    anything this mapping doesn't cover is handled by the escape hatches:
    ``resources.extra_args`` (appended) and ``resources.command_override`` (full replacement,
    per machine key) — so an author is never blocked waiting for a new field per vLLM flag.
    """
    cmd: List[str] = ["python3", _VLLM_SERVER_SCRIPT]
    if weights:
        cmd += ["--model", weights]
    res = manifest.resources
    if res is not None:
        if res.max_model_len is not None:
            cmd += ["--max_model_len", str(res.max_model_len)]
        if res.max_num_seqs is not None:
            cmd += ["--max_num_seqs", str(res.max_num_seqs)]
        if res.block_size is not None:
            cmd += ["--block_size", str(res.block_size)]
        if res.trace_region_bytes is not None:
            cmd += ["--trace_region_size", str(res.trace_region_bytes)]
    cap = manifest.capabilities
    if cap is not None:
        if cap.tool_parser:
            cmd += ["--tool_parser", cap.tool_parser]
        if cap.reasoning_parser:
            cmd += ["--reasoning_parser", cap.reasoning_parser]
    if res is not None and res.extra_args:
        cmd += [str(a) for a in res.extra_args]
    return cmd


def _compose_launch_env(manifest: "Manifest") -> Dict[str, str]:
    """Compose the serving env: the V1 engine default, then the manifest's ``env`` overlaid.

    Topology-specific vars (``MESH_DEVICE`` etc.) are author-supplied via ``manifest.env`` —
    tt-model does not invent device-mapping names it can't verify from ``mesh`` alone.
    """
    env: Dict[str, str] = {"VLLM_USE_V1": "1"}
    env.update({k: str(v) for k, v in (manifest.env or {}).items()})
    return env


def render_vllm_metadata(manifest: "Manifest") -> dict:
    """Render the plugin-owned ``vllm_metadata.json`` dict from a v4 unified manifest.

    tt-model becomes the source of truth: rather than ship an author-written metadata file
    verbatim, it *generates* the plugin schema from the one authoritative manifest —
    ``entrypoint.arch_name`` -> ``arch``, ``entrypoint.cls`` -> ``main_class``,
    ``weights.repo`` -> ``hf_weights``, and a ``launch`` map composed from
    ``resources``/``capabilities``/``env`` (see ``_compose_launch_*``). A per-machine
    ``command_override`` becomes its own launch entry; the composed command is always emitted
    under ``default`` (unless overridden there). ``select_launch`` then picks the right entry
    for the serving machine exactly as it does for a hand-written file.
    """
    ep = manifest.entrypoint
    if ep is None:
        raise ValueError("render_vllm_metadata requires a v4 manifest with an 'entrypoint'.")
    weights = manifest.weights.repo_id if manifest.weights else None
    base_cmd = _compose_launch_command(manifest, weights)
    env = _compose_launch_env(manifest)

    overrides: Dict[str, List[str]] = (
        dict(manifest.resources.command_override) if manifest.resources else {}
    )
    launch: dict = {}
    launch["default"] = {
        "command": [str(a) for a in overrides.get("default", base_cmd)],
        "env": env,
    }
    for key, argv in overrides.items():
        if key == "default":
            continue
        launch[key] = {"command": [str(a) for a in argv], "env": env}

    return {
        "arch": ep.arch_name,
        "main_class": ep.cls,
        "hf_weights": weights,
        "launch": launch,
    }


def write_vllm_metadata(bundle_folder: Path, metadata: dict) -> Path:
    """Write a rendered ``vllm_metadata.json`` into an installed bundle folder."""
    path = bundle_folder / VLLM_METADATA_NAME
    path.write_text(json.dumps(metadata, indent=2))
    return path


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

    ``TT_MODEL_MACHINE`` (explicit override) > ``<arch>-<n>card`` > ``<arch>`` > ``default``.
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
    "render_vllm_metadata",
    "write_vllm_metadata",
    "machine_candidates",
    "select_launch",
    "LaunchSpec",
    "VllmMetadata",
]
