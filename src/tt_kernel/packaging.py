# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Assemble a v5 self-contained ("fat") bundle — "package what's on your box".

Unlike the v4 push path (which references a host-provisioned tt-metal/vLLM and ships only the
serving metadata), this stages ONE running folder that carries the author's actual built
artifacts: their ttnn wheel (custom C++/LLK kernels compiled in), the empty-target base vLLM
wheel, the vLLM plugin wheel, and their modified tt-metal-community tree — plus a generated
``install.sh``/``run.sh`` and a v5 manifest. A consumer needs only a TT card + firmware.

Weights are NEVER embedded: the manifest carries the HF repo id and ``pull`` downloads them.

This module does the filesystem staging only (no network); ``cli.package`` calls it then uploads
via ``hub`` (git-LFS handles the large wheels automatically). Kept import-light and hardware-free
so it is unit-testable offline.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import shutil
import socket
from pathlib import Path
from typing import Dict, List, Optional

from . import bundles
from .manifest import (
    BundledPlatform,
    Entrypoint,
    Manifest,
    Mesh,
    Producer,
    Resources,
    WeightsRef,
    WheelArtifact,
)

# Where the shipped wheels and the embedded metal tree live inside the bundle.
WHEELS_DIR = "wheels"
METAL_DIR = "metal"
INSTALL_SCRIPT = "install.sh"
RUN_SCRIPT = "run.sh"
REQUIREMENTS = "requirements.txt"
# Per-model vLLM bundle folders live under here; this dir (not the bundle root) is EXTRA_MODELS_DIR
# so the plugin's child-scan finds exactly the model metadata and not metal/, wheels/, venv/.
METADATA_DIR = "vllm_models"

# torch is the CPU build for Tenstorrent (never CUDA); requirements install uses this index.
_PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# A wheel filename is: {distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl
# (PEP 427). We only need the trailing three compatibility tags.
_WHEEL_RE = re.compile(r"^(?P<dist>.+?)-(?P<ver>[^-]+)(-\d[^-]*)?-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$")


def host_python_tag() -> str:
    """This interpreter's CPython wheel tag, e.g. ``cp312``."""
    import sys

    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def host_incompatible_wheels(bundled: "BundledPlatform") -> List[str]:  # noqa: F821
    """Shipped wheels whose interpreter/platform tags don't match this host.

    The shipped wheels are the author's build (e.g. cp312/linux_x86_64), NOT universal — a
    consumer on a different Python minor or OS can't install them. Universal wheels
    (``py3-none-any`` like the plugin) are skipped. Returns human-readable reasons; empty ==
    all installable here.
    """
    import sys

    problems: List[str] = []
    host_py = host_python_tag()
    host_is_linux = sys.platform.startswith("linux")
    for w in bundled.wheels:
        if not w.python_tag or w.python_tag.startswith("py") or w.abi_tag in (None, "none"):
            continue  # universal / non-CPython-pinned wheel
        if w.python_tag != host_py:
            problems.append(f"{Path(w.path).name}: built for {w.python_tag}, host is {host_py}")
        if w.platform_tag and w.platform_tag != "any" and "linux" in w.platform_tag and not host_is_linux:
            problems.append(f"{Path(w.path).name}: built for {w.platform_tag}, host is {sys.platform}")
    return problems


def sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    """Streaming sha256 of a file (wheels are large — don't slurp)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def parse_wheel_tags(filename: str) -> Dict[str, Optional[str]]:
    """Extract (python_tag, abi_tag, platform_tag) from a wheel filename.

    Returns None tags for a non-conforming name rather than raising — the artifact still ships;
    it just carries no install-time compatibility gate.
    """
    m = _WHEEL_RE.match(Path(filename).name)
    if not m:
        return {"python_tag": None, "abi_tag": None, "platform_tag": None}
    return {"python_tag": m["py"], "abi_tag": m["abi"], "platform_tag": m["plat"]}


def make_wheel_artifact(src: Path, rel_path: str) -> WheelArtifact:
    """Build a WheelArtifact (path within bundle + sha + size + tags) for one wheel file."""
    tags = parse_wheel_tags(src.name)
    return WheelArtifact(
        path=rel_path,
        sha256=sha256_file(src),
        size=src.stat().st_size,
        python_tag=tags["python_tag"],
        abi_tag=tags["abi_tag"],
        platform_tag=tags["platform_tag"],
    )


def render_install_sh() -> str:
    """A self-contained installer: make a venv, install the shipped wheels, then the deps.

    Path-relative and idempotent so it works wherever ``pull`` materializes the folder. Takes an
    optional venv path as ``$1`` (default ``./venv``) so a caller (tt-model pull) can point it at
    a managed location.
    """
    return f"""#!/usr/bin/env bash
# Install this self-contained TT model package into a fresh venv.
# Usage: ./{INSTALL_SCRIPT} [venv-path]   (default: ./venv)
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
VENV="${{1:-$HERE/venv}}"

python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
# The author's built wheels: ttnn (custom kernels compiled in), base empty-vLLM, plugin.
"$VENV/bin/pip" install "$HERE/{WHEELS_DIR}"/*.whl
# Dependencies (torch is the CPU build — Tenstorrent does not use CUDA torch).
"$VENV/bin/pip" install --extra-index-url {_PYTORCH_CPU_INDEX} -r "$HERE/{REQUIREMENTS}"
echo "installed into $VENV"
"""


def render_run_sh(manifest: Manifest) -> str:
    """A standalone launcher that wires the engine env and serves the OpenAI endpoint.

    Sets the non-obvious env this stack needs (LD_PRELOAD of _ttnncpp.so; TT_METAL_HOME at the
    installed ttnn; EXTRA_MODELS_DIR at this folder so the plugin finds vllm_metadata.json;
    single-chip fabric-off defaults) plus any model-specific ``manifest.env``, then launches vLLM.
    Works with only tt-model absent — ``tt-model serve`` is the managed path, this is the raw one.
    """
    weights = manifest.weights.repo_id if manifest.weights else ""
    mesh_device = (manifest.mesh.topology if manifest.mesh and manifest.mesh.topology else "") or ""
    extra_env = "".join(
        f'export {k}="{v}"\n' for k, v in (manifest.env or {}).items()
    )
    # The tt_transformers adapter reads HF_MODEL from the env (not vLLM's --model), so export it.
    hf_export = f'export HF_MODEL="${{HF_MODEL:-{weights}}}"\n' if weights else ""
    # The TT vLLM backend REQUIRES a supported batch size and a concrete block_size (its default
    # of 256 / None both fail), so always emit them — from the manifest's resources, with the
    # known-good tt_transformers defaults when unset.
    res = manifest.resources
    max_num_seqs = (res.max_num_seqs if res and res.max_num_seqs else 32)
    block_size = (res.block_size if res and res.block_size else 64)
    serving = f"--max_num_seqs {max_num_seqs} --block_size {block_size}"
    if res and res.max_model_len:
        serving += f" --max_model_len {res.max_model_len}"
    if res and res.extra_args:
        serving += " " + " ".join(str(a) for a in res.extra_args)
    return f"""#!/usr/bin/env bash
# Serve this model on TT hardware. Assumes ./{INSTALL_SCRIPT} has been run.
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
VENV="${{VENV:-$HERE/venv}}"
PYBIN="$VENV/bin/python"

# Locate ttnn WITHOUT importing it — importing loads _ttnn.so, which is exactly what needs the
# LD_PRELOAD below (chicken-and-egg). find_spec resolves the path without executing the module.
TTNN_DIR="$("$PYBIN" -c 'import importlib.util,os;print(os.path.dirname(importlib.util.find_spec("ttnn").origin))')"
export LD_PRELOAD="$TTNN_DIR/build/lib/_ttnncpp.so"   # glibc static-TLS workaround
export TT_METAL_HOME="$TTNN_DIR"
# EXTRA_MODELS_DIR is a PARENT of per-model bundle folders; the plugin scans its children for
# each vllm_metadata.json (so the metadata lives in {METADATA_DIR}/<model>/, not the bundle root).
export EXTRA_MODELS_DIR="$HERE/{METADATA_DIR}"
export TT_VLLM_BUILTIN_MODELS=0
export VLLM_PLUGINS=tt,tt_model_registry
export PYTHONPATH="$HERE/{METAL_DIR}:${{PYTHONPATH:-}}"   # resolves the adapter's models.* imports
export MESH_DEVICE="${{MESH_DEVICE:-{mesh_device}}}"
export TT_METAL_VISIBLE_DEVICES="${{TT_METAL_VISIBLE_DEVICES:-0}}"
{hf_export}{extra_env}exec "$PYBIN" -m vllm.entrypoints.openai.api_server --model "{weights}" {serving} "$@"
"""


def stage_package(
    staged: Path,
    *,
    name: str,
    arch: str,
    ttnn_wheel: Path,
    metal_dir: Path,
    vllm_metadata: dict,
    tt_kernel_version: str,
    vllm_wheel: Optional[Path] = None,
    plugin_wheel: Optional[Path] = None,
    extra_wheels: Optional[List[Path]] = None,
    weights: Optional[WeightsRef] = None,
    device_count: int = 1,
    mesh: Optional[Mesh] = None,
    env: Optional[Dict[str, str]] = None,
    resources: Optional[Resources] = None,
    tt_metal_version: str = "unknown",
    firmware_min: Optional[str] = None,
) -> Manifest:
    """Materialize the running-folder layout under ``staged`` and return the v5 manifest.

    Copies the author's wheels into ``wheels/``, the modified metal tree into ``metal/``, writes
    ``vllm_metadata.json`` (the EXTRA_MODELS_DIR contract), the generated ``install.sh``/``run.sh``,
    a ``requirements.txt`` (from the metal tree if present), and ``tt_kernel_manifest.json``.
    No network.
    """
    staged.mkdir(parents=True, exist_ok=True)
    wheels_root = staged / WHEELS_DIR
    wheels_root.mkdir(exist_ok=True)

    def _copy_wheel(src: Path) -> WheelArtifact:
        dest = wheels_root / src.name
        shutil.copy2(src, dest)
        return make_wheel_artifact(dest, f"{WHEELS_DIR}/{src.name}")

    ttnn_art = _copy_wheel(ttnn_wheel)
    vllm_art = _copy_wheel(vllm_wheel) if vllm_wheel else None
    plugin_art = _copy_wheel(plugin_wheel) if plugin_wheel else None
    extra_arts = [_copy_wheel(w) for w in (extra_wheels or [])]

    # Embed the author's modified metal-community tree (skip caches/venvs/artifacts).
    shutil.copytree(
        metal_dir,
        staged / METAL_DIR,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".git", "venv", ".venv", "model_cache",
            "generated", "*.log", ".pytest_cache", "dist", "build_*",
        ),
    )

    # requirements.txt: prefer the metal tree's, else a minimal note.
    req_src = metal_dir / "requirements.txt"
    if req_src.is_file():
        shutil.copy2(req_src, staged / REQUIREMENTS)
    else:
        (staged / REQUIREMENTS).write_text("# add pip deps here (torch is installed CPU-only)\n")

    # The plugin's EXTRA_MODELS_DIR contract: it scans the *children* of EXTRA_MODELS_DIR for a
    # vllm_metadata.json in each. So the metadata goes in a per-model SUBFOLDER under METADATA_DIR
    # (which run.sh points EXTRA_MODELS_DIR at) — NOT the bundle root, where it would be missed.
    safe_key = name.replace("/", "__")
    model_bundle = staged / METADATA_DIR / safe_key
    model_bundle.mkdir(parents=True, exist_ok=True)
    (model_bundle / bundles.VLLM_METADATA_NAME).write_text(json.dumps(vllm_metadata, indent=2))

    bundled = BundledPlatform(
        ttnn_wheel=ttnn_art,
        vllm_wheel=vllm_art,
        plugin_wheel=plugin_art,
        extra_wheels=extra_arts,
        metal_dir=METAL_DIR,
        requirements=REQUIREMENTS,
        install_script=INSTALL_SCRIPT,
        run_script=RUN_SCRIPT,
        firmware_min=firmware_min,
    )
    entrypoint = Entrypoint(
        **{"class": vllm_metadata["main_class"], "arch_name": vllm_metadata["arch"]}
    )
    manifest = Manifest(
        schema_version="5",
        name=name,
        tt_metal_version=tt_metal_version,
        arch=arch,
        device_count=device_count,
        build_key=None,  # self-contained/kernels-less: kernels are inside the shipped ttnn wheel
        kernel_count=0,
        fast_path_kernels=None,
        producer=Producer(
            tt_kernel_version=tt_kernel_version,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            hostname=socket.gethostname(),
        ),
        weights=weights,
        entrypoint=entrypoint,
        mesh=mesh,
        env=env or {},
        resources=resources,
        bundled=bundled,
    )

    # Generated scripts (run.sh reads the manifest's weights/mesh/env).
    (staged / INSTALL_SCRIPT).write_text(render_install_sh())
    (staged / RUN_SCRIPT).write_text(render_run_sh(manifest))
    for s in (INSTALL_SCRIPT, RUN_SCRIPT):
        (staged / s).chmod(0o755)

    (staged / "tt_kernel_manifest.json").write_text(manifest.to_json())
    return manifest


__all__ = [
    "WHEELS_DIR",
    "METAL_DIR",
    "sha256_file",
    "parse_wheel_tags",
    "make_wheel_artifact",
    "host_python_tag",
    "host_incompatible_wheels",
    "render_install_sh",
    "render_run_sh",
    "stage_package",
]
