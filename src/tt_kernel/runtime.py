# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Install the *runtime* half of a bundle: the Python runner wheel and the weights.

This module deliberately holds everything that is NOT kernel-cache plumbing (cache.py)
or bundle-repo I/O (hub.py): downloading an arbitrary HF *model* repo, pip-installing
the shipped runner wheel into the active venv, and composing the ready-to-run serve
command. It NEVER imports the dispatch serving package — the runner spec is an opaque
string and dispatch is only *detected*, never imported (the decoupling boundary).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .manifest import WeightsRef

ENV_MODELS_DIR = "TT_KERNEL_MODELS_DIR"
# tt-kernel's own minimal OpenAI server for a legacy (dispatch-contract) runner. This
# replaces the retired dispatch runtime the runner used to hand off to. Used to BUILD the
# serve command; the module is only run as a subprocess, never imported here.
_LEGACY_SERVE_MODULE = "tt_kernel.legacy_serve"
# The env var the Tenstorrent vLLM plugin reads to discover extra model bundle folders.
# tt-kernel points it at the local bundles_dir at serve time; the plugin scans it and
# registers every model folder found there.
ENV_EXTRA_MODELS_DIR = "EXTRA_MODELS_DIR"
_VLLM_PKG = "vllm"
_VLLM_PLUGIN_PKG = "vllm_tt_plugin"


def resolve_models_dir(models_dir: Optional[str], repo_id: str) -> Path:
    """Where to download a model's weights.

    Resolution (env-then-flag, mirroring cache.resolve_out_root): ``--models-dir`` >
    ``TT_KERNEL_MODELS_DIR`` > ``~/.cache/tt-kernel/models``. The repo id is nested as
    ``<base>/<org>/<name>`` (no slash-flattening) so the path round-trips cleanly for
    ``rm``/serve and never collides.
    """
    explicit = models_dir if models_dir is not None else os.environ.get(ENV_MODELS_DIR)
    if explicit:
        base = Path(explicit).expanduser()
    else:
        home = os.environ.get("HOME")
        base = (Path(home) / ".cache" if home else Path("/tmp")) / "tt-kernel" / "models"
    # repo_id is "org/name" (or just "name"); keep its structure under base.
    return base.joinpath(*repo_id.split("/"))


def download_weights(weights: WeightsRef, dest: Path) -> Path:
    """Download a model's weights from the Hub into ``dest`` (resumable).

    Thin wrapper over ``huggingface_hub.snapshot_download`` — content-addressed and
    resumable, so a half-finished download just continues on a re-pull.
    """
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=weights.repo_id,
        repo_type=weights.repo_type,
        revision=weights.revision,
        allow_patterns=weights.allow_patterns,
        ignore_patterns=weights.ignore_patterns,
        local_dir=str(dest),
    )
    return Path(path)


def pip_install_wheels(
    wheel_paths: List[Path],
    *,
    python: Optional[str] = None,
    pip_args: Optional[List[str]] = None,
) -> None:
    """pip-install the shipped runner wheel(s) into the target interpreter's env.

    ``--no-deps`` is deliberate: the runner wheel is tree-shaken/self-contained, and we
    must NOT let pip pull a conflicting ``ttnn`` from PyPI (ttnn/tt-metal is the platform
    the version warning points at, never a vendored dep). ``python`` overrides the target
    interpreter (default: the venv tt-kernel itself runs in, where ttnn should live);
    ``pip_args`` is an escape hatch for the rare case the wheel really needs extra flags.
    Raises CalledProcessError on a non-zero pip exit.
    """
    if not wheel_paths:
        return
    exe = python or sys.executable
    cmd = [exe, "-m", "pip", "install", "--no-deps"]
    if pip_args:
        cmd.extend(pip_args)
    cmd.extend(str(p) for p in wheel_paths)
    subprocess.run(cmd, check=True)


def ttnn_importable(python: Optional[str] = None) -> bool:
    """Whether ``ttnn`` is importable from the target interpreter.

    Used to warn when pip would install the runner into a venv that lacks ttnn (e.g.
    tt-kernel was installed via pipx into its own env). For the default interpreter we
    check this process directly; for an explicit ``--python`` we shell out.
    """
    if python is None or python == sys.executable:
        return importlib.util.find_spec("ttnn") is not None
    try:
        proc = subprocess.run(
            [python, "-c", "import importlib.util,sys;"
             "sys.exit(0 if importlib.util.find_spec('ttnn') else 1)"],
            timeout=30,
        )
        return proc.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def legacy_serve_available() -> bool:
    """Whether the legacy-runner server (``tt_kernel.legacy_serve``) can actually run here.

    The module itself always imports (it ships with tt-kernel), so what matters is its
    web-server dependencies. DETECTION only — ``find_spec`` never imports them.
    """
    try:
        return (importlib.util.find_spec("fastapi") is not None
                and importlib.util.find_spec("uvicorn") is not None)
    except (ImportError, ValueError):
        return False


def runner_spec_importable(spec: str, python: Optional[str] = None) -> bool:
    """Whether a *reference* runner's module is importable in the target interpreter.

    Used by ``pull`` to verify a not-shipped (reference) runner is actually present
    before claiming the bundle is ready. The module is the part of ``spec`` before the
    ``:`` (``"pkg.mod:Runner"``) or the dotted prefix (``"pkg.mod.Runner"``). DETECTION
    only — ``find_spec`` never imports the module. Mirrors ``ttnn_importable``: checks
    this process directly for the default interpreter, else shells out.
    """
    module = spec.split(":", 1)[0] if ":" in spec else spec.rsplit(".", 1)[0]
    if python is None or python == sys.executable:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False
    try:
        proc = subprocess.run(
            [python, "-c", "import importlib.util,sys;"
             f"sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"],
            timeout=30,
        )
        return proc.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def serve_argv(
    model: str,
    *,
    runner_spec: str,
    python: Optional[str] = None,
) -> List[str]:
    """Argv for the legacy-runner server — ``tt_kernel.legacy_serve``.

    ``runner_spec`` is required: the shim serves one specific runner (there is no dynamic
    / bare-repo path anymore — that was the retired dispatch runtime). ``model`` is the
    local weights dir the runner loads.
    """
    return [python or "python", "-m", _LEGACY_SERVE_MODULE,
            "--runner", runner_spec, "--model", str(model)]


def serve_command(runner_spec: str, weights_path: Path) -> str:
    """The exact ready-to-run line for the legacy-runner OpenAI server."""
    return " ".join(serve_argv(str(weights_path), runner_spec=runner_spec))


# --------------------------------------------------------------------------- vLLM
def vllm_available() -> bool:
    """Whether the vLLM serving stack (vLLM + the Tenstorrent plugin) is importable here.

    DETECTION only (``find_spec``) — never imports vLLM. Both ``vllm`` and ``vllm_tt_plugin``
    must be present for the serve handoff to work. Provenance-agnostic: upstream vLLM plus
    the standalone plugin is the supported path; the legacy fork also satisfies this.
    """
    try:
        return (
            importlib.util.find_spec(_VLLM_PKG) is not None
            and importlib.util.find_spec(_VLLM_PLUGIN_PKG) is not None
        )
    except (ImportError, ValueError):
        return False


def vllm_serve_env(bundles_dir: Path, launch_env: Optional[dict] = None) -> dict:
    """The full environment for a vLLM serve subprocess.

    Overlays ``EXTRA_MODELS_DIR`` (pointed at ``bundles_dir`` so the plugin discovers the
    pulled model) and the bundle's per-machine launch env (``MESH_DEVICE``,
    ``TT_*_VER``, ``VLLM_USE_V1``, weights-dir vars, …) onto the current environment.
    """
    env = dict(os.environ)
    env[ENV_EXTRA_MODELS_DIR] = str(bundles_dir)
    if launch_env:
        env.update({str(k): str(v) for k, v in launch_env.items()})
    return env


def vllm_serve_argv(launch_command: List[str], *, python: Optional[str] = None) -> List[str]:
    """The argv to launch the vLLM OpenAI server, from a bundle's per-machine command.

    The bundle's ``launch.command`` is authoritative (e.g. ``["python3",
    "server_example_tt.py", "--model", ...]``). ``python`` optionally overrides the
    interpreter when the command's first token is a bare ``python``/``python3``.
    """
    argv = [str(c) for c in launch_command]
    if python and argv and argv[0] in ("python", "python3"):
        argv[0] = python
    return argv


def health_check(base_url: str, *, timeout: float = 5.0) -> tuple[bool, str]:
    """Probe an OpenAI-compatible server's ``/v1/models`` (cheap liveness check).

    Returns ``(ok, detail)``. Uses only the stdlib so tt-kernel adds no HTTP dependency.
    """
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — localhost health probe
            code = resp.getcode()
            return (200 <= code < 300, f"GET {url} -> {code}")
    except urllib.error.URLError as exc:
        return (False, f"GET {url} failed: {exc}")
    except (OSError, ValueError) as exc:
        return (False, f"GET {url} failed: {exc}")


__all__ = [
    "resolve_models_dir",
    "download_weights",
    "pip_install_wheels",
    "ttnn_importable",
    "runner_spec_importable",
    "legacy_serve_available",
    "serve_argv",
    "serve_command",
    "vllm_available",
    "vllm_serve_env",
    "vllm_serve_argv",
    "health_check",
    "ENV_EXTRA_MODELS_DIR",
]
