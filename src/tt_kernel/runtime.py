"""Install the *runtime* half of a v2 bundle: the Python runner wheel and the weights.

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
# The dotted path of dispatch's serve entry point — used only to BUILD the printed
# command string and to DETECT availability. Never imported.
_DISPATCH_SERVE_MODULE = "tt_inference_server.dispatch.serve"
_DISPATCH_PKG = "tt_inference_server.dispatch"


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


def dispatch_available() -> bool:
    """Whether the dispatch serving package is importable here (DETECTION only).

    Uses ``find_spec`` so we never import (and never execute) dispatch — tt-kernel must
    not depend on it.
    """
    try:
        return importlib.util.find_spec(_DISPATCH_PKG) is not None
    except (ImportError, ValueError):
        return False


def serve_command(runner_spec: str, weights_path: Path) -> str:
    """The exact ready-to-run line for dispatch's OpenAI-compatible server."""
    return (
        f"python -m {_DISPATCH_SERVE_MODULE} serve --unsafe "
        f"--runner {runner_spec} {weights_path}"
    )


__all__ = [
    "resolve_models_dir",
    "download_weights",
    "pip_install_wheels",
    "ttnn_importable",
    "dispatch_available",
    "serve_command",
]
