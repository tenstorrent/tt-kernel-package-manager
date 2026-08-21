# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""``tt-model install`` — provision the serving stack on this host.

This is the one place in tt-model that *installs* the surrounding platform, and it is
deliberately explicit and opt-in. Everything else stays declarative: ``doctor``,
``toolchain`` and ``instances`` only ever check and report, which is what makes their
verdicts trustworthy.

What it sets up, on top of an existing tt-metal (ttnn) environment:

- the Tenstorrent vLLM fork + TT plugin (the serving layer)
- tt-model itself (the distribution layer)

then verifies with ``tt-model doctor``.

tt-metal is *not* installed here beyond offering the PyPI ``ttnn`` route: building it is
the heavy, hardware-coupled part. That assumption is ENFORCED rather than documented —
without a usable interpreter and an importable ttnn we stop before spending ~450MB on an
environment that could never serve a model.

PROTECTED FACT: the Tenstorrent vLLM plugin work lives on the ``dev`` branch. We clone and
install ``dev`` — never ``main``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from . import console, instances, metal

VLLM_REPO = "https://github.com/tenstorrent/vllm"
DEFAULT_VLLM_REF = "dev"
DEFAULT_VLLM_DIR = "~/dispatch/vllm"
TTNN_PYPI_SPEC = "ttnn>=0.72"

# One fixed roadmap, so k/N never drifts with flags.
PHASES = ["Preflight", "vLLM fork", "Serving layer", "tt-model", "Verify"]
PHASE_DETAIL = {
    "Preflight": "target interpreter, ttnn, vLLM ref",
    "vLLM fork": f"clone or reuse tenstorrent/vllm@{DEFAULT_VLLM_REF}",
    "Serving layer": "vLLM fork + TT plugin  (~450MB)",
    "tt-model": "editable install",
    "Verify": "tt-model doctor",
}

# Exit codes, documented so callers (and the bootstrap shim) can rely on them.
EXIT_OK = 0
EXIT_PREFLIGHT = 1
EXIT_USAGE = 2
EXIT_INADEQUATE = 3


# --------------------------------------------------------------- interpreter resolution
@dataclass
class Target:
    """The interpreter we would install into, and how we chose it."""
    python: Optional[str]
    source: str
    # Why it is unusable, if it is. Empty means "runs".
    problem: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.python is not None and self.problem is None


def _runs(python: str) -> bool:
    """True if this path is actually an interpreter we can execute."""
    try:
        return subprocess.run([python, "-c", "0"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def can_import(python: str, module: str) -> bool:
    """Whether ``module`` imports in the target interpreter. Never imports it into ours —
    ttnn pulls in the whole device stack."""
    try:
        return subprocess.run([python, "-c", f"import {module}"],
                              capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def resolve_target(venv: Optional[str] = None) -> Target:
    """Pick the interpreter to install into: --venv > $VIRTUAL_ENV > a scanned tt-metal
    venv with importable ttnn > the running interpreter.

    An explicit --venv is *validated here* rather than assumed. Passing a path that does
    not exist used to be announced as "Using python: ..." and then produce a misleading
    "ttnn is not importable" — true, but only because there was no interpreter at all —
    before dying three steps later with a raw shell error. Naming the real problem at the
    first opportunity is the whole point of a preflight.
    """
    if venv:
        root = Path(venv).expanduser()
        python = root / "bin" / "python3"
        if not root.exists():
            return Target(str(python), "--venv", f"no such directory: {root}")
        if not python.is_file():
            alt = root / "bin" / "python"
            if alt.is_file():
                python = alt
            else:
                return Target(str(python), "--venv", f"no bin/python3 under {root}")
        if not _runs(str(python)):
            return Target(str(python), "--venv", f"{python} exists but will not run")
        return Target(str(python), "--venv")

    active = os.environ.get("VIRTUAL_ENV")
    if active:
        python = Path(active) / "bin" / "python3"
        if python.is_file() and _runs(str(python)):
            return Target(str(python), "the active $VIRTUAL_ENV")

    for _home, py in instances.scan_checkouts():
        if py and can_import(py, "ttnn"):
            return Target(py, "auto-detected tt-metal venv (ttnn importable)")

    import sys as _sys
    return Target(_sys.executable, "the running interpreter")


def ttnn_candidates() -> List[Tuple[str, str]]:
    """``(tt_metal_home, python)`` pairs on this box whose python can import ttnn.

    Used to turn a preflight failure into a concrete suggestion instead of a dead end.
    """
    out = []
    for home, py in instances.scan_checkouts():
        if py and can_import(py, "ttnn"):
            out.append((home, py))
    return out


# ------------------------------------------------------------------------------ preflight
@dataclass
class Preflight:
    """The verdict of the checks that run before anything is installed."""
    target: Target
    ttnn_ok: bool = False
    vllm_ref: str = DEFAULT_VLLM_REF
    blockers: List[str] = field(default_factory=list)
    # (command, why) pairs. Kept structured rather than pre-formatted strings so the
    # renderer can put the note on its own line — a command that wraps mid-flag is not
    # copy-pasteable, which defeats the point of suggesting it.
    routes: List[Tuple[str, str]] = field(default_factory=list)
    escape: Optional[Tuple[str, str]] = None

    @property
    def ok(self) -> bool:
        return not self.blockers


def check(venv: Optional[str] = None, *, vllm_ref: str = DEFAULT_VLLM_REF,
          allow_no_ttnn: bool = False) -> Preflight:
    """Everything that must hold before we spend ~450MB. Pure enough to unit-test: the only
    I/O is probing the filesystem and running the candidate interpreter.
    """
    target = resolve_target(venv)
    pre = Preflight(target=target, vllm_ref=vllm_ref)

    if not target.usable:
        pre.blockers.append(target.problem or "no usable interpreter")
        cands = ttnn_candidates()
        if cands:
            pre.routes.append((f"tt-model install --venv {Path(cands[0][1]).parent.parent}",
                               "a tt-metal venv on this box with ttnn"))
        else:
            pre.routes.append(("tt-model instances list",
                               "the interpreters tt-model can see"))
        return pre

    pre.ttnn_ok = can_import(target.python, "ttnn")
    if not pre.ttnn_ok:
        # Two real ways forward. The PyPI one is a single command that people have been
        # missing entirely, because the old wording framed tt-metal as built-separately and
        # out of scope — which reads as a dead end rather than a one-liner.
        pre.routes.append((f'pip install "{TTNN_PYPI_SPEC}"', "~250MB from PyPI, no build"))
        cands = ttnn_candidates()
        if cands:
            pre.routes.append((f"tt-model install --venv {Path(cands[0][1]).parent.parent}",
                               "a tt-metal already built on this box"))
        else:
            pre.routes.append(("tt-model install --venv <tt-metal>/python_env",
                               "a tt-metal you built yourself"))
        if not allow_no_ttnn:
            pre.blockers.append("ttnn is not importable")
            pre.escape = ("tt-model install --allow-no-ttnn",
                          "install the serving layers anyway — cannot serve a model")
    return pre


# ------------------------------------------------------------------- pip stream aggregator
class PipProgress:
    """Turn `pip install` chatter into one activity label.

    Pure: ``feed(line) -> str | None``. Deliberately reports two different things, because
    pip only tells us a real total once it starts installing:

    - while resolving, the count of packages collected so far and no bar — we do not know
      the denominator yet and will not invent one;
    - while installing, ``k/N`` from pip's own "Installing collected packages: a, b, c"
      line, which *is* exact.

    Bytes are a plain running counter for the same reason: pip reports per-wheel sizes with
    no total, so a percentage would be fabricated.
    """

    _SIZE = re.compile(r"\((\d+(?:\.\d+)?)\s*([kKMG]?B)\)")
    _UNITS = {"B": 1, "kB": 1e3, "KB": 1e3, "MB": 1e6, "GB": 1e9}

    def __init__(self, label="Installing"):
        self.label = label
        self.collected = 0
        self.installing_total = 0
        self.installed = 0
        self.bytes = 0
        self.done = False

    def feed(self, line: str) -> Optional[str]:
        s = line.strip()
        if s.startswith("Collecting "):
            self.collected += 1
        elif s.startswith("Installing collected packages:"):
            names = s.split(":", 1)[1]
            self.installing_total = len([n for n in names.split(",") if n.strip()])
        elif s.startswith("Successfully installed"):
            self.installed = self.installing_total
            self.done = True
        elif s.startswith(("Downloading ", "Using cached ")):
            m = self._SIZE.search(s)
            if m:
                self.bytes += float(m.group(1)) * self._UNITS.get(m.group(2), 1)
        elif self.installing_total and s.startswith(("Attempting uninstall:", "  Attempting uninstall:")):
            self.installed = min(self.installed + 1, self.installing_total)
        else:
            return None
        return self.activity()

    def activity(self) -> str:
        size = f" · {console.fmt_bytes(self.bytes)}" if self.bytes else ""
        if self.installing_total:
            bar = console.progress_bar(self.installed, self.installing_total)
            return (f"{self.label}  {bar}  {self.installed}/{self.installing_total} "
                    f"packages{size}")
        if self.collected:
            return f"{self.label}  resolving · {self.collected} collected{size}"
        return f"{self.label}{size}"


# ---------------------------------------------------------------------------- the install
def clone_or_reuse_vllm(vllm_dir: Path, ref: str) -> Tuple[bool, str]:
    """Clone the fork if absent. Returns ``(cloned, detail)``; reuse is not an error."""
    if (vllm_dir / ".git").is_dir():
        try:
            branch = subprocess.run(["git", "-C", str(vllm_dir), "branch", "--show-current"],
                                    capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            branch = "unknown"
        return False, f"already at {vllm_dir} ({branch or 'detached'}) — reusing"
    vllm_dir.parent.mkdir(parents=True, exist_ok=True)
    rc, out = console.run_with_activity(
        ["git", "clone", "-b", ref, VLLM_REPO, str(vllm_dir)],
        label=f"Cloning tenstorrent/vllm@{ref}",
    )
    if rc != 0:
        raise RuntimeError(f"git clone failed (exit {rc}):\n{out[-2000:]}")
    return True, f"cloned {ref} -> {vllm_dir}"


def pip_install(python: str, args: List[str], *, label: str,
                env: Optional[dict] = None) -> Tuple[int, str]:
    """Run one pip install, reporting progress on a single line we own."""
    progress = PipProgress(label)
    return console.run_with_activity(
        [python, "-m", "pip", "install", *args],
        label=label, env=env, parse=progress.feed,
    )


# ---------------------------------------------------------------------------- verify
@dataclass
class Verdict:
    """What ``doctor`` found in the environment we just installed into."""
    report: object                 # toolchain.ToolchainReport
    conflicts: list = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return EXIT_OK if getattr(self.report, "ok", False) else EXIT_INADEQUATE


def verify(python: str) -> Verdict:
    """Re-check the toolchain *in the interpreter we installed into*.

    Deliberately probes ``python`` rather than our own process: `tt-model install --venv X`
    installs into X, and reporting on the manager's environment instead would describe a
    different machine state than the one just created.
    """
    from . import toolchain

    import sys as _sys

    if os.path.realpath(python) == os.path.realpath(_sys.executable):
        report = toolchain.check_toolchain()
    else:
        report = _remote_report(python)
    return Verdict(report=report, conflicts=toolchain.check_environment(python))


def _remote_report(python: str):
    """Build a ToolchainReport for another interpreter by asking it directly."""
    from . import toolchain

    probe = (
        "import importlib.util as u, json\n"
        "def v(*names):\n"
        "    import importlib.metadata as md\n"
        "    for n in names:\n"
        "        try: return md.version(n)\n"
        "        except Exception: pass\n"
        "    return None\n"
        "print(json.dumps({'ttnn': v('ttnn','tt-metal','tt_metal','metal-libs'),"
        " 'vllm': v('vllm'), 'plugin': v('vllm_tt_plugin','vllm-tt-plugin'),"
        " 'has_ttnn': u.find_spec('ttnn') is not None,"
        " 'has_vllm': u.find_spec('vllm') is not None,"
        " 'has_plugin': u.find_spec('vllm_tt_plugin') is not None}))"
    )
    data = {}
    try:
        out = subprocess.run([python, "-c", probe], capture_output=True, text=True, timeout=180)
        if out.returncode == 0:
            import json
            data = json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError):
        data = {}

    ttnn_v = data.get("ttnn")
    vllm_v = data.get("vllm")
    has_plugin = bool(data.get("has_plugin"))
    components = [
        toolchain._component("tt-metal", found=bool(ttnn_v) or bool(data.get("has_ttnn")),
                             version=ttnn_v),
    ]
    vllm_ok = bool(data.get("has_vllm")) and has_plugin
    components.append(toolchain.ComponentReport(
        name="vllm", found=bool(data.get("has_vllm")), version=vllm_v,
        required="tenstorrent/vllm@dev + plugin", adequate=vllm_ok,
        message=("ok (vllm + TT plugin present)" if vllm_ok
                 else "not found — install the Tenstorrent vLLM fork + plugin"),
    ))
    return toolchain.ToolchainReport(components=components)


def pip_error_line(output: str) -> str:
    """The most useful single line from a failed pip run.

    A card is not a log viewer: pick the line that names the failure and leave the rest in
    the captured output printed beneath.
    """
    lines = [ln.strip() for ln in (output or "").splitlines() if ln.strip()]
    for marker in ("ERROR: ", "error: ", "No matching distribution",
                   "Could not find a version", "Permission denied"):
        for ln in reversed(lines):
            if marker in ln:
                return ln[:160]
    return lines[-1][:160] if lines else ""
