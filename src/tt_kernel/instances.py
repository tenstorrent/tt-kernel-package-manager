# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The tt-metal *instance registry* — the supply side of version resolution.

A v4 manifest declares the *demand* (``platform.ttnn`` / ``runtime.version`` /
``runtime.plugin_version`` ranges). This module describes what's actually installed on **this
host** so tt-kernel can link a model to the right build. An *instance* is an **activation** —
an interpreter plus the env (``TT_METAL_HOME`` / ``PYTHONPATH`` / ``LD_LIBRARY_PATH``) that
makes one specific ttnn (and its vLLM + plugin) importable — not just a version string, because
tt-metal is frequently a source build.

Instances come from three sources, unioned:

- **active** — the interpreter tt-kernel is running under (always a candidate, so a box with
  one build behaves exactly as before this module existed);
- **registry** — explicit entries in ``~/.config/tt-kernel/instances.json`` (the manager owns
  this file; tt-cli or the user writes entries auto-scan can't find);
- **scan** — auto-discovered tt-metal checkouts under a set of scan roots.

Version resolution for a non-active instance happens **out-of-process** (the manager has one
interpreter), by running ``<python> -c ...`` under that instance's env — a fusion of the
``python=``-parameterized shell-out in :mod:`tt_kernel.runtime` and the capture-stdout idiom in
:mod:`tt_kernel.metal` / :mod:`tt_kernel.device`.

This module is **declarative**: it discovers, probes, and selects. It never installs a
tt-metal — provisioning is a future tt-cli concern.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import metal
from .toolchain import _parse_version, version_satisfies

# Marker that identifies a tt-metal source checkout (mirrors metal.detect_tt_metal_home).
_CHECKOUT_MARKER = ("tt_metal", "hw", "inc")
# Interpreters to look for inside a scanned checkout, most-canonical first.
_CHECKOUT_PYTHONS = ("build/python_env/bin/python", "python_env/bin/python")
_PROBE_TIMEOUT = 30


# --------------------------------------------------------------------------- model
@dataclass
class Instance:
    """One activatable tt-metal build."""

    name: str
    python: str  # absolute interpreter path
    tt_metal_home: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    source: str = "registry"  # "active" | "scan" | "registry"

    @property
    def is_active(self) -> bool:
        return self.source == "active"

    def activation_env(self) -> Dict[str, str]:
        """The extra env this instance layers on top of the process environment.

        From ``tt_metal_home`` we derive the full activation a tt-metal source build needs —
        ``TT_METAL_HOME``, ``PYTHONPATH`` (the checkout root), and ``LD_LIBRARY_PATH``
        (``build/lib``, where ``libtt_metal.so`` lives). Without the last two, serving a
        *scanned* checkout (whose ``env`` is empty) would inherit the shell's ambient
        ``LD_LIBRARY_PATH`` from a different build and load the wrong ``.so`` at import. These
        derived values are marked for **prepend** at launch (``vllm_serve_env``) so they add to,
        rather than clobber, any inherited path. An explicit ``env`` entry always wins.
        """
        out: Dict[str, str] = {}
        if self.tt_metal_home:
            home = self.tt_metal_home
            out["TT_METAL_HOME"] = home
            # Prepend markers (see PREPEND_VARS / vllm_serve_env).
            out["PYTHONPATH"] = home
            out["LD_LIBRARY_PATH"] = os.path.join(home, "build", "lib")
        out.update(self.env)  # explicit registry env overrides the derived defaults
        return out

    def to_entry(self) -> dict:
        """Registry-file JSON form (active/scan instances are never persisted)."""
        return {"name": self.name, "python": self.python,
                "tt_metal_home": self.tt_metal_home, "env": self.env}


@dataclass
class InstanceVersions:
    """The three versions that determine whether an instance satisfies a manifest."""

    ttnn: Optional[str] = None
    vllm: Optional[str] = None
    plugin: Optional[str] = None


@dataclass
class Candidate:
    instance: Instance
    versions: InstanceVersions
    satisfies: bool


@dataclass
class SelectionResult:
    """Verdict of resolving a manifest's ranges against the host's instances."""

    chosen: Optional[Instance]
    candidates: List[Candidate]
    reason: str


# --------------------------------------------------------------------------- registry file
def _registry_path() -> Path:
    """``$XDG_CONFIG_HOME/tt-kernel/instances.json`` else ``~/.config/tt-kernel/instances.json``.

    The manager owns this file; it is the first ``~/.config`` user in the codebase (the bundle
    index and caches live under ``~/.cache``). Config, not cache: it records user/tt-cli intent.
    """
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "tt-kernel" / "instances.json"


def _load() -> dict:
    path = _registry_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        # A corrupt registry is surfaced (not silently treated as empty), because the next
        # write would otherwise erase every registered instance. We move it aside so a stray
        # ``instances add`` can't clobber it, and continue from empty.
        try:
            backup = path.with_suffix(".json.corrupt")
            path.replace(backup)
            sys.stderr.write(f"tt-kernel: registry {path} was corrupt; moved to {backup}\n")
        except OSError:
            pass
        return {}
    except OSError:
        return {}


def _save(data: dict) -> None:
    """Atomically persist the registry (tempfile + os.replace).

    A plain ``write_text`` truncates then writes, so a Ctrl-C mid-write leaves a corrupt file
    (which ``_load`` would then read as empty, and the next write would make permanent). The
    atomic replace guarantees readers only ever see the old or the new whole file.
    """
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".instances-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(data, indent=2))
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def default_scan_roots() -> List[str]:
    """Where auto-scan looks for tt-metal checkouts by default (overridable in the file)."""
    home = os.path.expanduser("~")
    roots = [home, os.path.join(home, "tt-metal"), "/opt"]
    for env in ("TT_METAL_HOME", "TT_METAL_RUNTIME_ROOT"):
        val = os.environ.get(env)
        if val:
            roots.append(val)
    # de-dupe, preserve order
    seen: set = set()
    return [r for r in roots if r and not (r in seen or seen.add(r))]


# --------------------------------------------------------------------------- registry CRUD
def add_instance(name: str, python: str, *, tt_metal_home: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None) -> Instance:
    """Add (or replace) a manual registry entry. Returns the stored Instance."""
    inst = Instance(name=name, python=str(Path(python).expanduser()),
                    tt_metal_home=tt_metal_home, env=env or {}, source="registry")
    data = _load()
    entries = [e for e in data.get("instances", []) if e.get("name") != name]
    entries.append(inst.to_entry())
    data["instances"] = entries
    _save(data)
    return inst


def remove_instance(name: str) -> bool:
    """Remove a manual registry entry by name. Returns True if something was removed."""
    data = _load()
    entries = data.get("instances", [])
    kept = [e for e in entries if e.get("name") != name]
    if len(kept) == len(entries):
        return False
    data["instances"] = kept
    _save(data)
    return True


# --------------------------------------------------------------------------- discovery
def active_instance() -> Instance:
    """The interpreter tt-kernel runs under — always a candidate (preserves prior behavior)."""
    return Instance(name="active", python=sys.executable,
                    tt_metal_home=metal.detect_tt_metal_home(), env={}, source="active")


def registry_instances() -> List[Instance]:
    out: List[Instance] = []
    for e in _load().get("instances", []):
        py = e.get("python")
        if not py:
            continue
        out.append(Instance(name=e.get("name") or py, python=py,
                            tt_metal_home=e.get("tt_metal_home"),
                            env=dict(e.get("env") or {}), source="registry"))
    return out


def scan_checkouts(roots: Optional[List[str]] = None) -> List[Tuple[str, Optional[str]]]:
    """Find tt-metal checkouts under ``roots``. Returns ``(tt_metal_home, python|None)`` pairs.

    A checkout is a dir whose ``tt_metal/hw/inc`` exists (the marker
    :func:`metal.detect_tt_metal_home` uses). Scan is shallow — each root itself and its
    immediate children — so a home dir full of projects is cheap to probe. ``python`` is the
    first of ``build/python_env`` / ``python_env`` that exists, or None (found but not
    launchable — surfaced to the user, never auto-selected).
    """
    roots = roots if roots is not None else default_scan_roots()
    found: List[Tuple[str, Optional[str]]] = []
    seen: set = set()
    for root in roots:
        rp = Path(root).expanduser()
        # Default roots include $HOME and /opt, which aren't guaranteed listable on a shared
        # or hardened host — an unreadable root must be skipped, not crash `pull`/`serve`
        # (which reach scan via the main path) with a PermissionError.
        candidates = [rp]
        if rp.is_dir():
            try:
                candidates += list(rp.iterdir())
            except OSError:
                continue
        for c in candidates:
            try:
                if not c.is_dir() or c.joinpath(*_CHECKOUT_MARKER).is_dir() is False:
                    continue
            except OSError:
                continue
            home = str(c.resolve())
            if home in seen:
                continue
            seen.add(home)
            py: Optional[str] = None
            for rel in _CHECKOUT_PYTHONS:
                p = c / rel
                if p.is_file():
                    py = str(p.resolve())
                    break
            found.append((home, py))
    return found


def scan_instances(roots: Optional[List[str]] = None) -> List[Instance]:
    """Launchable instances from :func:`scan_checkouts` (skips checkouts with no interpreter)."""
    out: List[Instance] = []
    for home, py in scan_checkouts(roots):
        if not py:
            continue
        out.append(Instance(name=f"scan:{Path(home).name}", python=py,
                            tt_metal_home=home, env={}, source="scan"))
    return out


def all_instances(roots: Optional[List[str]] = None) -> List[Instance]:
    """Union of active + registry + scan, deduped by (realpath(python), tt_metal_home).

    Precedence on a collision: active > registry > scan (an explicit entry overrides a scanned
    one; the active interpreter is always kept as itself). Order otherwise preserves
    active-first, then registry, then scan.
    """
    scan_roots = roots
    if scan_roots is None:
        scan_roots = _load().get("scan_roots") or default_scan_roots()
    ordered = [active_instance()] + registry_instances() + scan_instances(scan_roots)
    out: List[Instance] = []
    seen: set = set()

    def _key(i: Instance):
        try:
            rp = os.path.realpath(i.python)
        except OSError:
            rp = i.python
        return (rp, i.tt_metal_home or "")

    for inst in ordered:
        k = _key(inst)
        if k in seen:
            continue
        seen.add(k)
        out.append(inst)
    return out


# --------------------------------------------------------------------------- probing
# One-liner run under a candidate interpreter: print "ttnn|vllm|plugin" versions (blank if
# absent). Kept dependency-free so it runs under any tt-metal venv.
_PROBE_SRC = (
    "import importlib.metadata as m\n"
    "def v(*names):\n"
    " for n in names:\n"
    "  try: return m.version(n)\n"
    "  except Exception: pass\n"
    " return ''\n"
    "print('|'.join([v('ttnn','tt-metal','tt_metal','metal-libs'),"
    "v('vllm'),v('vllm_tt_plugin','vllm-tt-plugin')]))\n"
)


def _cache_key(inst: Instance) -> str:
    """Cache key for an instance's probed versions.

    Keyed on the SAME identity ``all_instances`` dedupes on — ``(realpath(python),
    tt_metal_home)`` — so two registry entries sharing one interpreter but differing in
    ``TT_METAL_HOME`` don't collide on a single slot (and get each other's versions). The
    interpreter's mtime is folded in too, so rebuilding a checkout in place busts the cache
    instead of serving a stale ttnn version forever.
    """
    try:
        rp = os.path.realpath(inst.python)
    except OSError:
        rp = inst.python
    try:
        mtime = str(int(os.path.getmtime(rp)))
    except OSError:
        mtime = "0"
    return f"{rp}::{inst.tt_metal_home or ''}::{mtime}"


def probe_versions(inst: Instance, *, use_cache: bool = True) -> InstanceVersions:
    """Resolve an instance's ttnn / vLLM / plugin versions.

    The active instance resolves in-process (cheap). Any other instance is probed
    **out-of-process** under its activation env; ttnn falls back to ``git describe`` in
    ``tt_metal_home`` (mirroring :func:`metal.resolve_version`). Results are cached (keyed by
    :func:`_cache_key`) in the registry file's ``version_cache`` (``use_cache=False``
    refreshes). Never raises — an unresolved version is ``None`` and treated as "assume OK".
    """
    if inst.is_active:
        return InstanceVersions(ttnn=metal.resolve_version(),
                                vllm=metal._vllm_version(),
                                plugin=metal._vllm_plugin_version())

    key = _cache_key(inst)
    if use_cache:
        cached = _load().get("version_cache", {}).get(key)
        if isinstance(cached, dict):
            return InstanceVersions(ttnn=cached.get("ttnn"), vllm=cached.get("vllm"),
                                    plugin=cached.get("plugin"))

    versions = InstanceVersions()
    probe_ok = False
    env = {**os.environ, **inst.activation_env()}
    try:
        out = subprocess.run([inst.python, "-c", _PROBE_SRC], capture_output=True, text=True,
                             timeout=_PROBE_TIMEOUT, env=env, check=True)
        parts = (out.stdout.strip().split("|") + ["", "", ""])[:3]
        versions = InstanceVersions(ttnn=parts[0] or None, vllm=parts[1] or None,
                                    plugin=parts[2] or None)
        probe_ok = True
    except (subprocess.SubprocessError, OSError):
        pass  # missing .so / timeout / bad interpreter — treated as "unknown", not cached
    if versions.ttnn is None and inst.tt_metal_home:
        versions.ttnn = _git_describe(inst.tt_metal_home)

    # Only cache a result we actually resolved. Caching an all-None from a probe that merely
    # timed out would permanently mark the instance "unknown = assume OK".
    if probe_ok or any((versions.ttnn, versions.vllm, versions.plugin)):
        data = _load()
        data.setdefault("version_cache", {})[key] = {
            "ttnn": versions.ttnn, "vllm": versions.vllm, "plugin": versions.plugin}
        _save(data)
    return versions


def _git_describe(home: str) -> Optional[str]:
    import shutil
    if not shutil.which("git"):
        return None
    for argv in (["git", "-C", home, "describe", "--tags", "--always", "--dirty"],
                 ["git", "-C", home, "rev-parse", "HEAD"]):
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=True)
            val = out.stdout.strip()
            if val:
                return val
        except (subprocess.SubprocessError, OSError):
            continue
    return None


# --------------------------------------------------------------------------- selection
def select(*, ttnn: Optional[str] = None, vllm: Optional[str] = None,
           plugin: Optional[str] = None, roots: Optional[List[str]] = None,
           use_cache: bool = True) -> SelectionResult:
    """Choose the **newest** instance satisfying every declared range.

    An instance satisfies when each supplied range accepts the corresponding installed version
    (via :func:`toolchain.version_satisfies`; an unresolvable/None version is "assume OK", so a
    dev checkout is never falsely excluded). Candidates are sorted by parsed ttnn version
    descending; the first satisfying one wins. Returns every candidate (for a declarative
    report) plus the choice and a human-readable reason.
    """
    candidates: List[Candidate] = []
    for inst in all_instances(roots):
        v = probe_versions(inst, use_cache=use_cache)
        ok = (version_satisfies(v.ttnn, ttnn) is not False
              and version_satisfies(v.vllm, vllm) is not False
              and version_satisfies(v.plugin, plugin) is not False)
        candidates.append(Candidate(instance=inst, versions=v, satisfies=ok))

    def _sort_key(c: Candidate):
        parsed = _parse_version(c.versions.ttnn or "") or ()
        return (parsed, 0 if c.instance.is_active else -1)  # tie-break: prefer non-active newer

    satisfying = sorted([c for c in candidates if c.satisfies], key=_sort_key, reverse=True)
    if satisfying:
        chosen = satisfying[0].instance
        v = satisfying[0].versions
        return SelectionResult(chosen=chosen, candidates=candidates,
                               reason=f"{chosen.name} (ttnn={v.ttnn}, vllm={v.vllm}, "
                                      f"plugin={v.plugin})")
    ranges = ", ".join(f"{k}{r}" for k, r in
                       (("ttnn", ttnn), ("vllm", vllm), ("plugin", plugin)) if r)
    return SelectionResult(chosen=None, candidates=candidates,
                           reason=f"no installed tt-metal instance satisfies: {ranges or '(none)'}")


__all__ = [
    "Instance", "InstanceVersions", "Candidate", "SelectionResult",
    "active_instance", "registry_instances", "scan_instances", "all_instances",
    "add_instance", "remove_instance", "default_scan_roots",
    "probe_versions", "select",
]
