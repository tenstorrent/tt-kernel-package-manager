# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Validate the surrounding toolchain — and only ever *warn*, never install.

tt-kernel is the front door, but it is not a package installer for the platform: it
expects tt-metal, tt-lang, and tt-dispatch to already be present on the system
and merely checks they are *adequate* versions, warning (with the required version) when
they are not. This keeps tt-kernel's dependency surface tiny and never mutates the user's
environment.

The version lock below mirrors ``~/dispatch/docs/hardware.md`` ("Toolchain Version Lock")
— bump both together. It is plain constants (no TOML/parser dependency, 3.9-safe).
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from . import metal

# Minimum adequate versions of the surrounding stack. Mirror docs/hardware.md.
LOCK = {
    "tt-metal": "0.72.0",
    "tt-lang": "1.1.3",
    "tt-dispatch": "0.15.0",
}

# Distribution names each component may be installed under (first match wins).
_TT_LANG_DISTS = ("ttl", "tt-lang", "ttlang", "tt_lang")
_TT_IS_DISTS = ("tt-dispatch", "tt_dispatch")


@dataclass
class ComponentReport:
    name: str
    found: bool
    version: Optional[str]
    required: str
    adequate: bool
    message: str


@dataclass
class ToolchainReport:
    components: List[ComponentReport]

    @property
    def ok(self) -> bool:
        return all(c.adequate for c in self.components)

    @property
    def problems(self) -> List[ComponentReport]:
        return [c for c in self.components if not c.adequate]


def _parse_version(s: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Extract the leading dotted-numeric version from a string.

    Tolerates a ``v`` prefix and git-describe suffixes ("0.72.0-5-gabc" -> (0,72,0)).
    Returns None when there is no leading numeric component (e.g. a bare git sha).
    """
    if not s:
        return None
    s = s.strip().lstrip("vV")
    # Drop git-describe / build / prerelease suffixes ("0.72.0-5-gabc", "1.1.3+light").
    for sep in ("+", "-"):
        s = s.split(sep, 1)[0]
    nums: List[int] = []
    for part in s.split("."):
        if part.isdigit():
            nums.append(int(part))
        else:
            break
    return tuple(nums) if nums else None


def _meets(version: Optional[str], minimum: str) -> Optional[bool]:
    """True/False if ``version`` >= ``minimum``; None if ``version`` is unparseable."""
    v = _parse_version(version)
    if v is None:
        return None
    return v >= _parse_version(minimum)


def _dist_version(dists: Tuple[str, ...]) -> Optional[str]:
    for dist in dists:
        try:
            return md.version(dist)
        except md.PackageNotFoundError:
            continue
        except Exception:  # noqa: BLE001 — never let metadata lookup break a check
            continue
    return None


def _spec_present(*module_names: str) -> bool:
    for m in module_names:
        try:
            if importlib.util.find_spec(m) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return False


def _tt_dispatch_version() -> Optional[str]:
    """Version of tt-dispatch. It is normally run from a checkout (not pip
    installed), so fall back from dist metadata to the repo ``VERSION`` file located
    relative to the importable package."""
    v = _dist_version(_TT_IS_DISTS)
    if v:
        return v
    try:
        spec = importlib.util.find_spec("tt_dispatch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    pkg_dir = Path(list(spec.submodule_search_locations)[0])
    version_file = pkg_dir.parent / "VERSION"  # <repo>/VERSION, sibling of the package
    if version_file.is_file():
        try:
            return version_file.read_text().strip()
        except OSError:
            return None
    return None


def _component(name: str, *, found: bool, version: Optional[str]) -> ComponentReport:
    required = LOCK[name]
    if not found:
        return ComponentReport(name, False, None, required, False,
                               f"not found — install {name} >= {required}")
    verdict = _meets(version, required)
    if verdict is None:
        return ComponentReport(name, True, version, required, True,
                               f"version {version!r} not comparable; assuming OK (require >= {required})")
    if verdict:
        return ComponentReport(name, True, version, required, True, "ok")
    return ComponentReport(name, True, version, required, False,
                           f"version {version} is older than required {required} — upgrade")


def check_toolchain() -> ToolchainReport:
    """Inspect the local tt-metal / tt-lang / tt-dispatch. Never imports the
    heavy modules and never installs anything — detection via metadata, find_spec, and
    the tt-metal version resolver already used by ``compare``."""
    tt_metal_version = metal.resolve_version()
    tt_lang_version = _dist_version(_TT_LANG_DISTS)

    return ToolchainReport(components=[
        _component("tt-metal", found=bool(tt_metal_version) or _spec_present("ttnn"),
                   version=tt_metal_version),
        _component("tt-lang", found=_spec_present("ttl") or bool(tt_lang_version),
                   version=tt_lang_version),
        _component("tt-dispatch",
                   found=_spec_present("tt_dispatch"),
                   version=_tt_dispatch_version()),
    ])


__all__ = ["LOCK", "ComponentReport", "ToolchainReport", "check_toolchain"]
