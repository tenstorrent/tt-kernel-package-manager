# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Validate the surrounding toolchain — and only ever *warn*, never install.

tt-kernel is the front door, but it is not a package installer for the platform: it
expects the serving stack (tt-metal and the vLLM fork/plugin) to already be present on the
system and merely checks it is *adequate*, warning (with the required version) when it is
not. This keeps tt-kernel's dependency surface tiny and never mutates the user's environment.
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import metal

# The default serving stack is tt-metal plus the vLLM fork/plugin. tt-lang and tt-api are
# leftovers from an earlier serving prototype and are not part of the vLLM path, so they are
# not checked here (checking them only produced spurious "missing dependency" warnings).
LOCK = {
    "tt-metal": "0.72.0",
}

# Distribution names each component may be installed under (first match wins).
_VLLM_DISTS = ("vllm",)


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


def _vllm_component() -> ComponentReport:
    """Presence check for the Tenstorrent vLLM serving stack (fork + plugin).

    The fork tracks the ``dev`` branch, so this is presence-based rather than a strict
    version floor: both ``vllm`` and the ``vllm_tt_plugin`` package must be importable.
    """
    required = "tenstorrent/vllm@dev + plugin"
    version = _dist_version(_VLLM_DISTS)
    if not _spec_present("vllm"):
        return ComponentReport(
            "vllm", False, None, required, False,
            "not found — install the Tenstorrent vLLM fork + plugin (see scripts/install.sh)",
        )
    if not _spec_present("vllm_tt_plugin"):
        return ComponentReport(
            "vllm", True, version, required, False,
            "vllm present but the TT plugin (vllm_tt_plugin) is not importable — "
            "pip install -e plugins/vllm-tt-plugin",
        )
    return ComponentReport("vllm", True, version, required, True, "ok (vllm + TT plugin present)")


def check_toolchain() -> ToolchainReport:
    """Inspect the local tt-metal + vLLM serving stack. Never imports the heavy modules and
    never installs anything — detection via metadata, find_spec, and the tt-metal version
    resolver already used by ``compare``. tt-lang and tt-api (leftovers from an earlier
    serving prototype) are not part of the vLLM path and are not checked."""
    tt_metal_version = metal.resolve_version()
    return ToolchainReport(components=[
        _component("tt-metal", found=bool(tt_metal_version) or _spec_present("ttnn"),
                   version=tt_metal_version),
        _vllm_component(),
    ])


__all__ = ["LOCK", "ComponentReport", "ToolchainReport", "check_toolchain"]
