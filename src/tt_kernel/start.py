# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""``tt-model start`` — one guided path from nothing to a served model.

This orchestrates existing commands rather than reimplementing them: ``auth`` for the
token, ``toolchain``/``metal`` for the environment, ``hub`` to resolve the bundle, then the
same pull and serve code paths everything else uses. Behaviour is unchanged; what is new is
that the four steps are named up front, reported as they happen, and stop at the first one
that cannot succeed.

Interactive, but never *required*: every prompt has a non-interactive path (``--token``,
``$HF_TOKEN``, an existing HF token store, ``--yes``), and a non-TTY stdin skips prompting
entirely so this stays usable in CI.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import auth, console, hub, localdb, metal, runtime, toolchain

PHASES = ["Account", "Validate", "Model", "Serve"]
PHASE_DETAIL = {
    "Account": "Hugging Face token",
    "Validate": "tt-metal, vLLM, hardware",
    "Model": "resolve + pull the bundle",
    "Serve": "launch the OpenAI server",
}


def stdin_is_interactive() -> bool:
    """Whether we may prompt at all.

    A prompt on a closed or piped stdin does not fail — it hangs, or reads EOF and takes a
    default the user never saw. Checking up front lets the caller degrade to "explain what
    is missing and exit" instead.
    """
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


# ----------------------------------------------------------------------------- account
@dataclass
class Account:
    name: Optional[str]
    source: str            # how we got the token
    logged_in: bool = False


def resolve_account(token: Optional[str] = None, *, allow_prompt: bool = True) -> Account:
    """Establish an HF identity, prompting only if there is no other way.

    Order: an explicit --token, then whatever huggingface_hub already has (its own token
    store or $HF_TOKEN), then a prompt. The token is read with getpass and handed straight
    to huggingface_hub — it is never echoed, logged, or held anywhere we render from.
    """
    if token:
        auth.login(token=token)
        me = auth.whoami()
        return Account(name=(me or {}).get("name"), source="--token", logged_in=bool(me))

    me = auth.whoami()
    if me:
        source = "$HF_TOKEN" if os.environ.get("HF_TOKEN") else "the HF token store"
        return Account(name=me.get("name"), source=source, logged_in=True)

    if not allow_prompt:
        return Account(name=None, source="none", logged_in=False)

    # A prompt must never run inside a capturing step() — it would be hidden and the CLI
    # would appear to hang. Callers arrange that; this only reads.
    secret = console.secret("Hugging Face token (input hidden): ")
    if not secret.strip():
        return Account(name=None, source="none", logged_in=False)
    auth.login(token=secret.strip())
    me = auth.whoami()
    return Account(name=(me or {}).get("name"), source="prompt", logged_in=bool(me))


# ---------------------------------------------------------------------------- validate
@dataclass
class Environment:
    report: object                       # toolchain.ToolchainReport
    arch: Optional[str]
    device_count: int
    device_source: Optional[str]
    port: int
    port_free: bool
    conflicts: list

    @property
    def blockers(self) -> List[str]:
        """What would stop a serve, in the order the user should fix it."""
        out = []
        for c in self.report.components:
            if not c.adequate:
                out.append(f"{c.name} is not adequate: {c.message}")
        if not self.port_free:
            out.append(f"port {self.port} is already in use")
        return out


def validate(port: int = 8000, *, arch_override: Optional[str] = None) -> Environment:
    """Check everything a serve needs, in one pass, before touching the network."""
    dev = metal.detect_device(arch_override=arch_override)
    return Environment(
        report=toolchain.check_toolchain(),
        arch=dev.arch,
        device_count=dev.device_count,
        device_source=dev.source,
        port=port,
        port_free=not runtime.port_in_use(port),
        conflicts=toolchain.check_environment(),
    )


# ------------------------------------------------------------------------------- model
def resolve_bundle(model: str) -> Tuple[str, str]:
    """Map what the user typed to an installed-or-installable bundle id.

    Returns ``(repo_id, how)``. Accepts a bundle id directly; for a bare HF model id it
    looks for an already-installed bundle that serves it, so `tt-model start Qwen/Qwen3-32B`
    works once `mando2222/Qwen3-32B-blackhole` is installed.
    """
    entry = localdb.get(model)
    if entry:
        return model, "installed"

    tail = model.split("/")[-1].lower()
    for e in localdb.all_entries():
        rid = e.get("repo_id") or ""
        if rid.split("/")[-1].lower().startswith(tail):
            return rid, f"installed bundle matching {model}"
    return model, "to pull"


def is_installed(repo_id: str) -> bool:
    entry = localdb.get(repo_id)
    return bool(entry and entry.get("bundle_path"))


# ------------------------------------------------------------------- model discovery
@dataclass
class Choice:
    repo_id: str
    label: str


def installed_choices() -> List[Choice]:
    """Installed bundles, newest-looking first, as menu entries.

    `tt-model start` with no argument used to be a bare "Missing argument 'model'." — which
    is the one thing a guided command should not do. If there is something to serve, offer
    it; the user should not have to run `list` to find out what they already have.
    """
    out: List[Choice] = []
    for e in localdb.all_entries():
        repo_id = e.get("repo_id")
        if not repo_id or not e.get("bundle_path"):
            continue
        bits = [b for b in (e.get("backend"), e.get("arch")) if b]
        if e.get("self_contained"):
            bits.append("self-contained")
        suffix = f"  ({' · '.join(bits)})" if bits else ""
        out.append(Choice(repo_id=repo_id, label=f"{repo_id}{suffix}"))
    return sorted(out, key=lambda c: c.repo_id)
