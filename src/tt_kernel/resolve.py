# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Detect-only bundle resolution — "does a bundle exist for this id, and what's in it?"

This is the front-door query the orchestrator (``tt-kernel run``) and the dispatch
courtesy notice both build on. It is **detection only**: it reads the local install
index and (optionally) the published manifest, and it NEVER imports a runner or opens a
device. The verdict it returns is what drives the resolution ladder:

  Tier 1 — bundle with a runner   -> run exactly what the author built
  Tier 2 — bundle, kernels-only   -> dispatch runner + precompiled cache (hit on disk)
  Tier 3 — no bundle              -> dispatch's dynamic path on the bare HF repo

``resolve`` composes existing pieces (``localdb.get`` + ``hub.fetch_manifest``); it adds
no new I/O primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import localdb


@dataclass
class BundleResolution:
    """The detect-only verdict for one model identifier.

    ``source`` is ``"local"`` (recorded as installed), ``"hub"`` (published but not
    installed), or ``None`` (no bundle found). ``weights_path`` is only set for an
    installed bundle whose weights were downloaded.
    """

    repo_id: str
    exists: bool = False
    installed: bool = False
    source: Optional[str] = None
    has_runner: bool = False
    is_packaged: Optional[bool] = None
    runner_spec: Optional[str] = None
    entry_point: Optional[str] = None
    weights_repo: Optional[str] = None
    weights_path: Optional[str] = None
    arch: Optional[str] = None
    tt_metal_version: Optional[str] = None
    build_key: Optional[int] = None
    # Serving backend of the bundle's runner: "dispatch" (default/legacy) or "vllm".
    backend: Optional[str] = None
    # For an installed vLLM bundle: the local folder laid into bundles_dir (== a model
    # folder under EXTRA_MODELS_DIR). None until pulled.
    bundle_path: Optional[str] = None

    @property
    def is_vllm(self) -> bool:
        return self.backend == "vllm"

    @property
    def tier(self) -> int:
        """Which resolution tier this bundle drives (3 == no bundle / dynamic floor)."""
        if self.exists and self.has_runner:
            return 1
        if self.exists:
            return 2
        return 3

    @property
    def serve_target(self) -> Optional[str]:
        """The identifier to hand to dispatch's dynamic path: the local weights dir if
        present, else the bundle's HF weights repo. ``None`` when neither is known
        (caller falls back to the original repo_id)."""
        return self.weights_path or self.weights_repo


def _from_local(repo_id: str, entry: dict) -> BundleResolution:
    spec = entry.get("runner_spec")
    backend = entry.get("backend") or "dispatch"
    bundle_path = entry.get("bundle_path")
    # A vLLM bundle "has a runner" by virtue of its installed bundle folder, even though
    # its dispatch ``runner_spec`` is empty.
    has_runner = bool(spec) or (backend == "vllm" and bundle_path is not None)
    return BundleResolution(
        repo_id=repo_id,
        exists=True,
        installed=True,
        source="local",
        has_runner=has_runner,
        is_packaged=entry.get("python_installed"),
        runner_spec=spec,
        entry_point=entry.get("entry_point"),
        weights_repo=entry.get("weights_repo"),
        weights_path=entry.get("weights_path"),
        arch=entry.get("arch"),
        tt_metal_version=entry.get("tt_metal_version"),
        build_key=entry.get("build_key"),
        backend=backend,
        bundle_path=bundle_path,
    )


def _from_manifest(repo_id: str, manifest) -> BundleResolution:
    runner = manifest.runner
    weights = manifest.weights
    backend = runner.backend if runner else None
    return BundleResolution(
        repo_id=repo_id,
        exists=True,
        installed=False,
        source="hub",
        has_runner=runner is not None,
        is_packaged=runner.is_packaged if runner else None,
        runner_spec=runner.spec if runner else None,
        entry_point=runner.entry_point if runner else None,
        weights_repo=weights.repo_id if weights else None,
        weights_path=None,
        arch=manifest.arch,
        tt_metal_version=manifest.tt_metal_version,
        build_key=manifest.build_key,
        backend=backend,
        bundle_path=None,
    )


def resolve(
    repo_id: str, *, revision: Optional[str] = None, local_only: bool = False
) -> BundleResolution:
    """Resolve a model identifier to its bundle verdict (detect-only).

    Order: the local install index first (no network), then — unless ``local_only`` —
    the published manifest on the Hub. Any Hub failure (not a bundle / 404 / offline)
    resolves to "no bundle" rather than raising, so callers can treat resolution as a
    cheap, never-throwing probe. ``local_only=True`` is the right mode for a hot path
    (e.g. dispatch's courtesy notice) that must not make network calls.
    """
    entry = localdb.get(repo_id)
    if entry is not None:
        return _from_local(repo_id, entry)

    if not local_only:
        try:
            from . import hub

            manifest = hub.fetch_manifest(repo_id, revision)
            return _from_manifest(repo_id, manifest)
        except Exception:  # noqa: BLE001 — not a bundle / offline / 404 => "no bundle"
            pass

    return BundleResolution(repo_id=repo_id, exists=False)


__all__ = ["BundleResolution", "resolve"]
