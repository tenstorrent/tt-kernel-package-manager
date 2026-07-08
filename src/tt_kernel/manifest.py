# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The compatibility manifest — the correctness core of tt-kernel.

A manifest pins everything the cached binaries depend on so that ``pull`` can refuse
an install that would silently miss (or, worse, load wrong binaries). It records the
``build_key`` that names the cache subtree on disk *and* the inputs that determine it,
because a pure-Python consumer cannot compute its local ``build_key`` without opening a
device. See ``compute_build_key`` in tt-metal ``build_env_manager.cpp:164-184`` and the
per-kernel hash in ``program_descriptors.cpp:126-141``.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "3"


class FileEntry(BaseModel):
    """One file in the cache subtree, indexed for integrity verification.

    ``path`` is relative to the ``<build_key>/`` root of the subtree.
    """

    path: str
    sha256: str
    size: int


class BuildKeyInputs(BaseModel):
    """The inputs to tt-metal's ``compute_build_key`` (build_env_manager.cpp:164-184).

    ``harvesting_mask`` only participates in the build_key when coordinate
    virtualization is disabled, so it is excluded from comparison when
    ``coordinate_virtualization_enabled`` is true (mirrors the C++ logic).
    """

    dispatch_core_type: str = "WORKER"
    dispatch_core_axis: str = "ROW"
    num_hw_cqs: int = 1
    coordinate_virtualization_enabled: bool = True
    harvesting_mask: int = 0
    compile_hash_string: str = ""


class Producer(BaseModel):
    tt_kernel_version: str
    created_at: str
    hostname: Optional[str] = None
    # Absolute path of the producer's tt-metal source root, as embedded in the kernel
    # cache's .dephash dependency paths. Lets a consumer on a different host (different
    # checkout path / HOME) rewrite those tree-dep prefixes to its own tt-metal so the
    # pulled cache hits instead of recompiling. None => producer couldn't detect it
    # (consumer falls back to in-cache relocation only, correct on the same host).
    tt_metal_home: Optional[str] = None


class RunnerPayload(BaseModel):
    """The runner a bundle serves through. ``backend`` selects which serving layer and,
    with it, which runner contract the payload satisfies:

    - ``backend == "dispatch"`` (default, legacy): a Python runner for the dispatch
      serving layer (``tt_api.serve``), contract ``generate()``/``generate_stream()``/
      ``benchmark()``. Two modes:

      - **packaged**: ``wheels`` is non-empty — the wheel(s) are stored under ``python/``
        in the bundle and indexed in ``Manifest.files`` (path prefix ``python/``) so the
        existing integrity check covers them. ``pull`` pip-installs them.
      - **reference**: ``wheels`` is empty — the runner is *not* shipped; the consumer is
        expected to already have it or to install it from ``source``.

      ``spec`` is the opaque ``"module:Class"`` string dispatch selects via
      ``serve --runner``; tt-kernel records it but never imports it.

    - ``backend == "vllm"``: the model is served through the Tenstorrent vLLM plugin. The
      payload is a self-contained bundle *folder* (``bundle_dir``) holding a plugin-owned
      ``vllm_metadata.json`` (arch name, main-class path, per-machine launch command, HF
      weights ref) plus the ``VllmGeneratorAdapter`` class and its dependencies. tt-kernel
      lays the folder into ``EXTRA_MODELS_DIR`` at serve time; the plugin scans it and
      registers the model. ``vllm_metadata.json`` — not this payload — is the source of
      truth for the serving contract, so ``spec``/``wheels`` are unused for vLLM.
    """

    spec: str = ""  # "module:Class" for dispatch --runner (dispatch backend only)
    wheels: List[str] = Field(default_factory=list)  # filenames under python/; empty => reference
    entry_point: Optional[str] = None  # name registered under tt_models.runners
    source: Optional[str] = None  # where to get a reference (not-shipped) runner: pip name / git URL
    requires_python: Optional[str] = None  # informational
    backend: str = "dispatch"  # "dispatch" | "vllm"
    # For backend=="vllm": path (within the bundle repo) of the folder holding
    # vllm_metadata.json + the adapter class + its deps. None for dispatch backend.
    bundle_dir: Optional[str] = None

    @property
    def is_packaged(self) -> bool:
        """True when the bundle ships the runner wheel(s); False => reference-only.

        For the vLLM backend the folder itself (``bundle_dir``) is the shipped payload."""
        return bool(self.wheels) or (self.backend == "vllm" and self.bundle_dir is not None)

    @property
    def is_vllm(self) -> bool:
        return self.backend == "vllm"


class WeightsRef(BaseModel):
    """Where to fetch model weights from the Hub.

    The single, actionable record of which model a bundle targets: a normal HF model
    repo, downloaded separately from the kernel bundle (skippable with ``--no-weights``).
    """

    repo_id: str
    revision: Optional[str] = None
    allow_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None
    repo_type: str = "model"


class Manifest(BaseModel):
    """Root document of a bundle (``tt_kernel_manifest.json``)."""

    schema_version: str = SCHEMA_VERSION
    name: str
    tt_metal_version: str  # MUST match local (per-kernel hash dependency)
    arch: str  # blackhole | wormhole_b0 | ...
    device_count: int = 1
    # uint64 naming the cache subtree on disk. ``None`` => a kernels-less bundle (no
    # precompiled cache shipped): the vLLM path JITs at first-run warmup into tt-metal's
    # own local cache, and a dispatch bundle without kernels falls back to dynamic JIT.
    build_key: Optional[int] = None
    build_key_inputs: BuildKeyInputs = Field(default_factory=BuildKeyInputs)
    kernel_count: int = 0
    # Whether the cache carries the traced-decode / on-device-lm_head kernels a fast-path
    # consumer (DISPATCH_TRACE/DISPATCH_ONDEVICE_LMHEAD) needs (#6). None => not recorded
    # (older bundle); False => baseline-only (fast-path serving will re-JIT those kernels).
    fast_path_kernels: Optional[bool] = None
    files: List[FileEntry] = Field(default_factory=list)
    producer: Producer
    # Runtime payload (both optional): a runner to dispatch to (packaged or reference)
    # and the model weights to fetch. Absent => a pure kernel (warm compile-cache) bundle.
    runner: Optional[RunnerPayload] = None
    weights: Optional[WeightsRef] = None

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        """Parse and validate a manifest, rejecting any unsupported schema version.

        tt-kernel is pre-release: there is exactly one supported schema. A bundle from a
        different (older/newer) schema is refused outright rather than silently
        half-read — re-publish it with a matching tt-kernel.
        """
        m = cls.model_validate_json(text)
        if m.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported bundle schema_version {m.schema_version!r}; this tt-kernel "
                f"requires schema {SCHEMA_VERSION!r}. Re-publish the bundle with a current "
                "tt-kernel."
            )
        return m

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)


class Incompatibility(BaseModel):
    """A single reason a bundle may not be usable on the local environment."""

    field: str
    expected: str  # from the manifest
    detected: str  # from the local environment
    fatal: bool  # True => never installable; False => guarded behind --force


class CompatibilityReport(BaseModel):
    """Verdict of comparing a manifest against the detected local environment."""

    compatible: bool  # no issues at all
    issues: List[Incompatibility] = Field(default_factory=list)

    @property
    def has_fatal(self) -> bool:
        return any(i.fatal for i in self.issues)

    @property
    def forceable(self) -> bool:
        """True when the only blockers are non-fatal (installable with --force)."""
        return bool(self.issues) and not self.has_fatal


def compare(manifest: Manifest, local: "LocalEnv") -> CompatibilityReport:  # noqa: F821
    """Compare a manifest against the detected local environment.

    ``local`` is a ``metal.LocalEnv`` (imported lazily to avoid a cycle). Comparison
    rules, from the verified tt-metal source:

    - ``arch`` mismatch is **fatal** — the binaries are for a different ISA.
    - ``tt_metal_version`` mismatch is a hard block (non-fatal: forceable) — per-kernel
      hashes won't match, so the cache would silently miss.
    - build_key inputs that differ change the build_key integer, so the consumer's
      tt-metal would look under a different directory => silent miss (forceable).
    - ``device_count`` mismatch is a warning (forceable).

    A **kernels-less** bundle (``build_key is None``) ships no precompiled cache, so none
    of the cache-dependent gates apply: only ``arch`` (still fatal — the adapter/kernels
    JIT for a specific ISA) and ``device_count`` are checked. ``tt_metal_version`` is left
    to ``runner_version_advisory`` (a warning), because a kernels-less runner is reusable
    and the vLLM path re-warms its own kernels.
    """
    issues: List[Incompatibility] = []
    inp = manifest.build_key_inputs

    if local.arch and manifest.arch != local.arch:
        issues.append(
            Incompatibility(field="arch", expected=manifest.arch, detected=local.arch, fatal=True)
        )

    if manifest.build_key is None:
        # Kernels-less: skip every cache-dependent gate; only device_count is a warning.
        if local.device_count and manifest.device_count != local.device_count:
            issues.append(
                Incompatibility(
                    field="device_count",
                    expected=str(manifest.device_count),
                    detected=str(local.device_count),
                    fatal=False,
                )
            )
        return CompatibilityReport(compatible=not issues, issues=issues)

    if local.tt_metal_version and manifest.tt_metal_version != local.tt_metal_version:
        issues.append(
            Incompatibility(
                field="tt_metal_version",
                expected=manifest.tt_metal_version,
                detected=local.tt_metal_version,
                fatal=False,
            )
        )

    if local.device_count and manifest.device_count != local.device_count:
        issues.append(
            Incompatibility(
                field="device_count",
                expected=str(manifest.device_count),
                detected=str(local.device_count),
                fatal=False,
            )
        )

    # harvesting_mask only affects the build_key when virtualization is disabled.
    if not inp.coordinate_virtualization_enabled and local.harvesting_mask is not None:
        if inp.harvesting_mask != local.harvesting_mask:
            issues.append(
                Incompatibility(
                    field="harvesting_mask",
                    expected=str(inp.harvesting_mask),
                    detected=str(local.harvesting_mask),
                    fatal=False,
                )
            )

    # If --probe gave us a real local build_key, an integer mismatch is decisive.
    if local.build_key is not None and manifest.build_key != local.build_key:
        issues.append(
            Incompatibility(
                field="build_key",
                expected=str(manifest.build_key),
                detected=str(local.build_key),
                fatal=False,
            )
        )

    return CompatibilityReport(compatible=not issues, issues=issues)


def runner_version_advisory(manifest: Manifest, local: "LocalEnv") -> Optional[Incompatibility]:  # noqa: F821
    """Non-fatal version check for the runner wheel + weights.

    Unlike the kernel ``compare()`` gate (which hard-blocks a tt_metal_version
    mismatch because mismatched kernels are useless), the runner wheel and weights
    are reusable and not as hard-locked, so a mismatch installs anyway with a loud
    warning — the user resolves the version blocker themselves. Returns an
    informational ``Incompatibility`` (always ``fatal=False``) or None.
    """
    if manifest.runner is None and manifest.weights is None:
        return None
    if local.tt_metal_version and manifest.tt_metal_version != local.tt_metal_version:
        return Incompatibility(
            field="runner_tt_metal_version",
            expected=manifest.tt_metal_version,
            detected=local.tt_metal_version,
            fatal=False,
        )
    return None
