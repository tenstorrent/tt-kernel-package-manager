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

SCHEMA_VERSION = "1"


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


class Manifest(BaseModel):
    """Root document of a bundle (``tt_kernel_manifest.json``)."""

    schema_version: str = SCHEMA_VERSION
    name: str
    model: Optional[str] = None  # informational
    tt_metal_version: str  # MUST match local (per-kernel hash dependency)
    arch: str  # blackhole | wormhole_b0 | ...
    device_count: int = 1
    build_key: int  # uint64; names the cache subtree on disk
    build_key_inputs: BuildKeyInputs = Field(default_factory=BuildKeyInputs)
    kernel_count: int = 0
    files: List[FileEntry] = Field(default_factory=list)
    producer: Producer

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        return cls.model_validate_json(text)

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
    """
    issues: List[Incompatibility] = []
    inp = manifest.build_key_inputs

    if local.arch and manifest.arch != local.arch:
        issues.append(
            Incompatibility(field="arch", expected=manifest.arch, detected=local.arch, fatal=True)
        )

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
