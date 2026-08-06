# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""``tt-kernel`` command-line interface."""

from __future__ import annotations

import datetime
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import typer

from . import MANIFEST_NAME, TT_KERNEL_CATALOG_TAG, TT_KERNEL_TAG, __version__
from . import auth, bundles, cache, hub, localdb, metal, resolve as resolve_mod, runtime, toolchain
from .manifest import (
    CompatibilityReport,
    FileEntry,
    Manifest,
    Producer,
    RunnerPayload,
    WeightsRef,
    compare,
    runner_version_advisory,
)

app = typer.Typer(
    name="tt-kernel",
    help="Publish and pull precompiled tt-metal kernel caches over Hugging Face Hub.",
    no_args_is_help=True,
    add_completion=False,
)


def _err(msg: str) -> "typer.Exit":
    typer.secho(msg, fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


def _print_report(report: CompatibilityReport) -> None:
    if report.compatible:
        typer.secho("✓ compatible with the local environment", fg=typer.colors.GREEN)
        return
    typer.secho("Compatibility issues:", fg=typer.colors.YELLOW)
    for i in report.issues:
        tag = "FATAL" if i.fatal else "warn"
        color = typer.colors.RED if i.fatal else typer.colors.YELLOW
        typer.secho(
            f"  [{tag}] {i.field}: bundle={i.expected!r} local={i.detected!r}", fg=color
        )


# --------------------------------------------------------------------------- login
@app.command()
def login(
    token: Optional[str] = typer.Option(None, help="HF token; omit for interactive login."),
) -> None:
    """Log in to Hugging Face (reuses HF's token store)."""
    auth.login(token=token)
    me = auth.whoami()
    if me:
        typer.secho(f"Logged in as {me.get('name')}", fg=typer.colors.GREEN)
    else:
        raise _err("Login did not produce a valid identity.")


# ---------------------------------------------------------------------------- push
@app.command()
def push(
    repo_id: str = typer.Argument(..., help="Target repo as namespace/name."),
    private: bool = typer.Option(False, "--private/--public", help="Repo visibility."),
    publish: bool = typer.Option(
        False, "--publish", help="List this bundle in the community catalog (requires "
        "--public). Adds an opt-in tag; the catalog indexes a pointer to your repo — it "
        "stores nothing and your repo stays under your governance. Delist with `tt-kernel unpublish`."
    ),
    cache_dir: Optional[str] = typer.Option(None, help="Override the tt-metal cache root."),
    build_key: Optional[int] = typer.Option(None, help="Which build_key subtree to publish."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Override arch detection."),
    num_hw_cqs: Optional[int] = typer.Option(None, help="Hardware command queues used (default 1)."),
    name: Optional[str] = typer.Option(None, help="Bundle name (defaults to the repo name)."),
    tt_metal_version: Optional[str] = typer.Option(
        None, "--tt-metal-version", help="Override the detected tt-metal version (e.g. for testing)."
    ),
    python_package: Optional[List[str]] = typer.Option(
        None, "--python-package", help="Path to a prebuilt runner wheel/sdist to ship "
        "(repeatable). Omit for a reference runner (--runner-spec only)."
    ),
    runner_spec: Optional[str] = typer.Option(
        None, "--runner-spec", help="Runner as module:Class for dispatch --runner. With "
        "--python-package it is packaged (shipped); alone it is a reference the consumer resolves."
    ),
    runner_source: Optional[str] = typer.Option(
        None, "--runner-source", help="For a reference runner: where to get it (pip name / git URL)."
    ),
    entry_point: Optional[str] = typer.Option(
        None, "--entry-point", help="Entry-point name the wheel registers under tt_models.runners."
    ),
    capability: Optional[List[str]] = typer.Option(
        None, "--capability", help="Model-capability tag to surface in the catalog "
        "(repeatable), e.g. --capability moe --capability sliding-window-attention. Added as "
        "a repo tag; the catalog renders known ones as badges and filters."
    ),
    weights: Optional[str] = typer.Option(
        None, "--weights", help="HF model repo id whose weights this bundle targets."
    ),
    weights_revision: Optional[str] = typer.Option(None, "--weights-revision"),
    weights_allow: Optional[List[str]] = typer.Option(None, "--weights-allow"),
    weights_ignore: Optional[List[str]] = typer.Option(None, "--weights-ignore"),
    backend: str = typer.Option(
        "dispatch", "--backend", help="Serving backend: 'dispatch' (kernel-cache bundle) or "
        "'vllm' (a kernels-less bundle folder served through the Tenstorrent vLLM plugin)."
    ),
    bundle_dir: Optional[str] = typer.Option(
        None, "--bundle-dir", help="For --backend vllm: local folder holding vllm_metadata.json "
        "+ the adapter class + its deps. Shipped verbatim; laid into EXTRA_MODELS_DIR on pull."
    ),
) -> None:
    """Package a bundle and publish it.

    With ``--backend dispatch`` (default): package the local kernel cache for one build_key;
    the bundle may also declare a runner (packaged or reference) and a --weights ref so a
    single pull installs kernels + runner + weights.

    With ``--backend vllm``: package the ``--bundle-dir`` folder (vllm_metadata.json + the
    ``VllmGeneratorAdapter`` class + deps) as a **kernels-less** bundle — no precompiled cache
    is shipped; the vLLM plugin JITs at first-run warmup.
    """
    # A catalog listing is public by definition — refuse to list a private repo.
    if publish and private:
        raise _err("--publish lists the bundle in the public community catalog and requires "
                   "--public. Re-run with --public, or drop --publish to push privately.")

    if backend not in ("dispatch", "vllm"):
        raise _err(f"--backend must be 'dispatch' or 'vllm', not {backend!r}.")
    if backend == "vllm":
        _push_vllm(
            repo_id, private=private, publish=publish, bundle_dir=bundle_dir, arch=arch,
            name=name, tt_metal_version=tt_metal_version, weights=weights,
            weights_revision=weights_revision, weights_allow=weights_allow,
            weights_ignore=weights_ignore, capability=capability,
        )
        return
    if bundle_dir:
        raise _err("--bundle-dir is only valid with --backend vllm.")

    # Validate runtime payload args before any device/cache work or upload.
    wheel_paths: List[Path] = []
    if python_package:
        if not runner_spec:
            raise _err("--python-package requires --runner-spec module:Class (the wheel is "
                       "useless to dispatch without a runner spec).")
        for pkg in python_package:
            p = Path(pkg).expanduser()
            if not p.is_file() or p.suffix not in (".whl",) and not p.name.endswith(".tar.gz"):
                raise _err(f"--python-package {pkg!r} must be an existing .whl or .tar.gz file.")
            wheel_paths.append(p)
    if runner_spec and (":" not in runner_spec and "." not in runner_spec):
        raise _err(f"--runner-spec {runner_spec!r} must be 'module:Class' (or 'module.Class').")
    if entry_point and not runner_spec:
        raise _err("--entry-point requires --runner-spec.")
    if runner_source and not runner_spec:
        raise _err("--runner-source requires --runner-spec (it says where to get the reference runner).")

    out_root = cache.resolve_out_root(cache_dir)
    try:
        key = cache.select_build_key(out_root, build_key)
    except (FileNotFoundError, ValueError) as exc:
        raise _err(str(exc))

    subtree = cache.build_key_path(out_root, key)
    typer.echo(f"Packaging build_key {key} from {subtree}")
    # Isolation feedback + pre-push guard (#2): show what's being shipped and warn if the
    # cache does not look isolated to one model (sibling build_keys / the shared default).
    typer.echo(f"  {cache.count_kernels(subtree)} kernel group(s) in this subtree")
    default_cache = cache_dir is None and not os.environ.get("TT_METAL_CACHE")
    for warning in cache.publish_warnings(out_root, key, default_cache=default_cache):
        typer.secho(f"  ! {warning}", fg=typer.colors.YELLOW)

    dev = metal.detect_device(arch_override=arch)
    version = tt_metal_version or metal.resolve_version()
    if not version:
        raise _err(
            "Could not resolve tt-metal version. Set TT_METAL_HOME, install ttnn, or pass "
            "--tt-metal-version so the consumer can match it."
        )
    if not dev.arch:
        raise _err("Could not detect arch. Pass --arch (blackhole | wormhole_b0 | ...).")

    files = cache.index_subtree(subtree)

    # Runtime payload: build the runner block whenever a spec is given (packaged if wheels
    # were supplied, reference otherwise) and index any shipped wheels under python/.
    runner_block: Optional[RunnerPayload] = None
    if runner_spec:
        runner_block = RunnerPayload(
            spec=runner_spec,
            wheels=[p.name for p in wheel_paths],
            entry_point=entry_point,
            source=runner_source,
        )
        files = files + [
            FileEntry(path=f"python/{p.name}", sha256=cache.sha256_file(p), size=p.stat().st_size)
            for p in wheel_paths
        ]
    weights_block: Optional[WeightsRef] = None
    if weights:
        weights_block = WeightsRef(
            repo_id=weights,
            revision=weights_revision,
            allow_patterns=weights_allow or None,
            ignore_patterns=weights_ignore or None,
        )

    manifest = Manifest(
        name=name or repo_id.split("/")[-1],
        tt_metal_version=version,
        arch=dev.arch,
        device_count=dev.device_count or 1,
        build_key=key,
        build_key_inputs=metal.build_key_inputs(
            num_hw_cqs=num_hw_cqs, harvesting_mask=dev.harvesting_mask
        ),
        kernel_count=cache.count_kernels(subtree),
        fast_path_kernels=cache.detect_fast_path_kernels(subtree),
        files=files,
        producer=Producer(
            tt_kernel_version=__version__,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            hostname=socket.gethostname(),
            tt_metal_home=cache.detect_cache_tt_metal_root(subtree),
        ),
        runner=runner_block,
        weights=weights_block,
    )

    with tempfile.TemporaryDirectory() as td:
        staged = Path(td)
        # Mirror the subtree under <staged>/<build_key>/ so it installs cleanly.
        shutil.copytree(subtree, staged / str(key))
        # Ship the runner wheel(s) under python/ (uploaded automatically by upload_folder).
        if wheel_paths:
            (staged / "python").mkdir()
            for p in wheel_paths:
                shutil.copy2(p, staged / "python" / p.name)
        (staged / MANIFEST_NAME).write_text(manifest.to_json())

        typer.echo(f"Creating repo {repo_id} (private={private})")
        hub.create_repo(repo_id, private=private)
        hub.set_visibility(repo_id, private=private)
        typer.echo(
            f"Uploading {len(files)} files ({manifest.total_size / 1e6:.1f} MB) ..."
        )
        hub.push_folder(repo_id, staged, commit_message=f"tt-kernel push {manifest.name}")
        tags = [TT_KERNEL_TAG, dev.arch]
        if publish:
            tags.append(TT_KERNEL_CATALOG_TAG)
        if capability:
            tags.extend(c.strip().lower() for c in capability if c.strip())
        try:
            hub.tag_repo(repo_id, tags)
        except Exception as exc:  # tagging is best-effort
            typer.secho(f"  (could not write tags: {exc})", fg=typer.colors.YELLOW)

    typer.secho(f"✓ Pushed {repo_id} (build_key {key})", fg=typer.colors.GREEN)
    if publish:
        typer.secho(
            "✓ Listed in the community catalog. It indexes a pointer to this public repo — "
            "it stores none of your content, which stays under your governance. "
            "Delist any time with `tt-kernel unpublish " + repo_id + "`.",
            fg=typer.colors.GREEN,
        )


def _push_vllm(
    repo_id: str,
    *,
    private: bool,
    publish: bool,
    bundle_dir: Optional[str],
    arch: Optional[str],
    name: Optional[str],
    tt_metal_version: Optional[str],
    weights: Optional[str],
    weights_revision: Optional[str],
    weights_allow: Optional[List[str]],
    weights_ignore: Optional[List[str]],
    capability: Optional[List[str]],
) -> None:
    """Package and publish a kernels-less vLLM bundle folder.

    The folder (``vllm_metadata.json`` + the adapter class + deps) is shipped verbatim under
    ``BUNDLE_SUBDIR/`` in the repo and indexed for integrity. No kernel cache is packaged —
    the vLLM plugin JITs at first-run warmup.
    """
    if not bundle_dir:
        raise _err("--backend vllm requires --bundle-dir pointing at the bundle folder.")
    folder = Path(bundle_dir).expanduser()
    if not folder.is_dir():
        raise _err(f"--bundle-dir {bundle_dir!r} is not a directory.")
    try:
        md = bundles.read_vllm_metadata(folder)
    except (FileNotFoundError, ValueError) as exc:
        raise _err(str(exc))
    if not md.arch or not md.main_class:
        raise _err(
            f"{bundles.VLLM_METADATA_NAME} must set both 'arch' (HF architecture name) and "
            "'main_class' (\"module:Class\")."
        )

    dev = metal.detect_device(arch_override=arch)
    if not dev.arch:
        raise _err("Could not detect arch. Pass --arch (blackhole | wormhole_b0 | ...).")
    # A vLLM bundle ships no kernels, so tt_metal_version is advisory only; still record it
    # when resolvable so the consumer sees the build it was authored against.
    version = tt_metal_version or metal.resolve_version() or "unknown"

    # Index the bundle folder under a fixed subdir so pull can locate + integrity-check it.
    subdir = "vllm_bundle"
    indexed = cache.index_subtree(folder)
    files = [
        FileEntry(path=f"{subdir}/{e.path}", sha256=e.sha256, size=e.size) for e in indexed
    ]

    weights_target = weights or md.hf_weights
    weights_block: Optional[WeightsRef] = None
    if weights_target:
        weights_block = WeightsRef(
            repo_id=weights_target,
            revision=weights_revision,
            allow_patterns=weights_allow or None,
            ignore_patterns=weights_ignore or None,
        )

    manifest = Manifest(
        name=name or repo_id.split("/")[-1],
        tt_metal_version=version,
        arch=dev.arch,
        device_count=dev.device_count or 1,
        build_key=None,  # kernels-less
        kernel_count=0,
        fast_path_kernels=None,
        files=files,
        producer=Producer(
            tt_kernel_version=__version__,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            hostname=socket.gethostname(),
        ),
        runner=RunnerPayload(backend="vllm", bundle_dir=subdir),
        weights=weights_block,
    )

    typer.echo(f"Packaging vLLM bundle from {folder} ({len(files)} file(s))")
    typer.echo(f"  arch registration: {md.arch}  ->  {md.main_class}")
    with tempfile.TemporaryDirectory() as td:
        staged = Path(td)
        shutil.copytree(folder, staged / subdir)
        (staged / MANIFEST_NAME).write_text(manifest.to_json())

        typer.echo(f"Creating repo {repo_id} (private={private})")
        hub.create_repo(repo_id, private=private)
        hub.set_visibility(repo_id, private=private)
        typer.echo(f"Uploading {len(files)} files ({manifest.total_size / 1e6:.1f} MB) ...")
        hub.push_folder(repo_id, staged, commit_message=f"tt-kernel push {manifest.name} (vllm)")
        tags = [TT_KERNEL_TAG, dev.arch, "vllm"]
        if publish:
            tags.append(TT_KERNEL_CATALOG_TAG)
        if capability:
            tags.extend(c.strip().lower() for c in capability if c.strip())
        try:
            hub.tag_repo(repo_id, tags)
        except Exception as exc:  # tagging is best-effort
            typer.secho(f"  (could not write tags: {exc})", fg=typer.colors.YELLOW)

    typer.secho(f"✓ Pushed vLLM bundle {repo_id}", fg=typer.colors.GREEN)
    typer.secho(f"  Serve it:  tt-kernel serve {repo_id}", fg=typer.colors.CYAN)


# ---------------------------------------------------------------------------- pull
@app.command()
def pull(
    repo_id: str = typer.Argument(..., help="Source repo as namespace/name[@revision]."),
    force: bool = typer.Option(False, "--force", help="Install despite non-fatal mismatches."),
    cache_dir: Optional[str] = typer.Option(None, help="Override the tt-metal cache root."),
    probe: bool = typer.Option(False, "--probe", help="Open a device to read the true build_key."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Override arch detection."),
    models_dir: Optional[str] = typer.Option(None, "--models-dir", help="Where to download weights."),
    bundles_dir: Optional[str] = typer.Option(
        None, "--bundles-dir", help="For a vLLM bundle: where to lay the model folder "
        "(== EXTRA_MODELS_DIR). Default: $TT_KERNEL_BUNDLES_DIR or ~/.cache/tt-kernel/bundles."
    ),
    with_weights: bool = typer.Option(
        False, "--with-weights", help="For a vLLM bundle: also download the HF weights now "
        "(default: skip — the model class fetches them from the HF id at load)."
    ),
    no_python: bool = typer.Option(False, "--no-python", help="Skip installing the runner wheel."),
    no_weights: bool = typer.Option(False, "--no-weights", help="Skip downloading weights."),
    kernels_only: bool = typer.Option(
        False, "--kernels-only", help="Install only the kernel cache (implies --no-python and --no-weights)."
    ),
    python_exe: Optional[str] = typer.Option(
        None, "--python", help="Target interpreter for the runner pip install (default: this venv)."
    ),
) -> None:
    """Download a bundle and install everything it carries: kernels, runner, weights.

    A single pull installs the kernel cache, sets up the runner (pip-installs a packaged
    wheel, or verifies a reference runner resolves), and downloads the model weights, then
    prints the exact `serve` command. Skip parts with --no-python / --no-weights /
    --kernels-only.
    """
    if kernels_only:
        no_python = no_weights = True

    repo_id, revision = _split_revision(repo_id)
    runner_installed = False  # we pip-installed a packaged wheel
    runner_ready = False  # runner is usable (installed, or reference that resolves)
    weights_path: Optional[Path] = None
    with tempfile.TemporaryDirectory() as td:
        snapshot = hub.download_bundle(repo_id, revision, dest=td)
        manifest_path = snapshot / MANIFEST_NAME
        if not manifest_path.is_file():
            raise _err(f"{repo_id} is not a tt-kernel bundle (no {MANIFEST_NAME}).")
        manifest = Manifest.from_json(manifest_path.read_text())

        # vLLM bundles carry no kernel cache: install the model folder into bundles_dir
        # instead of the tt-metal cache, then return.
        if manifest.runner and manifest.runner.is_vllm:
            _install_vllm_bundle(
                repo_id, snapshot, manifest, force=force, arch=arch,
                models_dir=models_dir, bundles_dir=bundles_dir,
                with_weights=with_weights and not no_weights,
            )
            return

        env = metal.local_env(arch_override=arch, probe=probe)
        report = compare(manifest, env)
        _print_report(report)
        _warn_toolchain()  # complements the kernel compat check with the serving-stack versions

        if report.has_fatal:
            raise _err("Refusing to install: fatal incompatibility (see above).")
        if report.issues and not force:
            raise _err("Refusing to install: re-run with --force to override the warnings above.")

        staged = snapshot / str(manifest.build_key)
        if not staged.is_dir():
            raise _err(f"Bundle is missing its build_key subtree {manifest.build_key}/.")

        # Partition the file index: kernels live under the build_key subtree; runner
        # wheels under python/ (verified relative to the snapshot root).
        wheel_entries = [f for f in manifest.files if f.path.startswith("python/")]
        kernel_entries = [f for f in manifest.files if not f.path.startswith("python/")]

        typer.echo(f"Verifying {len(kernel_entries)} kernel files ...")
        problems = cache.verify_files(staged, kernel_entries)
        if problems:
            for p in problems[:20]:
                typer.secho(f"  {p}", fg=typer.colors.RED)
            raise _err(f"Integrity check failed ({len(problems)} problem(s)).")

        out_root = cache.resolve_out_root(cache_dir)
        target = cache.install_subtree(staged, out_root, manifest.build_key)
        typer.secho(f"✓ kernels -> {target}", fg=typer.colors.GREEN)
        if manifest.fast_path_kernels is False:
            typer.secho(
                "  ! baseline-only bundle: it lacks the traced-decode / on-device-lm_head "
                "kernels, so serving on the fast path (DISPATCH_TRACE / "
                "DISPATCH_ONDEVICE_LMHEAD) will re-JIT them. Produce a fast-path bundle by "
                "warming with those flags enabled.",
                fg=typer.colors.YELLOW,
            )

        # Cross-host dep relocation: if this bundle was built against a tt-metal at a
        # different path than ours, rewrite the tree-dep prefix so the cache hits here too
        # (in-cache paths were already relocated by install_subtree).
        producer_home = manifest.producer.tt_metal_home if manifest.producer else None
        if producer_home:
            consumer_home = metal.detect_tt_metal_home()
            if (consumer_home and os.path.isdir(consumer_home)
                    and os.path.normpath(consumer_home) != os.path.normpath(producer_home)):
                n = cache.relocate_tt_metal_tree(target, producer_home, consumer_home)
                if n:
                    typer.secho(
                        f"  ↻ relocated tt-metal tree deps in {n} dephash file(s): "
                        f"{producer_home} -> {consumer_home}",
                        fg=typer.colors.CYAN,
                    )

        # ---- runtime payload ----
        advisory = runner_version_advisory(manifest, env)
        if advisory is not None and (manifest.runner or manifest.weights):
            typer.secho(
                f"  ! runner/weights target tt-metal {advisory.expected!r}; you have "
                f"{advisory.detected!r}. Installing anyway — it will NOT run until the "
                "serving environment matches.",
                fg=typer.colors.YELLOW,
            )

        # Runner: packaged => verify + pip install the shipped wheel(s); reference => the
        # runner is not shipped, so just verify it resolves in the target env (install nothing).
        if manifest.runner and not no_python:
            if manifest.runner.is_packaged:
                wp = cache.verify_files(snapshot, wheel_entries)
                if wp:
                    for p in wp[:20]:
                        typer.secho(f"  {p}", fg=typer.colors.RED)
                    raise _err(f"Runner wheel integrity check failed ({len(wp)} problem(s)).")
                if not runtime.ttnn_importable(python_exe):
                    tgt = python_exe or "this interpreter"
                    typer.secho(
                        f"  ! ttnn is not importable from {tgt}; the runner will install but "
                        "not run there. Use --python to target the tt-metal venv.",
                        fg=typer.colors.YELLOW,
                    )
                wheels = [snapshot / e.path for e in wheel_entries]
                try:
                    typer.echo(f"Installing runner: {manifest.runner.spec} ({len(wheels)} wheel(s)) ...")
                    runtime.pip_install_wheels(wheels, python=python_exe)
                    runner_installed = True
                    runner_ready = True
                    typer.secho("✓ runner installed", fg=typer.colors.GREEN)
                except Exception as exc:  # noqa: BLE001 — record partial progress, don't roll back kernels
                    _record_pull(repo_id, manifest, out_root, runner_installed=False,
                                 weights_path=None, last_error=f"pip install failed: {exc}")
                    raise _err(
                        f"Kernels are installed, but the runner pip install failed: {exc}\n"
                        f"  Re-run `tt-kernel pull {repo_id} --no-weights` to retry just the runner."
                    )
            else:
                # Reference runner: nothing ships in the bundle; confirm it's importable.
                if runtime.runner_spec_importable(manifest.runner.spec, python_exe):
                    runner_ready = True
                    typer.secho(
                        f"✓ runner {manifest.runner.spec} resolved (reference; not shipped)",
                        fg=typer.colors.GREEN,
                    )
                else:
                    src = f" Install it from {manifest.runner.source}." if manifest.runner.source else ""
                    typer.secho(
                        f"  ! runner {manifest.runner.spec} is a reference (not shipped) and is "
                        f"not importable in the target env.{src}",
                        fg=typer.colors.YELLOW,
                    )

        # Weights: download into a resolvable models dir (resumable).
        if manifest.weights and not no_weights:
            dest = runtime.resolve_models_dir(models_dir, manifest.weights.repo_id)
            try:
                typer.echo(f"Downloading weights {manifest.weights.repo_id} -> {dest} ...")
                weights_path = runtime.download_weights(manifest.weights, dest)
                typer.secho(f"✓ weights -> {weights_path}", fg=typer.colors.GREEN)
            except Exception as exc:  # noqa: BLE001 — kernels+runner remain usable
                _record_pull(repo_id, manifest, out_root, runner_installed=runner_installed,
                             weights_path=None, last_error=f"weights download failed: {exc}")
                raise _err(
                    f"Kernels{' and runner' if runner_installed else ''} installed, but the "
                    f"weights download failed: {exc}\n"
                    f"  Re-run `tt-kernel pull {repo_id}` to resume the download."
                )

    _record_pull(repo_id, manifest, out_root, runner_installed=runner_installed,
                 weights_path=weights_path, last_error=None)

    # Ready-to-run guidance.
    typer.secho(f"✓ Installed {repo_id}", fg=typer.colors.GREEN)
    if manifest.runner and runner_ready and weights_path is not None:
        typer.echo("\nRun it:")
        typer.secho("  " + runtime.serve_command(manifest.runner.spec, weights_path),
                    fg=typer.colors.CYAN)
        if not runtime.legacy_serve_available():
            typer.secho("  (the legacy-runner server needs fastapi + uvicorn: "
                        "pip install 'tt-kernel[serve]')",
                        fg=typer.colors.YELLOW)
    elif manifest.runner:
        missing = []
        if not runner_ready:
            if manifest.runner.is_packaged:
                missing.append("runner (re-run without --no-python)")
            else:
                src = f" — install from {manifest.runner.source}" if manifest.runner.source else ""
                missing.append(f"runner {manifest.runner.spec} (reference{src})")
        if weights_path is None and manifest.weights:
            missing.append("weights (re-run without --no-weights)")
        if missing:
            typer.secho(f"  pending: {', '.join(missing)}", fg=typer.colors.YELLOW)


def _record_pull(repo_id, manifest, out_root, *, runner_installed, weights_path, last_error,
                 bundle_path=None) -> None:
    """Write the install binding to the local index (overwrites on re-pull)."""
    entry = {
        "name": manifest.name,
        "build_key": manifest.build_key,
        "arch": manifest.arch,
        "tt_metal_version": manifest.tt_metal_version,
        "out_root": out_root,
        "schema_version": manifest.schema_version,
        "runner_spec": manifest.runner.spec if manifest.runner else None,
        "entry_point": manifest.runner.entry_point if manifest.runner else None,
        "backend": manifest.runner.backend if manifest.runner else None,
        "bundle_path": str(bundle_path) if bundle_path else None,
        "weights_repo": manifest.weights.repo_id if manifest.weights else None,
        "weights_path": str(weights_path) if weights_path else None,
        "python_installed": runner_installed,
        "weights_installed": weights_path is not None,
        "installed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if last_error:
        entry["last_error"] = last_error
    localdb.record(repo_id, entry)


def _install_vllm_bundle(
    repo_id, snapshot, manifest, *, force, arch, models_dir, bundles_dir, with_weights
) -> None:
    """Install a kernels-less vLLM bundle: verify + lay the model folder into bundles_dir.

    No tt-metal cache is touched. Optionally downloads weights (default: skip — the model
    class fetches them from the HF id at load). Records the install for `run`/`serve`/`rm`.
    """
    # arch is the only fatal gate for a kernels-less bundle (see manifest.compare).
    env = metal.local_env(arch_override=arch, probe=False)
    report = compare(manifest, env)
    _print_report(report)
    _warn_toolchain()
    if report.has_fatal:
        raise _err("Refusing to install: fatal incompatibility (see above).")
    if report.issues and not force:
        raise _err("Refusing to install: re-run with --force to override the warnings above.")

    subdir = manifest.runner.bundle_dir or "vllm_bundle"
    staged = snapshot / subdir
    if not (staged / bundles.VLLM_METADATA_NAME).is_file():
        raise _err(f"Bundle is missing its folder {subdir}/{bundles.VLLM_METADATA_NAME}.")

    bundle_entries = [f for f in manifest.files if f.path.startswith(f"{subdir}/")]
    typer.echo(f"Verifying {len(bundle_entries)} bundle file(s) ...")
    # verify_files takes paths relative to a root; strip the subdir prefix by verifying
    # against the snapshot root with the full-prefixed paths.
    problems = cache.verify_files(snapshot, bundle_entries)
    if problems:
        for p in problems[:20]:
            typer.secho(f"  {p}", fg=typer.colors.RED)
        raise _err(f"Integrity check failed ({len(problems)} problem(s)).")

    bdir = bundles.resolve_bundles_dir(bundles_dir)
    key = bundles.model_key(repo_id)
    dest = bundles.install_bundle(staged, bdir, key)
    typer.secho(f"✓ vLLM bundle -> {dest}", fg=typer.colors.GREEN)

    md = bundles.read_vllm_metadata(dest)
    typer.secho(f"  registers {md.arch} -> {md.main_class}", fg=typer.colors.CYAN)

    weights_path = None
    if with_weights and manifest.weights:
        wdest = runtime.resolve_models_dir(models_dir, manifest.weights.repo_id)
        try:
            typer.echo(f"Downloading weights {manifest.weights.repo_id} -> {wdest} ...")
            weights_path = runtime.download_weights(manifest.weights, wdest)
            typer.secho(f"✓ weights -> {weights_path}", fg=typer.colors.GREEN)
        except Exception as exc:  # noqa: BLE001 — bundle is still usable (model self-fetches)
            typer.secho(f"  ! weights download failed (model will fetch at load): {exc}",
                        fg=typer.colors.YELLOW)

    _record_pull(repo_id, manifest, out_root="", runner_installed=False,
                 weights_path=weights_path, last_error=None, bundle_path=str(dest))
    typer.secho(f"✓ Installed {repo_id}", fg=typer.colors.GREEN)
    typer.secho(f"  Serve it:  tt-kernel serve {repo_id}", fg=typer.colors.CYAN)


# ------------------------------------------------------------------------- doctor
def _warn_toolchain() -> None:
    """Warn (never abort) about an inadequate surrounding toolchain. Called by run/pull
    so a version skew is surfaced without blocking the user's action."""
    for c in toolchain.check_toolchain().problems:
        typer.secho(f"  ! {c.name}: {c.message}", fg=typer.colors.YELLOW)


@app.command()
def doctor() -> None:
    """Report whether the surrounding toolchain (tt-metal, vLLM)
    and hardware are adequate. tt-kernel never installs these — it only checks and warns.

    Exits non-zero if any component is missing or below the required version.
    """
    report = toolchain.check_toolchain()
    typer.secho("Toolchain:", bold=True)
    for c in report.components:
        ok = c.adequate
        mark = "✓" if ok else "✗"
        color = typer.colors.GREEN if ok else typer.colors.RED
        ver = c.version or "—"
        typer.secho(f"  {mark} {c.name}: {ver} (require >= {c.required}) — {c.message}", fg=color)

    dev = metal.detect_device()
    typer.secho("\nHardware:", bold=True)
    if dev.arch:
        typer.secho(f"  ✓ arch={dev.arch} devices={dev.device_count} (via {dev.source})",
                    fg=typer.colors.GREEN)
    else:
        typer.secho("  ! no Tenstorrent device detected (tt-smi/ARCH_NAME unavailable)",
                    fg=typer.colors.YELLOW)

    if not report.ok:
        raise typer.Exit(code=1)
    typer.secho("\n✓ toolchain adequate", fg=typer.colors.GREEN)


# ----------------------------------------------------------------------------- run
def _handoff(argv: List[str], *, print_only: bool, why: str) -> None:
    """Print or execute the legacy-runner server handoff (``tt_kernel.legacy_serve``).
    Execution replaces this process's foreground with the server (blocks until it exits)."""
    typer.secho(f"[{why}]", fg=typer.colors.CYAN)
    if print_only:
        typer.echo(" ".join(argv))
        return
    if not runtime.legacy_serve_available():
        raise _err(
            "Cannot serve: the legacy-runner server needs fastapi + uvicorn "
            "(pip install 'tt-kernel[serve]'). Use `--print` to emit the command."
        )
    try:
        raise typer.Exit(code=subprocess.run(argv).returncode)
    except KeyboardInterrupt:  # graceful Ctrl-C of the served process
        raise typer.Exit(code=130)


def _endpoint_from_command(command: List[str]) -> str:
    """Best-effort OpenAI endpoint URL from a launch command's --host/--port (default 8000)."""
    host, port = "localhost", "8000"
    for i, tok in enumerate(command):
        if tok == "--port" and i + 1 < len(command):
            port = command[i + 1]
        elif tok.startswith("--port="):
            port = tok.split("=", 1)[1]
        elif tok in ("--host",) and i + 1 < len(command):
            h = command[i + 1]
            host = "localhost" if h in ("0.0.0.0", "") else h
        elif tok.startswith("--host="):
            h = tok.split("=", 1)[1]
            host = "localhost" if h in ("0.0.0.0", "") else h
    return f"http://{host}:{port}"


def _ensure_vllm_pulled(repo_id: str, revision: Optional[str], *, arch: Optional[str],
                        bundles_dir: Optional[str]) -> dict:
    """Return the local install entry for a vLLM bundle, pulling it first if absent."""
    entry = localdb.get(repo_id)
    if entry and entry.get("bundle_path") and Path(entry["bundle_path"]).is_dir():
        return entry
    with tempfile.TemporaryDirectory() as td:
        snapshot = hub.download_bundle(repo_id, revision, dest=td)
        mpath = snapshot / MANIFEST_NAME
        if not mpath.is_file():
            raise _err(f"{repo_id} is not a tt-kernel bundle (no {MANIFEST_NAME}).")
        manifest = Manifest.from_json(mpath.read_text())
        if not (manifest.runner and manifest.runner.is_vllm):
            raise _err(f"{repo_id} is not a vLLM bundle.")
        _install_vllm_bundle(repo_id, snapshot, manifest, force=False, arch=arch,
                             models_dir=None, bundles_dir=bundles_dir, with_weights=False)
    entry = localdb.get(repo_id)
    if not entry or not entry.get("bundle_path"):
        raise _err(f"Failed to install vLLM bundle {repo_id}.")
    return entry


def _serve_vllm(repo_id: str, revision: Optional[str], *, print_only: bool, local_only: bool,
                arch: Optional[str], bundles_dir: Optional[str], do_health: bool) -> None:
    """The vLLM one-command serve flow: pull-if-needed, launch, (optional) health, endpoint."""
    if local_only:
        entry = localdb.get(repo_id)
        if not entry or not entry.get("bundle_path"):
            raise _err(f"No installed vLLM bundle for {repo_id} (and --local-only forbids a pull).")
    else:
        entry = _ensure_vllm_pulled(repo_id, revision, arch=arch, bundles_dir=bundles_dir)

    bundle_path = Path(entry["bundle_path"])
    if not bundle_path.is_dir():
        raise _err(f"Installed bundle folder is missing: {bundle_path}. Re-run `tt-kernel pull {repo_id}`.")
    extra_models_dir = bundle_path.parent  # == EXTRA_MODELS_DIR (holds this model folder)
    md = bundles.read_vllm_metadata(bundle_path)
    mkey, launch = bundles.select_launch(md, arch)
    if launch is None:
        cands = ", ".join(bundles.machine_candidates(arch))
        raise _err(
            f"{bundles.VLLM_METADATA_NAME} has no launch command for this machine "
            f"(tried: {cands}). Add one, or set a 'default'."
        )

    argv = runtime.vllm_serve_argv(launch.command)
    env = runtime.vllm_serve_env(extra_models_dir, launch.env)
    endpoint = _endpoint_from_command(launch.command)

    typer.secho(f"[vLLM: {md.arch} via {mkey}; EXTRA_MODELS_DIR={extra_models_dir}]",
                fg=typer.colors.CYAN)
    typer.secho(f"  OpenAI endpoint (once up): {endpoint}", fg=typer.colors.CYAN)
    if print_only:
        # Shell-quote both halves: a launch command may legitimately carry spaces and quotes
        # (a --tt-config JSON blob, an --additional-server-args string, a `bash -c` payload).
        # A plain join would emit something that re-parses into different argv than the one
        # `serve` actually execs — and --print exists to be pasted into a shell.
        exports = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in
                           {runtime.ENV_EXTRA_MODELS_DIR: str(extra_models_dir), **launch.env}.items())
        typer.echo(f"{exports} " + shlex.join(argv))
        return
    if not runtime.vllm_available():
        raise _err(
            "Cannot serve: the Tenstorrent vLLM stack (vllm + vllm_tt_plugin) is not importable "
            "here. Install it (see scripts/install.sh), or use --print to emit the command."
        )
    try:
        raise typer.Exit(code=subprocess.run(argv, env=env).returncode)
    except KeyboardInterrupt:  # graceful Ctrl-C of the served process
        raise typer.Exit(code=130)


@app.command()
def serve(
    repo_id: str = typer.Argument(..., help="vLLM bundle id (namespace/name[@rev]) to serve."),
    print_only: bool = typer.Option(False, "--print", help="Print the launch command instead of running it."),
    local_only: bool = typer.Option(False, "--local-only", help="Do not pull; require an installed bundle."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Override arch/machine detection."),
    bundles_dir: Optional[str] = typer.Option(None, "--bundles-dir", help="Override EXTRA_MODELS_DIR location."),
    health_check: bool = typer.Option(False, "--health-check", help="(reserved) probe the server after launch."),
) -> None:
    """Serve a vLLM bundle through the Tenstorrent vLLM plugin (the primary path).

    One command: pull the bundle folder if needed, point EXTRA_MODELS_DIR at it, and launch
    the OpenAI-compatible server with the bundle's per-machine launch command. Repeat
    invocations skip the pull and go straight to launch.
    """
    _warn_toolchain()
    repo_id, revision = _split_revision(repo_id)
    _serve_vllm(repo_id, revision, print_only=print_only, local_only=local_only,
                arch=arch, bundles_dir=bundles_dir, do_health=health_check)


@app.command()
def run(
    repo_id: str = typer.Argument(
        ..., help="Model to run: a tt-kernel bundle id (namespace/name[@rev]) or a bare HF model id."
    ),
    print_only: bool = typer.Option(
        False, "--print", help="Print the serve command instead of executing it."
    ),
    local_only: bool = typer.Option(
        False, "--local-only", help="Do not query the Hub; resolve only against installed bundles."
    ),
) -> None:
    """Serve a model through the right path.

    - **vLLM bundle** -> the Tenstorrent vLLM plugin (the default; same as `tt-kernel serve`).
    - **legacy runner bundle** (a runner following the legacy contract in
      docs/authoring_runners.md), once installed -> tt-kernel's own OpenAI-compatible
      legacy-runner server (`tt_kernel.legacy_serve`).
    - anything else (kernels-only bundle, or a bare HF repo) -> not servable by tt-kernel;
      publish a vLLM bundle.

    The old dynamic dispatch path (`tt_api.serve`) is retired.
    """
    _warn_toolchain()
    repo_id, revision = _split_revision(repo_id)
    res = resolve_mod.resolve(repo_id, revision=revision, local_only=local_only)

    # vLLM bundle -> the plugin (default path).
    if res.is_vllm:
        _serve_vllm(repo_id, revision, print_only=print_only, local_only=local_only,
                    arch=None, bundles_dir=None, do_health=False)
        return

    # Legacy runner bundle -> tt-kernel's legacy-runner server. It needs the runner
    # installed and the weights on disk, so it only works once the bundle is pulled.
    if res.has_runner:
        if res.installed and res.weights_path:
            argv = runtime.serve_argv(res.weights_path, runner_spec=res.runner_spec,
                                      python=sys.executable)
            _handoff(argv, print_only=print_only,
                     why=f"legacy runner {res.runner_spec} via tt_kernel.legacy_serve")
            return
        if res.installed:
            raise _err(
                f"{repo_id} is installed but its weights are not on disk. Re-run "
                f"`tt-kernel pull {repo_id}` (without --no-weights) so the runner can load."
            )
        typer.secho(
            f"A tt-kernel bundle exists for {repo_id} (legacy runner {res.runner_spec}).",
            fg=typer.colors.YELLOW,
        )
        typer.secho(f"  Install it first:  tt-kernel pull {repo_id}", fg=typer.colors.YELLOW)
        typer.secho("  then `tt-kernel run` serves it via the legacy-runner server.",
                    fg=typer.colors.YELLOW)
        return

    # No runner (kernels-only bundle, or a bare HF repo): the dynamic dispatch path is
    # retired. tt-kernel serves vLLM bundles and legacy-runner bundles only.
    raise _err(
        f"Nothing to serve for {repo_id}. tt-kernel serves vLLM bundles "
        f"(`tt-kernel serve <id>`) and legacy-runner bundles. To serve this model, publish "
        "it as a vLLM bundle — see docs/authoring_runners.md."
    )


# ---------------------------------------------------------------------------- info
@app.command()
def info(
    repo_id: str = typer.Argument(..., help="Repo as namespace/name[@revision]."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Override arch detection."),
    probe: bool = typer.Option(False, "--probe", help="Open a device to read the true build_key."),
) -> None:
    """Print a bundle's manifest and its compatibility verdict vs the local env."""
    repo_id, revision = _split_revision(repo_id)
    manifest = hub.fetch_manifest(repo_id, revision)
    typer.echo(manifest.to_json())
    typer.echo("")
    report = compare(manifest, metal.local_env(arch_override=arch, probe=probe))
    _print_report(report)


# ---------------------------------------------------------------------------- list
@app.command(name="list")
def list_installed() -> None:
    """List locally installed bundles."""
    entries = localdb.all_entries()
    if not entries:
        typer.echo("No bundles installed.")
        return
    for e in entries:
        backend = e.get("backend") or "dispatch"
        if backend == "vllm":
            typer.echo(
                f"{e['repo_id']}  backend=vllm  arch={e.get('arch')}  "
                f"bundle={e.get('bundle_path')}"
            )
        else:
            typer.echo(
                f"{e['repo_id']}  build_key={e.get('build_key')}  arch={e.get('arch')}  "
                f"tt_metal={e.get('tt_metal_version')}"
            )


# -------------------------------------------------------------------------- search
@app.command()
def search(
    query: str = typer.Argument("", help="Free-text query over tt-kernel cache repos."),
    limit: int = typer.Option(50, help="Max results."),
    catalog: bool = typer.Option(
        False, "--catalog", help="Restrict to repos listed in the community catalog "
        "(the set the web frontend shows), not every pushed bundle."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Search the Hub for published tt-kernel caches."""
    results = hub.search(query, limit=limit, catalog_only=catalog)
    if as_json:
        typer.echo(json.dumps(results, indent=2))
        return
    if not results:
        typer.echo("No matching bundles found.")
        return
    for r in results:
        vis = "private" if r.get("private") else "public"
        typer.echo(f"{r['id']}  [{vis}]  downloads={r.get('downloads')}")


# ----------------------------------------------------------------------- publish
@app.command()
def publish(
    repo_id: str = typer.Argument(..., help="An already-pushed public bundle as namespace/name."),
) -> None:
    """List an existing public bundle in the community catalog (opt-in).

    Use this to add a bundle you pushed earlier without ``--publish``. The catalog only
    ever holds a pointer to your public HF repo; it stores none of your content, and your
    repo stays entirely under your governance. Delist with ``tt-kernel unpublish``.
    """
    try:
        if hub.is_private(repo_id):
            raise _err(f"{repo_id} is private; the catalog is public. Make it public first "
                       "(`tt-kernel push ... --public`) before listing.")
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _err(f"Could not read {repo_id} on the Hub: {exc}")
    hub.set_catalog_listing(repo_id, listed=True)
    typer.secho(
        f"✓ Listed {repo_id} in the community catalog (pointer only; content stays yours). "
        f"Delist with `tt-kernel unpublish {repo_id}`.",
        fg=typer.colors.GREEN,
    )


# --------------------------------------------------------------------- unpublish
@app.command()
def unpublish(
    repo_id: str = typer.Argument(..., help="A listed bundle as namespace/name."),
) -> None:
    """Remove a bundle from the community catalog. The repo itself is untouched."""
    hub.set_catalog_listing(repo_id, listed=False)
    typer.secho(
        f"✓ Delisted {repo_id} from the community catalog (it drops off on the next crawl). "
        "The repo and its content are unchanged.",
        fg=typer.colors.GREEN,
    )


# ------------------------------------------------------------------------------ rm
@app.command()
def rm(
    repo_id: str = typer.Argument(..., help="Installed bundle as namespace/name."),
    cache_dir: Optional[str] = typer.Option(None, help="Override the tt-metal cache root."),
) -> None:
    """Remove a locally installed bundle and its index entry.

    For a dispatch bundle this removes the kernel-cache subtree; for a vLLM bundle it
    removes the model folder from bundles_dir (EXTRA_MODELS_DIR).
    """
    entry = localdb.get(repo_id)
    if not entry:
        raise _err(f"{repo_id} is not recorded as installed.")

    # vLLM bundle: no cache subtree — remove the installed model folder instead.
    if (entry.get("backend") == "vllm") or entry.get("build_key") is None:
        bundle_path = entry.get("bundle_path")
        removed = False
        if bundle_path:
            p = Path(bundle_path)
            removed = bundles.remove_bundle(p.parent, p.name)
        localdb.remove(repo_id)
        if removed:
            typer.secho(f"✓ Removed vLLM bundle {repo_id} ({bundle_path})", fg=typer.colors.GREEN)
        else:
            typer.secho("Index entry removed; bundle folder was already gone.",
                        fg=typer.colors.YELLOW)
        return

    # The stored out_root is already a full prefix; only re-resolve if --cache-dir given.
    out_root = cache.resolve_out_root(cache_dir) if cache_dir else (
        entry.get("out_root") or cache.resolve_out_root(None)
    )
    removed = cache.remove_subtree(out_root, int(entry["build_key"]))
    localdb.remove(repo_id)
    if removed:
        typer.secho(f"✓ Removed {repo_id} (build_key {entry['build_key']})", fg=typer.colors.GREEN)
    else:
        typer.secho(
            "Index entry removed; cache subtree was already gone.", fg=typer.colors.YELLOW
        )


# --------------------------------------------------------------------------- clean
@app.command()
def clean(
    build_key: Optional[int] = typer.Option(
        None, "--build-key", help="Remove this build_key subtree from the cache."
    ),
    all_keys: bool = typer.Option(
        False, "--all", help="Remove ALL build_key subtrees under the cache root."
    ),
    cache_dir: Optional[str] = typer.Option(None, help="Override the tt-metal cache root."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt for --all."),
) -> None:
    """Clear kernel-cache subtrees to force a clean state before a run/produce.

    The tt-metal JIT cache is keyed by build_key (the build environment), and every model
    run on a build shares one subtree — so to produce a model-specific bundle you must start
    from a clean cache. Use this to wipe a stale subtree (or all of them) first:

      tt-kernel clean --build-key N            # remove one build_key subtree
      tt-kernel clean --all                    # remove every build_key subtree
      tt-kernel clean --all --cache-dir DIR    # ... under a specific cache root

    For removing an *installed bundle* (and its index entry), use `tt-kernel rm` instead.
    """
    if all_keys and build_key is not None:
        raise _err("Pass either --build-key N or --all, not both.")
    out_root = cache.resolve_out_root(cache_dir)
    keys = cache.list_build_keys(out_root)
    if all_keys:
        if not keys:
            typer.echo(f"No build_key subtrees under {out_root}; nothing to clean.")
            return
        if not yes:
            typer.confirm(
                f"Remove ALL {len(keys)} build_key subtree(s) under {out_root}?", abort=True
            )
        for k in keys:
            cache.remove_subtree(out_root, k)
        typer.secho(
            f"✓ removed {len(keys)} build_key subtree(s) from {out_root}", fg=typer.colors.GREEN
        )
    elif build_key is not None:
        if cache.remove_subtree(out_root, build_key):
            typer.secho(f"✓ removed build_key {build_key} from {out_root}", fg=typer.colors.GREEN)
        else:
            typer.secho(
                f"build_key {build_key} not present under {out_root}.", fg=typer.colors.YELLOW
            )
    else:
        raise _err("Specify --build-key N or --all.")


# ---------------------------------------------------------------------------- utils
def _split_revision(repo_id: str) -> "tuple[str, Optional[str]]":
    """Split ``namespace/name@revision`` into (repo_id, revision|None)."""
    if "@" in repo_id:
        rid, rev = repo_id.rsplit("@", 1)
        return rid, rev
    return repo_id, None


@app.command()
def version() -> None:
    """Print the tt-kernel version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
