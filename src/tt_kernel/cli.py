"""``tt-kernel`` command-line interface."""

from __future__ import annotations

import datetime
import json
import shutil
import socket
import tempfile
from pathlib import Path
from typing import List, Optional

import typer

from . import MANIFEST_NAME, TT_KERNEL_TAG, __version__
from . import auth, cache, hub, localdb, metal, runtime
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
    cache_dir: Optional[str] = typer.Option(None, help="Override the tt-metal cache root."),
    build_key: Optional[int] = typer.Option(None, help="Which build_key subtree to publish."),
    model: Optional[str] = typer.Option(None, help="Informational model id, e.g. google/gemma-..."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Override arch detection."),
    num_hw_cqs: Optional[int] = typer.Option(None, help="Hardware command queues used (default 1)."),
    name: Optional[str] = typer.Option(None, help="Bundle name (defaults to the repo name)."),
    tt_metal_version: Optional[str] = typer.Option(
        None, "--tt-metal-version", help="Override the detected tt-metal version (e.g. for testing)."
    ),
    python_package: Optional[List[str]] = typer.Option(
        None, "--python-package", help="Path to a prebuilt runner wheel/sdist to ship (repeatable)."
    ),
    runner_spec: Optional[str] = typer.Option(
        None, "--runner-spec", help="Runner as module:Class, recorded for dispatch --runner."
    ),
    entry_point: Optional[str] = typer.Option(
        None, "--entry-point", help="Entry-point name the wheel registers under tt_inference_server.runners."
    ),
    weights: Optional[str] = typer.Option(
        None, "--weights", help="HF model repo id whose weights this bundle targets."
    ),
    weights_revision: Optional[str] = typer.Option(None, "--weights-revision"),
    weights_allow: Optional[List[str]] = typer.Option(None, "--weights-allow"),
    weights_ignore: Optional[List[str]] = typer.Option(None, "--weights-ignore"),
) -> None:
    """Package the local kernel cache for one build_key and publish it.

    A v2 bundle may also ship a self-contained runner wheel (--python-package +
    --runner-spec) and a weights reference (--weights), so a single `tt-kernel pull`
    installs kernels + runner + weights.
    """
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

    out_root = cache.resolve_out_root(cache_dir)
    try:
        key = cache.select_build_key(out_root, build_key)
    except (FileNotFoundError, ValueError) as exc:
        raise _err(str(exc))

    subtree = cache.build_key_path(out_root, key)
    typer.echo(f"Packaging build_key {key} from {subtree}")

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

    # v2 runtime payload: index shipped wheels under python/ and build the runner/weights blocks.
    runner_block: Optional[RunnerPayload] = None
    if wheel_paths:
        runner_block = RunnerPayload(
            spec=runner_spec,
            wheels=[p.name for p in wheel_paths],
            entry_point=entry_point,
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
        model=model or weights,
        tt_metal_version=version,
        arch=dev.arch,
        device_count=dev.device_count or 1,
        build_key=key,
        build_key_inputs=metal.build_key_inputs(
            num_hw_cqs=num_hw_cqs, harvesting_mask=dev.harvesting_mask
        ),
        kernel_count=cache.count_kernels(subtree),
        files=files,
        producer=Producer(
            tt_kernel_version=__version__,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            hostname=socket.gethostname(),
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
        try:
            hub.tag_repo(repo_id, [TT_KERNEL_TAG, dev.arch])
        except Exception as exc:  # tagging is best-effort
            typer.secho(f"  (could not write tags: {exc})", fg=typer.colors.YELLOW)

    typer.secho(f"✓ Pushed {repo_id} (build_key {key})", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------- pull
@app.command()
def pull(
    repo_id: str = typer.Argument(..., help="Source repo as namespace/name[@revision]."),
    force: bool = typer.Option(False, "--force", help="Install despite non-fatal mismatches."),
    cache_dir: Optional[str] = typer.Option(None, help="Override the tt-metal cache root."),
    probe: bool = typer.Option(False, "--probe", help="Open a device to read the true build_key."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Override arch detection."),
    models_dir: Optional[str] = typer.Option(None, "--models-dir", help="Where to download weights."),
    no_python: bool = typer.Option(False, "--no-python", help="Skip installing the runner wheel."),
    no_weights: bool = typer.Option(False, "--no-weights", help="Skip downloading weights."),
    kernels_only: bool = typer.Option(
        False, "--kernels-only", help="Install only the kernel cache (back-compat; implies the skips)."
    ),
    python_exe: Optional[str] = typer.Option(
        None, "--python", help="Target interpreter for the runner pip install (default: this venv)."
    ),
) -> None:
    """Download a bundle and install everything it carries: kernels, runner, weights.

    A single pull installs the kernel cache, pip-installs the shipped runner wheel, and
    downloads the model weights, then prints the exact `serve` command. Skip parts with
    --no-python / --no-weights / --kernels-only.
    """
    if kernels_only:
        no_python = no_weights = True

    repo_id, revision = _split_revision(repo_id)
    runner_installed = False
    weights_path: Optional[Path] = None
    with tempfile.TemporaryDirectory() as td:
        snapshot = hub.download_bundle(repo_id, revision, dest=td)
        manifest_path = snapshot / MANIFEST_NAME
        if not manifest_path.is_file():
            raise _err(f"{repo_id} is not a tt-kernel bundle (no {MANIFEST_NAME}).")
        manifest = Manifest.from_json(manifest_path.read_text())

        env = metal.local_env(arch_override=arch, probe=probe)
        report = compare(manifest, env)
        _print_report(report)

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

        # ---- runtime payload (schema v2) ----
        advisory = runner_version_advisory(manifest, env)
        if advisory is not None and (manifest.runner or manifest.weights):
            typer.secho(
                f"  ! runner/weights target tt-metal {advisory.expected!r}; you have "
                f"{advisory.detected!r}. Installing anyway — it will NOT run until the "
                "serving environment matches.",
                fg=typer.colors.YELLOW,
            )

        # Runner wheel: verify integrity, warn if the target venv lacks ttnn, then pip install.
        if manifest.runner and not no_python:
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
                typer.secho("✓ runner installed", fg=typer.colors.GREEN)
            except Exception as exc:  # noqa: BLE001 — record partial progress, don't roll back kernels
                _record_pull(repo_id, manifest, out_root, runner_installed=False,
                             weights_path=None, last_error=f"pip install failed: {exc}")
                raise _err(
                    f"Kernels are installed, but the runner pip install failed: {exc}\n"
                    f"  Re-run `tt-kernel pull {repo_id} --no-weights` to retry just the runner."
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
    if manifest.runner and runner_installed and weights_path is not None:
        typer.echo("\nRun it:")
        typer.secho("  " + runtime.serve_command(manifest.runner.spec, weights_path),
                    fg=typer.colors.CYAN)
        if not runtime.dispatch_available():
            typer.secho("  (install the dispatch serving package: pip install tt-inference-server)",
                        fg=typer.colors.YELLOW)
    elif manifest.runner:
        missing = []
        if not runner_installed:
            missing.append("runner (re-run without --no-python)")
        if weights_path is None and manifest.weights:
            missing.append("weights (re-run without --no-weights)")
        if missing:
            typer.secho(f"  pending: {', '.join(missing)}", fg=typer.colors.YELLOW)


def _record_pull(repo_id, manifest, out_root, *, runner_installed, weights_path, last_error) -> None:
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
        "weights_repo": manifest.weights.repo_id if manifest.weights else None,
        "weights_path": str(weights_path) if weights_path else None,
        "python_installed": runner_installed,
        "weights_installed": weights_path is not None,
        "installed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if last_error:
        entry["last_error"] = last_error
    localdb.record(repo_id, entry)


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
        typer.echo(
            f"{e['repo_id']}  build_key={e.get('build_key')}  arch={e.get('arch')}  "
            f"tt_metal={e.get('tt_metal_version')}"
        )


# -------------------------------------------------------------------------- search
@app.command()
def search(
    query: str = typer.Argument("", help="Free-text query over tt-kernel cache repos."),
    limit: int = typer.Option(50, help="Max results."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Search the Hub for published tt-kernel caches."""
    results = hub.search(query, limit=limit)
    if as_json:
        typer.echo(json.dumps(results, indent=2))
        return
    if not results:
        typer.echo("No matching bundles found.")
        return
    for r in results:
        vis = "private" if r.get("private") else "public"
        typer.echo(f"{r['id']}  [{vis}]  downloads={r.get('downloads')}")


# ------------------------------------------------------------------------------ rm
@app.command()
def rm(
    repo_id: str = typer.Argument(..., help="Installed bundle as namespace/name."),
    cache_dir: Optional[str] = typer.Option(None, help="Override the tt-metal cache root."),
) -> None:
    """Remove a locally installed cache subtree and its index entry."""
    entry = localdb.get(repo_id)
    if not entry:
        raise _err(f"{repo_id} is not recorded as installed.")
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
            f"Index entry removed; cache subtree was already gone.", fg=typer.colors.YELLOW
        )


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
