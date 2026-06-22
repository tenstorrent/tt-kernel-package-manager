"""``tt-kernel`` command-line interface."""

from __future__ import annotations

import datetime
import json
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Optional

import typer

from . import MANIFEST_NAME, TT_KERNEL_TAG, __version__
from . import auth, cache, hub, localdb, metal
from .manifest import CompatibilityReport, Manifest, Producer, compare

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
) -> None:
    """Package the local kernel cache for one build_key and publish it."""
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
    manifest = Manifest(
        name=name or repo_id.split("/")[-1],
        model=model,
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
    )

    with tempfile.TemporaryDirectory() as td:
        staged = Path(td)
        # Mirror the subtree under <staged>/<build_key>/ so it installs cleanly.
        shutil.copytree(subtree, staged / str(key))
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
) -> None:
    """Download a bundle, validate it against the local environment, and install."""
    repo_id, revision = _split_revision(repo_id)
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

        typer.echo(f"Verifying {len(manifest.files)} files ...")
        problems = cache.verify_files(staged, manifest.files)
        if problems:
            for p in problems[:20]:
                typer.secho(f"  {p}", fg=typer.colors.RED)
            raise _err(f"Integrity check failed ({len(problems)} problem(s)).")

        out_root = cache.resolve_out_root(cache_dir)
        target = cache.install_subtree(staged, out_root, manifest.build_key)

    localdb.record(
        repo_id,
        {
            "name": manifest.name,
            "build_key": manifest.build_key,
            "arch": manifest.arch,
            "tt_metal_version": manifest.tt_metal_version,
            "out_root": out_root,
            "installed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )
    typer.secho(f"✓ Installed {repo_id} -> {target}", fg=typer.colors.GREEN)


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
