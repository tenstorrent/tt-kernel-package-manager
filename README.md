# tt-kernel

A wrapper around Hugging Face that lets users upload and distribute versioned kernels
and configs for Tenstorrent, so you can pull a pre-tuned cache and run models in known-good
configurations.

Concretely: it publishes and pulls **precompiled tt-metal kernel caches** over Hugging Face
Hub, so a model's first run on Tenstorrent hardware is a cache **hit** instead of a slow JIT
recompile.

tt-metal JIT-compiles every kernel on first run and caches the RISC-V binaries on disk.
Those binaries are deterministic for a fixed `(tt-metal build, arch, device config,
compile-time args)` tuple. `tt-kernel` packages that cache, publishes it as a Hugging
Face model repo addressed `namespace/name`, and on pull validates compatibility before
installing it into the local cache. **Only the kernel cache + a compatibility manifest
are shipped — never model weights.**

## Install

```bash
pip install tt-kernel        # from this repo: pip install -e .
```

## Usage

```bash
tt-kernel login                                   # reuses huggingface_hub's token store
tt-kernel push you/smallmodel-blackholex1 --public
tt-kernel info  you/smallmodel-blackholex1        # manifest + compatibility verdict
tt-kernel pull  you/smallmodel-blackholex1        # validate, then install into the cache
tt-kernel search gemma                            # discover published caches
tt-kernel list                                    # locally installed bundles
tt-kernel rm    you/smallmodel-blackholex1        # remove an installed cache subtree
```

Typical workflow: run your model once to populate the cache, `push`, then on another
matching host `pull` and re-run — the kernels load from cache with no recompile.

## How compatibility is enforced

A cached binary is only valid when the consumer's environment matches the producer's.
`tt-kernel` records this in `tt_kernel_manifest.json` and checks it on `pull`:

| Field | Source of truth | On mismatch |
|-------|-----------------|-------------|
| `arch` | tt-smi → `ARCH_NAME` → `--arch` | **fatal** — binaries are a different ISA |
| `tt_metal_version` | package metadata → `git describe` | blocked (use `--force`) — per-kernel hashes won't match |
| `build_key` inputs | tt-smi + env + flags | blocked (use `--force`) — names a different cache dir |
| `device_count` | tt-smi | warning (use `--force`) |

`build_key` (which names the on-disk cache subtree, `<cache_root>/<build_key>/`) is
computed in C++ and not exposed to Python, so `pull` reconstructs its **inputs** —
`arch`, dispatch core type/axis, `num_hw_cqs`, `harvesting_mask` (only when coordinate
virtualization is disabled), and a compile-flag fingerprint — and refuses to install on
a mismatch. Pass `--probe` to open a device and read the true local `build_key` for an
exact integer check.

These rules mirror the verified tt-metal source: cache root in `rtoptions.cpp` /
`build.cpp`, layout in `jit_compile_server.cpp`, `build_key` in `build_env_manager.cpp`,
and the per-kernel hash in `program_descriptors.cpp`.

## Cache location

Resolved exactly as tt-metal does: `TT_METAL_CACHE` → `$HOME/.cache/tt-metal-cache/` →
`/tmp/tt-metal-cache/`. Override with `--cache-dir`.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest
```

## Testing without hardware

You don't need a Tenstorrent card or a tt-metal build to exercise the full
push/pull round-trip — generate a synthetic cache and stamp the version by hand:

```bash
# 1. Make fake cache data laid out like a real tt-metal cache
scripts/make_test_cache.sh /tmp/ttk-test-cache 4242

# 2. Auth (use your own current HF token)
export HF_TOKEN=hf_...
tt-kernel login

# 3. Publish it — --arch and --tt-metal-version stand in for hardware/build detection
tt-kernel push <you>/kernel-selftest --private \
  --cache-dir /tmp/ttk-test-cache --arch blackhole \
  --tt-metal-version v0.99-test --model google/gemma-test

# 4. Inspect + compatibility verdict
tt-kernel info <you>/kernel-selftest --arch blackhole

# 5. Pull into a DIFFERENT empty cache dir (simulates another machine)
tt-kernel pull <you>/kernel-selftest --cache-dir /tmp/ttk-restore --arch blackhole
diff -r /tmp/ttk-test-cache/4242 /tmp/ttk-restore/4242 && echo "round-trip OK"

# 6. Local bookkeeping + teardown
tt-kernel list
tt-kernel rm <you>/kernel-selftest --cache-dir /tmp/ttk-restore
```

Try the guard rails too: `tt-kernel pull ... --arch wormhole_b0` fails fatally
(wrong ISA), and a `--tt-metal-version` that differs from the bundle's blocks the
install until you add `--force`.
