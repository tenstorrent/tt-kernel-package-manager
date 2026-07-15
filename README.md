# tt-kernel

> ⚠️ **Experimental — no support, no guarantees.** tt-kernel is an early, experimental
> project. Nothing here is officially supported, and we make no claim of correctness,
> stability, or fitness for any purpose. APIs, the bundle format, and behavior may change
> or break at any time without notice. Use it at your own risk.

`tt-kernel` distributes models over the Hugging Face Hub and serves them on Tenstorrent
hardware. The **default serving path is the Tenstorrent vLLM plugin** — an OpenAI-compatible
server. One command pulls a model and brings the server up:

```bash
tt-kernel serve <namespace>/<model>     # pull the bundle, register it with vLLM, launch the server
```

### vLLM bundles (the default)

A **vLLM bundle** is a small, self-contained folder: a plugin-owned `vllm_metadata.json`
(the HF architecture, the generator-adapter class, a per-machine launch command, and a
reference to the HF weights) plus the adapter code and its dependencies. It ships **no kernel
cache and no weights** — vLLM JIT-compiles kernels at first-run warmup into tt-metal's own
local cache, and the model fetches weights from the referenced HF repo. On `pull`, the folder
is placed into a local bundles directory that the vLLM plugin discovers via `EXTRA_MODELS_DIR`
and auto-registers, so no per-model edit to the plugin is needed. **Weights are never stored in
a bundle** — only referenced by their HF repo id. To author one, see
**[docs/authoring_runners.md](docs/authoring_runners.md)**.

### Kernel-cache bundles (legacy dispatch path)

`tt-kernel` also publishes and pulls **precompiled tt-metal kernel caches** for the older
"dispatch" serving runtime (`tt_api.serve`), so a model's first run is a cache **hit** instead
of a slow JIT recompile. tt-metal JIT-compiles every kernel on first run and caches the RISC-V
binaries; they are deterministic for a fixed `(tt-metal build, arch, device config,
compile-time args)` tuple. `tt-kernel` packages that cache, publishes it as an HF model repo
addressed `namespace/name`, and validates compatibility before installing it locally. See
[Kernel-cache bundles (legacy)](#kernel-cache-bundles-kernels--runner--weights-legacy-dispatch-path)
below.

`tt-kernel serve` (and `tt-kernel run`) is the **front door**: it resolves whether a bundle
exists and routes accordingly — the vLLM plugin by default, the dispatch runtime for a
kernel-cache bundle, or the dynamic runtime on a bare Hugging Face repo. Custom implementations
win; nothing overrides them.

## Install

```bash
pip install tt-kernel        # from this repo: pip install -e .
```

To serve, you also need the Tenstorrent **vLLM fork + plugin** on top of a working tt-metal
env. `scripts/install.sh` sets up the whole serving stack (vLLM fork + plugin + tt-kernel) and
runs `tt-kernel doctor`:

```bash
scripts/install.sh           # installs fork + plugin + tt-kernel into the tt-metal venv
```

## Usage

```bash
tt-kernel login                                   # reuses huggingface_hub's token store
tt-kernel doctor                                  # check tt-metal/tt-lang/tt-api/vLLM + hardware

# vLLM (default) — serve a model through the Tenstorrent vLLM plugin
tt-kernel serve you/mymodel                        # pull if needed, register, launch the OpenAI server
tt-kernel serve you/mymodel --print                # print the launch command instead of running it
tt-kernel push  you/mymodel --backend vllm \       # publish a vLLM bundle folder
  --bundle-dir ./bundle --weights some-org/mymodel

# Shared / discovery
tt-kernel pull  you/mymodel                        # download + install a bundle locally
tt-kernel info  you/mymodel                        # manifest + compatibility verdict
tt-kernel search gemma                             # discover published bundles
tt-kernel search gemma --catalog                   # only bundles listed in the community catalog
tt-kernel list                                     # locally installed bundles
tt-kernel rm    you/mymodel                        # remove an installed bundle

# Kernel-cache (legacy dispatch path) — see below
tt-kernel run   you/smallmodel-blackholex1         # dispatch runtime + precompiled cache
tt-kernel clean --all                              # wipe cache subtrees for a clean producer state
```

## Serving with vLLM (the default)

`tt-kernel serve <id>` is the one-command path. It pulls the bundle folder if it isn't already
installed, lays it into the local bundles directory, points the vLLM plugin at that directory
via `EXTRA_MODELS_DIR`, and launches the OpenAI-compatible server with the bundle's
per-machine launch command:

```bash
tt-kernel serve you/mymodel                   # pull-if-needed -> register -> launch; prints the endpoint
tt-kernel serve you/mymodel --print           # emit the exact launch command + env instead of running
tt-kernel serve you/mymodel --local-only      # require an installed bundle; never hit the Hub
tt-kernel serve you/mymodel --bundles-dir DIR # override the EXTRA_MODELS_DIR location
```

Repeat invocations skip the pull and go straight to launch. `tt-kernel run <id>` routes a vLLM
bundle to this same path.

### Publishing a vLLM bundle

Author a bundle folder — a `vllm_metadata.json` plus the adapter class (or a reference to an
existing tt-metal generator) — then push it. It is **kernels-less**: no precompiled cache and
no weights are shipped. See **[docs/authoring_runners.md](docs/authoring_runners.md)** for the
metadata schema and the adapter contract.

```bash
tt-kernel push you/mymodel --private --backend vllm \
  --bundle-dir ./bundle \                  # folder with vllm_metadata.json + adapter code
  --weights some-org/mymodel               # HF weights the model loads at runtime

tt-kernel pull you/mymodel                  # lay the folder into the local bundles dir
tt-kernel pull you/mymodel --with-weights   # ...and also pre-download the weights (default: skip)
```

The plugin auto-registers every bundle it finds under `EXTRA_MODELS_DIR`, so no per-model edit
to the plugin is required. `vllm_metadata.json` is owned by the plugin; `tt-kernel` ships it
verbatim and reads only the architecture and the per-machine launch command.

## Kernel-cache bundles (kernels + runner + weights, legacy dispatch path)

> **Legacy.** This path serves through the older dispatch runtime (`tt_api.serve`), not vLLM.
> Prefer a [vLLM bundle](#serving-with-vllm-the-default) for new models.

A kernel-cache bundle ships a precompiled tt-metal cache and can add a runner and a weights
reference so one `pull` installs everything. The runner is either **packaged** (a wheel shipped
in the bundle, via `--python-package`) or a **reference** (a `--runner-spec` the consumer
already has or installs from `--runner-source`). **Producing the runner is governed by
[docs/authoring_runners.md](docs/authoring_runners.md)** — read it before pushing one; a
runner that doesn't follow the contract won't install or serve.

```bash
# Producer (on a host whose kernel cache is populated, with the runner wheel built):
tt-kernel push you/mymodel-blackhole --private \
  --python-package dist/ttrunner_mymodel-0.1-py3-none-any.whl \
  --runner-spec ttrunner_mymodel.runner:MyRunner \
  --weights some-org/mymodel

# Consumer:
tt-kernel pull you/mymodel-blackhole       # kernels + pip-install runner + download weights
#   -> prints the exact `serve --unsafe --runner ...` command to run
```

`pull` partial-install flags: `--no-python`, `--no-weights`, `--kernels-only`,
`--models-dir DIR`, `--python PATH` (target interpreter for the runner install).

**Version coupling:** the runner and the kernels are co-versioned (the kernels were compiled
from the tt-metal build whose `ttnn` the runner calls). A kernel-version mismatch hard-blocks;
the runner/weights install anyway with a warning. `tt-kernel` does not fix a mismatch — build
and serve on the same tt-metal build. See the guide for details.

## The front door: `serve` and `run`

For a vLLM bundle, [`tt-kernel serve <id>`](#serving-with-vllm-the-default) is the default and
the recommended entry point.

`tt-kernel run <id>` is the general resolver. It routes a **vLLM bundle to the vLLM plugin**
(the same path as `serve`); anything else falls down the legacy **dispatch three-tier ladder** —
a curated kernel-cache bundle always wins, and a bare Hugging Face repo falls through to the
dynamic dispatch runtime. A completely custom implementation is therefore never overridden.

| Tier | Trigger | What runs |
|------|---------|-----------|
| **vLLM (default)** | a vLLM bundle | the Tenstorrent vLLM plugin (OpenAI server) — see [Serving with vLLM](#serving-with-vllm-the-default) |
| **1 — custom bundle** | a kernel-cache bundle carries a runner | the author's runner + their precompiled kernels (dispatch) |
| **2 — kernels-only** | a kernel-cache bundle with no runner | the dynamic dispatch runtime, with the precompiled cache hitting on disk |
| **3 — no bundle** | a bare HF id / local path | the dynamic dispatch runtime on the model as-is |

```bash
tt-kernel serve you/mymodel                  # vLLM bundle -> the plugin (default)
tt-kernel run   you/mymodel-blackhole        # kernel-cache bundle -> author's runner + kernels (dispatch)
tt-kernel run   meta-llama/Llama-3.1-8B      # no bundle -> dynamic dispatch runtime on the bare repo
tt-kernel run   you/mymodel --print          # print the serve command instead of executing
tt-kernel run   you/mymodel --local-only     # resolve only against installed bundles (no Hub call)
```

On the legacy dispatch path, when a tuned bundle is **published but not installed**, `run`
tells you it exists (`tt-kernel pull <id>` to use it) and then does exactly what you asked —
running the dynamic path on the bare repo rather than silently downloading. That handoff
targets the dispatch runtime (`tt_api.serve`); `tt-kernel` only *detects* that package, never
imports it — the runner spec is an opaque string.

## Community catalog (web front end)

`web/` is a static, searchable browser for community-published bundles — like
`ollama.com/models`, backed entirely by the Hugging Face Hub. It is a **pure index**: it
hosts and stores nothing, and queries the HF public API live from the visitor's browser.
Every card is a pointer to a public HF repo that remains under its author's governance.

Listing is an explicit opt-in, separate from `push`:

```bash
tt-kernel push you/mymodel-blackhole --public --publish   # push and list in one step
tt-kernel publish   you/mymodel-blackhole                  # list a repo pushed earlier
tt-kernel unpublish you/mymodel-blackhole                  # delist (repo untouched)
```

`--publish` requires `--public` and adds the `tt-kernel-catalog` tag; the catalog shows only
repos carrying it. Deploy the front end by copying `web/` to any static server — no backend,
no build step. See **[web/README.md](web/README.md)**.

## Checking your toolchain

`tt-kernel` expects the surrounding stack — tt-metal, tt-lang, tt-api, and the vLLM fork +
plugin — to already be present on the system. It does **not** install them (use
`scripts/install.sh` for that); it checks they are adequate and warns when they are not.

```bash
tt-kernel doctor
```

```
Toolchain:
  ✓ tt-metal: 0.72.1.dev3 (require >= 0.72.0) — ok
  ✓ tt-lang: 1.1.3 (require >= 1.1.3) — ok
  ✓ tt-api: 0.1.0 (require >= 0.1.0) — ok
  ✓ vllm: 0.11.0 (require >= tenstorrent/vllm@dev + plugin) — ok (vllm + TT plugin present)

Hardware:
  ✓ arch=blackhole devices=1 (via tt-smi)
```

The vLLM check is presence-based (the fork tracks the `dev` branch): both `vllm` and the
`vllm_tt_plugin` package must be importable.

`doctor` exits non-zero if any component is missing or below the required version. `run` and
`pull` run the same check and emit a warning (they do not abort) so a version skew is visible
before it bites. tt-api is detected by import and its `VERSION` file (it is
normally used from a checkout, not pip-installed).

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
  --tt-metal-version v0.99-test

# 4. Inspect + compatibility verdict
tt-kernel info <you>/kernel-selftest --arch blackhole

# 5. Pull into a DIFFERENT empty cache dir (simulates another machine)
tt-kernel pull <you>/kernel-selftest --cache-dir /tmp/ttk-restore --arch blackhole
diff -r /tmp/ttk-test-cache/tt-metal-cache4242 /tmp/ttk-restore/tt-metal-cache4242 \
  && echo "round-trip OK"

# 6. Local bookkeeping + teardown
tt-kernel list
tt-kernel rm <you>/kernel-selftest --cache-dir /tmp/ttk-restore
```

Try the guard rails too: `tt-kernel pull ... --arch wormhole_b0` fails fatally
(wrong ISA), and a `--tt-metal-version` that differs from the bundle's blocks the
install until you add `--force`.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Reporting bugs via GitHub Issues
- Submitting pull requests
- Coding standards and testing requirements

Pull requests are reviewed weekly. For questions, feel free to open an issue or discussion.

## License

This project is licensed under the **Apache License 2.0** - see [LICENSE](LICENSE) for the complete license text.

For clarification on how this license applies to commercial use, modifications, and patent grants, see [LICENSE_understanding.txt](LICENSE_understanding.txt).

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.
