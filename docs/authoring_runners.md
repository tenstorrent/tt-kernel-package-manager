# Authoring a model for `tt-kernel`

There are two ways to make a model `pull`-and-serve with `tt-kernel`:

- **vLLM bundle — the default.** Author a `VllmGeneratorAdapter` and ship a small bundle
  folder; `tt-kernel serve` runs it through the Tenstorrent vLLM plugin (batching, paging,
  the real serving path). **This is what you almost certainly want.** Jump to
  **[Authoring a vLLM bundle](#authoring-a-vllm-bundle-the-default---backend-vllm)**.
- **Legacy runner bundle.** A duck-typed `generate()`/`generate_stream()` runner, served by
  tt-kernel's own minimal `legacy_serve` server. This path is **retired-but-supported**: it
  exists so an already-built legacy runner isn't stranded. It has none of vLLM's batching or
  performance — do not start here for a new model.

> ⚠️ If you are starting fresh, author a **vLLM bundle**. The legacy runner sections (1–5)
> below are only for an existing runner that already follows the old contract.

---

## Legacy runner bundles (retired serving path)

> **Legacy.** New models should be vLLM bundles (see the bottom of this guide). This section
> documents the older runner contract, served by `python -m tt_kernel.legacy_serve` (a small
> OpenAI-compatible shim that ships with tt-kernel — no external serving runtime required).

## 1. Implement the runner contract

Your runner is a plain Python class. It is **duck-typed** — you do not subclass anything.
It must expose exactly three methods and three attributes:

```python
class MyRunner:
    # --- required attributes, set in __init__ ---
    _tokenizer: object   # a transformers-like tokenizer
    _listed: bool        # True for a known/validated model; else False
    _community: bool     # True if unverified/community (drives the serve "community" tag)

    # --- required methods ---
    def generate(self, prompt: str, max_new_tokens: int = 50,
                 temperature: float = 1.0, chat: bool = True) -> str: ...

    def generate_stream(self, prompt: str, max_new_tokens: int = 50,
                        temperature: float = 1.0, chat: bool = True):
        # yield decoded text deltas (str), one per step;
        # the FINAL yielded item MUST be a dict:
        #   {"finish_reason": str, "prompt_tokens": int, "completion_tokens": int}
        ...

    def benchmark(self, prompt: str, n_tokens: int = 50) -> tuple[float, str]:
        return (tokens_per_second, output_text)
```

`generate_stream` is what the OpenAI-compatible server drives for both streaming and
non-streaming responses, so make sure the final usage dict is correct.

### Constructor

The serving layer constructs your runner as:

```python
MyRunner(model_path, device, max_seq=..., unsafe=..., force_novel=...,
         trace_region_size=..., device_ids=...)
```

Declare only the keyword args you use (or accept `**kwargs` and ignore the rest). `model_path`
is the local weights directory `tt-kernel pull` downloaded for you.

### Device ownership (mesh runners)

If your model needs a device topology dispatch doesn't open for you — e.g. a 1×1 **mesh** —
set `MANAGES_OWN_DEVICE = True`. Then `device` is `None` and you open/own the device yourself:

```python
class MyRunner:
    MANAGES_OWN_DEVICE = True
    def __init__(self, model_path, device, **kwargs):
        import ttnn, atexit
        self.mesh = ttnn.open_mesh_device(...)
        atexit.register(ttnn.close_mesh_device, self.mesh)   # <-- REQUIRED
        ...
```

**Closing the device cleanly is mandatory.** Serving runs one model per process and swaps
models by restarting; an ungraceful teardown can leave a locked device mutex
(`/dev/shm/tt_device_*`) that wedges the card until a manual `tt-smi -r`. Register an
`atexit` close (or close in your own lifecycle) so the next model starts on a pristine card.

---

## 2. Make it a self-contained, renamespaced package

The wheel you ship is installed with `pip install --no-deps` into the serving environment,
so it must stand on its own:

- **No `ttnn` / tt-metal in your dependencies.** They are the platform — already present in
  the serving venv, and never vendored. `import ttnn` is fine; depending on it is not.
- **Renamespace away from the tt-metal tree.** If your runner currently lives inside a
  tt-metal checkout with absolute imports like `from models.experimental.foo...`, move it
  under your own top-level package (e.g. `ttrunner_mymodel/`) with package-relative imports.
  Two runner packages that both vendor a `models/` tree will collide on that namespace and
  break multi-model installs.
- **Ship only what's needed.** Trace the import graph from your runner class and include only
  the modules it actually reaches. Drop unrelated models/utilities.
- **Pin a Python version range, not an exact build,** if you declare one.

Build it like any wheel:

```bash
python -m build --wheel        # or: pip wheel . --no-deps -w dist/
# -> dist/ttrunner_mymodel-<ver>-py3-none-any.whl
```

### Optional: register an entry point for auto-discovery

If you declare the `tt_models.runners` entry point, dispatch can auto-select your
runner from the model's HF config — `serve <model>` with no `--runner` flag just works:

```toml
# pyproject.toml of your runner package
[project.entry-points."tt_models.runners"]
my_model = "ttrunner_mymodel.runner:MyRunner"
```

Add a class hook so dispatch knows which models you claim (priority: `claims()` >
`supported_architectures` > `supported_model_types`):

```python
class MyRunner:
    supported_model_types = {"qwen3_5_moe"}
    # or, for finer control:
    # @classmethod
    # def claims(cls, hf_config): return hf_config.model_type == "qwen3_5_moe"
```

Even without an entry point, the explicit `--runner-spec` you record in the bundle (next
section) always works.

---

## 3. Versioning — the rule that makes or breaks a pull

Your runner calls a specific `ttnn` API, and the kernel cache in the same bundle was compiled
from that same tt-metal build. They are **co-versioned**: the bundle's single
`tt_metal_version` gates both. On `pull`:

- **Kernel cache**: a version mismatch is a hard block (mismatched binaries are useless).
- **Runner + weights**: install anyway, with a loud warning that it won't run until the
  serving environment matches.

`tt-kernel` does **not** fix a mismatch — that is the user's blocker to resolve (install the
matching tt-metal/ttnn). So: **build, push, and serve in environments with the same tt-metal
build.** The simplest reliable path is to produce the bundle on the same build you serve on.

---

## 4. Push the bundle

From a machine whose kernel cache is populated for your model (run it once to JIT-compile),
with your wheel built:

```bash
tt-kernel push <ns>/<model>-blackhole --private \
  --python-package dist/ttrunner_mymodel-0.1-py3-none-any.whl \
  --runner-spec ttrunner_mymodel.runner:MyRunner \
  --entry-point my_model \
  --weights <hf-org>/<hf-model>
```

- `--python-package` (repeatable) ships your wheel under `python/` in the bundle and
  integrity-indexes it.
- `--runner-spec module:Class` is the selector dispatch uses; **required** whenever you ship a
  wheel. Must be `module:Class` (or `module.Class`).
- `--entry-point` is informational/auto-discovery; optional.
- `--weights` records the HF model repo `pull` will download. Add `--weights-revision`,
  `--weights-allow`, `--weights-ignore` to scope it.

`--tt-metal-version` / `--arch` overrides exist for testing as with kernel-only bundles.

**Reference runners.** If the runner already ships in the consumer's environment (e.g. it's
registered in `tt_models`), omit `--python-package` and pass `--runner-spec` alone —
the bundle records a *reference* the consumer resolves rather than a shipped wheel. Add
`--runner-source <pip-name|git-url>` to tell the consumer where to get it. Reference mode trades
reproducibility for size: only a packaged wheel guarantees the consumer runs your exact runner
code (and thus reproduces your numbers).

---

## 5. What the consumer gets

```bash
tt-kernel pull <ns>/<model>-blackhole
```

installs the kernel cache, `pip install --no-deps` your wheel, downloads the weights, records
the binding, and prints the exact command to serve via tt-kernel's legacy-runner server
(needs `pip install 'tt-kernel[serve]'` for fastapi + uvicorn):

```
python -m tt_kernel.legacy_serve \
    --runner ttrunner_mymodel.runner:MyRunner --model <weights-path>
```

Skip flags (`--no-python`, `--no-weights`, `--kernels-only`) let users install parts. A
re-pull is idempotent.

---

## Authoring checklist

- [ ] Runner exposes `generate` / `generate_stream` / `benchmark` and sets `_tokenizer` /
      `_listed` / `_community` in `__init__`.
- [ ] `generate_stream` yields str deltas, final item is the usage dict.
- [ ] Constructor takes `(model_path, device, **kwargs)`.
- [ ] If a mesh/own device: `MANAGES_OWN_DEVICE = True` **and** a clean `atexit` close.
- [ ] Wheel is renamespaced (no `from models...`), self-contained, `ttnn` NOT a dependency.
- [ ] (Optional) `tt_models.runners` entry point + a `claims()`/`supported_*` hook.
- [ ] Bundle built and served on the **same tt-metal build** (co-versioned with the kernels).
- [ ] `tt-kernel push` run with `--python-package` + `--runner-spec` (+ `--weights`).

## Common pitfalls

- **`from models...` imports** — break on install; renamespace.
- **Depending on `ttnn`/tt-metal in the wheel** — `--no-deps` ignores it, or worse, a plain
  install pulls a conflicting `ttnn` from PyPI. Keep them out of `dependencies`.
- **`--python-package` without `--runner-spec`** — rejected at push; the wheel is unusable
  without a selector.
- **Producing the bundle on a different tt-metal build than you serve on** — runner won't
  import; the warning is honest, not a fix.
- **Not closing a mesh device** — wedges the card for the next model in a hot-swap.

---

## Authoring a vLLM bundle (the default, `--backend vllm`)

**This is the recommended path for any new model.** Serving goes through the Tenstorrent
**vLLM** plugin (`tenstorrent/vllm`), so the model's runner is a
`VllmGeneratorAdapter` — a low-level paged-attention adapter (`initialize_vllm_model`,
`prefill_forward`, `decode_forward`, `allocate_kv_cache`, `warmup_model_*`, …), *not* a
`generate()`/`generate_stream()` runner. See the contract at
`tt-metal/models/common/readiness_check/contract_vllm.py` and the canonical example
`tt-metal/models/tt_transformers/tt/generator_vllm.py`.

### The v4 unified manifest (`--manifest`, recommended)

Write **one** manifest that declares everything the model needs, and `tt-kernel` **renders**
the plugin-owned `vllm_metadata.json` from it on `pull` — you no longer hand-write two files.
The manifest is a *partial*: you declare the model's requirements; `tt-kernel` fills the
bookkeeping (producer, file index, arch/version detection) at push time.

```jsonc
// laguna.json — an authored v4 manifest
{
  "platform":     { "ttnn": ">=0.72,<0.76" },        // PEP 440 range, not an exact pin
  "runtime":      { "kind": "vllm", "version": ">=0.24",   // vLLM core range
                    "plugin_version": ">=0.3,<0.4" },      // Tenstorrent vLLM plugin range
  "target":       "p150x4",                            // searchable machine SKU
  "mesh":         { "devices": 4, "topology": "1x4", "fabric": "FABRIC_1D_RING" },
  "entrypoint":   { "class": "ttlaguna.generator_vllm:LagunaForCausalLM",
                    "arch_name": "LagunaForCausalLM" },  // -> main_class + arch
  "weights":      { "repo": "poolside/Laguna-XS-2.1" },
  "resources":    { "max_model_len": 131072, "max_num_seqs": 8, "block_size": 64,
                    "trace_region_bytes": 1500000000 },
  "capabilities": { "tool_parser": "poolside_v1", "reasoning_parser": "poolside_v1" },
  "env":          { "MESH_DEVICE": "P150", "TT_LAGUNA_PIPE_CHUNK": "2048" }
}
```

Push it (ship a `--bundle-dir` only if you have a custom adapter class / extension wheels —
omit it when `entrypoint.class` is a tt-metal built-in):

```bash
tt-kernel push you/laguna --private --backend vllm \
  --manifest ./laguna.json --bundle-dir ./adapter   # --bundle-dir optional for built-ins
tt-kernel pull  you/laguna                            # renders vllm_metadata.json locally
tt-kernel serve you/laguna                            # composes + launches the vLLM server
```

**How the launch command is composed.** `tt-kernel` turns `resources`/`capabilities`/`env`
into the `server_example_tt.py` launch command (underscore flags: `--max_model_len`,
`--max_num_seqs`, `--block_size`, `--trace_region_size`, `--tool_parser`, `--reasoning_parser`)
with `VLLM_USE_V1=1` plus your `env` overlaid. When the mapping doesn't cover something, use
the escape hatches under `resources`:

- `"extra_args": ["--enable-prefix-caching"]` — appended to the composed command.
- `"command_override": { "default": ["python3","my_server.py", ...],
   "blackhole-4card": [...] }` — replaces composition entirely, per machine key.

**How compatibility is resolved.** `platform.ttnn`, `runtime.version` (vLLM core), and
`runtime.plugin_version` (the `vllm_tt_plugin` package) are all *ranges*. On `pull`, an installed
version outside a range is a **forceable** block (`--force` overrides), never fatal — only an
`arch` mismatch is fatal. A dev checkout whose version is a bare git sha is treated as "assume
OK" and never falsely blocked. `tt-kernel doctor you/laguna` prints the required-vs-installed
verdict (and how to get in range) without installing anything. Omitting `plugin_version` keeps
the legacy presence-only plugin check (the fork tracks `dev`).

**Linking to the right tt-metal build (instances).** When a host has several tt-metal builds,
tt-kernel doesn't guess from whatever venv is active — it consults an **instance registry** (the
supply side). On `pull` it selects the **newest installed instance that satisfies all three
ranges** and **pins** that instance's activation (interpreter + `TT_METAL_HOME` / `PYTHONPATH` /
`LD_LIBRARY_PATH`) into the install record; `serve` then launches the server under that exact
build (re-resolving gracefully if the pinned build was removed). Manage instances with:

```bash
tt-kernel instances list --for you/laguna   # what's installed, and which satisfy this model
tt-kernel instances add --name metal-0.73 \  # register a build auto-scan can't find
  --python /opt/tt/0.73/venv/bin/python --tt-metal-home /opt/tt/0.73
tt-kernel instances scan                     # auto-discover tt-metal checkouts
tt-kernel pull  you/laguna --instance metal-0.73   # force a specific instance
tt-kernel serve you/laguna --instance metal-0.73
```

Instances are discovered from three sources, unioned: the **active** interpreter (always a
candidate — a single-build box behaves as before), explicit **registry** entries in
`~/.config/tt-kernel/instances.json` (the manager owns this file), and an **auto-scan** of
tt-metal checkouts under the scan roots. tt-kernel never *installs* a tt-metal — it only
selects among what's present and reports what's missing.

**Discovery.** `push` tags the repo with its `arch` and `target`, so consumers can ask
`tt-kernel search --target p150x4` / `--arch blackhole` for "what runs on my box".

### Legacy: a hand-written `vllm_metadata.json` (`--bundle-dir` only)

The older path — you author `vllm_metadata.json` yourself and `tt-kernel` ships it verbatim —
is still supported for existing bundles. Prefer the v4 manifest above for anything new.

A vLLM bundle is **kernels-less**: it ships no precompiled cache (vLLM JITs at first-run
warmup into tt-metal's own local cache). It is a self-contained *folder*:

```
my_bundle/
  vllm_metadata.json      # plugin-owned schema (below)
  generator_vllm.py       # the adapter class (+ any deps), or omit for a tt-metal built-in
```

`vllm_metadata.json` (the plugin owns this schema; `tt-kernel` ships it verbatim and reads
only `arch` + the per-machine `launch` command):

```json
{
  "arch": "LlamaForCausalLM",
  "main_class": "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM",
  "hf_weights": "meta-llama/Llama-3.1-8B-Instruct",
  "launch": {
    "blackhole": {
      "command": ["python3", "server_example_tt.py", "--model",
                  "meta-llama/Llama-3.1-8B-Instruct", "--max_num_seqs", "8"],
      "env": {"MESH_DEVICE": "P150", "VLLM_USE_V1": "1"}
    },
    "default": { "command": ["python3", "server_example_tt.py", "--model", "..."], "env": {} }
  }
}
```

- `arch` is the HF `architectures` name; the plugin registers it under its `TT`-prefix
  convention (`TT<arch>`). Reference an existing tt-metal generator via `main_class` and the
  folder needs no code; ship a novel adapter as `generator_vllm.py` in the folder.
- `launch` is keyed per machine; `tt-kernel serve` selects the entry for the local machine
  (`<arch>-<n>card` > `<arch>` > `default`, override with `TT_KERNEL_MACHINE`).

Push, then serve:

```bash
tt-kernel push you/mymodel --private --backend vllm --bundle-dir ./my_bundle \
  --weights meta-llama/Llama-3.1-8B-Instruct
tt-kernel serve you/mymodel     # pulls the folder, sets EXTRA_MODELS_DIR, launches vLLM
```

On the serving host the vLLM plugin must be the fork that supports `EXTRA_MODELS_DIR`
(`scripts/install.sh` sets this up). No plugin source edit is needed — the bundle registers
itself.
