# Self-contained (v5) model packages

A **self-contained bundle** ships the platform *inside* the package: the author's built `ttnn`
wheel (custom C++/LLK kernels compiled in), optionally the base vLLM + plugin wheels, and their
modified `tt-metal-community` tree — plus a generated `install.sh`/`run.sh` and a v5 manifest.
Weights stay a **pointer** (an HF repo id), downloaded at pull. A consumer needs only a TT card +
firmware. `tt-model` alone does the whole job — no tt-cli, no pre-provisioned tt-metal/vLLM.

## User flow

### Producer — "package what's on your box"
Build/bring up your model on `tt-metal-community` (your ttnn wheel now carries your kernels), then:

```bash
tt-model package <your-org>/<model-name> \        # HF target is a POSITIONAL arg (omit + --out to stage locally)
  --from-metal .                                   # your modified tt-metal-community tree
  --ttnn-wheel dist/ttnn-*.whl                      # your built engine wheel (required)
  --vllm-wheel dist/vllm-*.whl                      # optional: empty-target base vLLM
  --plugin-wheel dist/vllm_tt_plugin-*.whl          # optional: the TT vLLM plugin
  --arch-name LlamaForCausalLM                      # HF architecture -> vllm_metadata
  --main-class models.tt_transformers.tt.generator_vllm:LlamaForCausalLM \
  --weights unsloth/Llama-3.2-3B-Instruct           # POINTER — weights are not embedded
  --mesh P150 \
  --max-num-seqs 32 --block-size 64 --max-model-len 4096   # serving args (TT backend needs a
                                                            # supported batch + block size)
```
`--wheels-dir <dir>` auto-classifies `ttnn-*` / `vllm-*` / `vllm_tt_plugin-*` instead of the
explicit flags. Large wheels go to git-LFS automatically on push. The result is one HF **model**
repo (the "running folder"): `wheels/`, `metal/`, `install.sh`, `run.sh`, a per-model
`vllm_models/<name>/vllm_metadata.json` (the EXTRA_MODELS_DIR contract), and
`tt_kernel_manifest.json`. If you omit `--max-num-seqs`/`--block-size`, the launcher defaults to
32/64 (the known-good tt_transformers values).

### Consumer — pull + serve (only a card + firmware required)
```bash
tt-model pull  <org>/<model-name>     # installs the shipped wheels into the bundle's OWN venv,
                                        # (optionally --with-weights) downloads the weights
tt-model serve <org>/<model-name>     # runs the bundle's run.sh in that venv (OpenAI endpoint)
```
`serve` also install-then-serves a not-yet-pulled bundle. Everything runs from the bundle's venv;
the host's tt-metal/vLLM (if any) is never touched.

## Testing

### Offline (no hardware, no network)
The producer/consumer logic is fully unit-tested with mocked pip + HF:
```bash
pytest tests/test_v5.py                    # manifest v5 schema, self-contained compare() rules
pytest tests/test_packaging.py             # stage_package layout, wheel-tag parsing, CLI stage-only
pytest tests/test_self_contained_install.py # pull installs into a venv; serve runs run.sh
pytest                                     # full suite (expected: 201 passed, 1 skipped)
```

### Hardware smoke (a TT card)
Validates the real round-trip. Stage locally, install, and serve:
```bash
# 1. stage a bundle from a built ttnn wheel + a metal-community tree
tt-model package --from-metal <community-clone> --ttnn-wheel <ttnn.whl> \
  --arch blackhole --arch-name LlamaForCausalLM \
  --main-class models.tt_transformers.tt.generator_vllm:LlamaForCausalLM \
  --weights unsloth/Llama-3.2-3B-Instruct --mesh P150 --out /tmp/bundle

# 2. install the shipped platform into the bundle's own venv
bash /tmp/bundle/install.sh /tmp/bundle/venv

# 3. sanity: the bundle venv opens the device (find_spec avoids the import-before-preload trap)
TTNN=$(/tmp/bundle/venv/bin/python -c 'import importlib.util,os;print(os.path.dirname(importlib.util.find_spec("ttnn").origin))')
LD_PRELOAD=$TTNN/build/lib/_ttnncpp.so TT_METAL_HOME=$TTNN TT_METAL_VISIBLE_DEVICES=0 \
  /tmp/bundle/venv/bin/python -c "import ttnn; d=ttnn.open_mesh_device(ttnn.MeshShape(1,1)); print(d.arch()); ttnn.close_mesh_device(d)"
```
**Expected:** `Arch.BLACKHOLE`, clean close. A full generation (the tt_transformers demo run from
the bundle venv) produces coherent text at ~75 tok/s/user on a single p150.

### Notes / gotchas the tests encode
- The shipped `ttnn` wheel **must** bundle `_ttnncpp.so`; it's py/abi/arch-pinned (cp312/linux_x86_64),
  and `pull` refuses a wheel that doesn't match the host interpreter (`host_incompatible_wheels`).
- Locate ttnn via `importlib.util.find_spec`, never `import ttnn`, when computing `LD_PRELOAD` — the
  import is exactly what the preload fixes (glibc static-TLS). `run.sh` does this.
- Single-chip: fabric disabled + `TT_METAL_VISIBLE_DEVICES=0`.
- **`vllm_metadata.json` must be in a per-model subfolder** under `EXTRA_MODELS_DIR` (`vllm_models/<name>/`),
  not the bundle root — the plugin scans children, so a root-level file registers 0 architectures.
- **`HF_MODEL` must be exported** for serving: the tt_transformers adapter reads it from the env, not
  from vLLM's `--model` (both are set by `run.sh`).
- pip may warn that vLLM wants `pydantic>=2.12` while `tt_transformers` pins `2.9.2`; validated
  end-to-end on a p150 that vLLM serves correctly under 2.9.2 (the warning is a metadata mismatch,
  not a runtime break). Reconcile the pin if a future vLLM feature needs 2.12.

## Validated end-to-end
The full loop — `package` → push to HF → `pull` (install-the-platform into the bundle venv) →
`serve` (vLLM OpenAI endpoint) → `curl` a chat completion — was run on a Blackhole p150 and
returned coherent output. The fixes above (metadata subfolder, `HF_MODEL` export, serving-arg
defaults, `_ttnncpp.so` in the wheel, `find_spec` preload) all come from that run.
