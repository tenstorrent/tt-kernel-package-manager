<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Field report: packaging `tt-tnt` with `tt-kernel`

A record of what happened when a small, from-scratch model was packaged and served through
`tt-kernel` twice — once in August 2026 at a 512-token context, and again days later after a
retrain moved it to 2048. Written from the model author's side, kept here because the second
pass is the interesting one: it exercised what happens when a *published model changes*, which
is the part a first-time packaging never reaches.

## What was packaged

[`tt-tnt`](https://github.com/tsingletaryTT/tt-tnt) is a ~22M-parameter Llama-3-architecture
model — RoPE, RMSNorm, SwiGLU, grouped-query attention, 6 blocks, embedding dim 384 — trained
from random initialization with tt-metal's `ttml` trainer on a single Blackhole p300c, then
converted to a Hugging Face directory and packaged as a v4 vLLM bundle. It is deliberately small;
the point was never capability, but to show the whole path end to end without gaps: train on
Tenstorrent, package with `tt-kernel`, serve through the Tenstorrent vLLM plugin. The manifest
(`entrypoint.class`, `mesh`, `resources.max_model_len`, `env`) and a single-file adapter were the
entire packaging surface, and both passes served successfully — `/v1/models` reporting the right
context length, the adapter registered out of `EXTRA_MODELS_DIR`, real completions coming back.

## What worked, and why it mattered

The property that earned its keep was **the bundle carrying its own tt-metal change**. This
model's compute grid is harvested — 11×10 = 110 cores on this part — and upstream
`ModelArgs.find_grid` hardcodes an assumption that is wrong there. Rather than blocking on an
upstream fix, the bundle's adapter monkeypatches `find_grid` to read
`compute_with_storage_grid_size()` from the live device, documents why in its docstring, and says
what it would take to delete. `tt-kernel` never had to know. That is a genuinely useful property
for anyone bringing up a model on hardware that does not match an upstream default: the model can
ship a fix it depends on, at model scope, and the packaging path stays untouched. The same
mechanism later absorbed a second patch — scoping the `tt_transformers` converted-weight cache by
source revision — without any change to the bundle format or to `tt-kernel` itself.

## Where the friction was

All of it was **staleness that presents as success**, and all of it was found by a human noticing
a timestamp rather than by anything failing. `tt-kernel push` set repository visibility
unconditionally on every push, so a push with no flag could publish a private repo — fixed in
[#12](https://github.com/tenstorrent/tt-kernel-package-manager/pull/12), where the root cause was
a boolean that could not distinguish "asked for public" from "said nothing". `tt-kernel serve`
reuses a cached bundle without checking whether the source revision moved, so the retrained model
was very nearly served with the previous revision's `--max_model_len 512` — filed as
[#13](https://github.com/tenstorrent/tt-kernel-package-manager/issues/13). Alongside those, four
smaller things: the launch command is cwd-dependent, instance dedupe collapses distinct `uv`
venvs by `realpath`, `__pycache__` gets uploaded into bundles, and compat gating reads the active
environment rather than the selected instance. None of these are hard to fix; the common thread
worth naming is that a package manager's *quiet* paths — cache reuse, default flags, skipped
pulls — are where a model author gets a confident wrong answer, and the cheapest remedy is a line
of output saying which cached thing was used and where it came from.
