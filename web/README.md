<!-- SPDX-License-Identifier: Apache-2.0 -->
# tt-kernel catalog (web)

A searchable, browsable front end for community-published tt-kernel bundles — like
`ollama.com/models`, but backed entirely by the Hugging Face Hub.

## What it is (and what it is not)

This is a **pure index**. It is a static site (HTML + CSS + one vanilla-JS file) that, from
the *visitor's own browser*, queries the Hugging Face Hub public API for model repos carrying
the opt-in tag `tt-kernel-catalog` and renders them as searchable cards.

- **It hosts nothing.** No kernels, no weights, no manifests, no user data live on this
  server. The browser fetches everything live from `huggingface.co`.
- **It stores nothing.** There is no database and no build-time crawl — the listing is always
  exactly what is currently tagged on the Hub.
- **It maps, it does not own.** Every card is a pointer to a public HF repo. Those repos are
  owned and governed solely by whoever pushed them. Correctness, licensing, safety, and
  liability rest with the bundle authors and consumers — not with whoever hosts this page.

Because of this, hosting the catalog is just serving three static files.

## How a bundle gets listed

Listing is an explicit opt-in by the author, separate from a plain `push`:

```bash
tt-kernel push you/mymodel-blackhole --public --publish   # push + list in one go
tt-kernel publish you/mymodel-blackhole                    # list a repo pushed earlier
tt-kernel unpublish you/mymodel-blackhole                  # delist (repo untouched)
```

`--publish` requires `--public` (a private repo cannot be indexed). It adds the
`tt-kernel-catalog` tag to the repo's model card; the catalog shows only repos with that tag.
Delisting removes the tag — the bundle drops off on the next page load. The repo and its
content are never modified by (un)publishing beyond that one tag.

## Deploy

Copy this folder to any static web server. No build step, no backend, no runtime.

```bash
# Local preview
cd web && python3 -m http.server 8080     # then open http://localhost:8080

# Behind nginx
#   root /srv/tt-kernel-catalog;   # (the contents of this web/ folder)
#   index index.html;
```

Any static host works (nginx, Caddy, `python -m http.server`, an S3 bucket, an HF Space with
a static SDK, GitHub Pages, …). The only requirement is that visitors' browsers can reach the
Hugging Face API — which supports CORS for these public GET endpoints, so no proxy is needed.

## Configure

Edit [`config.js`](config.js). Common tweaks:

| Key | Meaning |
|-----|---------|
| `CATALOG_TAG` | The opt-in tag. Must match `tt_kernel.TT_KERNEL_CATALOG_TAG`. |
| `HF_ORIGIN` | Point at an HF mirror if you run one. |
| `LIST_LIMIT` | Max repos pulled from the list endpoint. |
| `ENRICH_CONCURRENCY` | Parallel manifest fetches for rich card details. |
| `BRAND` / `TAGLINE` | Header text. |

## How it works

1. **List** — `GET {HF}/api/models?filter=tt-kernel-catalog&full=true` returns the tagged
   repos with downloads, last-modified, and tags. Arch **and model capabilities** are read
   straight from the tags, so those badges and filter chips render immediately with no
   per-repo fetch.
2. **Enrich** — in the background, with limited concurrency, each repo's
   `resolve/main/tt_kernel_manifest.json` is fetched to fill in kernel count, tt-metal
   version, and whether it carries a runner (the one feature filter).
3. **Detail** — clicking a card opens a drawer led by the `tt-kernel pull` command
   (the primary action), the full manifest, `info`/`run` commands, and a secondary link to
   the source HF repo.

**Capability tags.** A producer marks model capabilities at push time:

```bash
tt-kernel push you/mixtral-blackhole --public --publish \
  --capability moe --capability sliding-window-attention
```

The catalog maps recognized capability tags to display labels via `CAPABILITIES` in
[`config.js`](config.js) (e.g. `moe` → "MoE", `sliding-window-attention` → "Sliding
window"). Add rows there as new capability kernels land. Weights are **never** shipped in a
bundle — only referenced — so there is no "weights" badge; the target model appears in the
detail drawer.

If the HF API is unreachable (offline, or a mirror without CORS), the page says so instead
of failing silently.
