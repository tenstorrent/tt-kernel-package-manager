# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""tt-kernel: publish and pull precompiled tt-metal kernel caches over Hugging Face Hub."""

__version__ = "0.1.0"

# HF model-repo tag that marks a repo as a tt-kernel cache bundle (used by search).
TT_KERNEL_TAG = "tt-kernel-cache"

# HF model-repo tag that OPTS a (public) bundle into the community catalog. This is a
# deliberate, separate act from `push`: pushing a bundle only tags it TT_KERNEL_TAG;
# `--publish` (or `tt-kernel publish`) additionally adds this tag. The web catalog lists
# ONLY repos carrying this tag. tt-kernel stores nothing — the catalog is a pure index of
# pointers to public HF repos, which remain under their owners' governance.
TT_KERNEL_CATALOG_TAG = "tt-kernel-catalog"

# Filename of the compatibility manifest at the root of every bundle.
MANIFEST_NAME = "tt_kernel_manifest.json"

# Detect-only bundle resolution — the front-door query other consumers (the `run`
# orchestrator, dispatch's courtesy notice) build on. Exposed as `resolve_bundle` so it
# never shadows the `tt_kernel.resolve` submodule.
from .resolve import BundleResolution, resolve as resolve_bundle  # noqa: E402

__all__ = [
    "TT_KERNEL_TAG",
    "TT_KERNEL_CATALOG_TAG",
    "MANIFEST_NAME",
    "BundleResolution",
    "resolve_bundle",
]
