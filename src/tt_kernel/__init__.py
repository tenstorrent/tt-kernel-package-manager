# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2024 Tenstorrent USA, Inc.

"""tt-kernel: publish and pull precompiled tt-metal kernel caches over Hugging Face Hub."""

__version__ = "0.1.0"

# HF model-repo tag that marks a repo as a tt-kernel cache bundle (used by search).
TT_KERNEL_TAG = "tt-kernel-cache"

# Filename of the compatibility manifest at the root of every bundle.
MANIFEST_NAME = "tt_kernel_manifest.json"
