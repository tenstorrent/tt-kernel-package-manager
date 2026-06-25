# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Hugging Face Hub I/O. All storage, auth, visibility, LFS, and search live here.

Each bundle is one HF model repo (``repo_type="model"``) named ``namespace/name`` and
tagged ``tt-kernel-cache`` so ``search`` can filter for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from . import MANIFEST_NAME, TT_KERNEL_TAG
from .manifest import Manifest

_REPO_TYPE = "model"


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def create_repo(repo_id: str, private: bool) -> str:
    """Create (or reuse) the model repo. Returns the repo URL."""
    from huggingface_hub import create_repo as hf_create_repo

    url = hf_create_repo(
        repo_id=repo_id, repo_type=_REPO_TYPE, private=private, exist_ok=True
    )
    return str(url)


def set_visibility(repo_id: str, private: bool) -> None:
    """Update repo visibility (newer hub uses update_repo_settings)."""
    api = _api()
    if hasattr(api, "update_repo_settings"):
        api.update_repo_settings(repo_id=repo_id, repo_type=_REPO_TYPE, private=private)
    else:  # older huggingface_hub
        api.update_repo_visibility(repo_id=repo_id, repo_type=_REPO_TYPE, private=private)


def push_folder(repo_id: str, folder: Path, commit_message: str) -> None:
    """Upload an entire staged bundle folder. Large binaries go to LFS automatically."""
    _api().upload_folder(
        repo_id=repo_id,
        repo_type=_REPO_TYPE,
        folder_path=str(folder),
        commit_message=commit_message,
    )


def tag_repo(repo_id: str, tags: List[str]) -> None:
    """Best-effort: write a model card with metadata tags so search can filter."""
    from huggingface_hub import ModelCard, ModelCardData

    try:
        card = ModelCard.load(repo_id)
    except Exception:
        card = ModelCard("")
    existing = list(getattr(card.data, "tags", None) or [])
    card.data = ModelCardData(tags=sorted(set(existing) | set(tags)))
    card.push_to_hub(repo_id, repo_type=_REPO_TYPE)


def download_bundle(repo_id: str, revision: Optional[str], dest: Optional[str] = None) -> Path:
    """Snapshot-download a bundle and return the local snapshot path."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id, repo_type=_REPO_TYPE, revision=revision, local_dir=dest
    )
    return Path(path)


def fetch_manifest(repo_id: str, revision: Optional[str]) -> Manifest:
    """Download just the manifest file and parse it (used by ``info``)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id, repo_type=_REPO_TYPE, filename=MANIFEST_NAME, revision=revision
    )
    return Manifest.from_json(Path(path).read_text())


def search(query: str, limit: int = 50) -> List[dict]:
    """List tt-kernel cache repos matching a query, newest first."""
    api = _api()
    results = api.list_models(filter=TT_KERNEL_TAG, search=query or None, limit=limit)
    out: List[dict] = []
    for m in results:
        out.append(
            {
                "id": getattr(m, "id", None) or getattr(m, "modelId", None),
                "private": getattr(m, "private", None),
                "downloads": getattr(m, "downloads", None),
                "last_modified": str(getattr(m, "last_modified", "") or ""),
            }
        )
    return out
