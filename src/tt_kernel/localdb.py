# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""A tiny local index of installed bundles, for ``list`` and ``rm``.

Stored at ``~/.cache/tt-model/installed.json``. This is a convenience record only —
the source of truth for what's usable is always the tt-metal cache on disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def _index_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "tt-model" / "installed.json"


def _load() -> Dict[str, dict]:
    path = _index_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save(data: Dict[str, dict]) -> None:
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def record(repo_id: str, entry: dict) -> None:
    """Record (or replace) an installed bundle keyed by ``namespace/name``."""
    data = _load()
    data[repo_id] = entry
    _save(data)


def get(repo_id: str) -> Optional[dict]:
    return _load().get(repo_id)


def remove(repo_id: str) -> bool:
    data = _load()
    if repo_id in data:
        del data[repo_id]
        _save(data)
        return True
    return False


def all_entries() -> List[dict]:
    """Return installed entries, each augmented with its ``repo_id``."""
    return [{"repo_id": k, **v} for k, v in sorted(_load().items())]
