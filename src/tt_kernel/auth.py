# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Thin wrapper over huggingface_hub auth. We store no token of our own."""

from __future__ import annotations

from typing import Optional


def login(token: Optional[str] = None) -> None:
    """Run the HF login flow (interactive if no token), reusing HF's token store."""
    from huggingface_hub import login as hf_login

    hf_login(token=token)


def logout() -> None:
    from huggingface_hub import logout as hf_logout

    hf_logout()


def whoami() -> Optional[dict]:
    """Return the current HF identity, or None if not logged in."""
    from huggingface_hub import whoami as hf_whoami

    try:
        return hf_whoami()
    except Exception:
        return None


def current_namespace() -> Optional[str]:
    """The logged-in user/org name, usable as a default repo namespace."""
    me = whoami()
    if not me:
        return None
    return me.get("name")
