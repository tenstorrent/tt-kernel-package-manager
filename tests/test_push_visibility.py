# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests that ``push`` never changes a repo's visibility as a side effect.

Before the tri-state fix, ``--private/--public`` was a plain bool defaulting to ``False``
and ``hub.set_visibility`` ran on *every* push, so ``tt-kernel push you/private-model``
with no visibility flag published a previously-private repo. These tests pin the rule:

    a push changes visibility only when the user asks, or when it creates the repo.

Everything Hub-side is faked (``_FakeHub``), so no test can reach the network — let alone
flip a real repo.
"""

import json

from typer.testing import CliRunner

from tt_kernel import bundles, cache, cli, hub, metal
from tt_kernel.device import DeviceInfo

runner = CliRunner()


class _FakeHub:
    """A Hub that models exactly one thing: each repo's visibility.

    ``repos`` maps ``repo_id -> private?``; a repo absent from it does not exist. Every
    visibility write is recorded so a test can assert that *nothing* was written, which is
    the real invariant — "still private" could otherwise be satisfied by a flip and a flip
    back.
    """

    def __init__(self, repos=None):
        self.repos = dict(repos or {})
        self.created = []            # [(repo_id, private)]
        self.visibility_calls = []   # [(repo_id, private)] — must stay empty unless asked
        self.pushed = []             # [repo_id]

    # --- the subset of hub.py that push touches -------------------------------------
    def repo_exists(self, repo_id):
        return repo_id in self.repos

    def create_repo(self, repo_id, private):
        self.created.append((repo_id, private))
        self.repos[repo_id] = bool(private)
        return f"https://huggingface.co/{repo_id}"

    def is_private(self, repo_id):
        if repo_id not in self.repos:
            raise RuntimeError(f"404 {repo_id}")
        return self.repos[repo_id]

    def set_visibility(self, repo_id, private):
        self.visibility_calls.append((repo_id, private))
        self.repos[repo_id] = bool(private)

    def push_folder(self, repo_id, folder, commit_message=""):
        self.pushed.append(repo_id)

    def tag_repo(self, repo_id, tags):
        pass

    def install(self, monkeypatch):
        for name in ("repo_exists", "create_repo", "is_private", "set_visibility",
                     "push_folder", "tag_repo"):
            monkeypatch.setattr(hub, name, getattr(self, name))
        # hub.is_private_safe() calls the module-level is_private, so it picks up the fake.
        return self


def _fake_device(monkeypatch):
    """No hardware in CI: a deterministic blackhole card and a known tt-metal version."""
    monkeypatch.setattr(metal, "detect_device",
                        lambda arch_override=None: DeviceInfo(arch="blackhole", device_count=1,
                                                              source="test"))
    monkeypatch.setattr(metal, "resolve_version", lambda: "0.72.0")


def _mk_kernel_cache(tmp_path, key=111):
    """A minimal one-build_key kernel cache; returns the value to pass to --cache-dir."""
    out = cache.resolve_out_root(str(tmp_path))
    kdir = cache.build_key_path(out, key) / "kernels"
    kdir.mkdir(parents=True)
    (kdir / "kernel.bin").write_bytes(b"\x00")
    return str(tmp_path)


def _push_dispatch(tmp_path, repo_id, *flags):
    return runner.invoke(cli.app, ["push", repo_id, "--cache-dir", _mk_kernel_cache(tmp_path),
                                   "--arch", "blackhole", *flags])


def _mk_vllm_bundle(tmp_path):
    src = tmp_path / "vllm_src"
    src.mkdir()
    (src / bundles.VLLM_METADATA_NAME).write_text(
        json.dumps({"arch": "LlamaForCausalLM", "main_class": "m:C"})
    )
    (src / "generator_vllm.py").write_text("# adapter code\n")
    return src


# ------------------------------------------------- existing repo, no visibility flag
def test_push_to_existing_private_repo_without_flag_keeps_it_private(monkeypatch, tmp_path):
    """The regression this fix exists for: a bare push must not publish a private repo."""
    fake = _FakeHub({"acme/secret": True}).install(monkeypatch)
    _fake_device(monkeypatch)

    res = _push_dispatch(tmp_path, "acme/secret")

    assert res.exit_code == 0, res.output
    assert fake.repos["acme/secret"] is True          # still private
    assert fake.visibility_calls == []                # and never even asked to change
    assert fake.pushed == ["acme/secret"]             # the content push still happened
    assert "leaving its visibility unchanged" in res.output


def test_vllm_push_to_existing_private_repo_keeps_it_private(monkeypatch, tmp_path):
    """Same invariant on the vLLM path, where the bug was actually hit."""
    fake = _FakeHub({"acme/secret": True}).install(monkeypatch)
    _fake_device(monkeypatch)

    res = runner.invoke(cli.app, ["push", "acme/secret", "--backend", "vllm",
                                  "--bundle-dir", str(_mk_vllm_bundle(tmp_path)),
                                  "--arch", "blackhole"])

    assert res.exit_code == 0, res.output
    assert fake.repos["acme/secret"] is True
    assert fake.visibility_calls == []


def test_push_to_existing_public_repo_without_flag_leaves_it_public(monkeypatch, tmp_path):
    """The rule is symmetric: an unasked-for push must not make a public repo private."""
    fake = _FakeHub({"acme/open": False}).install(monkeypatch)
    _fake_device(monkeypatch)

    res = _push_dispatch(tmp_path, "acme/open")

    assert res.exit_code == 0, res.output
    assert fake.repos["acme/open"] is False
    assert fake.visibility_calls == []


# ----------------------------------------------- existing repo, explicit visibility
def test_explicit_public_on_existing_private_repo_is_honoured_and_reported(monkeypatch, tmp_path):
    fake = _FakeHub({"acme/secret": True}).install(monkeypatch)
    _fake_device(monkeypatch)

    res = _push_dispatch(tmp_path, "acme/secret", "--public")

    assert res.exit_code == 0, res.output
    assert fake.visibility_calls == [("acme/secret", False)]
    assert fake.repos["acme/secret"] is False
    # A flip is never invisible: it has to show up in the output.
    assert "Changed visibility of acme/secret to public" in res.output


def test_explicit_private_on_existing_private_repo_is_a_no_op(monkeypatch, tmp_path):
    """Asking for the visibility a repo already has writes nothing."""
    fake = _FakeHub({"acme/secret": True}).install(monkeypatch)
    _fake_device(monkeypatch)

    res = _push_dispatch(tmp_path, "acme/secret", "--private")

    assert res.exit_code == 0, res.output
    assert fake.visibility_calls == []
    assert "already private" in res.output


# --------------------------------------------------------------- creation-time flag
def test_new_repo_is_created_private_when_asked(monkeypatch, tmp_path):
    fake = _FakeHub().install(monkeypatch)
    _fake_device(monkeypatch)

    res = _push_dispatch(tmp_path, "acme/new", "--private")

    assert res.exit_code == 0, res.output
    assert fake.created == [("acme/new", True)]
    assert fake.visibility_calls == []  # creation carries the visibility; no second write


def test_new_repo_without_flag_is_created_public(monkeypatch, tmp_path):
    """The documented default for a *new* repo is unchanged by this fix."""
    fake = _FakeHub().install(monkeypatch)
    _fake_device(monkeypatch)

    res = _push_dispatch(tmp_path, "acme/new")

    assert res.exit_code == 0, res.output
    assert fake.created == [("acme/new", False)]
    assert fake.visibility_calls == []


# ------------------------------------------------------------------------- --publish
def test_publish_on_existing_private_repo_errors_instead_of_flipping(monkeypatch, tmp_path):
    """The catalog is public, but that is a reason to ask — not to flip silently."""
    fake = _FakeHub({"acme/secret": True}).install(monkeypatch)
    _fake_device(monkeypatch)

    res = _push_dispatch(tmp_path, "acme/secret", "--publish")

    assert res.exit_code == 1
    assert "--public" in res.output
    assert fake.repos["acme/secret"] is True
    assert fake.visibility_calls == []
    assert fake.pushed == []  # refused before uploading anything


def test_publish_on_existing_public_repo_needs_no_flag(monkeypatch, tmp_path):
    fake = _FakeHub({"acme/open": False}).install(monkeypatch)
    _fake_device(monkeypatch)

    res = _push_dispatch(tmp_path, "acme/open", "--publish")

    assert res.exit_code == 0, res.output
    assert fake.visibility_calls == []
    assert fake.pushed == ["acme/open"]
