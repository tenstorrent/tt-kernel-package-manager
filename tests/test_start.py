# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""`tt-model start` — the guided flow, and the prompt paths where a wizard rots.

The failure mode of a badly-placed prompt is a *hang*, not an exception, so several of
these assert on completing at all rather than on output. They are cheap; a hang in CI is
not.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from tt_kernel import cli, console, localdb, start, toolchain

runner = CliRunner()
CLI = [sys.executable, "-m", "tt_kernel.cli"]


# ── prompting policy ─────────────────────────────────────────────────────────
class TestPromptPolicy:
    def test_explicit_token_never_prompts(self, monkeypatch):
        monkeypatch.setattr(start.auth, "login", lambda token=None: None)
        monkeypatch.setattr(start.auth, "whoami", lambda: {"name": "me"})
        monkeypatch.setattr(start.console, "secret", _must_not_be_called)
        acct = start.resolve_account("hf_x")
        assert acct.logged_in and acct.source == "--token"

    def test_existing_identity_never_prompts(self, monkeypatch):
        monkeypatch.setattr(start.auth, "whoami", lambda: {"name": "me"})
        monkeypatch.setattr(start.console, "secret", _must_not_be_called)
        assert start.resolve_account().logged_in

    def test_prompt_suppressed_when_not_allowed(self, monkeypatch):
        """--yes and a non-TTY stdin must both reach this path. A prompt here would read
        EOF and silently take a default the user never saw."""
        monkeypatch.setattr(start.auth, "whoami", lambda: None)
        monkeypatch.setattr(start.console, "secret", _must_not_be_called)
        acct = start.resolve_account(allow_prompt=False)
        assert not acct.logged_in and acct.source == "none"

    def test_empty_prompt_is_not_treated_as_a_token(self, monkeypatch):
        monkeypatch.setattr(start.auth, "whoami", lambda: None)
        monkeypatch.setattr(start.console, "secret", lambda p: "   ")
        monkeypatch.setattr(start.auth, "login", _must_not_be_called)
        assert not start.resolve_account(allow_prompt=True).logged_in

    def test_token_is_never_echoed_into_output(self, monkeypatch):
        """A secret that reaches the renderer reaches scrollback and any log."""
        monkeypatch.setattr(start.auth, "login", lambda token=None: None)
        monkeypatch.setattr(start.auth, "whoami", lambda: {"name": "me"})
        acct = start.resolve_account("hf_SUPERSECRET")
        assert "hf_SUPERSECRET" not in repr(acct)
        assert "hf_SUPERSECRET" not in str(acct.__dict__)


def _must_not_be_called(*a, **k):
    raise AssertionError("prompted when it must not")


def test_stdin_detection_survives_a_closed_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", None)
    assert start.stdin_is_interactive() is False


# ── the guided flow does not hang ────────────────────────────────────────────
class TestNoHang:
    def _env(self):
        return dict(os.environ, COLUMNS="100")

    def test_non_tty_stdin_completes(self):
        """The canonical wizard bug: a prompt inside a capturing step, or on a piped stdin,
        hangs instead of failing. Bounded so a regression shows up as a failure."""
        res = subprocess.run(CLI + ["start", "nope/nope", "--print"],
                             stdin=subprocess.DEVNULL, capture_output=True, text=True,
                             timeout=120, env=self._env())
        assert res.returncode is not None

    def test_yes_flag_completes(self):
        res = subprocess.run(CLI + ["start", "nope/nope", "--print", "--yes"],
                             capture_output=True, text=True, timeout=120, env=self._env())
        assert res.returncode is not None


# ── bundle resolution ────────────────────────────────────────────────────────
class TestResolveBundle:
    def test_an_installed_id_resolves_to_itself(self, monkeypatch):
        monkeypatch.setattr(start.localdb, "get", lambda rid: {"repo_id": rid})
        assert start.resolve_bundle("org/m") == ("org/m", "installed")

    def test_a_bare_model_id_finds_an_installed_bundle(self, monkeypatch):
        """`tt-model start Qwen/Qwen3-32B` should find mando2222/Qwen3-32B-blackhole rather
        than trying to pull a bundle id the user never typed."""
        monkeypatch.setattr(start.localdb, "get", lambda rid: None)
        monkeypatch.setattr(start.localdb, "all_entries",
                            lambda: [{"repo_id": "mando2222/Qwen3-32B-blackhole"}])
        repo, how = start.resolve_bundle("Qwen/Qwen3-32B")
        assert repo == "mando2222/Qwen3-32B-blackhole"
        assert "matching" in how

    def test_an_unknown_id_is_left_to_pull(self, monkeypatch):
        monkeypatch.setattr(start.localdb, "get", lambda rid: None)
        monkeypatch.setattr(start.localdb, "all_entries", lambda: [])
        assert start.resolve_bundle("org/new") == ("org/new", "to pull")


# ── validation gate ──────────────────────────────────────────────────────────
def _report(ok=True):
    return toolchain.ToolchainReport(components=[
        toolchain.ComponentReport("tt-metal", ok, "0.77.0" if ok else None, "0.72.0", ok,
                                  "ok" if ok else "not found"),
    ])


class TestValidate:
    def _env(self, *, ok=True, port_free=True):
        return start.Environment(report=_report(ok), arch="blackhole", device_count=4,
                                 device_source="tt-smi", port=8000, port_free=port_free,
                                 conflicts=[])

    def test_a_busy_port_is_a_blocker(self):
        assert "port 8000 is already in use" in self._env(port_free=False).blockers

    def test_an_inadequate_component_is_a_blocker(self):
        assert any("tt-metal" in b for b in self._env(ok=False).blockers)

    def test_a_healthy_environment_has_no_blockers(self):
        assert self._env().blockers == []

    def test_conflicts_are_reported_but_do_not_block(self):
        """An environment conflict may involve a package the TT path never imports, so it
        is surfaced and not enforced — the same call the doctor change makes."""
        env = self._env()
        env.conflicts = [toolchain.EnvConflict("opencv", "numpy>=2", "numpy 1.26.4")]
        assert env.blockers == []

    def test_blocked_validation_pulls_nothing(self, monkeypatch):
        """The gate exists so a doomed run stops before touching the Hub."""
        called = []
        monkeypatch.setattr(cli, "_ensure_vllm_pulled",
                            lambda *a, **k: called.append(a) or {})
        monkeypatch.setattr(start, "validate", lambda *a, **k: self._env(port_free=False))
        res = runner.invoke(cli.app, ["start", "org/m", "--print", "--yes"])
        assert res.exit_code == 1
        assert called == [], "pulled despite a failed validation"
        assert "Nothing was pulled or started" in res.output


# ── roadmap integrity ────────────────────────────────────────────────────────
def test_every_phase_has_a_description():
    """The upfront panel and the stepper read from the same list; a phase with no detail
    would render a blank row."""
    assert set(start.PHASES) == set(start.PHASE_DETAIL)
    assert all(start.PHASE_DETAIL[p] for p in start.PHASES)


def test_phase_count_is_fixed_and_not_flag_dependent():
    """k/N is only trustworthy if N cannot drift with flags."""
    assert len(start.PHASES) == 4


# ── did-you-mean for a slipped word (F5) ─────────────────────────────────────
class TestDidYouMean:
    def test_two_command_names_in_a_row_suggests_the_later_one(self):
        """`pull serve <id>` reads as "serve <id>" with a stray word; the trailing tokens
        follow the intent, not the typo."""
        assert cli._did_you_mean(["pull", "serve", "x/y"]) == "tt-model serve x/y"

    def test_the_direction_is_not_hardcoded(self):
        assert cli._did_you_mean(["serve", "pull", "x/y"]) == "tt-model pull x/y"

    @pytest.mark.parametrize("argv", [
        ["pull", "x/y"],                 # correct usage
        ["pull", "x/y", "z/w"],          # two repo ids: not a slipped command name
        ["install"],                     # too short
        ["pull", "pull", "x/y"],         # same word twice is not a slip we can resolve
        ["instances", "list"],           # a real two-word command
    ])
    def test_no_suggestion_when_there_is_nothing_to_infer(self, argv):
        """A wrong hint is worse than none: it sends the user somewhere they did not ask
        to go, on a command that may have failed for an unrelated reason."""
        assert cli._did_you_mean(argv) is None

    def test_end_to_end_hint_appears_on_the_usage_error(self):
        res = runner.invoke(cli.app, ["pull", "serve", "x/y"])
        assert res.exit_code != 0
        assert "Did you mean" in res.output
        assert "tt-model serve x/y" in res.output

    def test_valid_commands_never_get_a_hint(self):
        res = runner.invoke(cli.app, ["--help"])
        assert "Did you mean" not in res.output

    def test_usage_error_classes_cover_typers_vendored_fork(self):
        """Typer vendors its own click, so a subcommand parse failure raises
        typer._click.exceptions.UsageError — NOT a subclass of click.UsageError. Catching
        only the latter silently matched nothing."""
        names = {c.__module__ for c in cli._USAGE_ERRORS}
        assert any("typer" in n for n in names), names
        assert any(n.startswith("click") for n in names), names


# ── `tt-model start` with no model named ─────────────────────────────────────
class TestMenu:
    @pytest.mark.parametrize("keys,expected", [
        (["2"], 1),
        (["1"], 0),
        ([""], 0),                      # bare Enter takes the default
        (["q"], None),                  # declining is not an error
        (["9", "abc", "2"], 1),         # re-prompts, does not crash or take a default
    ])
    def test_selection(self, monkeypatch, keys, expected):
        it = iter(keys)
        monkeypatch.setattr("builtins.input", lambda prompt="": next(it))
        assert console.choose("Serve", ["a", "b"]) == expected

    def test_eof_declines_rather_than_looping(self, monkeypatch):
        def boom(prompt=""):
            raise EOFError
        monkeypatch.setattr("builtins.input", boom)
        assert console.choose("Serve", ["a", "b"]) is None


class TestPickModel:
    def _entries(self, monkeypatch, entries):
        monkeypatch.setattr(start.localdb, "all_entries", lambda: entries)

    def test_no_model_and_nothing_installed_explains_instead_of_erroring(self, monkeypatch):
        """`tt-model start` used to answer "Missing argument 'model'." — the one response a
        guided command must not give."""
        self._entries(monkeypatch, [])
        res = runner.invoke(cli.app, ["start"])
        assert res.exit_code == 2
        assert "Nothing is installed yet" in res.output
        assert "tt-model search --catalog" in res.output
        assert "Missing argument" not in res.output

    def test_a_single_installed_bundle_is_used_without_asking(self, monkeypatch):
        self._entries(monkeypatch, [
            {"repo_id": "org/only", "bundle_path": "/tmp/x", "backend": "vllm"}])
        monkeypatch.setattr(cli.console, "choose", _must_not_be_called)
        repo, note = cli._pick_model(interactive=True)
        assert repo == "org/only"
        assert "only installed" in note

    def test_several_installed_non_interactive_lists_them(self, monkeypatch):
        """No prompt is possible, so name the exact commands rather than failing vaguely."""
        self._entries(monkeypatch, [
            {"repo_id": "org/a", "bundle_path": "/tmp/x"},
            {"repo_id": "org/b", "bundle_path": "/tmp/x"}])
        res = runner.invoke(cli.app, ["start", "--yes"])
        assert res.exit_code == 2
        assert "tt-model start org/a" in res.output
        assert "tt-model start org/b" in res.output

    def test_several_installed_interactive_prompts(self, monkeypatch):
        self._entries(monkeypatch, [
            {"repo_id": "org/a", "bundle_path": "/tmp/x"},
            {"repo_id": "org/b", "bundle_path": "/tmp/x"}])
        monkeypatch.setattr(cli.console, "choose", lambda *a, **k: 1)
        repo, note = cli._pick_model(interactive=True)
        assert repo == "org/b"

    def test_declining_the_menu_exits_without_starting_anything(self, monkeypatch):
        self._entries(monkeypatch, [
            {"repo_id": "org/a", "bundle_path": "/tmp/x"},
            {"repo_id": "org/b", "bundle_path": "/tmp/x"}])
        monkeypatch.setattr(cli.console, "choose", lambda *a, **k: None)
        with pytest.raises(typer.Exit):
            cli._pick_model(interactive=True)

    def test_entries_without_a_bundle_path_are_not_offered(self, monkeypatch):
        """A recorded-but-not-materialised entry cannot be served, so offering it would
        send the user into a failure."""
        self._entries(monkeypatch, [
            {"repo_id": "org/ghost"},                          # no bundle_path
            {"repo_id": "org/real", "bundle_path": "/tmp/x"}])
        ids = [c.repo_id for c in start.installed_choices()]
        assert ids == ["org/real"]

    def test_labels_carry_enough_to_choose_between(self, monkeypatch):
        self._entries(monkeypatch, [
            {"repo_id": "org/a", "bundle_path": "/x", "backend": "vllm", "arch": "blackhole"}])
        assert start.installed_choices()[0].label == "org/a  (vllm · blackhole)"


def test_model_argument_is_optional():
    """The signature is the contract: a required argument makes the guided path impossible."""
    res = runner.invoke(cli.app, ["start", "--help"])
    assert "[{model}]" in res.output or "Omit to pick" in res.output


# ── never auto-pick a bundle that cannot serve ───────────────────────────────
class TestServabilityGate:
    def _entries(self, monkeypatch, entries):
        monkeypatch.setattr(start.localdb, "all_entries", lambda: entries)

    def _servability(self, monkeypatch, mapping):
        monkeypatch.setattr(start, "_servability",
                            lambda path: mapping.get(path, (True, None)))

    def test_the_only_bundle_being_unservable_is_not_auto_picked(self, monkeypatch):
        """Auto-selecting a bundle already known to be unrunnable walked the user through
        three phases to fail at the fourth on something knowable before the first. Only a
        caller that cannot be asked (--yes, or a piped stdin) gets the refusal."""
        self._entries(monkeypatch, [
            {"repo_id": "org/broken", "bundle_path": "/b"}])
        self._servability(monkeypatch, {"/b": (False, "models is not importable")})
        res = runner.invoke(cli.app, ["start", "--yes"])
        assert res.exit_code == 2
        assert "Nothing installed here can serve" in res.output
        assert "models is not importable" in res.output

    def test_the_reason_is_named_per_bundle(self, monkeypatch):
        self._entries(monkeypatch, [
            {"repo_id": "org/a", "bundle_path": "/a"},
            {"repo_id": "org/b", "bundle_path": "/b"}])
        self._servability(monkeypatch, {"/a": (False, "models is not importable"),
                                       "/b": (False, "models is not importable")})
        res = runner.invoke(cli.app, ["start", "--yes"])
        assert res.output.count("models is not importable") >= 2

    def test_unservable_bundles_are_still_offered_interactively(self, monkeypatch):
        """Declining to CHOOSE for the user is not the same as declining to LET them
        choose. They may be about to fix PYTHONPATH, or may just want to see the failure —
        so the menu still offers it, marked, rather than refusing outright."""
        self._entries(monkeypatch, [{"repo_id": "org/broken", "bundle_path": "/b"}])
        self._servability(monkeypatch, {"/b": (False, "models is not importable")})
        seen = {}
        monkeypatch.setattr(cli.console, "choose",
                            lambda prompt, labels, **k: (seen.update(labels=labels), 0)[1])
        repo, note = cli._pick_model(interactive=True)
        assert repo == "org/broken"
        assert "despite" in note, note
        assert any("✗" in l for l in seen["labels"]), "the menu did not mark it unrunnable"

    def test_a_single_servable_bundle_is_still_auto_picked(self, monkeypatch):
        self._entries(monkeypatch, [{"repo_id": "org/ok", "bundle_path": "/ok"}])
        self._servability(monkeypatch, {})
        repo, note = cli._pick_model(interactive=False)
        assert repo == "org/ok"

    def test_the_one_servable_bundle_wins_over_broken_siblings(self, monkeypatch):
        """With exactly one runnable candidate there is nothing to choose between, even
        though other bundles are installed."""
        self._entries(monkeypatch, [
            {"repo_id": "org/broken", "bundle_path": "/b"},
            {"repo_id": "org/works", "bundle_path": "/w"}])
        self._servability(monkeypatch, {"/b": (False, "models is not importable")})
        repo, note = cli._pick_model(interactive=False)
        assert repo == "org/works"
        assert "can serve here" in note

    def test_servable_bundles_sort_first_so_the_default_is_never_broken(self, monkeypatch):
        self._entries(monkeypatch, [
            {"repo_id": "org/aaa-broken", "bundle_path": "/b"},
            {"repo_id": "org/zzz-works", "bundle_path": "/w"}])
        self._servability(monkeypatch, {"/b": (False, "models is not importable")})
        choices = start.installed_choices()
        assert choices[0].repo_id == "org/zzz-works", "a broken bundle sorted to the default"

    def test_an_explicit_id_is_still_honoured(self, monkeypatch):
        """The gate is about *picking for* the user. Naming a bundle explicitly must still
        work — it stops at the serve preflight, which says the same thing with more detail."""
        self._entries(monkeypatch, [{"repo_id": "org/broken", "bundle_path": "/b"}])
        self._servability(monkeypatch, {"/b": (False, "models is not importable")})
        monkeypatch.setattr(start, "validate", lambda *a, **k: start.Environment(
            report=_report(True), arch="blackhole", device_count=4, device_source="tt-smi",
            port=8000, port_free=True, conflicts=[]))
        res = runner.invoke(cli.app, ["start", "org/broken", "--yes", "--print"])
        assert "Nothing installed here can serve" not in res.output

    def test_unreadable_metadata_does_not_mark_a_bundle_unservable(self, tmp_path):
        """Fail open: a metadata problem is a different failure, and guessing "unservable"
        would hide a bundle that works."""
        servable, reason = start._servability(str(tmp_path / "missing"))
        assert servable is True and reason is None

    def test_servability_check_can_be_skipped(self, monkeypatch):
        """It spawns a subprocess per bundle; callers that only need labels can opt out."""
        self._entries(monkeypatch, [{"repo_id": "org/a", "bundle_path": "/a"}])
        monkeypatch.setattr(start, "_servability", _must_not_be_called)
        assert start.installed_choices(check_servable=False)[0].servable is True
