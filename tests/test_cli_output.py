# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Render checks for the CLI's output layer.

Adapted from `.claude/skills/cli-design/reference/test_cli_output.py`. That file's render
cases are written against its own `demo.py`, so the target commands are parameterised here
instead of hard-coded — see TARGETS.

Three things these guard, none of which a normal unit test can see:

1. Piped output carries ZERO escape codes (a CLI that colours a pipe corrupts logs).
2. Machine-readable output (`--print`, `--json`) survives a narrow terminal unwrapped.
   Rich wraps at COLUMNS and would silently break a pasteable command or a JSON document.
3. Under a real PTY, no third-party progress bar writes to the terminal and nothing is
   left un-erased at exit. Interleaved bars once left bytes in the input buffer that bash
   then executed as commands, so this is correctness, not cosmetics.
"""

import os
import pty
import re
import subprocess
import sys

import pytest

from tt_kernel import console

# The seam: point these at whatever commands exercise the output layer.
CLI = [sys.executable, "-m", "tt_kernel.cli"]
TARGETS = {
    "plain": ["doctor"],                      # structured, no network
    "help": ["--help"],
    "version": ["--version"],
}
# Bars we must never see again: HF tqdm and hf_xet write these.
FOREIGN_BAR_MARKERS = ("Download complete", "Reconstruction complete", "Fetching ",
                       "it/s]", "B/s]")

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def plain(raw):
    """Visible text only. Assert on THIS, not raw bytes: styling puts escape codes between
    a glyph and its label, so `"✓ x" in raw` is never true."""
    return ANSI.sub("", raw).replace("\r", "\n")


def run(args, columns=None, env=None):
    e = dict(os.environ, **(env or {}))
    if columns:
        e["COLUMNS"] = str(columns)
    return subprocess.run(CLI + args, capture_output=True, text=True, env=e)


def run_in_pty(args, columns=100):
    """Run under a real PTY and return everything written to it."""
    chunks = []
    prev = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = str(columns)
    try:
        pty.spawn(CLI + args, lambda fd: (lambda d: (chunks.append(d), d)[1])(os.read(fd, 1024)))
    finally:
        if prev is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = prev
    return b"".join(chunks).decode("utf8", "replace")


# ── 1. pure helpers ──────────────────────────────────────────────────────────
class TestHelpers:
    def test_progress_bar_fills_left_to_right(self):
        assert console.progress_bar(0, 4, width=4) == "▕░░░░▏"
        assert console.progress_bar(2, 4, width=4) == "▕██░░▏"
        assert console.progress_bar(4, 4, width=4) == "▕████▏"

    def test_unknown_total_renders_no_bar(self):
        """Say nothing rather than fake a percentage. pip and xet both give byte counts
        with no reliable total, so this path is load-bearing, not theoretical."""
        assert console.progress_bar(3, 0) == ""

    def test_bytes_use_decimal_units(self):
        assert console.fmt_bytes(12_110_000) == "12.1 MB"
        assert console.fmt_bytes(0) == "0 B"

    def test_folding_is_off_outside_a_phase(self):
        """Our 12 single-shot commands run outside any phase, so detail must show there."""
        console.set_verbose(False)
        assert console.show_detail() is True
        console.set_verbose(True)
        assert console.show_detail() is True
        console.set_verbose(False)


# ── 2. non-TTY cleanliness ───────────────────────────────────────────────────
class TestPiped:
    @pytest.mark.parametrize("name", sorted(TARGETS))
    def test_no_escape_codes_when_piped(self, name):
        res = run(TARGETS[name])
        assert "\x1b" not in res.stdout, f"{name}: styled a pipe"
        assert "\x1b" not in res.stderr, f"{name}: styled stderr"

    def test_version_is_a_bare_string(self):
        """--version is scraped by scripts; it must be the version and nothing else."""
        res = run(TARGETS["version"])
        assert res.returncode == 0, res.stderr
        assert re.fullmatch(r"\d+\.\d+\.\d+\S*", res.stdout.strip()), res.stdout

    def test_no_foreign_progress_bars_reach_a_pipe(self):
        res = run(TARGETS["plain"])
        for marker in FOREIGN_BAR_MARKERS:
            assert marker not in res.stdout, f"third-party bar leaked: {marker}"


# ── 3. narrow terminals must not corrupt machine-readable output ─────────────
class TestNarrowTerminal:
    @pytest.mark.parametrize("columns", [40, 80, 200])
    def test_help_renders_at_any_width(self, columns):
        res = run(TARGETS["help"], columns=columns)
        assert res.returncode == 0, res.stderr

    @pytest.mark.parametrize("columns", [40, 200])
    def test_doctor_renders_at_any_width(self, columns):
        res = run(TARGETS["plain"], columns=columns)
        assert res.returncode in (0, 1), res.stderr   # 1 = inadequate toolchain, still valid
        assert "\x1b" not in res.stdout

    def test_raw_never_wraps(self):
        """console.raw() is the carve-out that keeps `--print` pasteable and `--json`
        parseable. A wrapped command line is not runnable, so this must hold at any width."""
        line = "A=1 B=2 " + " ".join(f"--flag-{i} value-{i}" for i in range(40))
        prog = (
            "import sys; sys.path.insert(0, 'src')\n"
            "from tt_kernel import console\n"
            f"console.raw({line!r})\n"
        )
        res = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True,
                             env=dict(os.environ, COLUMNS="40"))
        assert res.returncode == 0, res.stderr
        assert res.stdout.rstrip("\n") == line
        assert res.stdout.count("\n") == 1, "raw() output was wrapped"


# ── 4. real terminal behaviour ───────────────────────────────────────────────
class TestPty:
    def test_styled_when_attached_to_a_terminal(self):
        raw = run_in_pty(TARGETS["plain"])
        assert "\x1b[" in raw, "no styling on a real TTY"

    def test_no_foreign_bars_and_nothing_left_unerased(self):
        """The F2/F9 guard. Interleaved HF/xet bars used to survive the process and paint
        over the next shell prompt, at which point bash executed the residue as commands."""
        raw = run_in_pty(TARGETS["plain"])
        visible = plain(raw)
        for marker in FOREIGN_BAR_MARKERS:
            assert marker not in visible, f"third-party bar reached the terminal: {marker}"
        assert "█" not in visible, "a progress bar we do not own drew to the terminal"
        assert raw.rstrip().endswith(("\x1b[0m", "adequate", ")", ".", "]")) or raw.endswith("\n"), \
            "output did not end on a clean boundary"

    def test_no_color_strips_styling_even_on_a_tty(self):
        raw = run_in_pty(["--no-color"] + TARGETS["plain"])
        assert "\x1b[3" not in raw and "\x1b[9" not in raw, "--no-color still emitted colour"


# ── 5. --help is documentation ────────────────────────────────────────────────
EXPECTED_PANELS = ["Get started", "Run a model", "Get models", "Publish models",
                   "Environment", "Maintenance"]


class TestHelpGrouping:
    def _help(self):
        return run(["--help"], columns=100).stdout

    def test_every_panel_is_present(self):
        out = self._help()
        for panel in EXPECTED_PANELS:
            assert panel in out, f"missing help panel: {panel}"

    def test_get_started_comes_first(self):
        """Panel order follows source order of the first command in each. A new user should
        meet `start`/`install` before `push`."""
        out = self._help()
        positions = {p: out.index(p) for p in EXPECTED_PANELS if p in out}
        assert positions["Get started"] == min(positions.values())

    def test_every_command_lives_in_a_panel(self):
        """An ungrouped command falls into a generic "Commands" box, which is how a flat
        30-item list grows back."""
        out = self._help()
        assert "─ Commands ─" not in out, "some command is not assigned to a panel"

    def test_developer_fixtures_are_hidden(self):
        """`dev` fabricates test data; it should not crowd help for people running models."""
        out = self._help()
        assert "make-test-cache" not in out
        assert "\ndev " not in out

    def test_the_guided_entry_point_is_advertised(self):
        assert "start" in self._help()

    @pytest.mark.parametrize("cmd", ["start", "install", "serve", "pull", "doctor"])
    def test_each_command_help_renders_and_is_escape_free_when_piped(self, cmd):
        res = run([cmd, "--help"], columns=100)
        assert res.returncode == 0, res.stderr
        assert "\x1b" not in res.stdout
