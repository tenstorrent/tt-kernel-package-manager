"""Unit tests for the runtime payload helpers (runner wheel + weights install).

All hardware/network/pip side effects are monkeypatched — nothing is actually
installed or downloaded.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from tt_kernel import runtime
from tt_kernel.manifest import WeightsRef


# --------------------------------------------------------------------------- models dir

def test_resolve_models_dir_flag_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(runtime.ENV_MODELS_DIR, str(tmp_path / "env"))
    out = runtime.resolve_models_dir(str(tmp_path / "flag"), "org/model")
    assert out == tmp_path / "flag" / "org" / "model"


def test_resolve_models_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv(runtime.ENV_MODELS_DIR, str(tmp_path / "env"))
    out = runtime.resolve_models_dir(None, "org/model")
    assert out == tmp_path / "env" / "org" / "model"


def test_resolve_models_dir_default(monkeypatch):
    monkeypatch.delenv(runtime.ENV_MODELS_DIR, raising=False)
    monkeypatch.setenv("HOME", "/home/someone")
    out = runtime.resolve_models_dir(None, "org/model")
    assert out == Path("/home/someone/.cache/tt-kernel/models/org/model")


def test_resolve_models_dir_no_org():
    out = runtime.resolve_models_dir("/tmp/x", "just-a-name")
    assert out == Path("/tmp/x/just-a-name")


# --------------------------------------------------------------------------- pip install

def test_pip_install_wheels_argv(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)) or _ok())
    wheel = tmp_path / "r-0.1-py3-none-any.whl"
    wheel.write_bytes(b"x")
    runtime.pip_install_wheels([wheel])
    cmd, kw = calls[0]
    assert cmd[:5] == [sys.executable, "-m", "pip", "install", "--no-deps"]
    assert str(wheel) in cmd
    assert kw.get("check") is True


def test_pip_install_wheels_respects_python_and_pip_args(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _ok())
    wheel = tmp_path / "r.whl"
    wheel.write_bytes(b"x")
    runtime.pip_install_wheels([wheel], python="/opt/venv/bin/python", pip_args=["--force-reinstall"])
    cmd = calls[0]
    assert cmd[0] == "/opt/venv/bin/python"
    assert "--force-reinstall" in cmd


def test_pip_install_wheels_propagates_failure(monkeypatch, tmp_path):
    def boom(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", boom)
    wheel = tmp_path / "r.whl"
    wheel.write_bytes(b"x")
    with pytest.raises(subprocess.CalledProcessError):
        runtime.pip_install_wheels([wheel])


def test_pip_install_wheels_empty_is_noop(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("should not run"))
    runtime.pip_install_wheels([])


# --------------------------------------------------------------------------- weights

def test_download_weights_forwards_args(monkeypatch, tmp_path):
    seen = {}

    def fake_snapshot(**kwargs):
        seen.update(kwargs)
        return str(tmp_path / "dl")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    w = WeightsRef(repo_id="org/m", revision="abc", allow_patterns=["*.safetensors"])
    out = runtime.download_weights(w, tmp_path / "dest")
    assert seen["repo_id"] == "org/m"
    assert seen["revision"] == "abc"
    assert seen["allow_patterns"] == ["*.safetensors"]
    assert seen["repo_type"] == "model"
    assert out == tmp_path / "dl"


# --------------------------------------------------------------------------- serve cmd / detection

def test_serve_command_exact():
    cmd = runtime.serve_command("pkg.mod:Runner", Path("/models/org/m"))
    assert cmd == (
        "python -m tt_inference_server.dispatch.serve serve --unsafe "
        "--runner pkg.mod:Runner /models/org/m"
    )


def test_dispatch_available_uses_find_spec(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert runtime.dispatch_available() is True
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert runtime.dispatch_available() is False


def test_ttnn_importable_current_interpreter(monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert runtime.ttnn_importable(None) is True
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert runtime.ttnn_importable(None) is False


class _OK:
    returncode = 0


def _ok():
    return _OK()
