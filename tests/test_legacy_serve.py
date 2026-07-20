# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for the legacy-runner OpenAI shim (tt_kernel.legacy_serve). No device, no server."""

import sys
import types

import pytest

from tt_kernel import legacy_serve


class _FakeRunner:
    """A minimal runner following the legacy contract."""

    MANAGES_OWN_DEVICE = True  # so build_runner doesn't try to open a device

    def __init__(self, model_path, device, **kwargs):
        self.model_path = model_path
        self.device = device
        self.kwargs = kwargs
        self._tokenizer = None
        self._listed = True
        self._community = False

    def generate_stream(self, prompt, max_new_tokens=50, temperature=1.0, chat=True):
        for tok in ["Hel", "lo", "!"]:
            yield tok
        yield {"finish_reason": "stop", "prompt_tokens": 3, "completion_tokens": 3}


def _install_fake_runner_module(monkeypatch, cls=_FakeRunner):
    mod = types.ModuleType("fake_runner_pkg")
    mod.Runner = cls
    monkeypatch.setitem(sys.modules, "fake_runner_pkg", mod)


def test_load_runner_class(monkeypatch):
    _install_fake_runner_module(monkeypatch)
    assert legacy_serve._load_runner_class("fake_runner_pkg:Runner") is _FakeRunner


def test_build_runner_manages_own_device_gets_none(monkeypatch):
    _install_fake_runner_module(monkeypatch)
    r = legacy_serve.build_runner("fake_runner_pkg:Runner", "/w/model", device_ids=[0])
    assert isinstance(r, _FakeRunner)
    assert r.device is None and r.model_path == "/w/model"
    assert r.kwargs.get("device_ids") == [0]


def test_messages_to_prompt_fallback_without_template():
    prompt = legacy_serve._messages_to_prompt(
        [{"role": "user", "content": "hi"}], tokenizer=None
    )
    assert "user: hi" in prompt and prompt.strip().endswith("assistant:")


def test_messages_to_prompt_uses_chat_template():
    class _Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "TEMPLATED:" + messages[-1]["content"]

    out = legacy_serve._messages_to_prompt([{"role": "user", "content": "yo"}], _Tok())
    assert out == "TEMPLATED:yo"


def test_split_stream_yields_deltas_then_usage():
    r = _FakeRunner("/w", None)
    events = list(legacy_serve._split_stream(r, "hi", max_new_tokens=5, temperature=1.0))
    kinds = [k for k, _ in events]
    assert kinds == ["delta", "delta", "delta", "usage"]
    assert "".join(p for k, p in events if k == "delta") == "Hello!"
    usage = events[-1][1]
    assert usage["completion_tokens"] == 3 and usage["finish_reason"] == "stop"


def test_parse_args_requires_runner_and_model():
    ns = legacy_serve._parse_args(["--runner", "pkg:R", "--model", "/w/m", "--port", "8123"])
    assert ns.runner == "pkg:R" and ns.model == "/w/m" and ns.port == 8123
    with pytest.raises(SystemExit):
        legacy_serve._parse_args(["--model", "/w/m"])  # missing --runner


def test_build_app_chat_completion_non_streaming(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = legacy_serve.build_app(_FakeRunner("/w", None), model_name="m", default_max_tokens=16)
    client = TestClient(app)
    resp = client.post("/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Hello!"
    assert body["usage"]["total_tokens"] == 6
