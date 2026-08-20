# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""A minimal OpenAI-compatible server for a **legacy** (dispatch-contract) runner.

The dispatch serving runtime this used to hand off to is retired. This is a small,
self-contained replacement so a runner written against the legacy contract in
``docs/authoring_runners.md`` — a duck-typed class exposing ``generate()`` /
``generate_stream()`` / ``benchmark()`` plus ``_tokenizer`` / ``_listed`` /
``_community`` — can still be served over ``/v1/chat/completions``.

It is **transitional**, not the destination: it serves one request at a time with none
of the batching, paging, or performance of the vLLM path. Prefer authoring a vLLM bundle
(see the top of ``docs/authoring_runners.md``). This exists so an author who already built
a legacy runner is not stranded.

Run it directly, or let ``tt-model run <installed-dispatch-bundle>`` build the command:

    python -m tt_kernel.legacy_serve --runner pkg.mod:Runner --model /path/to/weights

Requires ``fastapi`` + ``uvicorn`` (``pip install 'tt-model[serve]'``) and, at runtime,
``ttnn`` and the runner's package in the environment.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from typing import Any, Iterable, List, Optional


def _load_runner_class(spec: str):
    """Import ``"module:Class"`` (or ``"module.Class"``) and return the class object."""
    if ":" in spec:
        module_name, cls_name = spec.split(":", 1)
    else:
        module_name, cls_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def _open_device(device_ids: List[int]):
    """Open a device for a runner that does not manage its own. Returns (device, close).

    Runners that need a specific topology set ``MANAGES_OWN_DEVICE = True`` and receive
    ``device=None``; for the rest we open a single device and register a clean close.
    """
    import atexit

    import ttnn

    device = ttnn.open_device(device_id=device_ids[0])
    close = lambda: ttnn.close_device(device)  # noqa: E731
    atexit.register(close)
    return device, close


def build_runner(spec: str, model_path: str, *, device_ids: List[int], **kwargs) -> Any:
    """Instantiate the runner per the legacy contract, opening a device if it needs one."""
    cls = _load_runner_class(spec)
    if getattr(cls, "MANAGES_OWN_DEVICE", False):
        device = None  # the runner opens and owns its device (and closes it via atexit)
    else:
        device, _ = _open_device(device_ids)
    return cls(model_path, device, device_ids=device_ids, **kwargs)


def _messages_to_prompt(messages: List[dict], tokenizer: Any) -> str:
    """Render OpenAI ``messages`` into a single prompt string.

    Prefer the runner tokenizer's chat template (correct multi-turn formatting); fall back
    to a plain role-tagged concatenation when no template is available.
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is not None:
        try:
            return apply(messages, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001 — fall back to plain formatting
            pass
    parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
    parts.append("assistant:")
    return "\n".join(parts)


def _split_stream(runner: Any, prompt: str, *, max_new_tokens: int, temperature: float):
    """Drive ``generate_stream``; yield (text_deltas..., usage_dict).

    The contract guarantees the final yielded item is the usage dict; everything before it
    is a decoded text delta. We pass ``chat=False`` because the prompt is already templated.
    """
    usage = {"finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0}
    stream: Iterable = runner.generate_stream(
        prompt, max_new_tokens=max_new_tokens, temperature=temperature, chat=False
    )
    for item in stream:
        if isinstance(item, dict):
            usage.update(item)
            break
        yield ("delta", item)
    yield ("usage", usage)


def build_app(runner: Any, *, model_name: str, default_max_tokens: int):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="tt-model legacy runner server")
    community = bool(getattr(runner, "_community", False))

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [
            {"id": model_name, "object": "model", "owned_by":
             "community" if community else "tenstorrent"}
        ]}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict):
        messages = body.get("messages", [])
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_tokens") or default_max_tokens)
        temperature = float(body.get("temperature", 1.0))
        prompt = _messages_to_prompt(messages, getattr(runner, "_tokenizer", None))
        created = int(time.time())
        cid = f"chatcmpl-{created}"

        if stream:
            def sse():
                for kind, payload in _split_stream(
                    runner, prompt, max_new_tokens=max_tokens, temperature=temperature
                ):
                    if kind == "delta":
                        chunk = {"id": cid, "object": "chat.completion.chunk",
                                 "created": created, "model": model_name,
                                 "choices": [{"index": 0, "delta": {"content": payload},
                                              "finish_reason": None}]}
                    else:  # usage -> final chunk
                        chunk = {"id": cid, "object": "chat.completion.chunk",
                                 "created": created, "model": model_name,
                                 "choices": [{"index": 0, "delta": {},
                                              "finish_reason": payload.get("finish_reason", "stop")}],
                                 "usage": {
                                     "prompt_tokens": payload.get("prompt_tokens", 0),
                                     "completion_tokens": payload.get("completion_tokens", 0),
                                     "total_tokens": payload.get("prompt_tokens", 0)
                                     + payload.get("completion_tokens", 0)}}
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        text, usage = [], {}
        for kind, payload in _split_stream(
            runner, prompt, max_new_tokens=max_tokens, temperature=temperature
        ):
            (text.append(payload) if kind == "delta" else usage.update(payload))
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(text)},
                         "finish_reason": usage.get("finish_reason", "stop")}],
            "usage": {"prompt_tokens": usage.get("prompt_tokens", 0),
                      "completion_tokens": usage.get("completion_tokens", 0),
                      "total_tokens": usage.get("prompt_tokens", 0)
                      + usage.get("completion_tokens", 0)},
        })

    return app


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tt_kernel.legacy_serve",
        description="Serve a legacy (dispatch-contract) runner over an OpenAI-compatible API.",
    )
    p.add_argument("--runner", required=True, help='Runner spec "module:Class".')
    p.add_argument("--model", required=True, help="Local weights dir (model_path) or HF id.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Default cap when a request omits max_tokens.")
    p.add_argument("--device-ids", default="0", help="Comma-separated device ids (default 0).")
    p.add_argument("--served-model-name", default=None, help="Name reported at /v1/models.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    try:
        import uvicorn  # noqa: F401
        import fastapi  # noqa: F401
    except ImportError as exc:  # pragma: no cover — env-dependent
        raise SystemExit(
            f"legacy_serve needs fastapi + uvicorn ({exc}). Install with: "
            "pip install 'tt-model[serve]'"
        )
    device_ids = [int(x) for x in str(args.device_ids).split(",") if x.strip() != ""]
    name = args.served_model_name or args.model
    print(f"[tt-model legacy_serve] loading runner {args.runner} on model {args.model} ...")
    runner = build_runner(args.runner, args.model, device_ids=device_ids)
    app = build_app(runner, model_name=name, default_max_tokens=args.max_new_tokens)
    print(f"[tt-model legacy_serve] OpenAI endpoint: http://{args.host}:{args.port}/v1")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
