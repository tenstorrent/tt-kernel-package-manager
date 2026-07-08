#!/usr/bin/env bash
# One-command install of the full serve-a-model stack on top of an existing tt-metal env.
#
# Layers set up here:
#   - the Tenstorrent vLLM fork + TT plugin (serving layer)
#   - tt-kernel (distribution layer)
# then a `tt-kernel doctor` to confirm the stack is adequate.
#
# This is orchestration only — tt-kernel's own modules never install anything. It assumes
# tt-metal (ttnn) is already built and importable in the target venv (that is the heavy,
# hardware-coupled part and is out of scope for this script).
#
# PROTECTED FACT: the Tenstorrent vLLM plugin work lives on the `dev` branch. This script
# clones and installs `dev` — never `main`.
#
# Usage:
#   scripts/install.sh [--venv PATH] [--vllm-dir PATH] [--vllm-ref dev]
#
#   --venv      Python venv to install into (default: $VIRTUAL_ENV, else the tt-metal
#               python_env if found under ~/dispatch/tt-metal, else the active python).
#   --vllm-dir  Where to clone the vLLM fork (default: ~/dispatch/vllm).
#   --vllm-ref  Branch/ref of the fork to use (default: dev — do not change to main).
set -euo pipefail

VENV="${VIRTUAL_ENV:-}"
VLLM_DIR="${HOME}/dispatch/vllm"
VLLM_REF="dev"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --vllm-dir) VLLM_DIR="$2"; shift 2 ;;
    --vllm-ref) VLLM_REF="$2"; shift 2 ;;
    -h|--help) sed -n '1,25p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ "$VLLM_REF" = "main" ]; then
  echo "ERROR: the TT vLLM plugin work lives on 'dev', not 'main'. Refusing --vllm-ref main." >&2
  exit 2
fi

# Resolve the target python.
if [ -z "$VENV" ] && [ -x "${HOME}/dispatch/tt-metal/python_env/bin/python3" ]; then
  VENV="${HOME}/dispatch/tt-metal/python_env"
fi
if [ -n "$VENV" ]; then
  PY="$VENV/bin/python3"
else
  PY="$(command -v python3)"
fi
echo ">> Using python: $PY"

if ! "$PY" -c "import ttnn" >/dev/null 2>&1; then
  echo "!! ttnn is not importable from this python. Build/activate the tt-metal env first;" >&2
  echo "   this script installs the serving+distribution layers on top of it." >&2
fi

# 1. Clone the vLLM fork (dev) if not already present, then editable-install fork + plugin.
if [ ! -d "$VLLM_DIR/.git" ]; then
  echo ">> Cloning tenstorrent/vllm@$VLLM_REF -> $VLLM_DIR"
  git clone -b "$VLLM_REF" https://github.com/tenstorrent/vllm "$VLLM_DIR"
else
  echo ">> vLLM fork already at $VLLM_DIR ($(git -C "$VLLM_DIR" branch --show-current))"
fi

echo ">> Installing vLLM fork + TT plugin (editable)"
"$PY" -m pip install -e "$VLLM_DIR"
"$PY" -m pip install -e "$VLLM_DIR/plugins/vllm-tt-plugin"

# 2. Install tt-kernel.
echo ">> Installing tt-kernel (editable)"
"$PY" -m pip install -e "$REPO_ROOT"

# 3. Report.
echo ">> Running tt-kernel doctor"
"$PY" -m tt_kernel.cli doctor || true

cat <<EOF

Done. Serve a model with:
  tt-kernel serve <namespace>/<model>
EOF
