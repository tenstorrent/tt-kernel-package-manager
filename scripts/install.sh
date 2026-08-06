#!/usr/bin/env bash
# One-command install of the full serve-a-model stack on top of an existing tt-metal env.
#
# Layers set up here:
#   - the Tenstorrent vLLM plugin (serving layer), on top of upstream vLLM
#   - tt-kernel (distribution layer)
# then a `tt-kernel doctor` to confirm the stack is adequate.
#
# This is orchestration only — tt-kernel's own modules never install anything. It assumes
# tt-metal (ttnn) is already built and importable in the target venv (that is the heavy,
# hardware-coupled part and is out of scope for this script).
#
# DEFAULT (recommended): upstream vLLM from PyPI, built for the `empty` device target, plus
# the standalone `tenstorrent/vllm-tt-plugin`. No fork is involved — the plugin registers the
# `tt` platform through vLLM's standard plugin entry points (`vllm.platform_plugins` /
# `vllm.general_plugins`), and everything TT-specific (platform, worker, scheduler, model
# registration, EXTRA_MODELS_DIR bundle discovery) lives in the plugin. Mirrors the plugin's
# own canonical docs/install-vllm-tt.sh.
#
# LEGACY: `--vllm-dir PATH` installs from an existing `tenstorrent/vllm` fork checkout (the
# fork plus its in-tree plugin at plugins/vllm-tt-plugin). The fork is being deprecated —
# development has moved to the standalone plugin repo — so use this only to reproduce an
# existing pinned-fork environment.
#
# Usage:
#   scripts/install.sh [--venv PATH] [--plugin-dir PATH] [--plugin-ref REF] [--vllm-version V]
#   scripts/install.sh --vllm-dir PATH [--vllm-ref dev]        # legacy fork path
#
#   --venv          Python venv to install into. Default: the active $VIRTUAL_ENV, else a
#                   tt-metal venv auto-detected by importable ttnn (searches $TT_METAL_HOME
#                   and the common ~/tt-metal, ~/dispatch/tt-metal, ~/projects/tt-metal
#                   layouts), else python3.
#   --plugin-dir    Where to clone/find tenstorrent/vllm-tt-plugin
#                   (default: ~/dispatch/vllm-tt-plugin).
#   --plugin-ref    Branch/ref of the plugin repo (default: main).
#   --vllm-version  Upstream vLLM version to build (default: 0.24.0 — the version the
#                   plugin pins and tests against).
#   --vllm-dir      LEGACY: use an existing tenstorrent/vllm fork checkout instead of
#                   upstream vLLM. Deprecated; prints a warning.
#   --vllm-ref      LEGACY: branch/ref of the fork (default: dev — never main).
set -euo pipefail

VENV="${VIRTUAL_ENV:-}"
PLUGIN_DIR="${HOME}/dispatch/vllm-tt-plugin"
PLUGIN_REF="main"
VLLM_VERSION="0.24.0"
VLLM_DIR=""          # non-empty selects the legacy fork path
VLLM_REF="dev"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --plugin-dir) PLUGIN_DIR="$2"; shift 2 ;;
    --plugin-ref) PLUGIN_REF="$2"; shift 2 ;;
    --vllm-version) VLLM_VERSION="$2"; shift 2 ;;
    --vllm-dir) VLLM_DIR="$2"; shift 2 ;;
    --vllm-ref) VLLM_REF="$2"; shift 2 ;;
    -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# The fork's plugin work lives on `dev`; `main` is stale. Only meaningful on the legacy path.
if [ -n "$VLLM_DIR" ] && [ "$VLLM_REF" = "main" ]; then
  echo "ERROR: the fork's TT plugin work lives on 'dev', not 'main'. Refusing --vllm-ref main." >&2
  exit 2
fi

# Resolve the target python: --venv > $VIRTUAL_ENV > the first tt-metal venv (in common
# locations) whose python can import ttnn > system python3. No fixed workspace layout is
# assumed — tt-metal lives in a different place on every machine.
_ttnn_ok() { [ -x "$1/bin/python3" ] && "$1/bin/python3" -c "import ttnn" >/dev/null 2>&1; }

if [ -z "$VENV" ]; then
  for cand in \
    ${TT_METAL_HOME:+"$TT_METAL_HOME/python_env"} \
    "$HOME/tt-metal/python_env" \
    "$HOME/dispatch/tt-metal/python_env" \
    "$HOME/projects/tt-metal/python_env"; do
    if _ttnn_ok "$cand"; then
      VENV="$cand"; echo ">> Auto-detected tt-metal venv (ttnn importable): $VENV"; break
    fi
  done
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

# tt-metal venvs are increasingly uv-managed and ship no `pip` module at all, so prefer
# `uv pip` (pointed explicitly at the target interpreter) and fall back to the venv's pip.
if command -v uv >/dev/null 2>&1; then
  _USE_UV=true; echo ">> Package manager: uv ($(uv --version 2>/dev/null))"
else
  _USE_UV=false; echo ">> Package manager: $PY -m pip"
fi
_pip_install()   { if $_USE_UV; then uv pip install --python "$PY" "$@"
                   else "$PY" -m pip install "$@"; fi }
_pip_uninstall() { if $_USE_UV; then uv pip uninstall --python "$PY" "$@" || true
                   else "$PY" -m pip uninstall -y "$@" || true; fi }

# Tenstorrent has NO CUDA. vLLM is built with the 'empty' device target (no CUDA/other
# device kernels compiled in) — all compute runs through the TT out-of-tree platform
# (device "tt"). A plain install would default to VLLM_TARGET_DEVICE=cuda, which is wrong
# here. VLLM_TARGET_DEVICE is build-time only; it is not needed at runtime.
export VLLM_TARGET_DEVICE=empty

if [ -n "$VLLM_DIR" ]; then
  # ---------------------------------------------------------------- legacy fork path
  echo "!! DEPRECATED: --vllm-dir installs the tenstorrent/vllm fork. The fork is being"  >&2
  echo "   retired; plugin development has moved to tenstorrent/vllm-tt-plugin, which"    >&2
  echo "   runs against upstream vLLM. Drop --vllm-dir to use the supported path."        >&2

  if [ ! -d "$VLLM_DIR/.git" ]; then
    echo ">> Cloning tenstorrent/vllm@$VLLM_REF -> $VLLM_DIR"
    git clone -b "$VLLM_REF" https://github.com/tenstorrent/vllm "$VLLM_DIR"
  else
    echo ">> vLLM fork already at $VLLM_DIR ($(git -C "$VLLM_DIR" branch --show-current))"
  fi

  echo ">> Installing vLLM fork + its in-tree TT plugin (editable)"
  _pip_install -e "$VLLM_DIR" --extra-index-url https://download.pytorch.org/whl/cpu
  _pip_install -e "$VLLM_DIR/plugins/vllm-tt-plugin"
else
  # ------------------------------------------------- default: upstream vLLM + plugin
  if [ ! -d "$PLUGIN_DIR/.git" ]; then
    echo ">> Cloning tenstorrent/vllm-tt-plugin@$PLUGIN_REF -> $PLUGIN_DIR"
    git clone -b "$PLUGIN_REF" https://github.com/tenstorrent/vllm-tt-plugin "$PLUGIN_DIR"
  else
    echo ">> Plugin already at $PLUGIN_DIR ($(git -C "$PLUGIN_DIR" branch --show-current))"
  fi

  # ttnn pins numpy<2 while vLLM's requirements pull an opencv that wants numpy>=2; the
  # plugin ships the override file that keeps numpy<2 (which tt-metal fixes and we cannot
  # move). Use it when present so the install can't rewrite the tt-metal env's numpy.
  OVERRIDES="$PLUGIN_DIR/docs/vllm-overrides.txt"
  echo ">> Installing upstream vLLM $VLLM_VERSION (VLLM_TARGET_DEVICE=empty)"
  if [ -f "$OVERRIDES" ]; then
    _pip_install --no-binary vllm --override "$OVERRIDES" "vllm==$VLLM_VERSION"
  else
    echo "   (no $OVERRIDES found — installing without dependency overrides)"
    _pip_install --no-binary vllm "vllm==$VLLM_VERSION"
  fi

  # torchaudio arrives as a CUDA wheel that cannot load next to CPU torch, and
  # transformers>=5.12 imports it if it is merely installed.
  echo ">> Removing torchaudio (CUDA wheel, unloadable beside CPU torch)"
  _pip_uninstall torchaudio

  echo ">> Installing the TT plugin (editable)"
  _pip_install -e "$PLUGIN_DIR"
fi

# Install tt-kernel.
echo ">> Installing tt-kernel (editable)"
_pip_install -e "$REPO_ROOT"

# Report.
echo ">> Running tt-kernel doctor"
"$PY" -m tt_kernel.cli doctor || true

cat <<EOF

Done. Serve a model with:
  tt-kernel serve <namespace>/<model>
EOF
