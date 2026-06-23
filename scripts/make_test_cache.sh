#!/usr/bin/env bash
# Generate a synthetic tt-metal kernel cache for testing tt-kernel without hardware.
#
# Lays out the real on-disk structure (jit_compile_server.cpp:109-120):
#   <root>/<build_key>/kernels/<kernel_name>/<target>/<files>
#   <root>/<build_key>/firmware/<target>/<files>
#
# Usage: scripts/make_test_cache.sh [ROOT] [BUILD_KEY] [--with-runner]
#   ROOT          cache root dir to create (default: /tmp/ttk-test-cache)
#   BUILD_KEY     numeric build_key dir name (default: 4242)
#   --with-runner also drop a fake runner wheel next to ROOT, for a v2 round-trip
set -euo pipefail

ROOT="${1:-/tmp/ttk-test-cache}"
BUILD_KEY="${2:-4242}"
WITH_RUNNER=0
for arg in "$@"; do [ "$arg" = "--with-runner" ] && WITH_RUNNER=1; done
BASE="$ROOT/$BUILD_KEY"

rm -rf "$BASE"
for kernel in reader writer compute; do
  for target in trisc0 trisc1 brisc; do
    mkdir -p "$BASE/kernels/$kernel/$target"
    # A couple of fake artifacts per kernel/target, with varied content + size.
    head -c $((RANDOM % 4096 + 512)) /dev/urandom > "$BASE/kernels/$kernel/$target/$kernel.elf"
    printf 'fake hex for %s/%s\n' "$kernel" "$target" > "$BASE/kernels/$kernel/$target/$kernel.hex"
  done
done

for target in trisc0 trisc1 brisc ncrisc erisc; do
  mkdir -p "$BASE/firmware/$target"
  head -c $((RANDOM % 2048 + 256)) /dev/urandom > "$BASE/firmware/$target/fw.elf"
done

echo "Created synthetic cache at: $BASE"
echo "  build_key : $BUILD_KEY"
echo "  kernels   : $(find "$BASE/kernels" -maxdepth 1 -mindepth 1 -type d | wc -l | tr -d ' ')"
echo "  files     : $(find "$BASE" -type f | wc -l | tr -d ' ')"
echo

if [ "$WITH_RUNNER" = "1" ]; then
  # A "wheel" is just a zip with a trivial METADATA — enough for the push/pull round-trip
  # (pull pip-installs it; for the hardware-free test, pip is the only real external step).
  WHEEL="$ROOT/fake_runner-0.1-py3-none-any.whl"
  TMPW="$(mktemp -d)"
  printf 'Metadata-Version: 2.1\nName: fake-runner\nVersion: 0.1\n' \
    > "$TMPW/METADATA"
  # A wheel is just a zip; build it with python's zipfile to avoid a `zip` dependency.
  python3 -m zipfile -c "$WHEEL" "$TMPW/METADATA"
  rm -rf "$TMPW"
  echo "Created fake runner wheel: $WHEEL"
  echo
  echo "Push a v2 bundle (kernels + runner + weights) with:"
  echo "  tt-kernel push <ns>/<name> --private --cache-dir \"$ROOT\" --arch blackhole \\"
  echo "    --tt-metal-version v0.99-test \\"
  echo "    --python-package \"$WHEEL\" --runner-spec pkg.mod:Runner --entry-point demo \\"
  echo "    --weights org/model"
else
  echo "Push it with:"
  echo "  tt-kernel push <ns>/<name> --private --cache-dir \"$ROOT\" --arch blackhole \\"
  echo "    --tt-metal-version v0.99-test --model google/gemma-test"
fi
