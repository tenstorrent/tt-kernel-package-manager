#!/usr/bin/env bash
# Generate a synthetic tt-metal kernel cache for testing tt-kernel without hardware.
#
# Lays out the real on-disk structure (jit_compile_server.cpp:109-120):
#   <root>/<build_key>/kernels/<kernel_name>/<target>/<files>
#   <root>/<build_key>/firmware/<target>/<files>
#
# Usage: scripts/make_test_cache.sh [ROOT] [BUILD_KEY]
#   ROOT       cache root dir to create (default: /tmp/ttk-test-cache)
#   BUILD_KEY  numeric build_key dir name (default: 4242)
set -euo pipefail

ROOT="${1:-/tmp/ttk-test-cache}"
BUILD_KEY="${2:-4242}"
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
echo "Push it with:"
echo "  tt-kernel push <ns>/<name> --private --cache-dir \"$ROOT\" --arch blackhole \\"
echo "    --tt-metal-version v0.99-test --model google/gemma-test"
