#!/usr/bin/env bash
# Generate a synthetic tt-metal kernel cache for testing tt-model without hardware.
#
# Lays out the real on-disk structure (jit_compile_server.cpp:109-120). The build dir
# is named "tt-metal-cache<build_key>": with --cache-dir/TT_METAL_CACHE set, tt-model
# glues the build_key onto the "tt-metal-cache" prefix, so the build dirs are siblings
# directly under ROOT (see cache.py resolve_out_root / _parent_and_prefix).
#   <root>/tt-metal-cache<build_key>/kernels/<kernel_name>/<target>/<files>
#   <root>/tt-metal-cache<build_key>/firmware/<target>/<files>
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
BASE="$ROOT/tt-metal-cache$BUILD_KEY"

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
  # A valid wheel needs a .dist-info/ dir (METADATA + WHEEL + RECORD), not just a bare
  # METADATA file, or `pip install` rejects it ("invalid wheel, .dist-info directory not
  # found"). Build a minimal but real wheel exposing fake_runner:Runner so the v2
  # round-trip actually pip-installs (pip is the only real external step here).
  WHEEL="$ROOT/fake_runner-0.1-py3-none-any.whl"
  python3 - "$WHEEL" <<'PYEOF'
import base64, hashlib, sys, zipfile

whl = sys.argv[1]
files = {
    "fake_runner/__init__.py": b'class Runner:\n    """Trivial fake runner for round-trip testing."""\n    pass\n',
    "fake_runner-0.1.dist-info/METADATA": b"Metadata-Version: 2.1\nName: fake-runner\nVersion: 0.1\nSummary: round-trip test runner\n",
    "fake_runner-0.1.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: make_test_cache\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
}
record = []
for name, data in files.items():
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    record.append(f"{name},sha256={digest},{len(data)}")
record.append("fake_runner-0.1.dist-info/RECORD,,")
with zipfile.ZipFile(whl, "w", zipfile.ZIP_DEFLATED) as z:
    for name, data in files.items():
        z.writestr(name, data)
    z.writestr("fake_runner-0.1.dist-info/RECORD", ("\n".join(record) + "\n").encode())
PYEOF
  echo "Created fake runner wheel: $WHEEL"
  echo
  echo "Push a v2 bundle (kernels + runner + weights) with:"
  echo "  tt-model push <ns>/<name> --private --cache-dir \"$ROOT\" --arch blackhole \\"
  echo "    --tt-metal-version v0.99-test \\"
  echo "    --python-package \"$WHEEL\" --runner-spec fake_runner:Runner --entry-point demo \\"
  echo "    --weights org/model"
else
  echo "Push it with:"
  echo "  tt-model push <ns>/<name> --private --cache-dir \"$ROOT\" --arch blackhole \\"
  echo "    --tt-metal-version v0.99-test --model google/gemma-test"
fi
