#!/usr/bin/env bash
#
# One-shot setup: create a virtualenv, install dependencies, download the two
# checkpoints and verify their checksums.
#
#   ./setup.sh
#
# Idempotent -- safe to re-run. Existing venv and already-downloaded checkpoints
# are reused, and checksums are re-verified either way.

set -euo pipefail

VENV_DIR=".venv"
CHECKPOINT_DIR="checkpoints"

# file name -> "url  sha256"
UVDOC_URL="https://github.com/tanguymagne/UVDoc/raw/main/model/best_model.pkl"
UVDOC_SHA="7e90861b8a516eb4bc51f84bd889cb77275743d2d1d3ca8091951ec9f2b7da23"
PADDLE_PARAMS_URL="https://huggingface.co/PaddlePaddle/UVDoc/resolve/main/inference.pdiparams"
PADDLE_PARAMS_SHA="810488899520e0da843b9bd9769ba4949f1c81e357f0eceb12d4a7da459c3eca"
PADDLE_GRAPH_URL="https://huggingface.co/PaddlePaddle/UVDoc/resolve/main/inference.json"
PADDLE_GRAPH_SHA="2c2bc3e0f15e782cf8f2ad411b5033d99ca504fe88648f8054a5e925ba2336e0"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        sha256sum "$1" | cut -d' ' -f1
    fi
}

# Download $1 to $2 unless already present with the right checksum, then verify.
fetch_and_verify() {
    local url="$1" destination="$2" expected="$3"
    if [ -f "$destination" ] && [ "$(sha256_of "$destination")" = "$expected" ]; then
        echo "  ok (cached)  $(basename "$destination")"
        return
    fi
    echo "  downloading  $(basename "$destination")"
    curl -fL --progress-bar -o "$destination" "$url"
    local actual
    actual="$(sha256_of "$destination")"
    if [ "$actual" != "$expected" ]; then
        echo "  CHECKSUM MISMATCH for $(basename "$destination")" >&2
        echo "    expected $expected" >&2
        echo "    actual   $actual" >&2
        exit 1
    fi
    echo "  ok           $(basename "$destination")"
}

# --------------------------------------------------------------- virtualenv
say "1/3  Python environment"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  created $VENV_DIR"
else
    echo "  reusing $VENV_DIR"
fi
PYTHON="$VENV_DIR/bin/python"
echo "  $("$PYTHON" --version)"

# ------------------------------------------------------------- dependencies
say "2/3  Dependencies"
"$PYTHON" -m pip install --quiet --upgrade pip

# On Linux the default index serves CUDA-enabled torch wheels (~2.5 GB). This
# project only reads a checkpoint with torch, so the CPU build is enough.
if [ "$(uname -s)" = "Linux" ]; then
    echo "  installing CPU-only torch (Linux)"
    "$PYTHON" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi
"$PYTHON" -m pip install --quiet -r requirements.txt
"$PYTHON" - <<'PY'
import importlib
for module in ("numpy", "paddle", "torch", "PIL"):
    print(f"  {module:8s} {getattr(importlib.import_module(module), '__version__', 'ok')}")
PY

# -------------------------------------------------------------- checkpoints
say "3/3  Checkpoints (64 MB, not stored in git)"
mkdir -p "$CHECKPOINT_DIR"
fetch_and_verify "$UVDOC_URL"          "$CHECKPOINT_DIR/best_model.pkl"      "$UVDOC_SHA"
fetch_and_verify "$PADDLE_PARAMS_URL"  "$CHECKPOINT_DIR/inference.pdiparams" "$PADDLE_PARAMS_SHA"
fetch_and_verify "$PADDLE_GRAPH_URL"   "$CHECKPOINT_DIR/inference.json"      "$PADDLE_GRAPH_SHA"

say "Ready. Run the checks:"
cat <<'EOF'
  .venv/bin/python compare.py              weights + architecture
  .venv/bin/python verify_determinism.py   same image -> same grid, bit for bit

Both exit 0 when every check passes.
EOF
