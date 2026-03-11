#!/usr/bin/env bash
# Verify ATOM health by running simple_inference.
# Uses local AITER and Triton when present (e.g. /dockerx/aiter, /dockerx/triton).
# Set AITER_DIR / TRITON_DIR if your paths differ.
#
# Runs with --level 0 --enforce-eager to avoid torch.compile/Triton issues.
# If you see "module_quant.so: undefined symbol: getCurrentHIPStream", AITER was
# built against a different PyTorch than the one in your env: rebuild AITER, or
# pass a non-FP8 model path (e.g. BF16) as first arg or set ATOM_HEALTH_CHECK_MODEL.

set -euo pipefail

# Ninja is required to load C++ extensions (e.g. PyTorch/torch)
if ! command -v ninja &>/dev/null; then
  echo "Ninja not found; installing via pip..."
  pip install ninja
fi

# Paths: ATOM repo, then local AITER, Triton (docker layout: /dockerx/*).
# Do not add local PyTorch to PYTHONPATH: AITER .so extensions are built against
# the installed PyTorch; mixing with a different torch causes undefined HIP symbols.
ATOM_DIR="${ATOM_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
AITER_DIR="${AITER_DIR:-/dockerx/aiter}"
TRITON_DIR="${TRITON_DIR:-/dockerx/triton}"

# Fallback: sibling aiter if /dockerx/aiter missing
if [[ ! -d "$AITER_DIR" ]]; then
  AITER_DIR="$(dirname "$ATOM_DIR")/aiter"
fi

# First arg overrides default; ATOM_HEALTH_CHECK_MODEL env overrides both (e.g. use a BF16 model if FP8 quant .so fails).
MODEL="${1:-${ATOM_HEALTH_CHECK_MODEL:-/data/hf_hub_cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}}"

if [[ ! -d "$AITER_DIR" ]]; then
  echo "AITER dir not found: $AITER_DIR (set AITER_DIR if different)" >&2
  exit 1
fi

# PYTHONPATH: only ATOM, AITER, Triton (do not append existing PYTHONPATH so that
# torch is loaded from the active env; otherwise aiter's module_quant.so can see
# a different torch and hit undefined symbol getCurrentHIPStream).
_pp="$ATOM_DIR:$AITER_DIR"
if [[ -d "$TRITON_DIR/python" ]]; then
  _pp="${_pp}:${TRITON_DIR}/python"
fi
export PYTHONPATH="${_pp}"

# Preflight: detect AITER .so vs PyTorch ABI mismatch (undefined symbol getCurrentHIPStream)
# so we exit with a clear message instead of a long traceback.
# if ! python -c "
# import sys
# try:
#     import aiter.jit.module_rmsnorm_quant
# except ImportError as e:
#     msg = str(e)
#     if 'getCurrentHIPStream' in msg or 'undefined symbol' in msg:
#         print('Health check skipped: AITER JIT extensions were built against a different PyTorch.', file=sys.stderr)
#         print('Rebuild AITER with the current PyTorch (pip install -e /dockerx/aiter from AITER repo).', file=sys.stderr)
#         sys.exit(2)
#     raise
# "; then
#   exit 2
# fi

# Eager mode so health check does not depend on torch.compile + Triton (avoids
# IndexError in Triton codegen and f64->f8 legalization issues with local PyTorch/Triton).
cd "$ATOM_DIR"
exec python -m atom.examples.simple_inference --model "$MODEL" --level 0 --enforce-eager
