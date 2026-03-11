#!/usr/bin/env bash
# Verify ATOM health by running simple_inference. When using a local AITER clone,
# set AITER_DIR so this script can add it to PYTHONPATH (avoids ImportError from
# aiter: destroy_dist_env, torch_compile_guard, etc.).

set -euo pipefail

# Ninja is required to load C++ extensions (e.g. PyTorch/torch)
if ! command -v ninja &>/dev/null; then
  echo "Ninja not found; installing via pip..."
  pip install ninja
fi

# Default: AITER at sibling ../aiter when run from ATOM repo root
ATOM_DIR="${ATOM_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
AITER_DIR="${AITER_DIR:-$(dirname "$ATOM_DIR")/aiter}"

MODEL="${1:-/data/hf_hub_cache/models--amd--Llama-3.1-8B-Instruct-FP8-KV/snapshots/fa42f9a9105c545755fea25cf69f49ac8c8b40e1}"

if [[ ! -d "$AITER_DIR" ]]; then
  echo "AITER dir not found: $AITER_DIR (set AITER_DIR if different)" >&2
  exit 1
fi

# export PYTHONPATH="${AITER_DIR}${PYTHONPATH:+:$PYTHONPATH}"
cd "$ATOM_DIR"
exec python -m atom.examples.simple_inference --model "$MODEL"
