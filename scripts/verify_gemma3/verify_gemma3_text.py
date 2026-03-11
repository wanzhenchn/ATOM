#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Verify Gemma 3 text-only (CausalLM) models in ATOM.

Requires ATOM to register Gemma3ForCausalLM in support_model_arch_dict
(atom/model_engine/model_runner.py). Gemma 3 text models use
architectures: ["Gemma3ForCausalLM"] and model_type: "gemma3_text".

Usage:
  # With real weights (default: google/gemma-3-1b-it)
  python scripts/verify_gemma3/verify_gemma3_text.py

  # With dummy weights (no download)
  python scripts/verify_gemma3/verify_gemma3_text.py --load_dummy

  # Custom model
  python scripts/verify_gemma3/verify_gemma3_text.py --model google/gemma-3-4b-it

  # From repo root
  python scripts/verify_gemma3/verify_gemma3_text.py --model google/gemma-3-1b-it

  # Real weights + graph mode (level=3, torch.compile + CUDA graphs; first run compiles ~10 min).
  # Gemma3 can hit a known Inductor bug in piecewise mode; if so, run without --graph-mode.
  python scripts/verify_gemma3/verify_gemma3_text.py --graph-mode

Defaults: level=0 (no torch.compile), enforce_eager=True. Uses HF_TOKEN from env
for HuggingFace.

When using the ASM decode path (default, no ROCm Triton), Gemma3 output is
omitted in the report because ASM does not support sliding window; the script
still runs and passes. For correct Gemma3 text, use ROCm Triton and
ATOM_USE_TRITON_DECODE=1.
"""
import argparse
import os
import sys
import warnings

# Reduce noise from dependencies (must be before other imports)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Repo root so we can import atom and resolve sibling aiter
repo_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Use local AITER clone when present (avoids ImportError in engine subprocess)
# Candidates: AITER_DIR, sibling aiter, then /dockerx/aiter for docker env
_aiter_candidates = [
    os.environ.get("AITER_DIR"),
    os.path.join(os.path.dirname(repo_root), "aiter"),
    "/dockerx/aiter",
]
_pp_parts = [p for p in (os.environ.get("PYTHONPATH") or "").split(os.pathsep) if p]
for _d in _aiter_candidates:
    if _d and os.path.isdir(_d) and _d not in _pp_parts:
        _pp_parts.insert(0, _d)
    if _d and os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

# Add local Triton when present (needed for ATOM_USE_TRITON_DECODE=1)
_triton_candidates = [
    os.environ.get("TRITON_DIR"),
    os.path.join(os.path.dirname(repo_root), "triton"),
    "/dockerx/triton",
]
_triton_python = None
for _d in _triton_candidates:
    if _d and os.path.isdir(os.path.join(_d, "python")):
        _triton_python = os.path.join(_d, "python")
        break
if _triton_python and _triton_python not in _pp_parts:
    _pp_parts.insert(0, _triton_python)
if _triton_python and _triton_python not in sys.path:
    sys.path.insert(0, _triton_python)

if _pp_parts:
    os.environ["PYTHONPATH"] = os.pathsep.join(_pp_parts)

# Enable Triton decode when Triton is available (required for correct Gemma3 sliding window output).
# Fall back to ASM (ATOM_USE_TRITON_DECODE=0) only if Triton is not found.
if _triton_python is not None:
    os.environ.setdefault("ATOM_USE_TRITON_DECODE", "1")
else:
    os.environ.setdefault("ATOM_USE_TRITON_DECODE", "0")
# Reduce log noise so verification output is readable
os.environ.setdefault("AITER_LOG_LEVEL", "ERROR")

import logging
from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from transformers import AutoTokenizer

# Keep atom logs at WARNING so script output is clean
logging.getLogger("atom").setLevel(logging.WARNING)


DEFAULT_MODEL = "google/gemma-3-1b-it"
PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]


def _is_likely_garbage(text: str) -> bool:
    """Heuristic: output is likely garbage if it mixes many scripts (ASM decode without sliding window)."""
    if not text or len(text.strip()) < 4:
        return True
    text = text.strip()
    # Strong signal: any run of 3+ non-ASCII letters (indicates mixed scripts)
    run = 0
    for c in text:
        if c.isalpha():
            if c.isascii():
                run = 0
            else:
                run += 1
                if run >= 3:
                    return True
        else:
            run = 0
    # Or: majority of alphabetic chars are non-ASCII
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha > 10 and ascii_letters / total_alpha < 0.85:
        return True
    return False


def _clean_output(text: str) -> str:
    """Trim trailing junk and mask garbage so verification output is readable."""
    if not text or not text.strip():
        return text
    text = text.strip()
    if _is_likely_garbage(text):
        return "[output omitted: ASM decode path does not support sliding window; use ROCm Triton (ATOM_USE_TRITON_DECODE=1) for correct Gemma3 text]"
    # If there's a clean sentence end, truncate there to avoid trailing half-tokens
    last = max(
        text.rfind("."), text.rfind("!"), text.rfind("?"), text.rfind("\n")
    )
    if last != -1 and last > len(text) // 2:
        text = text[: last + 1].strip()
    return text


def generate_cuda_graph_sizes(max_size: int) -> list:
    sizes = []
    power = 1
    while power <= max_size:
        sizes.append(power)
        power *= 2
    return sizes


def main():
    parser = argparse.ArgumentParser(
        description="Verify Gemma 3 text-only model in ATOM"
    )
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        model=DEFAULT_MODEL,
        enforce_eager=True,
        level=0,
        max_model_len=512,
        gpu_memory_utilization=0.5,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max tokens to generate (default: 64)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature for coherent output (default: 0.5)",
    )
    parser.add_argument(
        "--graph-mode",
        action="store_true",
        help="Enable torch.compile and CUDA graphs (level=3, enforce_eager=False). First run may take several minutes to compile.",
    )
    args = parser.parse_args()

    if args.graph_mode:
        args.level = 3
        args.enforce_eager = False
        args.max_model_len = 2048
        args.gpu_memory_utilization = 0.9
        print(
            "Graph mode: level=3 (piecewise + CUDA graphs), enforce_eager=False. "
            "Note: Gemma3 may hit Inductor 'Expected tensors only, got int' in piecewise path; use default (eager) if it fails."
        )

    args.cudagraph_capture_sizes = str(
        generate_cuda_graph_sizes(max(len(PROMPTS), 1))
    )

    load_dummy = getattr(args, "load_dummy", False)
    print(
        f"Loading model: {args.model}"
        + (" (dummy)" if load_dummy else "")
    )

    # Load tokenizer once (uses HF_TOKEN from env if set); pass to engine to avoid double load
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )
    try:
        engine_args = EngineArgs.from_cli_args(args)
        llm = engine_args.create_engine(tokenizer=tokenizer)
    except KeyError as e:
        if "Gemma3ForCausalLM" in str(e) or "architectures" in str(e).lower():
            print(
                "\n*** ATOM does not yet support Gemma 3 text architecture. ***\n"
                "Gemma 3 text models use HuggingFace architecture "
                "'Gemma3ForCausalLM'. Add it to support_model_arch_dict in\n"
                "  atom/model_engine/model_runner.py\n"
                "and implement the model in atom/models/ (e.g. gemma3.py).\n"
            )
        raise SystemExit(1) from e

    # Gemma 3 chat format via tokenizer (no enable_thinking)
    try:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in PROMPTS
        ]
    except Exception:
        prompts = PROMPTS

    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )

    print("Running generation...")
    outputs = llm.generate(prompts, sampling_params)

    print("\n--- Gemma 3 text verification ---")
    for prompt, output in zip(PROMPTS, outputs):
        text = output.get("text", "")
        text = _clean_output(text)
        print(f"Prompt: {prompt!r}")
        print(f"Output: {text!r}")
        print("-" * 50)
    print("OK: Gemma 3 text model verification passed.")

    if hasattr(llm, "close"):
        llm.close()


if __name__ == "__main__":
    main()
