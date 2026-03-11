#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Evaluate Gemma 3 VLM accuracy via ATOM's standalone VLM inference.

Test suites:
  synthetic  — Synthetic images with 100% verifiable ground truth (default, no internet)
    - Solid color identification (red/green/blue/etc. → color name in response)
    - Gradient detection (is this a gradient? → yes)
    - Counting (N-dot images → count answer)
    - Color comparison (two-color split images)

  vqa_lite   — Small subset of a public VQA-style dataset (requires internet)

Evaluation uses the same SimpleSDPAAttention standalone approach as
verify_gemma3_vlm.py (no paged attention infrastructure needed).

Usage:
  # Default: synthetic suite
  python scripts/verify_gemma3/eval_gemma3_vlm.py

  # VQA lite (requires HuggingFace internet access)
  python scripts/verify_gemma3/eval_gemma3_vlm.py --suite vqa_lite --n 50

  # Custom model
  python scripts/verify_gemma3/eval_gemma3_vlm.py --model /data/models/gemma-3-4b-it

  # Verbose output
  python scripts/verify_gemma3/eval_gemma3_vlm.py --verbose

Defaults: synthetic suite, model=/data/models/gemma-3-4b-it.
Expected accuracy: synthetic ≥ 70% (color + gradient questions).
"""
import argparse
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Path setup (same as verify_gemma3_vlm.py) ─────────────────────────────────
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

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

os.environ.setdefault("AITER_LOG_LEVEL", "ERROR")
os.environ.setdefault("TORCH_DYNAMO_DISABLE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import socket as _socket
def _free_port() -> int:
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
os.environ.setdefault("MASTER_PORT", str(_free_port()))

import logging
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from PIL import Image

try:
    torch._dynamo.config.disable = True
except Exception:
    pass

logging.getLogger("atom").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

# ── SimpleSDPAAttention (identical to verify_gemma3_vlm.py) ───────────────────

class SimpleSDPAAttention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads=None,
                 per_layer_sliding_window=None, prefix=None, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = per_layer_sliding_window
        self.layer_name = prefix or f"attn_{id(self)}"
        self._k_cache: list[torch.Tensor] = []
        self._v_cache: list[torch.Tensor] = []

    def reset_cache(self):
        self._k_cache.clear()
        self._v_cache.clear()

    def forward(self, query, key, value, positions=None, **kwargs):
        seq_len = query.shape[0]
        dtype = query.dtype
        q = query.view(seq_len, self.num_heads, self.head_dim)
        k = key.view(seq_len, self.num_kv_heads, self.head_dim)
        v = value.view(seq_len, self.num_kv_heads, self.head_dim)
        self._k_cache.append(k)
        self._v_cache.append(v)
        k_full = torch.cat(self._k_cache, dim=0)
        v_full = torch.cat(self._v_cache, dim=0)
        if self.sliding_window is not None and k_full.shape[0] > self.sliding_window:
            k_full = k_full[-self.sliding_window:]
            v_full = v_full[-self.sliding_window:]
        q_t = q.unsqueeze(0).transpose(1, 2)
        k_t = k_full.unsqueeze(0).transpose(1, 2)
        v_t = v_full.unsqueeze(0).transpose(1, 2)
        if self.num_kv_heads < self.num_heads:
            k_t = k_t.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v_t = v_t.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=self.scale,
                                             is_causal=(seq_len > 1))
        out = out.squeeze(0).transpose(0, 1).contiguous()
        return out.view(seq_len, self.num_heads * self.head_dim).to(dtype)


import atom.model_ops as _mops
_mops.Attention = SimpleSDPAAttention
_mops.PagedAttention = SimpleSDPAAttention

from atom.config import Config, set_current_atom_config
from atom.model_loader.loader import load_model
from atom.utils.forward_context import AttentionMetaData, Context, get_forward_context

try:
    from aiter import init_dist_env, destroy_dist_env
except (ImportError, AttributeError):
    from aiter.ops.communication import init_dist_env, destroy_dist_env

DEFAULT_MODEL = "/data/models/gemma-3-4b-it"

# ── Synthetic test generation ─────────────────────────────────────────────────

# Named solid colors (RGB)
SOLID_COLORS = {
    "red":    (220, 30,  30),
    "green":  (30,  180, 30),
    "blue":   (30,  30,  220),
    "yellow": (240, 220, 20),
    "white":  (240, 240, 240),
    "black":  (20,  20,  20),
    "orange": (230, 130, 20),
    "purple": (140, 30,  190),
    "pink":   (230, 100, 160),
    "cyan":   (30,  200, 210),
}


def make_solid_image(color_rgb: tuple, size: int = 224) -> Image.Image:
    arr = np.full((size, size, 3), color_rgb, dtype=np.uint8)
    return Image.fromarray(arr)


def make_gradient_image(c1: tuple, c2: tuple, size: int = 224) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        t = i / (size - 1)
        arr[i, :] = tuple(int(c1[j] * (1 - t) + c2[j] * t) for j in range(3))
    return Image.fromarray(arr)


def make_dot_image(n_dots: int, size: int = 224) -> Image.Image:
    """White image with n_dots black dots in a grid."""
    arr = np.full((size, size, 3), 240, dtype=np.uint8)
    dot_r = max(8, size // (n_dots * 4 + 2))
    # Place dots in a row at vertical center
    spacing = size // (n_dots + 1)
    cy = size // 2
    for i in range(n_dots):
        cx = spacing * (i + 1)
        rr, cc = np.ogrid[:size, :size]
        mask = (rr - cy) ** 2 + (cc - cx) ** 2 <= dot_r ** 2
        arr[mask] = [20, 20, 20]
    return Image.fromarray(arr)


def build_synthetic_suite() -> list[dict]:
    """Return list of {image, question, answer_check_fn, category, description}."""
    tests = []

    # ── 1. Color identification (10 solid colors) ─────────────────────────────
    for color_name, rgb in SOLID_COLORS.items():
        img = make_solid_image(rgb)
        def _check_color(text, cn=color_name):
            return cn.lower() in text.lower()
        tests.append({
            "image": img,
            "question": "What is the dominant color of this image? Answer with a single color name.",
            "answer_check": _check_color,
            "category": "color_id",
            "description": f"solid {color_name} → answer contains '{color_name}'",
            "expected": color_name,
        })

    # ── 2. Gradient detection (5 cases) ───────────────────────────────────────
    gradient_pairs = [
        ((30, 30, 200), (200, 30, 30)),   # blue→red
        ((30, 200, 30), (200, 200, 30)),   # green→yellow
        ((200, 200, 200), (30, 30, 30)),   # white→black
        ((30, 30, 200), (30, 200, 200)),   # blue→cyan
        ((200, 30, 200), (200, 200, 30)),  # purple→yellow
    ]
    for c1, c2 in gradient_pairs:
        img = make_gradient_image(c1, c2)
        tests.append({
            "image": img,
            "question": "Does this image show a smooth color gradient from one side to the other? Answer yes or no.",
            "answer_check": lambda t: re.search(r"\byes\b", t, re.IGNORECASE) is not None,
            "category": "gradient_detect",
            "description": f"gradient ({c1}→{c2}) → 'yes'",
            "expected": "yes",
        })

    # ── 3. Solid vs. gradient discrimination (5 solid, 5 gradient) ────────────
    for color_name, rgb in list(SOLID_COLORS.items())[:5]:
        img = make_solid_image(rgb)
        tests.append({
            "image": img,
            "question": "Is this image a uniform solid color with no gradient? Answer yes or no.",
            "answer_check": lambda t: re.search(r"\byes\b", t, re.IGNORECASE) is not None,
            "category": "solid_detect",
            "description": f"solid {color_name} → 'yes'",
            "expected": "yes",
        })
    for c1, c2 in gradient_pairs[:5]:
        img = make_gradient_image(c1, c2)
        tests.append({
            "image": img,
            "question": "Is this image a uniform solid color with no gradient? Answer yes or no.",
            "answer_check": lambda t: re.search(r"\bno\b", t, re.IGNORECASE) is not None,
            "category": "solid_detect",
            "description": f"gradient → 'no'",
            "expected": "no",
        })

    # ── 4. Dot counting (1–5 dots) ─────────────────────────────────────────────
    _num_words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    for n in range(1, 6):
        img = make_dot_image(n)
        def _check_count(text, n=n, nw=_num_words[n]):
            text_l = text.lower()
            return str(n) in text_l or nw in text_l
        tests.append({
            "image": img,
            "question": f"How many black dots are visible in this image? Answer with a number.",
            "answer_check": _check_count,
            "category": "counting",
            "description": f"{n} dot(s) → answer contains '{n}' or '{_num_words[n]}'",
            "expected": str(n),
        })

    return tests


# ── Inference helpers ─────────────────────────────────────────────────────────

def _setup_forward_context(positions: torch.Tensor, is_prefill: bool) -> None:
    fwd_ctx = get_forward_context()
    seq_len = positions.shape[0]
    cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=positions.device)
    fwd_ctx.attn_metadata = AttentionMetaData(
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_q,
        max_seqlen_q=seq_len, max_seqlen_k=seq_len,
    )
    fwd_ctx.context = Context(positions=positions, is_prefill=is_prefill, batch_size=1)


def reset_kv_caches(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, SimpleSDPAAttention):
            m.reset_cache()


def run_vlm_inference(
    model: nn.Module,
    processor,
    image: Image.Image,
    question: str,
    image_token_id: int,
    max_new_tokens: int = 32,
    device: torch.device = None,
) -> str:
    """Run a single VLM forward pass, return generated text."""
    if device is None:
        device = next(model.parameters()).device

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]}
    ]
    try:
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text_prompt = f"<image>\n{question}\n"

    inputs = processor(
        text=text_prompt, images=image, return_tensors="pt", padding=True
    )
    input_ids = inputs["input_ids"][0].to(device)
    pixel_values = inputs["pixel_values"].to(device).to(torch.bfloat16)

    reset_kv_caches(model)
    generated = []
    eos_id = processor.tokenizer.eos_token_id

    # Prefill
    positions = torch.arange(input_ids.shape[0], dtype=torch.long, device=device)
    _setup_forward_context(positions, is_prefill=True)
    with torch.no_grad():
        hidden = model(input_ids, positions, pixel_values=pixel_values)
        logits = model.compute_logits(hidden)
    next_token = int(logits[-1].argmax())
    generated.append(next_token)
    cur_pos = input_ids.shape[0]

    # Decode
    for _ in range(max_new_tokens - 1):
        if eos_id is not None and next_token == eos_id:
            break
        tok = torch.tensor([next_token], dtype=torch.long, device=device)
        pos = torch.tensor([cur_pos], dtype=torch.long, device=device)
        _setup_forward_context(pos, is_prefill=False)
        with torch.no_grad():
            hidden = model(tok, pos)
            logits = model.compute_logits(hidden)
        next_token = int(logits[-1].argmax())
        generated.append(next_token)
        cur_pos += 1

    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Category-level reporting ──────────────────────────────────────────────────

def report_by_category(results: list[dict]) -> None:
    from collections import defaultdict
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["correct"])
    print("\n  Per-category breakdown:")
    for cat, vals in sorted(by_cat.items()):
        acc = sum(vals) / len(vals)
        print(f"    {cat:20s}: {acc:.0%}  ({sum(vals)}/{len(vals)})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma 3 VLM accuracy in ATOM"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--suite", default="synthetic",
        choices=["synthetic"],
        help="Test suite (default: synthetic)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=24,
        help="Max tokens to generate per question (default: 24)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=1024,
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each question / prediction",
    )
    args = parser.parse_args()

    print(f"=== Gemma 3 VLM accuracy eval ===")
    print(f"Model: {args.model}")
    print(f"Suite: {args.suite}")
    print()

    # ── 1. Init distributed ───────────────────────────────────────────────────
    print("Initializing distributed env (TP=1)...")
    init_dist_env(
        tensor_model_parallel_size=1, rankID=0,
        backend="cpu:gloo,cuda:nccl",
        distributed_init_method="env://", local_rank=0,
    )

    # ── 2. Build ATOM config & model ──────────────────────────────────────────
    print("Building model...")
    config = Config(model=args.model, max_model_len=args.max_model_len)
    set_current_atom_config(config)
    hf_cfg = config.hf_config
    image_token_id = getattr(hf_cfg, "image_token_index", None) or getattr(
        hf_cfg, "image_token_id", None
    )

    from atom.models.gemma3 import Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration(config)
    model = model.to(torch.bfloat16).cuda().eval()
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

    print("Loading weights...")
    load_model(model, args.model, hf_cfg)

    print("Loading processor...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)

    # ── 3. Build test suite ───────────────────────────────────────────────────
    print("Building test suite...")
    tests = build_synthetic_suite()
    print(f"  {len(tests)} test cases")

    # ── 4. Run evaluation ─────────────────────────────────────────────────────
    import time
    results = []
    t0 = time.time()

    for i, test in enumerate(tests):
        try:
            response = run_vlm_inference(
                model=model,
                processor=processor,
                image=test["image"],
                question=test["question"],
                image_token_id=image_token_id,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            response = f"[ERROR: {exc}]"

        correct = test["answer_check"](response)
        results.append({
            "category": test["category"],
            "description": test["description"],
            "question": test["question"],
            "expected": test["expected"],
            "response": response,
            "correct": correct,
        })

        if args.verbose or not correct:
            status = "OK  " if correct else "FAIL"
            print(
                f"  [{i+1:2d}/{len(tests)}] [{status}] {test['description']}\n"
                f"          Q: {test['question'][:70]}\n"
                f"          A: {response[:80]!r}"
            )
        else:
            # Progress dot
            print(f"  [{i+1:2d}/{len(tests)}] [OK  ] {test['description']}")

    elapsed = time.time() - t0

    # ── 5. Report ─────────────────────────────────────────────────────────────
    correct_count = sum(r["correct"] for r in results)
    accuracy = correct_count / len(results)

    print()
    print(f"=== Results: {args.suite} ({args.model}) ===")
    print(f"  Total:    {len(results)}")
    print(f"  Correct:  {correct_count}")
    print(f"  Accuracy: {accuracy:.1%}  ({correct_count}/{len(results)})")
    print(f"  Time:     {elapsed:.1f}s  ({elapsed/len(results):.1f}s/example)")
    report_by_category(results)

    baseline = 0.70
    if accuracy >= baseline:
        print(f"\nOK: VLM accuracy {accuracy:.1%} >= baseline {baseline:.0%}.")
    else:
        print(
            f"\nWARN: VLM accuracy {accuracy:.1%} < baseline {baseline:.0%}. "
            "Check vision tower / projector weight loading."
        )

    try:
        destroy_dist_env()
    except Exception:
        pass


if __name__ == "__main__":
    main()
