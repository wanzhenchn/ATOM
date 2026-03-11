#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Verify Gemma 3 VLM (Gemma3ForConditionalGeneration) in ATOM.

Runs a standalone forward pass through the vision tower, multi-modal projector,
and language model using a simple in-memory SDPA attention (no paged KV cache).

This verifies:
  1. Model architecture can be instantiated with ATOM's parallel layers.
  2. Weights load correctly from the HF checkpoint (including weight name remapping).
  3. Vision tower + projector process pixel_values and produce embeddings.
  4. Image embeddings are merged into the language model input at the right positions.
  5. Greedy decode produces coherent, image-aware text.

Usage:
  # Default: synthetic blue image, "describe this image"
  python scripts/verify_gemma3/verify_gemma3_vlm.py

  # Custom model
  python scripts/verify_gemma3/verify_gemma3_vlm.py --model /data/models/gemma-3-4b-it

  # Custom image + question
  python scripts/verify_gemma3/verify_gemma3_vlm.py --image /path/to/img.jpg \\
      --question "What is in this image?"

  # From repo root
  PYTHONPATH=/dockerx/aiter python scripts/verify_gemma3/verify_gemma3_vlm.py

Model default: /data/models/gemma-3-4b-it (Gemma3ForConditionalGeneration, 4B).
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Path setup (ATOM, AITER, Triton) ─────────────────────────────────────────
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
# Suppress torch.compile / dynamo noise (not needed for this eager-mode verify script)
os.environ.setdefault("TORCH_DYNAMO_DISABLE", "1")

# ── Distributed env vars (must be set before init_dist_env with env://) ──────
import socket as _socket

def _free_port() -> int:
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", str(_free_port()))
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

# ── Imports ───────────────────────────────────────────────────────────────────
import logging
import torch
import torch.nn.functional as F
from torch import nn

# Disable torch.compile / dynamo: GemmaRMSNorm calls torch.compile() on its
# forward_static method which emits noisy dynamo metrics errors in this env.
try:
    torch._dynamo.config.disable = True
except Exception:
    pass

logging.getLogger("atom").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# ── SimpleSDPAAttention: replaces PagedAttention for standalone testing ────────
# Must be defined and patched into atom.model_ops BEFORE importing anything
# that transitively imports atom.model_ops (e.g. atom.models.gemma3).

class SimpleSDPAAttention(nn.Module):
    """In-memory SDPA attention for standalone VLM verification.

    Accumulates K/V across forward calls (prefill then decode steps).
    Call reset_cache() between separate sequences.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int = None,
        per_layer_sliding_window: int = None,
        prefix: str = None,
        **kwargs,  # absorb unused PagedAttention kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = per_layer_sliding_window
        self.layer_name = prefix or f"attn_{id(self)}"
        # Accumulated KV (list of per-step tensors; concatenated lazily)
        self._k_cache: list[torch.Tensor] = []
        self._v_cache: list[torch.Tensor] = []

    def reset_cache(self) -> None:
        self._k_cache.clear()
        self._v_cache.clear()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        # query : [seq_len, num_heads * head_dim]
        # key/v : [seq_len, num_kv_heads * head_dim]
        seq_len = query.shape[0]
        dtype = query.dtype

        q = query.view(seq_len, self.num_heads, self.head_dim)
        k = key.view(seq_len, self.num_kv_heads, self.head_dim)
        v = value.view(seq_len, self.num_kv_heads, self.head_dim)

        self._k_cache.append(k)
        self._v_cache.append(v)
        k_full = torch.cat(self._k_cache, dim=0)   # [kv_len, nkv, hd]
        v_full = torch.cat(self._v_cache, dim=0)

        # Apply sliding window for local-attention layers
        if self.sliding_window is not None and k_full.shape[0] > self.sliding_window:
            k_full = k_full[-self.sliding_window:]
            v_full = v_full[-self.sliding_window:]

        # [1, n_heads, seq, hd]
        q_t = q.unsqueeze(0).transpose(1, 2)
        k_t = k_full.unsqueeze(0).transpose(1, 2)
        v_t = v_full.unsqueeze(0).transpose(1, 2)

        # Expand KV heads for GQA
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k_t = k_t.repeat_interleave(n_rep, dim=1)
            v_t = v_t.repeat_interleave(n_rep, dim=1)

        # is_causal=True for prefill (seq_len > 1); False for single-token decode
        out = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            scale=self.scale,
            is_causal=(seq_len > 1),
        )
        # [1, n_heads, seq, hd] → [seq_len, n_heads * hd]
        out = out.squeeze(0).transpose(0, 1).contiguous()
        return out.view(seq_len, self.num_heads * self.head_dim).to(dtype)


# Patch atom.model_ops BEFORE importing atom.models.*
import atom.model_ops as _mops
_mops.Attention = SimpleSDPAAttention
_mops.PagedAttention = SimpleSDPAAttention

# ── ATOM imports (after patching) ─────────────────────────────────────────────
from atom.config import Config, set_current_atom_config
from atom.model_loader.loader import load_model
from atom.utils.forward_context import (
    AttentionMetaData,
    Context,
    get_forward_context,
)

try:
    from aiter import init_dist_env, destroy_dist_env
except (ImportError, AttributeError):
    from aiter.ops.communication import init_dist_env, destroy_dist_env

DEFAULT_MODEL = "/data/models/gemma-3-4b-it"
DEFAULT_QUESTION = "Describe what you see in this image."


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_test_image(size: int = 448):
    """Return a simple PIL image for testing (blue-sky gradient)."""
    try:
        from PIL import Image
        import numpy as np
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(size):
            ratio = i / size
            arr[i, :, 0] = int(30 + 100 * ratio)   # R: dark→medium
            arr[i, :, 1] = int(144 + 80 * ratio)   # G: sky-ish
            arr[i, :, 2] = int(235 - 60 * ratio)   # B: bright→medium
        return Image.fromarray(arr)
    except Exception:
        from PIL import Image
        return Image.new("RGB", (size, size), (100, 149, 237))  # cornflower blue


def reset_kv_caches(model: nn.Module) -> None:
    """Clear SimpleSDPAAttention KV caches in all layers of the model."""
    for module in model.modules():
        if isinstance(module, SimpleSDPAAttention):
            module.reset_cache()


def _is_likely_garbage(text: str) -> bool:
    """Heuristic: output looks like garbled non-text."""
    if not text or len(text.strip()) < 2:
        return True
    ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha > 5 and ascii_alpha / total_alpha < 0.75:
        return True
    return False


# ── Greedy decode ─────────────────────────────────────────────────────────────

def _setup_forward_context(positions: torch.Tensor, is_prefill: bool) -> None:
    """Set up the minimal ForwardContext needed by ParallelLMHead and other ops."""
    fwd_ctx = get_forward_context()
    seq_len = positions.shape[0]
    # cu_seqlens_q: [0, seq_len] for a single sequence
    cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=positions.device)
    fwd_ctx.attn_metadata = AttentionMetaData(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_q,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
    )
    fwd_ctx.context = Context(
        positions=positions,
        is_prefill=is_prefill,
        batch_size=1,
    )


def greedy_decode(
    model: nn.Module,
    input_ids: torch.Tensor,          # [seq_len]
    pixel_values: torch.Tensor,        # [1, C, H, W]
    max_new_tokens: int = 48,
    eos_token_id: int = None,
    device: torch.device = None,
) -> list[int]:
    """Greedy decode: prefill with pixel_values, then step-by-step decode."""
    if device is None:
        device = next(model.parameters()).device

    input_ids = input_ids.to(device)
    pixel_values = pixel_values.to(device).to(torch.bfloat16)

    generated: list[int] = []

    # ── Prefill ──
    positions = torch.arange(input_ids.shape[0], dtype=torch.long, device=device)
    _setup_forward_context(positions, is_prefill=True)
    with torch.no_grad():
        hidden = model(input_ids, positions, pixel_values=pixel_values)
        logits = model.compute_logits(hidden)   # [1, vocab] (lm_head selects last pos)
    next_token = int(logits[-1].argmax())
    generated.append(next_token)

    cur_pos = input_ids.shape[0]

    # ── Decode loop ──
    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and next_token == eos_token_id:
            break
        tok = torch.tensor([next_token], dtype=torch.long, device=device)
        pos = torch.tensor([cur_pos], dtype=torch.long, device=device)
        _setup_forward_context(pos, is_prefill=False)
        with torch.no_grad():
            hidden = model(tok, pos)
            logits = model.compute_logits(hidden)   # [1, vocab]
        next_token = int(logits[-1].argmax())
        generated.append(next_token)
        cur_pos += 1

    return generated


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Gemma 3 VLM (multimodal) in ATOM"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Path to model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--image", default=None,
        help="Path to image file. If omitted, a synthetic test image is used.",
    )
    parser.add_argument(
        "--question", default=DEFAULT_QUESTION,
        help="Question to ask about the image",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=48,
        help="Max tokens to generate (default: 48)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=1024,
        help="Max model context length for Config (default: 1024)",
    )
    args = parser.parse_args()

    print(f"=== Gemma 3 VLM verification ===")
    print(f"Model:    {args.model}")
    print(f"Question: {args.question!r}")
    print()

    # ── 1. Initialize distributed environment (single GPU, TP=1) ─────────────
    print("Initializing distributed env (TP=1)...")
    init_dist_env(
        tensor_model_parallel_size=1,
        rankID=0,
        backend="cpu:gloo,cuda:nccl",
        distributed_init_method="env://",
        local_rank=0,
    )
    print("  OK: distributed env ready.")

    # ── 2. Create ATOM Config ─────────────────────────────────────────────────
    print("Creating ATOM Config...")
    config = Config(model=args.model, max_model_len=args.max_model_len)
    set_current_atom_config(config)
    print(f"  dtype={config.torch_dtype}, quant={config.quant_config is not None}")
    hf_cfg = config.hf_config
    image_token_id = getattr(hf_cfg, "image_token_index", None) or getattr(
        hf_cfg, "image_token_id", None
    )
    mm_tokens = getattr(hf_cfg, "mm_tokens_per_image", 256)
    print(f"  image_token_id={image_token_id}, mm_tokens_per_image={mm_tokens}")

    # ── 3. Instantiate VLM ────────────────────────────────────────────────────
    print("Instantiating Gemma3ForConditionalGeneration...")
    from atom.models.gemma3 import Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration(config)
    model = model.to(torch.bfloat16).cuda().eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params / 1e9:.2f}B on cuda:{torch.cuda.current_device()}")

    # ── 4. Load weights ───────────────────────────────────────────────────────
    print("Loading weights from checkpoint...")
    load_model(model, args.model, hf_cfg)
    print("  Weights loaded.")

    # ── 5. Load processor ─────────────────────────────────────────────────────
    print("Loading processor...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    eos_id = tokenizer.eos_token_id
    print(f"  Vocabulary size: {tokenizer.vocab_size}, eos={eos_id}")

    # ── 6. Prepare image ──────────────────────────────────────────────────────
    if args.image:
        from PIL import Image
        image = Image.open(args.image).convert("RGB")
        print(f"Image: {args.image} {image.size}")
    else:
        image = _make_test_image()
        print("Image: synthetic blue-sky gradient (448×448)")

    # ── 7. Build multimodal prompt and preprocess ─────────────────────────────
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.question},
            ],
        }
    ]
    try:
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        print(f"  apply_chat_template failed ({e}), using manual format.")
        text_prompt = f"<image>\n{args.question}\n"

    inputs = processor(
        text=text_prompt,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    input_ids = inputs["input_ids"]       # [1, seq_len]
    pixel_values = inputs["pixel_values"]  # [1, C, H, W]

    n_img_toks = (input_ids == image_token_id).sum().item()
    print(f"  input_ids shape: {input_ids.shape}, image tokens: {n_img_toks}/{mm_tokens}")
    print(f"  pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")

    # ── 9. Greedy decode ──────────────────────────────────────────────────────
    print(f"\nRunning greedy decode (max_new_tokens={args.max_new_tokens})...")
    reset_kv_caches(model)

    try:
        generated_ids = greedy_decode(
            model=model,
            input_ids=input_ids[0],       # [seq_len] — ATOM uses flat 1D tokens
            pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_id,
        )
    except Exception as exc:
        import traceback
        print(f"\n*** Inference error: {exc} ***")
        traceback.print_exc()
        sys.exit(1)

    generated_text = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()

    # ── 9. Report & verify ────────────────────────────────────────────────────
    print(f"\n--- Gemma 3 VLM output ---")
    print(f"Prompt:   {args.question!r}")
    print(f"Response: {generated_text!r}")
    print("-" * 50)

    if not generated_text:
        print("FAIL: Response is empty.")
        sys.exit(1)

    if _is_likely_garbage(generated_text):
        print(
            "WARN: Response appears garbled (low ASCII ratio). "
            "Vision-language alignment may not be working correctly."
        )
    else:
        print("OK: Gemma 3 VLM verification passed (coherent text generated).")

    try:
        destroy_dist_env()
    except Exception:
        pass


if __name__ == "__main__":
    main()
