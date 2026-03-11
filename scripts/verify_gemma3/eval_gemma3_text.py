#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Evaluate Gemma 3 text model accuracy via ATOM's LLMEngine.

Benchmarks:
  arc_challenge  — ARC Challenge (4-choice science QA, harder)
  arc_easy       — ARC Easy (4-choice science QA, easier)
  mmlu           — MMLU subset (5-shot zero-shot, 4-choice)

Evaluation method: generate up to 8 tokens, parse the first A/B/C/D letter.

Usage:
  # Quick smoke test (20 examples, ARC-Challenge)
  python scripts/verify_gemma3/eval_gemma3_text.py --n 20

  # Full ARC-Challenge test split (~1170 examples)
  python scripts/verify_gemma3/eval_gemma3_text.py --benchmark arc_challenge --n 0

  # MMLU subset
  python scripts/verify_gemma3/eval_gemma3_text.py --benchmark mmlu --n 100

  # Custom model
  python scripts/verify_gemma3/eval_gemma3_text.py --model /data/models/gemma-3-4b-it --n 50

Defaults: arc_challenge, n=50, model=/data/models/gemma-3-4b-it.
Expected accuracy: Gemma3-4B-IT ARC-Challenge ~60-65%, MMLU ~55-65%.
"""
import argparse
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
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

if _triton_python is not None:
    os.environ.setdefault("ATOM_USE_TRITON_DECODE", "1")
else:
    os.environ.setdefault("ATOM_USE_TRITON_DECODE", "0")

os.environ.setdefault("AITER_LOG_LEVEL", "ERROR")

import logging
logging.getLogger("atom").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from transformers import AutoTokenizer

DEFAULT_MODEL = "/data/models/gemma-3-4b-it"

# ── Benchmark loaders ─────────────────────────────────────────────────────────

def load_arc(split_name: str, n: int) -> list[dict]:
    """Load ARC-Easy or ARC-Challenge examples."""
    from datasets import load_dataset
    name = "ARC-Challenge" if "challenge" in split_name else "ARC-Easy"
    ds = load_dataset("ai2_arc", name, split="test", streaming=(n > 0))
    examples = []
    for ex in ds:
        choices = ex["choices"]
        labels = choices["label"]   # ['A','B','C','D']
        texts = choices["text"]
        examples.append({
            "question": ex["question"],
            "choices": dict(zip(labels, texts)),
            "answer": ex["answerKey"],  # 'A','B','C','D' or '1','2','3','4'
        })
        if n > 0 and len(examples) >= n:
            break
    return examples


def load_mmlu(n: int) -> list[dict]:
    """Load a mixed MMLU subset across several subjects."""
    from datasets import load_dataset
    subjects = [
        "high_school_mathematics", "high_school_science",
        "world_history", "college_biology", "logical_fallacies",
        "high_school_geography", "astronomy", "nutrition",
    ]
    examples = []
    per_subject = max(1, (n if n > 0 else 200) // len(subjects))
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", streaming=True)
            for ex in ds:
                labels = ["A", "B", "C", "D"]
                choices = dict(zip(labels, ex["choices"]))
                examples.append({
                    "question": ex["question"],
                    "choices": choices,
                    "answer": labels[ex["answer"]],  # int index → letter
                })
                if len([e for e in examples]) % per_subject == 0:
                    break
                if n > 0 and len(examples) >= n:
                    break
        except Exception as e:
            print(f"  Warning: could not load mmlu/{subj}: {e}")
        if n > 0 and len(examples) >= n:
            break
    return examples[:n] if n > 0 else examples


# ── Prompt formatting ─────────────────────────────────────────────────────────

LABELS = ["A", "B", "C", "D"]

def format_mcq_prompt(question: str, choices: dict, tokenizer) -> str:
    """Format a multiple-choice question using the model's chat template."""
    choice_lines = "\n".join(
        f"{lbl}. {choices[lbl]}"
        for lbl in LABELS
        if lbl in choices
    )
    content = (
        f"{question}\n\n"
        f"{choice_lines}\n\n"
        "Answer with the letter of the correct choice only (A, B, C, or D)."
    )
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return content + "\nAnswer:"


def parse_answer(text: str) -> str | None:
    """Extract the first A/B/C/D from the generated text."""
    text = text.strip()
    # Direct single-letter answer
    if text and text[0].upper() in LABELS:
        return text[0].upper()
    # Pattern: "Answer: X" or "(X)" or "X."
    for pat in [r"\b([ABCD])\b", r"\(([ABCD])\)", r"^([ABCD])[.\s]"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


# ── Normalise answer key ──────────────────────────────────────────────────────

def normalise_key(key: str) -> str | None:
    """Map answer keys to A/B/C/D (ARC sometimes uses '1','2','3','4')."""
    k = str(key).strip().upper()
    if k in ("1", "A"):
        return "A"
    if k in ("2", "B"):
        return "B"
    if k in ("3", "C"):
        return "C"
    if k in ("4", "D"):
        return "D"
    return k if k in LABELS else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma 3 text accuracy in ATOM"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--benchmark", default="arc_challenge",
        choices=["arc_challenge", "arc_easy", "mmlu"],
        help="Benchmark to evaluate (default: arc_challenge)",
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Number of examples (0 = full split, default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Requests per ATOM engine batch (default: 16)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=1024,
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each question/prediction",
    )
    args = parser.parse_args()

    print(f"=== Gemma 3 text accuracy eval ===")
    print(f"Model:     {args.model}")
    print(f"Benchmark: {args.benchmark} (n={args.n or 'all'})")
    print()

    # ── Load examples ─────────────────────────────────────────────────────────
    print("Loading benchmark...")
    if args.benchmark in ("arc_challenge", "arc_easy"):
        examples = load_arc(args.benchmark, args.n)
    else:
        examples = load_mmlu(args.n)
    print(f"  Loaded {len(examples)} examples.")

    # ── Start ATOM engine ─────────────────────────────────────────────────────
    print("Starting ATOM engine...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    engine_args = EngineArgs(
        model=args.model,
        enforce_eager=True,
        level=0,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.7,
    )
    llm = engine_args.create_engine(tokenizer=tokenizer)
    print("  Engine ready.")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=8)

    # ── Evaluate in batches ───────────────────────────────────────────────────
    correct = 0
    skipped = 0
    total = len(examples)

    import time
    t0 = time.time()

    for batch_start in range(0, total, args.batch_size):
        batch = examples[batch_start : batch_start + args.batch_size]
        prompts = [
            format_mcq_prompt(ex["question"], ex["choices"], tokenizer)
            for ex in batch
        ]
        outputs = llm.generate(prompts, sampling_params)

        for ex, output in zip(batch, outputs):
            gen_text = output.get("text", "").strip()
            predicted = parse_answer(gen_text)
            gold = normalise_key(ex["answer"])

            if predicted is None:
                skipped += 1
                if args.verbose:
                    print(f"  [SKIP] Q: {ex['question'][:60]}... | gen={gen_text!r}")
            elif predicted == gold:
                correct += 1
                if args.verbose:
                    print(f"  [OK]   Q: {ex['question'][:60]}... | pred={predicted} gold={gold}")
            else:
                if args.verbose:
                    print(f"  [FAIL] Q: {ex['question'][:60]}... | pred={predicted} gold={gold} | gen={gen_text!r}")

        done = min(batch_start + args.batch_size, total)
        elapsed = time.time() - t0
        acc_so_far = correct / (done - skipped) if done > skipped else 0.0
        print(
            f"  [{done}/{total}] acc={acc_so_far:.1%}  correct={correct}  "
            f"skipped={skipped}  elapsed={elapsed:.1f}s"
        )

    # ── Final report ──────────────────────────────────────────────────────────
    answered = total - skipped
    accuracy = correct / answered if answered > 0 else 0.0

    print()
    print(f"=== Results: {args.benchmark} ({args.model}) ===")
    print(f"  Total:    {total}")
    print(f"  Answered: {answered}  ({skipped} skipped / unparseable)")
    print(f"  Correct:  {correct}")
    print(f"  Accuracy: {accuracy:.1%}  ({correct}/{answered})")

    # Rough expected baselines for sanity check
    _baselines = {
        "arc_challenge": 0.50,
        "arc_easy": 0.65,
        "mmlu": 0.45,
    }
    baseline = _baselines.get(args.benchmark, 0.40)
    if accuracy >= baseline:
        print(f"  OK: accuracy {accuracy:.1%} >= baseline {baseline:.0%}.")
    else:
        print(
            f"  WARN: accuracy {accuracy:.1%} < baseline {baseline:.0%}. "
            "May indicate model quality issues."
        )

    if hasattr(llm, "close"):
        llm.close()


if __name__ == "__main__":
    main()
