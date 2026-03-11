#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Check whether ATOM supports Gemma 3 multi-modal (vision-language) models.

Verifies registry support for Gemma3ForConditionalGeneration and optionally
checks a HuggingFace model config for vision/multimodal capabilities.

Usage:
  python scripts/verify_gemma3/check_gemma3_multimodal_support.py
  python scripts/verify_gemma3/check_gemma3_multimodal_support.py --model google/gemma-3-4b-it
"""
import argparse
import os
import sys

repo_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Use local AITER clone when present (avoids ImportError from aiter)
_aiter_dir = os.environ.get("AITER_DIR") or os.path.join(
    os.path.dirname(repo_root), "aiter"
)
if os.path.isdir(_aiter_dir):
    _existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = _aiter_dir + (f":{_existing}" if _existing else "")
    if _aiter_dir not in sys.path:
        sys.path.insert(0, _aiter_dir)

# Default HuggingFace model id for Gemma 3 multimodal (4B instruction-tuned)
DEFAULT_VLM_MODEL = "/data/models/gemma-3-4b-it" #"google/gemma-3-4b-it"
GEMMA3_VLM_ARCH = "Gemma3ForConditionalGeneration"
GEMMA3_TEXT_ARCH = "Gemma3ForCausalLM"


def _has_vision_config(config) -> bool:
    """Return True if config indicates vision/multimodal (e.g. vision encoder)."""
    return getattr(config, "vision_config", None) is not None or hasattr(
        config, "image_size"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Check if ATOM supports Gemma 3 multi-modal (VLM) for a given model"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_VLM_MODEL,
        help=f"HuggingFace model id or path (default: {DEFAULT_VLM_MODEL})",
    )
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Only check ATOM registry; do not load HuggingFace config (avoids gated repo / network).",
    )
    args = parser.parse_args()

    from atom.model_engine.model_runner import support_model_arch_dict

    print("Gemma 3 multi-modal (VLM) support check")
    print("=" * 50)
    print("ATOM registry (support_model_arch_dict):")
    vlm_supported = GEMMA3_VLM_ARCH in support_model_arch_dict
    text_supported = GEMMA3_TEXT_ARCH in support_model_arch_dict
    print(f"  {GEMMA3_VLM_ARCH}: {'yes' if vlm_supported else 'no'}")
    print(f"  {GEMMA3_TEXT_ARCH} (text-only): {'yes' if text_supported else 'no'}")

    if args.skip_config:
        print("\n(Skipping HuggingFace config; use without --skip-config to check a model.)")
        if not vlm_supported:
            print(
                "\n*** ATOM does not yet support Gemma 3 multi-modal. ***\n"
                "Add Gemma3ForConditionalGeneration to support_model_arch_dict in\n"
                "atom/model_engine/model_runner.py and implement the VLM model in atom/models/."
            )
            sys.exit(1)
        print("\nOK: ATOM registry includes Gemma 3 VLM architecture.")
        return

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            args.model, trust_remote_code=True
        )
        archs = getattr(config, "architectures", None) or []
        model_type = getattr(config, "model_type", "")
        has_vision = _has_vision_config(config)

        print(f"\nModel: {args.model}")
        print("HuggingFace config:")
        print(f"  architectures: {archs}")
        print(f"  model_type: {model_type}")
        print(f"  has vision/multimodal config: {has_vision}")

        if not archs:
            print("\n  Could not determine architecture.")
            sys.exit(1)

        primary = archs[0]

        if primary == GEMMA3_VLM_ARCH:
            if vlm_supported:
                print(f"\nOK: ATOM supports this Gemma 3 VLM model (architecture {primary!r}).")
            else:
                print(
                    f"\n*** ATOM does not yet support Gemma 3 multi-modal ({primary!r}). ***"
                )
                print(
                    "Add it to support_model_arch_dict in "
                    "atom/model_engine/model_runner.py and implement the VLM model"
                    " (vision encoder + language model) in atom/models/."
                )
                print("See docs and Gemma3ForCausalLM in atom/models/gemma3.py for patterns.")
                sys.exit(1)
        elif primary == GEMMA3_TEXT_ARCH:
            if has_vision:
                print(
                    "\n  This checkpoint reports text architecture but has vision config;"
                    " it may be a multimodal checkpoint. ATOM VLM support: no."
                )
                sys.exit(1)
            print(
                "\n  This is a text-only Gemma 3 model. For multi-modal verification,"
                f" use a VLM model (e.g. --model {DEFAULT_VLM_MODEL})."
            )
        else:
            print(f"\n  Unknown architecture {primary!r}; ATOM VLM support: no.")
            sys.exit(1)

    except Exception as e:
        err = str(e).strip()
        print(f"\nFailed to load config for {args.model}: {err}")
        if "gated" in err.lower() or "401" in err or "access" in err.lower():
            print(
                "\n  Model may be gated. Log in with: huggingface-cli login\n"
                "  Or run with --skip-config to only check ATOM registry."
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
