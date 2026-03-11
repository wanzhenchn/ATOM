# Gemma 3 verification and evaluation scripts for ATOM

Scripts to verify and evaluate [Gemma 3](https://huggingface.co/google/gemma-3-4b-it) text and vision-language models on ATOM.

Both `Gemma3ForCausalLM` (text-only) and `Gemma3ForConditionalGeneration` (VLM) are fully supported.

## Prerequisites

- ATOM installed and runnable (see repo root `README.md` and `docs/`).
- For real weights: model files available locally (default: `/data/models/gemma-3-4b-it`).
- ROCm/AMD GPU environment with aiter available (default: `/dockerx/aiter`).
- For Triton decode (needed for correct sliding-window text output): ROCm Triton at `/dockerx/triton` or set `TRITON_DIR`.

## Scripts

| Script | Purpose |
|--------|---------|
| `verify_gemma3_text.py` | Smoke test: run Gemma 3 text generation and inspect output |
| `verify_gemma3_vlm.py` | Smoke test: run Gemma 3 VLM inference on a sample image |
| `eval_gemma3_text.py` | Accuracy eval: ARC-Challenge / ARC-Easy / MMLU via ATOM LLMEngine |
| `eval_gemma3_vlm.py` | Accuracy eval: synthetic vision suite (colors, gradients, counting) |
| `check_gemma3_support.py` | Quick registry check (no GPU needed) |
| `check_gemma3_multimodal_support.py` | Quick VLM registry check (no GPU needed) |

## Quick start

```bash
cd /dockerx/ATOM

# Smoke test: text generation
python scripts/verify_gemma3/verify_gemma3_text.py

# Smoke test: VLM inference (uses AutoProcessor; requires torchvision)
python scripts/verify_gemma3/verify_gemma3_vlm.py

# Accuracy eval: ARC-Challenge, 50 examples (expected ~70%+)
python scripts/verify_gemma3/eval_gemma3_text.py --n 50

# Accuracy eval: full ARC-Challenge test split (~1170 examples)
python scripts/verify_gemma3/eval_gemma3_text.py --benchmark arc_challenge --n 0

# Accuracy eval: synthetic VLM suite (30 cases, expected ~90%+)
python scripts/verify_gemma3/eval_gemma3_vlm.py
```

## Accuracy baselines

| Script | Benchmark | Baseline | Observed |
|--------|-----------|----------|----------|
| `eval_gemma3_text.py` | ARC-Challenge (50 ex) | ≥50% | ~72% |
| `eval_gemma3_text.py` | ARC-Easy (50 ex) | ≥65% | — |
| `eval_gemma3_text.py` | MMLU (50 ex) | ≥45% | — |
| `eval_gemma3_vlm.py` | Synthetic (30 cases) | ≥70% | ~93% |

## Model paths and options

Scripts default to `/data/models/gemma-3-4b-it`. Override with `--model`:

```bash
python scripts/verify_gemma3/eval_gemma3_text.py --model /path/to/gemma-3-12b-it --n 100
```

For `eval_gemma3_text.py`, additional flags:

```
--benchmark {arc_challenge,arc_easy,mmlu}   Benchmark to run (default: arc_challenge)
--n N                                        Examples to evaluate; 0 = full split (default: 50)
--batch-size N                               Requests per engine batch (default: 16)
--max-model-len N                            Max token length (default: 1024)
--verbose                                    Print per-question predictions
```

## ATOM support details

### Text models (`Gemma3ForCausalLM`)

Registered in `atom/model_engine/model_runner.py` as `atom.models.gemma3.Gemma3ForCausalLM`.

Key implementation notes:

- **Sliding window attention:** Gemma 3 alternates local (sliding window, 1024 tokens) and global attention layers. The `is_sliding` flag per layer selects the window size.
- **RoPE:** Local layers use `rope_local_base_freq` (10000.0) without scaling; global layers use `rope_theta` from `rope_parameters["full_attention"]`. The `"linear"` and `"default"` rope types are mapped to `rope_scaling=None` because aiter's `LinearScalingRotaryEmbedding` returns a concatenated cache tensor incompatible with the `(cos, sin)` unpack in `RotaryEmbeddingBase`.
- **Triton decode:** Correct sliding-window output during decode requires `ATOM_USE_TRITON_DECODE=1`. The scripts auto-detect Triton and set this flag. Without Triton, output may be garbled.

### VLM models (`Gemma3ForConditionalGeneration`)

Registered as `atom.models.gemma3.Gemma3ForConditionalGeneration`.

Implemented via standalone inference (monkey-patched `SimpleSDPAAttention`) rather than the ATOM `LLMEngine`. The `eval_gemma3_vlm.py` and `verify_gemma3_vlm.py` scripts use this path.

Key implementation notes:

- **Architecture:** `Gemma3ForConditionalGeneration` wraps a SigLIP vision encoder (`SiglipVisionModel`) and the Gemma 3 language model (`Gemma3ForCausalLM`), connected by a two-layer MLP projector.
- **Image preprocessing:** `AutoProcessor` handles tokenization and image resizing to 896×896 (SigLIP format). Image tokens appear as a run of 256 `<image_soft_token>` IDs (token ID 262144) in `input_ids`.
- **VLM config flattening:** The outer `Gemma3Config` (VLM) stores language-model attributes under `text_config`. ATOM's `Config.__post_init__` promotes these to the top level so the model runner and attention backend can access them uniformly.
- **Weight loading:** `Gemma3ForConditionalGeneration.get_parameter()` remaps HuggingFace checkpoint prefixes (`model.language_model.` → `language_model.`, `model.vision_tower.` → `vision_tower.`, etc.) and provides no-op `PPMissingLayer`-style parameters for unknown keys.
