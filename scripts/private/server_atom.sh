#!/bin/bash
export HIP_VISIBLE_DEVICES=4,5,6,7

export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

#model_path=/data/models/Qwen3-235B-A22B-Instruct-2507-FP8/
model_path=/home/hatwu/models/Kimi-K2-Thinking-MXFP4/

export ATOM_PROFILER_MORE=1
PROFILER_DIR="./profiler_atom"
mkdir -p "$PROFILER_DIR"

python -m atom.entrypoints.openai_server \
    --model $model_path \
    --host localhost \
    --server-port 8001 \
    --trust-remote-code \
    -tp 4 \
    --kv_cache_dtype fp8 \
    --torch-profiler-dir "$PROFILER_DIR" \
    2>&1 | tee log.serve.log &

#   --enable-expert-parallel \
#   --max-model-len 16384 \
#   --max-num-batched-tokens 16384 \
