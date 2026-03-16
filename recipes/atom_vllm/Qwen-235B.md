# Qwen3-235B-A22B with ATOM vLLM OOT Platform

This recipe shows how to run `Qwen3-235B-A22B-Instruct-2507-FP8` with the ATOM vLLM out-of-tree platform. For the overall OOT design and plugin flow, see [vLLM-ATOM-OOT-Plugin-Backend](./vLLM-ATOM-OOT-Plugin-Backend.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Download the Model if Needed

```bash
model_id=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
model_path=/data/models/Qwen3-235B-A22B-Instruct-2507-FP8

hf download ${model_id} --local-dir ${model_path}
```

## Step 3: Launch vLLM Server

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

model_path=/data/models/Qwen3-235B-A22B-Instruct-2507-FP8

vllm serve $model_path \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --kv-cache-dtype fp8 \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --max-model-len 16384 \
    --max-num-batched-tokens 20000 \
    --no-enable-prefix-caching \
    2>&1 | tee log.serve.log &
```
