# GLM-4-MoE with ATOM vLLM OOT Platform

This recipe shows how to run a `GLM-4-MoE` checkpoint with the ATOM vLLM out-of-tree platform. For the overall OOT design and plugin flow, see [vLLM-ATOM-OOT-Plugin-Backend](./vLLM-ATOM-OOT-Plugin-Backend.md).

The checkpoint used here should expose the `Glm4MoeForCausalLM` architecture so it can be picked up by the ATOM OOT model override.

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Download the Model if Needed

```bash
model_id=zai-org/GLM-4.7-FP8
model_path=/data/models/glm4-moe

hf download ${model_id} --local-dir ${model_path}
```

## Step 3: Launch vLLM Server

```bash
model_path=/data/models/glm4-moe

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
    --no-enable-prefix-caching \
    2>&1 | tee log.serve.log &
```
