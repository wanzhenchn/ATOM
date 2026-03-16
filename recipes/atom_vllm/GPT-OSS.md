# GPT-OSS with ATOM vLLM OOT Platform

This recipe shows how to run `GPT-OSS-120B` with the ATOM vLLM out-of-tree platform. For the overall OOT design and plugin flow, see [vLLM-ATOM-OOT-Plugin-Backend](./vLLM-ATOM-OOT-Plugin-Backend.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Download the Model if Needed

```bash
model_id=openai/gpt-oss-120b
model_path=/data/models/gpt-oss-120b

hf download ${model_id} --local-dir ${model_path}
```

## Step 3: Launch vLLM Server

```bash
model_path=/data/models/gpt-oss-120b

vllm serve $model_path \
    --host localhost \
    --port 8000 \
    --kv-cache-dtype fp8 \
    --disable-log-requests \
    --gpu_memory_utilization 0.3 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --no-enable-prefix-caching \
    2>&1 | tee log.serve.log &
```
