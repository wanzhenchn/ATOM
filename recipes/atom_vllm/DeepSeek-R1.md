# DeepSeek-R1 with ATOM vLLM OOT Platform

This recipe shows how to run `deepseek-ai/DeepSeek-R1-0528` or `amd/DeepSeek-R1-0528-MXFP4` with the ATOM vLLM out-of-tree platform. For the overall OOT design and plugin flow, see [vLLM-ATOM-OOT-Plugin-Backend](./vLLM-ATOM-OOT-Plugin-Backend.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Download the Model if Needed

### BF16

```bash
model_id=deepseek-ai/DeepSeek-R1-0528
model_path=/data/models/DeepSeek-R1-0528

hf download ${model_id} --local-dir ${model_path}
```

### MXFP4

```bash
model_id=amd/DeepSeek-R1-0528-MXFP4
model_path=/data/models/DeepSeek-R1-0528-MXFP4

hf download ${model_id} --local-dir ${model_path}
```

## Step 3: Launch vLLM Server

### BF16

```bash
model_path=/data/models/DeepSeek-R1-0528

vllm serve $model_path \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --no-enable-prefix-caching \
    2>&1 | tee log.serve.log &
```

### MXFP4

```bash
model_path=/data/models/DeepSeek-R1-0528-MXFP4

vllm serve $model_path \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --no-enable-prefix-caching \
    2>&1 | tee log.serve.log &
```
