# Qwen3-235B-A22B with ATOM vLLM OOT Platform

This recipe shows how to run `Qwen3-235B-A22B-Instruct-2507-FP8` with the ATOM vLLM out-of-tree platform. For background on the OOT backend, see [vLLM OOT Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

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

The vLLM OOT plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

```bash
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

model_path=/data/models/Qwen3-235B-A22B-Instruct-2507-FP8

vllm serve $model_path \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --kv-cache-dtype fp8 \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --max-model-len 16384 \
    --no-enable-prefix-caching \
    2>&1 | tee log.serve.log &
```

### Optional: Enable Profiling
If you want to collect profiles, add `--profiler-config "$profiler_config"` to the `vllm serve` command above.

```bash
profiler_dir=./

profiler_config=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":true,"torch_profiler_record_shapes":true}' \
    "${profiler_dir}")
```

## Step 4: Validate Accuracy With lm_eval

```bash
addr=localhost
port=8000
url=http://${addr}:${port}/v1/completions
model=/data/models/Qwen3-235B-A22B-Instruct-2507-FP8
task=gsm8k

lm_eval --model local-completions \
        --model_args model=${model},base_url=${url},num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks ${task} \
        --num_fewshot 3 \
        2>&1 | tee log.lmeval.log
```
