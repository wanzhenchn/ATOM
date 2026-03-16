# Kimi-K2-Thinking with ATOM vLLM OOT Platform

This recipe shows how to run `Kimi-K2-Thinking` with the ATOM vLLM out-of-tree platform. For the overall OOT design and plugin flow, see [vLLM-ATOM-OOT-Plugin-Backend](./vLLM-ATOM-OOT-Plugin-Backend.md).

This model uses remote code, so the launch command keeps `--trust-remote-code`.

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Download the Model if Needed

```bash
model_id=amd/Kimi-K2-Thinking-MXFP4
model_path=/data/models/Kimi-K2-Thinking-MXFP4

hf download ${model_id} --local-dir ${model_path}
```

## Step 3: Launch vLLM Server

```bash
model_path=/data/models/Kimi-K2-Thinking-MXFP4

vllm serve $model_path \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --no-enable-prefix-caching \
    2>&1 | tee log.serve.log &
```

### Optional: Enable OOT Profiling
If you want to collect OOT profiles, export the following env vars and add `--profiler-config "$profiler_config"` to the `vllm serve` command above.

```bash
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_TORCH_PROFILER_DIR=./

profiler_config=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":%s,"torch_profiler_record_shapes":%s}' \
    "${VLLM_TORCH_PROFILER_DIR}" \
    "$([[ "${VLLM_TORCH_PROFILER_WITH_STACK:-0}" == "1" ]] && echo true || echo false)" \
    "$([[ "${VLLM_TORCH_PROFILER_RECORD_SHAPES:-0}" == "1" ]] && echo true || echo false)")
```

## Step 4: Validate Accuracy With lm_eval

```bash
addr=localhost
port=8000
url=http://${addr}:${port}/v1/completions
model=/data/models/Kimi-K2-Thinking-MXFP4
task=gsm8k

lm_eval --model local-completions \
        --model_args model=${model},base_url=${url},num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks ${task} \
        --num_fewshot 3 \
        2>&1 | tee log.lmeval.log
```
