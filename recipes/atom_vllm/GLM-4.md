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
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
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
model=/data/models/glm4-moe
task=gsm8k

lm_eval --model local-completions \
        --model_args model=${model},base_url=${url},num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks ${task} \
        --num_fewshot 3 \
        2>&1 | tee log.lmeval.log
```
