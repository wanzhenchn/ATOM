#export ATOM_DISABLE_VLLM_PLUGIN=1

export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER=1
export VLLM_RPC_TIMEOUT=1800000

export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor
rm -rf /root/.cache/

#model_path=/data/models/Qwen3-235B-A22B-Instruct-2507-FP8/
model_path=/home/hatwu/models/Kimi-K2-Thinking-MXFP4/

export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
PROFILER_DIR="./profiler_oot"
mkdir -p "$PROFILER_DIR"
PROFILER_CONFIG="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILER_DIR}\",\"torch_profiler_with_stack\":true,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_flops\":false,\"torch_profiler_use_gzip\":true,\"torch_profiler_dump_cuda_time_total\":true,\"torch_profiler_with_memory\":false,\"ignore_frontend\":false,\"delay_iterations\":0,\"max_iterations\":0}"


vllm serve $model_path \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --load-format fastsafetensors \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --no-enable-prefix-caching \
    --attention-backend ROCM_AITER_MLA \
    $PROFILER_CONFIG \
    2>&1 | tee log.serve.log &

#   --enable-expert-parallel \
