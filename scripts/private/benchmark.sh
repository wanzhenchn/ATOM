export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

#model="/data/models/Qwen3-235B-A22B-Instruct-2507-FP8/"
model="/models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8/"
vllm bench serve \
    --host localhost \
    --port 8000 \
    --model ${model} \
    --dataset-name random \
    --random-input-len 4000 \
    --random-output-len 1000 \
    --max-concurrency 64 \
    --num-prompts 256 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos 2>&1 | tee log.client.log
