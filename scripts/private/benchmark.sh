export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

ISL=${ISL:-8000}
OSL=${OSL:-1000}
CONCURRENCY=${CONCURRENCY:-32}
NUM_PROMPTS=$((CONCURRENCY * 10))

#model="/data/models/Qwen3-235B-A22B-Instruct-2507-FP8/"
model="/models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8/"
model=/home/hatwu/models/Kimi-K2-Thinking-MXFP4/

vllm bench serve \
    --host localhost \
    --port 8000 \
    --model ${model} \
    --dataset-name random \
    --random-input-len ${ISL} \
    --random-output-len ${OSL} \
    --random-range-ratio 0.8 \
    --max-concurrency ${CONCURRENCY} \
    --num-prompts ${NUM_PROMPTS} \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos 2>&1 | tee log.client.log
