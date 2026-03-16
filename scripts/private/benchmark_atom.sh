#!/bin/bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

ISL=${ISL:-8000}
OSL=${OSL:-1000}
CONCURRENCY=${CONCURRENCY:-32}
NUM_PROMPTS=$((CONCURRENCY * 10))
PORT=${PORT:-8001}

model=/home/hatwu/models/Kimi-K2-Thinking-MXFP4/

result_dir="./benchmark_results"
mkdir -p "$result_dir"
RESULT_FILENAME="kimi_k2_isl${ISL}_osl${OSL}_conc${CONCURRENCY}"

python -m atom.benchmarks.benchmark_serving \
    --model ${model} \
    --backend vllm \
    --base-url http://localhost:${PORT} \
    --trust-remote-code \
    --dataset-name random \
    --random-input-len ${ISL} \
    --random-output-len ${OSL} \
    --random-range-ratio 0.8 \
    --max-concurrency ${CONCURRENCY} \
    --num-prompts ${NUM_PROMPTS} \
    --request-rate inf \
    --ignore-eos \
    --save-result \
    --result-dir ${result_dir} \
    --result-filename ${RESULT_FILENAME}.json \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    2>&1 | tee log.client.log
