MODEL=/home/hatwu/models/Kimi-K2-Thinking-MXFP4/
PORT=8001
log_dir=./
RESULT_FILENAME=atom_result.json

python /home/hatwu/bench_serving/benchmark_serving.py \
    --model=$MODEL --backend=vllm --base-url=http://localhost:$PORT \
    --trust-remote-code \
    --dataset-name=random \
    --random-input-len=8192 --random-output-len=1024 \
    --random-range-ratio 0.8 \
    --num-prompts=320 \
    --max-concurrency=32 \
    --request-rate=inf --ignore-eos \
    --save-result --result-dir=${log_dir} --result-filename=$RESULT_FILENAME.json \
    --percentile-metrics="ttft,tpot,itl,e2el"
