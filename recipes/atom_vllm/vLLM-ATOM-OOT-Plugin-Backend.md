# vLLM out-of-tree ATOM Plugin Backend
ATOM can work as the OOT plugin backend of vLLM. The OOT register mechanism is quite mature and most of accelerators have leveraged this design to register their devices into vLLM without any code changes in upper framework. ATOM follows this design and provide the layer/op, even the entire model implementations to vLLM. The frontend users can launch vLLM server like before and there is no need to specify any arguments or env flags. Meanwhile the ATOM platform can leverage most of the vLLM features and focus more on model- and kernel-level optimizations. For the overall design, here is a RFC to enable ATOM work as the OOT plugin platform of vLLM: https://github.com/ROCm/ATOM/issues/201

## Preparing environment for vLLM with ATOM Plugin Backend
Pull the docker image for vLLM with ATOM Plugin Backend. The docker image is automatically released on ATOM side.
```bash
docker pull rocm/atom-dev:vllm-latest
```

### Launching server of vLLM with ATOM OOT Plugin Platform
There is no code change to vLLM side, so you can launch the vLLM server like before without any specific argument and env flags
```bash
model_path=<your model file path>

vllm serve $model_path \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --trust-remote-code \
    --disable-log-requests \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --no-enable-prefix-caching \
    2>&1 | tee log.serve.log &
```
If your model has not been downloaded, please use below command to directly download your model weight first in docker
```bash
hf download <your model name> --local-dir <your model file path>
```

If you want to disable the ATOM OOT plugin platform and model register, you can use below env flags. The default value is 0
```bash
export ATOM_DISABLE_VLLM_PLUGIN=1
```
If you only want to disable the ATOM Attention Backend, you can use below env flags.  The default value is 0. In most cases, it is not recommended to disable the ATOM Attention
```bash
export ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1
```

### Launching client for validating the accuracy
After server launched, you can begin your workloads. Here is an example for testing the accuracy for model
```bash
addr=localhost
port=8000
url=http://${addr}:${port}/v1/completions
model=<your model file path>
task=gsm8k
lm_eval --model local-completions \
        --model_args model=${model},base_url=${url},num_concurrent=65,max_retries=1,tokenized_requests=False \
        --tasks ${task} \
        --num_fewshot 3 \
        2>&1 | tee log.lmeval.log
```
