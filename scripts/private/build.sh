cd /home/hatwu/vllm
pip install -r requirements/rocm-build.txt
python3 setup.py develop

pip uninstall -y aiter
cd /home/hatwu/aiter
python3 setup.py develop

#pip install --upgrade triton
#pip uninstall -u transformers
#pip install transformers==5.0.0
pip install git+https://github.com/foundation-model-stack/fastsafetensors.git

cd /home/hatwu/ATOM
pip install -e . 2>&1 | tee build.log
