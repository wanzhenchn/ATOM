pip uninstall -y aiter
cd aiter
python3 setup.py develop
cd ..

pip install --upgrade triton
pip uninstall -u transformers
pip install transformers==5.0.0
pip install git+https://github.com/foundation-model-stack/fastsafetensors.git

cd ATOM
pip install -e . 2>&1 | tee build.log
