#!/bin/python3

import os
import venv
import subprocess

models_dir = "models"
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

# set default model cache for hugging face
os.environ['HF_HOME'] = './models'

venv.create("env", with_pip=True)

# subprocess.run(['source env/bin/activate && pip3 install wheel packaging numpy torch'], check=True, shell=True)
# pip3 install git+https://github.com/huggingface/transformers@main && 
subprocess.run(['source env/bin/activate && python \
download_models_helper.py'], check=True, shell=True)

# subprocess.run(['deactivate'], check=True, shell=True)