#!/bin/bash
set -e
# Create venv
python3 -m venv .venv && \
# Activate venv
source .venv/bin/activate && \
# Upgrade pip
pip install --upgrade pip && \
# Install vllm
pip install vllm==0.6.4 && \
# Install vllm-scheduler building dependencies
pip install -r vllm-scheduler/requirements-build.txt && \
# Move to vllm-scheduler directory
cd vllm-scheduler && \
# Use existing pytorch
python use_existing_torch.py && \
# Make directory of vllm_flash_attn
mkdir -p vllm/vllm_flash_attn && \
# Change setup.py file
rm setup.py && cp ../new_setup.py ./setup.py && \
# Install vllm-scheduler
pip install -e . --no-build-isolation && \
# Install vidur dependencies
pip install -r vidur/requirements.txt
