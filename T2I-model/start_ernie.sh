#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ernie-sglang

sglang serve \
  --model-path ./ERNIE-Image \
  --host 0.0.0.0 \
  --port 3010 \
  --num-gpus 8 \
  --tp-size 8