#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export CLIP_MODEL_PATH="models/clip-vit-large-patch14"
export CLIP_SERVER_PORT=8030
export CLIP_MAX_BATCH_SIZE=16
export CLIP_BATCH_TIMEOUT=0.05

python clip_server.py