#!/bin/bash

export CUDA_VISIBLE_DEVICES=6
export HPS_MODEL_DIR="models/HPSv2"
export HPS_SERVER_PORT=8020
export HPS_MAX_BATCH_SIZE=16
export HPS_BATCH_TIMEOUT=0.05

python hps_server.py