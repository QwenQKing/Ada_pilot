#!/bin/bash

export CUDA_VISIBLE_DEVICES=5
export GDINO_MODEL_PATH="models/GroundingDINO"

uvicorn gdino_server:app --host 0.0.0.0 --port 8010