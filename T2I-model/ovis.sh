export CUDA_VISIBLE_DEVICES=4,5,6,7,0,1,2,3
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

vllm serve ./Ovis-Image-7B/Ovis-Image-7B \
  --omni \
  --port 4011 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --served-model-name Ovis-Image-7B \
  --trust-remote-code