export CUDA_VISIBLE_DEVICES=4,5,6,7

# 对于 Qwen-Image-Edit-2511，使用 vllm serve + --omni
vllm serve ./FLUX.2-klein-base-9B \
  --omni \
  --port 4011 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --served-model-name FLUX.2