export CUDA_VISIBLE_DEVICES=5,6

# 对于 Qwen-Image-Edit-2511，使用 vllm serve + --omni
vllm serve ./z-image \
  --omni \
  --port 4011 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --served-model-name z-image