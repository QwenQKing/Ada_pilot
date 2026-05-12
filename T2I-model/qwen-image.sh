export CUDA_VISIBLE_DEVICES=0,1,2,3

# 对于 Qwen-Image-Edit-2511，使用 vllm serve + --omni
vllm serve ./Qwen-Image-2512 \
  --omni \
  --port 4011 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8 \
  --dtype bfloat16 \
  --served-model-name Qwen-Image