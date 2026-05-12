export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 对于 Qwen-Image-Edit-2511，使用 vllm serve + --omni
vllm serve ./LongCat-Image/LongCat-Image \
  --omni \
  --port 4011 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --served-model-name LongCat-Image