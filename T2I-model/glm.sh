export CUDA_VISIBLE_DEVICES=4,5,6,7

vllm serve ./GLM-Image \
--omni \
--port 1070 \
--tensor-parallel-size 4 \
--gpu-memory-utilization 0.8 \
--dtype bfloat16 \
--served-model-name GLM-Image