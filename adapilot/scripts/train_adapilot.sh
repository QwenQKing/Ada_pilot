#!/bin/bash

export BASE_MODEL="${BASE_MODEL:-/path/to/model}"
export PROJECT_NAME='adapilot'
export EXPERIMENT_NAME='adapilot_train'
export CUDA_VISIBLE_DEVICES=0,1
export RAY_PORT=6565
export RAY_DASHBOARD_PORT=8266
export RAY_TEMP_DIR=/tmp/ray_adapilot
export TRAIN_DATA_PATH=train-data/train.parquet
export VAL_DATA_PATH=train-data/val.parquet

export IMAGE_GEN_BASE_URL="${IMAGE_GEN_BASE_URL:-}"
export IMAGE_GEN_SIZE="1024x1024"
export IMAGE_GEN_STEPS=50
export IMAGE_GEN_GUIDANCE=4.5
export IMAGE_GEN_TIMEOUT=300
export IMAGE_GEN_MAX_CONCURRENT=4

export CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-models/clip-vit-large-patch14}"
export CLIP_SERVER_URL="${CLIP_SERVER_URL:-}"
export CLIP_TIMEOUT=30
export CLIP_MAX_WORKERS=32
export HPS_MODEL_DIR="${HPS_MODEL_DIR:-models/HPSv2}"
export HPS_SERVER_URL="${HPS_SERVER_URL:-}"
export HPS_TIMEOUT=60
export HPS_MAX_WORKERS=32
export GLM_OCR_API_BASE="${GLM_OCR_API_BASE:-}"
export GLM_OCR_MODEL="glm-ocr"
export GDINO_MODEL_PATH="${GDINO_MODEL_PATH:-models/GroundingDINO}"
export GDINO_SERVER_URL="${GDINO_SERVER_URL:-}"
export GDINO_TIMEOUT=60
export GDINO_MAX_WORKERS=8
export VLM_API_BASE="${VLM_API_BASE:-}"
export VLM_MODEL="${VLM_MODEL:-}"
export LLM_OCR_MAX_WORKERS=64
export LLM_AESTHETIC_MAX_WORKERS=64
export LLM_AESTHETIC_MAX_RETRIES=3
export VLM_ENTITY_MAX_RETRIES=3
export GDINO_MAX_RETRIES=3

export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR:-/tmp/adapilot/wandb}"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_ALGO=Ring
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true

export RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"/tmp/adapilot/ray_spill"}}'

mkdir -p /tmp/adapilot/ray_spill

OLD_PID=$(lsof -ti :${RAY_PORT} 2>/dev/null || true)
if [ -n "$OLD_PID" ]; then
    echo "Port ${RAY_PORT} is occupied (PID: ${OLD_PID}), cleaning up..."
    kill -9 $OLD_PID 2>/dev/null || true
    sleep 2
fi

rm -rf ${RAY_TEMP_DIR}
mkdir -p ${RAY_TEMP_DIR}

set -xe

unset RAY_ADDRESS

NODE_IP=$(hostname -i | awk '{print $1}')

ray start --head \
    --node-ip-address=${NODE_IP} \
    --num-cpus=8 \
    --num-gpus=4 \
    --port=${RAY_PORT} \
    --include-dashboard=true \
    --dashboard-port=${RAY_DASHBOARD_PORT} \
    --temp-dir=${RAY_TEMP_DIR} \
    --object-spilling-directory=${SPILL_DIR} \
    --object-store-memory=8000000000

export RAY_ADDRESS="${NODE_IP}:${RAY_PORT}"

sleep 10

ray status --address="${NODE_IP}:${RAY_PORT}" || { echo "Ray startup failed, exiting"; exit 1; }

python3 -m adapilot.trainer.multimodal.main_ppo \
    +actor_rollout_ref.rollout.generate.max_concurrent=16 \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$VAL_DATA_PATH \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.image_key=images \
    data.user_prompt_round_1=adapilot/prompts/round_1_user_prompt_qwenvl.pkl \
    +actor_rollout_ref.rollout.generate.user_prompt_after_image_generate=adapilot/prompts/user_prompt_after_image_generate.pkl \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_warmup=True \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_ratio=0.95 \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_warmup_steps=45 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_multi_turn_response_mask=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_multiturn_adapilot \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_gen_round=6 \
    actor_rollout_ref.rollout.response_length_total=16384 \
    +actor_rollout_ref.rollout.generate.image_generate_limit=4 \
    +actor_rollout_ref.rollout.generate.parallel_tool_call=True \
    +actor_rollout_ref.rollout.generate.parallel_tool_call_threads=8 \
    +actor_rollout_ref.rollout.generate.image_save_dir=adapilot/data/mixed/generated_images \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    'trainer.logger=[console,wandb]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    +trainer.max_actor_ckpt_to_keep=3 \
    +trainer.max_critic_ckpt_to_keep=3 \
    trainer.test_freq=20 \
    trainer.total_epochs=2 \
    +trainer.val_before_train=False \
    +algorithm.filter_groups.enable=False
