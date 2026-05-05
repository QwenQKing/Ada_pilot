#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

export BASE_MODEL="${BASE_MODEL:-/path/to/model}"
export CKPT_DIR="${CKPT_DIR:-/path/to/checkpoints/adapilot/adapilot_train}"
export PROJECT_NAME='adapilot'
export EXPERIMENT_NAME='adapilot_eval'

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RAY_PORT="${RAY_PORT:-6566}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8268}"
export RAY_TEMP_DIR=/tmp/ray_adapilot_eval

export DATA_ROOT="${DATA_ROOT:-adapilot/data}"
export RESULT_ROOT="${RESULT_ROOT:-adapilot/results/eval}"

export FORCE_RERUN="${FORCE_RERUN:-0}"

export IMAGE_GEN_BASE_URL="${IMAGE_GEN_BASE_URL:-}"
export IMAGE_GEN_SIZE="1024x1024"
export IMAGE_GEN_STEPS=50
export IMAGE_GEN_GUIDANCE=4.5
export IMAGE_GEN_TIMEOUT=300
export IMAGE_GEN_MAX_CONCURRENT=4

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_ALGO=Ring
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true

export RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"/tmp/adapilot/ray_spill"}}'

mkdir -p /tmp/adapilot/ray_spill
mkdir -p "${RESULT_ROOT}"

set -xe

for PORT in ${RAY_PORT} ${RAY_DASHBOARD_PORT}; do
    OLD_PID=$(lsof -ti :"${PORT}" 2>/dev/null || true)
    if [ -n "${OLD_PID}" ]; then
        kill -9 "${OLD_PID}" 2>/dev/null || true
        sleep 2
    fi
done
rm -rf "${RAY_TEMP_DIR}"
mkdir -p "${RAY_TEMP_DIR}"

unset RAY_ADDRESS
NODE_IP=$(hostname -i | awk '{print $1}')

RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-0}"
if [ "${RAY_INCLUDE_DASHBOARD}" = "1" ]; then
    ray start --head \
        --node-ip-address="${NODE_IP}" \
        --num-cpus=8 \
        --num-gpus=4 \
        --port="${RAY_PORT}" \
        --include-dashboard=true \
        --dashboard-port="${RAY_DASHBOARD_PORT}" \
        --temp-dir="${RAY_TEMP_DIR}" \
        --object-store-memory=8000000000 \
        --disable-usage-stats
else
    ray start --head \
        --node-ip-address="${NODE_IP}" \
        --num-cpus=8 \
        --num-gpus=4 \
        --port="${RAY_PORT}" \
        --include-dashboard=false \
        --temp-dir="${RAY_TEMP_DIR}" \
        --object-store-memory=8000000000 \
        --disable-usage-stats
fi

export RAY_ADDRESS="${NODE_IP}:${RAY_PORT}"

sleep 15
ray status --address="${NODE_IP}:${RAY_PORT}" || { echo "Ray startup failed"; ray stop --force; exit 1; }

DATASETS=()
PARQUETS=()
for subdir in $(ls -d "${DATA_ROOT}"/*/); do
    dataset_name=$(basename "${subdir}")
    if [ "${dataset_name}" = "mixed" ] || [ "${dataset_name}" = "datatxt" ]; then
        echo "[Skip] ${dataset_name}"
        continue
    fi
    parquet=$(find "${subdir}" -maxdepth 1 -name "*_test.parquet" 2>/dev/null | head -1)
    if [ -z "${parquet}" ]; then
        parquet=$(find "${subdir}" -maxdepth 1 -name "*_test.pq" 2>/dev/null | head -1)
    fi
    DATASETS+=("${dataset_name}")
    PARQUETS+=("${parquet}")
    echo "[Found] ${dataset_name} <- ${parquet}"
done

echo "============================================"
if [ -n "${DATASETS_FILTER:-}" ]; then
    IFS="," read -ra _FILTER <<< "${DATASETS_FILTER}"
    _FILTERED_DS=(); _FILTERED_PQ=()
    for i in "${!DATASETS[@]}"; do
        for f in "${_FILTER[@]}"; do
            if [ "${DATASETS[$i]}" = "${f}" ]; then
                _FILTERED_DS+=("${DATASETS[$i]}")
                _FILTERED_PQ+=("${PARQUETS[$i]}")
                break
            fi
        done
    done
    DATASETS=("${_FILTERED_DS[@]}")
    PARQUETS=("${_FILTERED_PQ[@]}")
    echo "[FILTER] DATASETS_FILTER=${DATASETS_FILTER}, remaining: ${#DATASETS[@]}"
fi
echo "Found ${#DATASETS[@]} datasets"
echo "============================================"

SKIPPED=()
RAN=()
for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    TEST_DATA="${PARQUETS[$i]}"
    SAVE_DIR="${RESULT_ROOT}/${DATASET}"
    OUTPUT_JSON="${SAVE_DIR}/eval_output.json"

    if [ "${FORCE_RERUN}" != "1" ] && [ -f "${OUTPUT_JSON}" ]; then
        echo "[Skip] ${DATASET} already done -> ${OUTPUT_JSON}"
        SKIPPED+=("${DATASET}")
        continue
    fi
    RAN+=("${DATASET}")

    mkdir -p "${SAVE_DIR}"
    mkdir -p "${SAVE_DIR}/generated_images"

    echo "============================================"
    echo "[Run] ${DATASET} <- ${TEST_DATA}"
    echo "============================================"

    python3 -m adapilot.trainer.multimodal.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="${TEST_DATA}" \
        data.val_files="${TEST_DATA}" \
        data.train_batch_size=32 \
        data.max_prompt_length=8192 \
        data.max_response_length=8192 \
        data.image_key=images \
        data.user_prompt_round_1=adapilot/prompts/round_1_user_prompt_qwenvl.pkl \
        +actor_rollout_ref.rollout.generate.user_prompt_after_image_generate=adapilot/prompts/user_prompt_after_image_generate.pkl \
        actor_rollout_ref.model.path="${BASE_MODEL}" \
        actor_rollout_ref.actor.optim.lr=2e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_multi_turn_response_mask=True \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm_multiturn_adapilot \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.max_gen_round=5 \
        actor_rollout_ref.rollout.response_length_total=16384 \
        actor_rollout_ref.rollout.max_model_len=32768 \
        actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
        +actor_rollout_ref.rollout.generate.image_generate_limit=3 \
        +actor_rollout_ref.rollout.generate.parallel_tool_call=True \
        +actor_rollout_ref.rollout.generate.parallel_tool_call_threads=8 \
        +actor_rollout_ref.rollout.generate.image_save_dir="${SAVE_DIR}/generated_images" \
        +actor_rollout_ref.rollout.generate.max_concurrent=16 \
        algorithm.kl_ctrl.kl_coef=0.0 \
        trainer.critic_warmup=0 \
        'trainer.logger=[console]' \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${EXPERIMENT_NAME}" \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=99999 \
        trainer.test_freq=1 \
        trainer.total_epochs=1 \
        trainer.default_local_dir="${CKPT_DIR}" \
        trainer.default_hdfs_dir=null \
        trainer.resume_mode=auto \
        +trainer.val_before_train=True \
        trainer.val_only=True \
        +trainer.val_only_save_dir="${SAVE_DIR}" \
        trainer.val_generations_to_log_to_wandb=0 \
        +algorithm.filter_groups.enable=False

    echo "[Done] ${DATASET} -> ${OUTPUT_JSON}"
done

ray stop --force

echo "============================================"
echo "Inference complete: ran ${#RAN[@]}, skipped ${#SKIPPED[@]}"
[ ${#RAN[@]} -gt 0 ]     && echo "  Ran    : ${RAN[*]}"
[ ${#SKIPPED[@]} -gt 0 ] && echo "  Skipped: ${SKIPPED[*]}"
echo "Results: ${RESULT_ROOT}/"
echo "Next   : bash adapilot/eval_score.sh"
echo "============================================"
