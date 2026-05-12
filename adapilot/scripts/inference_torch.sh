#!/bin/bash
set -e

export PROJECT_ROOT="${PROJECT_ROOT:-/path/to/project}"
cd "${PROJECT_ROOT}"
export BASE_MODEL="${BASE_MODEL:-/path/to/model}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/adapilot/adapilot_train}"
GLOBAL_STEP="${GLOBAL_STEP:-120}"

DATA_ROOT="${DATA_ROOT:-test-data}"

RESULT_ROOT="${RESULT_ROOT:-results/adapilot_results}"

export CUDA_VISIBLE_DEVICES=3,2

export IMAGE_GEN_BASE_URL="${IMAGE_GEN_BASE_URL:-}"
export IMAGE_GEN_SIZE="1024x1024"
export IMAGE_GEN_STEPS=50
export IMAGE_GEN_GUIDANCE=4.5
export IMAGE_GEN_TIMEOUT=300
export IMAGE_GEN_MAX_CONCURRENT=4

export TOKENIZERS_PARALLELISM=true

mkdir -p "${RESULT_ROOT}"

STEP_DIR="${CKPT_DIR}/global_step_${GLOBAL_STEP}"
ACTOR_DIR="${STEP_DIR}/actor"
HF_DIR="${ACTOR_DIR}/huggingface"

if [ ! -d "${STEP_DIR}" ]; then
    echo "[error] checkpoint not found: ${STEP_DIR}"
    exit 1
fi

if [ -f "${HF_DIR}/config.json" ]; then
    echo "[ok] HF checkpoint already exists: ${HF_DIR}"
    MODEL_PATH="${HF_DIR}"
elif ls "${ACTOR_DIR}"/model_world_size_*_rank_0.pt >/dev/null 2>&1; then
    echo "[merge] FSDP shards detected at ${ACTOR_DIR}"
    echo "[merge] running verl/scripts/model_merger.py to produce HF format ..."
    python verl/scripts/model_merger.py merge \
        --backend fsdp \
        --local_dir "${ACTOR_DIR}" \
        --target_dir "${HF_DIR}"
    MODEL_PATH="${HF_DIR}"
else
    echo "[warn] neither HF nor FSDP shards detected under ${ACTOR_DIR}; passing ${STEP_DIR} as-is"
    MODEL_PATH="${STEP_DIR}"
fi

echo "============================================"
echo "BASE_MODEL  : ${BASE_MODEL}"
echo "MODEL_PATH  : ${MODEL_PATH}"
echo "DATA_ROOT   : ${DATA_ROOT}"
echo "RESULT_ROOT : ${RESULT_ROOT}"
echo "GPU         : ${CUDA_VISIBLE_DEVICES}"
echo "============================================"

if [ -f "adapilot/scripts/inference_torch_demo.py" ]; then
    INFER_SCRIPT="adapilot/scripts/inference_torch_demo.py"
elif [ -f "scripts/inference_torch_demo.py" ]; then
    INFER_SCRIPT="scripts/inference_torch_demo.py"
else
    echo "[error] inference_torch_demo.py not found"
    exit 1
fi
echo "INFER_SCRIPT: ${INFER_SCRIPT}"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

python "${INFER_SCRIPT}" \
    --data_root   "${DATA_ROOT}" \
    --result_root "${RESULT_ROOT}" \
    --output_basename "val_raw_step${GLOBAL_STEP}.json" \
    --model_path  "${MODEL_PATH}" \
    --base_model_path "${BASE_MODEL}" \
    --prompt_path adapilot/prompts/round_1_user_prompt_qwenvl.pkl \
    --after_generate_prompt_path adapilot/prompts/user_prompt_after_image_generate.pkl \
    --max_gen_round 6 \
    --max_img_draw_round 3 \
    --max_new_tokens 1024

echo "============================================"
echo "Inference complete. Results saved to: ${RESULT_ROOT}"
echo "============================================"
