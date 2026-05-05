#!/bin/bash
export PYTHONUNBUFFERED=1

export CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-models/clip-vit-large-patch14}"
export CLIP_SERVER_URL="${CLIP_SERVER_URL:-}"
export CLIP_TIMEOUT=30
export CLIP_MAX_WORKERS=32

export HPS_MODEL_DIR="${HPS_MODEL_DIR:-models/HPSv2}"
export HPS_SERVER_URL="${HPS_SERVER_URL:-}"
export HPS_TIMEOUT=60
export HPS_MAX_WORKERS=32

export GLM_OCR_API_BASE="${GLM_OCR_API_BASE:-}"
export GLM_OCR_FALLBACK_BASE="${GLM_OCR_FALLBACK_BASE:-}"
export GLM_OCR_FALLBACK_BASE_2="${GLM_OCR_FALLBACK_BASE_2:-}"
export GLM_OCR_MODEL="glm-ocr"

export GDINO_MODEL_PATH="${GDINO_MODEL_PATH:-models/GroundingDINO}"
export GDINO_SERVER_URL="${GDINO_SERVER_URL:-}"
export GDINO_TIMEOUT=60
export GDINO_MAX_WORKERS=8

export USE_KIMI="${USE_KIMI:-0}"
export KIMI_API_BASE="${KIMI_API_BASE:-}"
export KIMI_MODEL="${KIMI_MODEL:-kimi-k2.6}"
export KIMI_MAX_WORKERS="${KIMI_MAX_WORKERS:-16}"
export KIMI_CACHE_DIR="${KIMI_CACHE_DIR:-kimi_cache}"

export VLM_API_BASE="${VLM_API_BASE:-}"
export VLM_MODEL="${VLM_MODEL:-}"
export LLM_OCR_MAX_WORKERS=64
export LLM_AESTHETIC_MAX_WORKERS=64
export LLM_AESTHETIC_MAX_RETRIES=3
export VLM_ENTITY_MAX_RETRIES=3

RESULT_ROOT="${RESULT_ROOT:-results/adapilot_results}"
WORKERS="${WORKERS:-64}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

USE_LLM_OCR="--use_llm_ocr"
USE_GDINO="--use_gdino"
USE_AESTHETIC=""
USE_REALISM_PHYSICS=""

set -e
cd "$(dirname "$0")"

echo "============================================"
echo "Result root     : ${RESULT_ROOT}"
echo "LLM OCR         : ${USE_LLM_OCR}"
echo "GDINO           : ${USE_GDINO}"
echo "Aesthetic       : ${USE_AESTHETIC}"
echo "Realism&Physics : ${USE_REALISM_PHYSICS}"
echo "Workers         : ${WORKERS}"
echo "============================================"

datasets=()
for json_file in ${RESULT_ROOT}/*/baseline_raw.json ${RESULT_ROOT}/*/val_raw*.json; do
    [ -f "$json_file" ] && datasets+=("$json_file")
done

if [ ${#datasets[@]} -eq 0 ]; then
    echo "[error] No result files found in ${RESULT_ROOT}"
    exit 1
fi

echo "Found ${#datasets[@]} datasets:"
for f in "${datasets[@]}"; do
    echo "  $f"
done
echo ""

SUCCESS=0
FAIL=0
for json_file in "${datasets[@]}"; do
    dataset_dir=$(dirname "$json_file")
    dataset_name=$(basename "$dataset_dir")
    output_path="${dataset_dir}/eval_metrics.json"

    echo ""
    echo "--------------------------------------------"
    echo "[eval] Dataset: ${dataset_name}"
    echo "  Input : ${json_file}"
    echo "  Output: ${output_path}"
    echo "--------------------------------------------"

    if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${output_path}" ]; then
        echo "  [skip] eval_metrics.json already exists"
        SUCCESS=$((SUCCESS+1))
        continue
    fi

    python eval_metrics.py \
        --json_path   "${json_file}" \
        --output_path "${output_path}" \
        --workers     ${WORKERS} \
        ${USE_LLM_OCR} \
        ${USE_GDINO} \
        ${USE_AESTHETIC} \
        ${USE_REALISM_PHYSICS}

    if [ $? -eq 0 ]; then
        SUCCESS=$((SUCCESS+1))
    else
        echo "[warn] ${dataset_name} evaluation failed"
        FAIL=$((FAIL+1))
    fi
done

echo ""
echo "============================================"
echo "Evaluation complete: ${SUCCESS} succeeded / ${FAIL} failed"
echo ""
echo "Result files:"
for json_file in "${datasets[@]}"; do
    dataset_dir=$(dirname "$json_file")
    dataset_name=$(basename "$dataset_dir")
    metrics_file="${dataset_dir}/eval_metrics.json"
    if [ -f "$metrics_file" ]; then
        echo "  OK ${dataset_name}: ${metrics_file}"
    else
        echo "  FAIL ${dataset_name}: not generated"
    fi
done
echo "============================================"

echo ""
echo "Generating summary table..."
python - << PYEOF
import os, json, glob
import pandas as pd

result_root = "${RESULT_ROOT}"
all_rows = []

for metrics_file in sorted(glob.glob(os.path.join(result_root, "*/eval_metrics.json"))):
    dataset_name = os.path.basename(os.path.dirname(metrics_file))
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    s = data.get('summary', {}).get('__all__', {})
    if s:
        row = {'dataset': dataset_name}
        row.update(s)
        all_rows.append(row)

if all_rows:
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(result_root, "all_datasets_summary.xlsx")
    df.to_excel(out_path, index=False)
    print(f"[summary] Saved: {out_path}")
    print(df.to_string(index=False))
else:
    print("[summary] No evaluation results found")
PYEOF
