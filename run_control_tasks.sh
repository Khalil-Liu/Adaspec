#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
TASKS="${TASKS:-hellaswag,mmlu}"
METHODS="${METHODS:-base svd_l2 gevd_l2 adaspec}"
BATCH_SIZE="${BATCH_SIZE:-auto:4}"
NUM_FEWSHOT="${NUM_FEWSHOT:-}"
LIMIT="${LIMIT:-}"
OUT_DIR="${OUT_DIR:-results/control_tasks}"
INCLUDE_PATH="${INCLUDE_PATH:-}"

SVD_POS="${SVD_POS:-projectors/svd/no_mean_sub_uu_positive.pt}"
SVD_NEG="${SVD_NEG:-projectors/svd/no_mean_sub_uu_negative.pt}"
GEVD_POS="${GEVD_POS:-projectors/gevd/no_mean_sub_uu_positive.pt}"
GEVD_NEG="${GEVD_NEG:-projectors/gevd/no_mean_sub_uu_negative.pt}"
PROBE="${PROBE:-weights/probe_router/probe_router.pt}"

mkdir -p "$OUT_DIR" logs

run_one() {
  local run_name="$1"
  local evaluator_method="$2"
  shift 2
  local output="$OUT_DIR/${run_name}.json"
  local limit_args=()
  local include_args=()
  local fewshot_args=()
  [[ -n "$LIMIT" ]] && limit_args=(--limit "$LIMIT")
  [[ -n "$INCLUDE_PATH" ]] && include_args=(--include-path "$INCLUDE_PATH")
  [[ -n "$NUM_FEWSHOT" ]] && fewshot_args=(--num-fewshot "$NUM_FEWSHOT")

  date "+[%F %T] START ${run_name}: ${TASKS}"
  CUDA_VISIBLE_DEVICES=0 python evaluate_edited_lm_eval.py \
    --method "$evaluator_method" --run-name "$run_name" --tasks "$TASKS" \
    --model-name "$MODEL" --output-json "$output" --batch-size "$BATCH_SIZE" \
    --max-length 2048 --device cuda --num-gpus 1 --max-gpu-memory 80 \
    "${limit_args[@]}" "${include_args[@]}" "${fewshot_args[@]}" "$@" \
    2>&1 | tee "logs/control_${run_name}.log"
  date "+[%F %T] END ${run_name}"
}

for method in $METHODS; do
  case "$method" in
    base)
      run_one base base
      ;;
    svd_l2)
      run_one svd_l2 sea --sea-positive-proj "$SVD_POS" --sea-negative-proj "$SVD_NEG" \
        --apply-sea-layers last-L --L 2 --combine-sea-embeddings l2_norm
      ;;
    gevd_l2)
      run_one gevd_l2 sea --sea-positive-proj "$GEVD_POS" --sea-negative-proj "$GEVD_NEG" \
        --apply-sea-layers last-L --L 2 --combine-sea-embeddings l2_norm
      ;;
    adaspec)
      run_one adaspec adaspec --svd-positive-proj "$SVD_POS" --svd-negative-proj "$SVD_NEG" \
        --gevd-positive-proj "$GEVD_POS" --gevd-negative-proj "$GEVD_NEG" \
        --probe-path "$PROBE" --lower-threshold 0.40 --upper-threshold 0.60 \
        --apply-sea-layers last-L --L 2 --combine-sea-embeddings l2_norm
      ;;
    *)
      echo "Unknown method: $method" >&2
      exit 2
      ;;
  esac
done

python summarize_control_task_results.py --results-dir "$OUT_DIR" | tee "$OUT_DIR/summary.md"
