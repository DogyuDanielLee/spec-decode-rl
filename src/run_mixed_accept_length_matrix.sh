#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPECFORGE_DIR="$ROOT_DIR/reference/code/SpecForge"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT_DIR/.artifacts/rl_project}"
DATA_DIR="${DATA_DIR:-$ARTIFACT_ROOT/data/domain_n1500_m150}"
CKPT_ROOT="${CKPT_ROOT:-$ARTIFACT_ROOT/outputs/domain_n1500_m150}"
RESULT_DIR="${RESULT_DIR:-$ARTIFACT_ROOT/results/domain_n1500_m150/accept_length_mixed}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-0.6B}"
MAX_STEPS="${MAX_STEPS:-7500}"
LIMIT="${LIMIT:-450}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PORT="${PORT:-30000}"

mkdir -p "$RESULT_DIR"

run_eval() {
  local label="$1"
  local draft_model="$2"
  local output_json="$RESULT_DIR/${label}.json"

  if [[ -f "$output_json" ]]; then
    echo "[$label] result exists, skipping: $output_json"
    return 0
  fi

  echo "[$label] eval start"
  TARGET_MODEL="$TARGET_MODEL" \
  DRAFT_MODEL="$draft_model" \
  MODE=spec \
  LABEL="$label" \
  EVAL_DATA="$DATA_DIR/mixed_eval.jsonl" \
  OUTPUT_JSON="$output_json" \
  LIMIT="$LIMIT" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  PORT="$PORT" \
    bash "$ROOT_DIR/src/run_accept_length_server_eval.sh"
  echo "[$label] eval done"
}

find_ckpt() {
  local domain="$1"
  local ckpt
  ckpt="$(find "$CKPT_ROOT/qwen3-0.6b-eagle3-$domain" -maxdepth 2 -type f -path "*/epoch_*_step_$MAX_STEPS/model.safetensors" -printf '%h\n' -quit 2>/dev/null || true)"
  if [[ -z "$ckpt" ]]; then
    echo "No checkpoint found for $domain at step $MAX_STEPS" >&2
    exit 1
  fi
  echo "$ckpt"
}

run_eval "eagle3_gsm8k" "$(find_ckpt gsm8k)"
run_eval "eagle3_mmlu" "$(find_ckpt mmlu)"
run_eval "eagle3_sharegpt" "$(find_ckpt sharegpt)"
run_eval "eagle3_all" "$(find_ckpt all)"

python "$ROOT_DIR/src/summarize_accept_length_matrix.py" \
  --result eagle3_gsm8k "$RESULT_DIR/eagle3_gsm8k.json" \
  --result eagle3_mmlu "$RESULT_DIR/eagle3_mmlu.json" \
  --result eagle3_sharegpt "$RESULT_DIR/eagle3_sharegpt.json" \
  --result eagle3_all "$RESULT_DIR/eagle3_all.json" \
  --output-dir "$RESULT_DIR"
