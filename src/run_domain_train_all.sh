#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPECFORGE_DIR="$ROOT_DIR/reference/code/SpecForge"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT_DIR/.artifacts/rl_project}"
DATA_DIR="${DATA_DIR:-$ARTIFACT_ROOT/data/domain_n1500_m150}"
OUT_ROOT="${OUT_ROOT:-$ARTIFACT_ROOT/outputs/domain_n1500_m150}"
LOG_DIR="${LOG_DIR:-$ARTIFACT_ROOT/results/domain_n1500_m150/logs}"
MAX_STEPS="${MAX_STEPS:-7500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
MAX_LENGTH="${MAX_LENGTH:-512}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-0.6B}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

run_one() {
  local domain="$1"
  local train_data="$2"
  local output_dir="$OUT_ROOT/qwen3-0.6b-eagle3-$domain"
  local final_ckpt
  local log_file="$LOG_DIR/train_$domain.log"

  final_ckpt="$(find "$output_dir" -maxdepth 2 -type f -path "*/epoch_*_step_$MAX_STEPS/model.safetensors" -print -quit 2>/dev/null || true)"

  if [[ -n "$final_ckpt" ]]; then
    echo "[$domain] final checkpoint exists, skipping: $final_ckpt"
    return 0
  fi

  echo "[$domain] training start"
  echo "  train_data=$train_data"
  echo "  output_dir=$output_dir"
  TARGET_MODEL="$TARGET_MODEL" \
  TRAIN_DATA="$train_data" \
  OUTPUT_DIR="$output_dir" \
  MAX_STEPS="$MAX_STEPS" \
  SAVE_INTERVAL="$SAVE_INTERVAL" \
  MAX_LENGTH="$MAX_LENGTH" \
    bash "$ROOT_DIR/src/run_eagle3_train.sh" 2>&1 | tee "$log_file"
  echo "[$domain] training done"
}

run_one "gsm8k" "$DATA_DIR/gsm8k_train.jsonl"
run_one "mmlu" "$DATA_DIR/mmlu_train.jsonl"
run_one "sharegpt" "$DATA_DIR/sharegpt_train.jsonl"
run_one "all" "$DATA_DIR/all_train.jsonl"
