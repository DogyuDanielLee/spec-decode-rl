#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPECFORGE_DIR="$ROOT_DIR/reference/code/SpecForge"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT_DIR/.artifacts/rl_project}"
CACHE_DIR="${CACHE_DIR:-$ARTIFACT_ROOT/cache/specforge}"
MODEL_DOWNLOAD_DIR="${MODEL_DOWNLOAD_DIR:-$ARTIFACT_ROOT/cache/hf}"
CONFIG_DIR="${CONFIG_DIR:-$ARTIFACT_ROOT/configs}"

cd "$SPECFORGE_DIR"

mkdir -p "$CACHE_DIR" "$MODEL_DOWNLOAD_DIR" "$CONFIG_DIR" "$ARTIFACT_ROOT/cache/compiled_kernels"

export TORCHINDUCTOR_CACHE_DIR="$ARTIFACT_ROOT/cache/compiled_kernels"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SPECFORGE_DRAFT_CONFIG_DIR="$CONFIG_DIR"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-0.6B}"
TRAIN_DATA="${TRAIN_DATA:-$ARTIFACT_ROOT/data/smoke_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACT_ROOT/outputs/smoke-qwen3-0.6b-eagle3}"
MAX_STEPS="${MAX_STEPS:-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
DRAFT_CONFIG="${DRAFT_CONFIG:-$CONFIG_DIR/$(basename "$TARGET_MODEL" | tr '[:upper:]' '[:lower:]')-eagle3-auto.json}"

python "$ROOT_DIR/src/create_eagle3_draft_config.py" \
  --specforge-dir "$SPECFORGE_DIR" \
  --target-model "$TARGET_MODEL" \
  --output "$DRAFT_CONFIG" \
  --model-download-dir "$MODEL_DOWNLOAD_DIR"

torchrun --standalone --nproc_per_node 1 \
  scripts/train_eagle3.py \
  --target-model-path "$TARGET_MODEL" \
  --draft-model-config "$DRAFT_CONFIG" \
  --train-data-path "$TRAIN_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --num-epochs 10 \
  --max-num-steps "$MAX_STEPS" \
  --total-steps "$MAX_STEPS" \
  --batch-size 1 \
  --tp-size 1 \
  --learning-rate 1e-4 \
  --max-length "$MAX_LENGTH" \
  --chat-template qwen3-instruct \
  --cache-dir "$CACHE_DIR" \
  --attention-backend sdpa \
  --target-model-backend hf \
  --log-interval 1 \
  --save-interval "$MAX_STEPS" \
  --eval-interval "$MAX_STEPS" \
  --seed 42 \
  --model-download-dir "$MODEL_DOWNLOAD_DIR"
