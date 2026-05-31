#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPECFORGE_DIR="$ROOT_DIR/reference/code/SpecForge"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT_DIR/.artifacts/rl_project}"

cd "$SPECFORGE_DIR"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-0.6B}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
PORT="${PORT:-30000}"
MODE="${MODE:-spec}"
EVAL_DATA="${EVAL_DATA:-$ARTIFACT_ROOT/data/gsm8k_1k/gsm8k_eval_100.jsonl}"
OUTPUT_JSON="${OUTPUT_JSON:-$ARTIFACT_ROOT/results/accept_length_${MODE}.json}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
LABEL="${LABEL:-}"

SERVER_ARGS=(
  --model-path "$TARGET_MODEL"
  --cuda-graph-max-bs 1
  --mem-fraction-static 0.25
  --tp-size 1
  --max-running-requests 1
  --trust-remote-code
  --attention-backend flashinfer
  --dtype bfloat16
  --device cuda
  --host localhost
  --port "$PORT"
)

if [[ "$MODE" == "spec" ]]; then
  if [[ -z "$DRAFT_MODEL" ]]; then
    echo "DRAFT_MODEL must be set when MODE=spec" >&2
    exit 1
  fi
  SERVER_ARGS+=(
    --speculative-algorithm EAGLE3
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --speculative-draft-model-path "$DRAFT_MODEL"
  )
fi

python3 -m sglang.launch_server "${SERVER_ARGS[@]}" &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
  wait "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

python - <<'PY'
import os
import time
import requests

port = os.environ.get("PORT", "30000")
url = f"http://localhost:{port}/health_generate"
for _ in range(120):
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200:
            raise SystemExit(0)
    except Exception:
        pass
    time.sleep(2)
raise SystemExit("server did not become healthy")
PY

cd "$ROOT_DIR"
python src/eval_accept_length_jsonl.py \
  --input "$EVAL_DATA" \
  --port "$PORT" \
  --batch-size 1 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT" \
  --label "$LABEL" \
  --output "$OUTPUT_JSON"
