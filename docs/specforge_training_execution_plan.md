# SpecForge EAGLE3 Training Execution Plan

This plan turns the revised project into concrete training/evaluation steps using the local SpecForge copy at `reference/code/SpecForge/`.

## 0. Scope

Goal: train a controlled pool of EAGLE3 drafters for one fixed target model, then evaluate the domain-by-drafter matrix.

Initial target:

- Target model `X`: `meta-llama/Llama-3.1-8B-Instruct` if access and GPU memory permit.
- Draft config: `reference/code/SpecForge/configs/llama3-8B-eagle3.json`.
- Chat template: `llama3`.
- Training mode: start with `online` for smoke tests; decide between online/offline for full runs after measuring disk/GPU constraints.

Planned drafters:

- `eagle3-gsm8k`
- `eagle3-mmlu`
- `eagle3-sharegpt`
- `eagle3-code` or `eagle3-translation`
- `eagle3-all`

## 1. Key SpecForge Facts

EAGLE3 is feature-space training. The draft model does not only see text; it learns from target-model hidden states.

SpecForge supports:

- Online training: target hidden states are generated during training. Low disk, higher GPU memory.
- Offline training: hidden states are generated first and stored on disk. High disk, lower GPU memory during training.

For this project, online training is easier to start because our datasets are small and it avoids a hidden-state storage management step. Offline training is useful if target-model forward passes dominate repeated training runs or if online memory usage is unstable.

## 2. Directory Layout

Use this layout under the repo root:

```text
reference/code/SpecForge/
  cache/dataset/rl_project/
    gsm8k_train.jsonl
    gsm8k_eval.jsonl
    mmlu_train.jsonl
    mmlu_eval.jsonl
    sharegpt_train.jsonl
    sharegpt_eval.jsonl
    code_train.jsonl
    code_eval.jsonl
    all_train.jsonl
  cache/hidden_states/rl_project/
    ...
  outputs/rl_project/
    eagle3-gsm8k/
    eagle3-mmlu/
    eagle3-sharegpt/
    eagle3-code/
    eagle3-all/
  results/rl_project/
```

Keep all run metadata in `outputs/rl_project/<run>/run_manifest.json`.

## 3. Environment Setup

From `reference/code/SpecForge/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -U sglang datasets transformers accelerate
```

Sanity checks:

```bash
python scripts/prepare_data.py --help
python scripts/train_eagle3.py --help
python benchmarks/bench_eagle3.py --help
```

If CUDA/FlashAttention packages fail, record the exact environment and fall back to the attention backend already used by the Llama example:

```bash
--attention-backend sdpa
```

## 4. Dataset Preparation

### 4.1 Supported Datasets

SpecForge already supports:

- `gsm8k`
- `sharegpt`
- `codealpaca-20k`

MMLU is available in the benchmark code, but not currently listed as a direct `prepare_data.py --dataset` training option. For MMLU training, prepare a custom JSONL in SpecForge conversation format.

SpecForge conversation format:

```json
{
  "id": "unique_id",
  "conversations": [
    {"role": "user", "content": "question or instruction"},
    {"role": "assistant", "content": "target answer"}
  ]
}
```

### 4.2 Generate Raw Domain Files

Run built-in processors where available:

```bash
cd reference/code/SpecForge

python scripts/prepare_data.py --dataset gsm8k --sample-size 700 \
  --output-path cache/dataset/rl_project/raw_gsm8k

python scripts/prepare_data.py --dataset sharegpt --sample-size 700 \
  --output-path cache/dataset/rl_project/raw_sharegpt

python scripts/prepare_data.py --dataset codealpaca-20k --sample-size 700 \
  --output-path cache/dataset/rl_project/raw_code
```

For MMLU, write a small conversion script that loads MMLU and writes 700 examples as conversations:

- User content: subject + question + options.
- Assistant content: correct answer letter and optionally the full answer.

Split each domain into:

- 500 train examples.
- 100-200 eval examples.

Use a fixed seed and save split IDs.

### 4.3 Decide Whether to Regenerate Responses

Recommended first full run: do not regenerate. This removes one moving part.

Optional stronger run: regenerate assistant responses with target model `X` to align data to the target model distribution.

Launch SGLang target server:

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --cuda-graph-bs 1 2 4 8 16 32 \
  --dtype bfloat16 \
  --mem-frac 0.8 \
  --port 30000
```

Regenerate one domain:

```bash
python scripts/regenerate_train_data.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --concurrency 32 \
  --max-tokens 1024 \
  --server-address localhost:30000 \
  --temperature 0.8 \
  --input-file-path cache/dataset/rl_project/gsm8k_train.jsonl \
  --output-file-path cache/dataset/rl_project/gsm8k_train_regen.jsonl \
  --resume
```

If using regenerated training data, use regenerated versions consistently for all domain drafters and `all_train.jsonl`.

## 5. Training Strategy

### 5.1 Smoke Test First

Before any 500-example run, run a 10-20 example smoke test.

```bash
torchrun --standalone --nproc_per_node 1 \
  scripts/train_eagle3.py \
  --target-model-path meta-llama/Llama-3.1-8B-Instruct \
  --draft-model-config configs/llama3-8B-eagle3.json \
  --train-data-path cache/dataset/rl_project/gsm8k_train_20.jsonl \
  --output-dir outputs/rl_project/smoke-eagle3-gsm8k \
  --num-epochs 1 \
  --max-num-steps 20 \
  --batch-size 1 \
  --tp-size 1 \
  --learning-rate 1e-4 \
  --max-length 2048 \
  --chat-template llama3 \
  --cache-dir cache \
  --attention-backend sdpa \
  --target-model-backend sglang \
  --log-interval 1 \
  --save-interval 20 \
  --sglang-mem-fraction-static 0.25
```

Smoke-test pass criteria:

- Dataset builds.
- Target model loads.
- Loss is finite.
- Checkpoint is written.
- The checkpoint can be loaded by SGLang for EAGLE3 decoding.

### 5.2 Full Online Training

Use `--max-num-steps` to equalize training budget across all drafters. This is better than matching epochs because sequence lengths differ by domain.

Initial conservative budget:

- `max_num_steps`: 2000 for every drafter.
- `batch_size`: 1.
- `max_length`: 2048 first; increase to 4096 only if memory allows.
- `learning_rate`: 1e-4 from SpecForge Llama example.
- `seed`: fixed, e.g. 42.

Command template:

```bash
DOMAIN=gsm8k
TRAIN_PATH=cache/dataset/rl_project/${DOMAIN}_train.jsonl
OUT=outputs/rl_project/eagle3-${DOMAIN}

torchrun --standalone --nproc_per_node 1 \
  scripts/train_eagle3.py \
  --target-model-path meta-llama/Llama-3.1-8B-Instruct \
  --draft-model-config configs/llama3-8B-eagle3.json \
  --train-data-path ${TRAIN_PATH} \
  --output-dir ${OUT} \
  --num-epochs 100 \
  --max-num-steps 2000 \
  --total-steps 2000 \
  --batch-size 1 \
  --tp-size 1 \
  --learning-rate 1e-4 \
  --max-length 2048 \
  --chat-template llama3 \
  --cache-dir cache \
  --attention-backend sdpa \
  --target-model-backend sglang \
  --log-interval 10 \
  --save-interval 500 \
  --eval-interval 500 \
  --seed 42 \
  --sglang-mem-fraction-static 0.25
```

Run for:

```bash
DOMAIN=gsm8k
DOMAIN=mmlu
DOMAIN=sharegpt
DOMAIN=code   # or translation
DOMAIN=all
```

For `all`, set:

```bash
TRAIN_PATH=cache/dataset/rl_project/all_train.jsonl
OUT=outputs/rl_project/eagle3-all
```

### 5.3 Offline Training Alternative

Use offline mode if online training OOMs or if hidden states can be reused across repeated runs.

Generate hidden states:

```bash
DOMAIN=gsm8k

torchrun --standalone --nproc_per_node 1 \
  scripts/prepare_hidden_states.py \
  --target-model-path meta-llama/Llama-3.1-8B-Instruct \
  --enable-aux-hidden-states \
  --data-path cache/dataset/rl_project/${DOMAIN}_train.jsonl \
  --output-path cache/hidden_states/rl_project/${DOMAIN}_train_llama3_8b \
  --chat-template llama3 \
  --max-length 2048 \
  --tp-size 1 \
  --batch-size 8
```

Train using stored hidden states:

```bash
torchrun --standalone --nproc_per_node 1 \
  scripts/train_eagle3.py \
  --target-model-path meta-llama/Llama-3.1-8B-Instruct \
  --draft-model-config configs/llama3-8B-eagle3.json \
  --train-data-path cache/dataset/rl_project/${DOMAIN}_train.jsonl \
  --train-hidden-states-path cache/hidden_states/rl_project/${DOMAIN}_train_llama3_8b \
  --output-dir outputs/rl_project/eagle3-${DOMAIN}-offline \
  --max-num-steps 2000 \
  --total-steps 2000 \
  --batch-size 1 \
  --tp-size 1 \
  --target-model-backend sglang \
  --learning-rate 1e-4 \
  --max-length 2048 \
  --chat-template llama3 \
  --cache-dir cache \
  --seed 42
```

## 6. Evaluation Matrix

Use `benchmarks/bench_eagle3.py` for first-pass evaluation. It can launch SGLang with a single EAGLE3 drafter checkpoint and run benchmarks such as `gsm8k`, `mmlu`, `humaneval`, `math500`, and `mtbench`.

Example:

```bash
python benchmarks/bench_eagle3.py \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path outputs/rl_project/eagle3-gsm8k \
  --port 30000 \
  --config-list 1,0,0,0 1,3,1,4 \
  --benchmark-list gsm8k:200 mmlu:200 mtbench:80 humaneval:100 \
  --dtype bfloat16 \
  --tp-size 1 \
  --trust-remote-code \
  --name eagle3-gsm8k \
  --output-dir results/rl_project
```

Repeat for:

- `outputs/rl_project/eagle3-gsm8k`
- `outputs/rl_project/eagle3-mmlu`
- `outputs/rl_project/eagle3-sharegpt`
- `outputs/rl_project/eagle3-code`
- `outputs/rl_project/eagle3-all`

Matrix rows:

- GSM8K
- MMLU
- ShareGPT/MT-Bench proxy
- Code/HumanEval or translation

Matrix columns:

- five trained drafters.

Primary reported metrics:

- average accept length, i.e. generated output tokens per target verification step.
- relative speedup / tokens per second.
- token-level acceptance rate, if exposed.

Fix `speculative-num-draft-tokens=4` for all runs. If `bench_eagle3.py` does not expose accept length directly, add a small parser around SGLang logs or modify the benchmarker to capture speculative counters. Do not rely only on wall-clock TPS unless the runs are stable and repeated.

## 7. Dynamic Selection Preparation

Do not implement dynamic selection before the matrix exists.

After the matrix:

1. Start with request-level routing.
2. Treat each request as one bandit round.
3. Reward: request-level average accept length or speedup proxy.
4. Compare uniform, greedy, UCB; add Thompson/EXP3 only if time allows.
5. Per-step selection remains a stretch goal because SGLang currently takes one `--speculative-draft-model-path` per server process.

Request-level implementation option:

- Run one SGLang server per drafter on different ports, or evaluate traces by launching one drafter at a time.
- Router chooses the drafter/server for each prompt.
- Log selected drafter, domain, reward, latency, and output length.

Offline replay option:

- First collect per-request metrics for every drafter and every prompt.
- Replay bandit policies over the table.
- This gives a clean algorithm comparison even before runtime multi-server routing is robust.

## 8. Milestones and Go/No-Go Gates

### Gate A: Environment

- `train_eagle3.py --help` works.
- `bench_eagle3.py --help` works.
- Target model can load in SGLang.

### Gate B: Data

- Four domain train/eval JSONL files exist.
- `all_train.jsonl` is the concatenation of the four train splits.
- Random seed and split IDs are saved.

### Gate C: Training Smoke Test

- 20-step EAGLE3 training succeeds.
- Checkpoint loads for speculative decoding.

### Gate D: Full Training

- Five drafters trained with same `max_num_steps` or same approximate training-token budget.
- Training loss curves saved.

### Gate E: Matrix

- Each drafter evaluated on each domain.
- Matrix generated.
- Decide whether specialization effect is strong enough for dynamic selection.

### Gate F: RL Selection

- If specialization is visible: implement request-level bandit first.
- If specialization is weak: analyze why `eagle3-all` dominates and treat selection as robustness/domain-shift study.

## 9. Immediate Next Commands

Start with:

```bash
cd reference/code/SpecForge
python scripts/prepare_data.py --dataset gsm8k --sample-size 700 --output-path cache/dataset/rl_project/raw_gsm8k
python scripts/prepare_data.py --dataset sharegpt --sample-size 700 --output-path cache/dataset/rl_project/raw_sharegpt
python scripts/prepare_data.py --dataset codealpaca-20k --sample-size 700 --output-path cache/dataset/rl_project/raw_code
```

Then create the custom MMLU JSONL and split all four domains into the final `*_train.jsonl` and `*_eval.jsonl` files.

Only after the four files are valid, run the 20-step smoke test.
