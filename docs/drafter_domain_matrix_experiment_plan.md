# Domain-Specialized EAGLE3 Drafter Experiment Plan

## Objective

Reframe the project as a reproduction and extension of adaptive speculative decoding in our own setup, primarily following MetaSD-style multi-drafter selection. ATLAS is also relevant motivation because it argues that static speculators can fall out of alignment as workload distributions shift, although its solution uses a static/adaptive speculator pair with runtime learning. Instead of optimizing the number of active drafters, we will keep a fixed pool of domain-specialized EAGLE3 drafters and study which reinforcement-learning or bandit selection policy chooses the most useful drafter.

The first required artifact is a domain-by-drafter performance matrix. It should show whether drafters trained on one domain are especially strong on that domain, and whether an all-domain drafter is more robust but less specialized.

## Target and Drafter Pool

- Target verifier `X`: one fixed 7B-8B instruction model compatible with SGLang/SpecForge, e.g. `Llama-3.1-8B-Instruct` if available.
- Base drafter type: EAGLE3 draft head trained for the fixed target model.
- Planned drafters:
  - `eagle3-gsm8k`: trained on math reasoning data.
  - `eagle3-mmlu`: trained on broad knowledge/exam-style QA data.
  - `eagle3-sharegpt`: trained on general instruction/chat data.
  - `eagle3-code` or `eagle3-translation`: fourth domain, to be finalized based on dataset and training stability.
  - `eagle3-all`: trained on the union of the four domain datasets.

`random` is intentionally excluded as a training domain because it is not a coherent specialization target.

## Dataset Plan

Use small controlled subsets first, with separate train/evaluation splits.

| Domain | Train size | Eval size | Dataset candidates | Role |
| --- | ---: | ---: | --- | --- |
| Math | 500 | 100-200 | GSM8K | Structured reasoning specialist |
| Knowledge QA | 500 | 100-200 | MMLU | Broad exam-style QA specialist |
| General chat | 500 | 100-200 | ShareGPT | General instruction specialist |
| Code or translation | 500 | 100-200 | Code Alpaca/HumanEval/MBPP or WMT | Distributionally distinct specialist |
| All-domain | 2000 | N/A | Union of the four train splits | Robust generalist drafter |

Evaluation must use held-out prompts. Training and evaluating on the same 500 examples would confound drafter specialization with memorization.

## Training Budget Matching

All five drafters should receive comparable training budget. Prefer matching by:

1. Total optimizer steps, or
2. Approximate number of training tokens.

If using example counts as a rough proxy:

- Domain-specific drafter: 500 examples repeated for more epochs.
- All-domain drafter: 2000 examples repeated for fewer epochs.

For example, a domain-specific drafter could train for 16 epochs over 500 examples, while `eagle3-all` trains for 4 epochs over 2000 examples. The report should phrase this as equalizing training tokens/steps, because sequence lengths differ by domain.

## Training Pipeline

EAGLE3 is a white-box speculative decoding method, so the drafter is trained using features from the fixed target model. The practical pipeline is:

1. Prepare domain datasets in the format expected by SpecForge.
2. Optionally regenerate assistant responses with the target model `X` so the training data follows the target model's output style.
3. Extract or stream target-model hidden states/logits depending on the SpecForge training mode.
4. Train each domain-specific EAGLE3 drafter from the same base initialization and training recipe.
5. Train `eagle3-all` on the union data with matched training budget.
6. Validate that each trained drafter can be loaded for speculative decoding with the same target model.

The milestone report can state that this training environment and protocol are being established; the actual five full training runs remain next work if not yet completed.

## Domain-by-Drafter Matrix

Main evaluation table:

| Eval domain | eagle3-gsm8k | eagle3-mmlu | eagle3-sharegpt | eagle3-code/trans | eagle3-all |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | TBD | TBD | TBD | TBD | TBD |
| MMLU | TBD | TBD | TBD | TBD | TBD |
| ShareGPT | TBD | TBD | TBD | TBD | TBD |
| Code/translation | TBD | TBD | TBD | TBD | TBD |

Primary metric:

- Average accept length: generated output tokens per target verification step. Fix `speculative-num-draft-tokens=4` for all comparisons.

Secondary metrics:

- Token-level acceptance rate.
- Relative speedup over autoregressive decoding.
- Mean/p50/p95 latency per generated token if available.
- Output quality guardrail where applicable, e.g. exact match for GSM8K, pass@1 for code, or lossless speculative decoding consistency.

Expected pattern:

- A domain-specific drafter should outperform `eagle3-all` on at least some matching domains.
- `eagle3-all` may have the best average robustness across all domains.
- If no specialization effect appears, dynamic selection has little room to help and the project should report that as a negative finding.

## Dynamic Selection Plan

After obtaining the matrix, use all five trained drafters and compare policies that select one drafter at a time. Whether selection happens per request or per speculative decoding step will be decided after checking the runtime integration cost.

- Per-request selection: easier; route each prompt to one drafter for the entire response.
- Per-step selection: closer to MetaSD; select a drafter at each speculative decoding round, but likely requires deeper serving-loop changes.

Tentative policy candidates:

- Uniform random.
- Greedy selection from observed rewards.
- UCB.
- Thompson sampling.
- EXP3 or sliding-window UCB if non-stationarity becomes central.

The final policy list is intentionally left open until the implementation path is clear.

## Logging Schema

Record one JSONL row per request or per speculative block, depending on the selection granularity.

```json
{
  "run_id": "2026-05-17_eagle3_ucb",
  "prompt_id": "gsm8k_eval_000123",
  "domain": "gsm8k",
  "target_model": "llama-3.1-8b-instruct",
  "selection_granularity": "request_or_step_tbd",
  "policy": "ucb",
  "candidate_drafters": ["eagle3-gsm8k", "eagle3-mmlu", "eagle3-sharegpt", "eagle3-code", "eagle3-all"],
  "selected_drafter": "eagle3-gsm8k",
  "step_index": 14,
  "draft_tokens": 4,
  "accepted_tokens": 3,
  "rejected_tokens": 1,
  "draft_time_ms": 9.4,
  "verify_time_ms": 42.7,
  "total_step_time_ms": 54.1,
  "reward": 3.0,
  "seed": 42
}
```

Also store run-level metadata: model paths, training checkpoint IDs, dataset split hashes, decoding parameters, GPU type, batch size, and SpecForge/SGLang versions.

## Milestone Deliverables

- Revised project framing: fixed drafter pool plus RL-based selection, rather than active-set-size minimization.
- Dataset choices and target model choice.
- EAGLE3 training protocol and fair training-budget rule.
- Evaluation matrix format and metrics.
- Remaining tasks: train five EAGLE3 drafters, run the matrix evaluation, decide per-request vs per-step selection, then implement and compare selected RL/bandit policies.

## Risks and Fallbacks

- EAGLE3 training is too expensive: reduce to three drafters, e.g. GSM8K, MMLU, all-domain.
- Runtime cannot switch drafters per step: use per-request routing as the main experiment and offline replay for per-step bandit analysis.
- Target model or EAGLE3 checkpoints are incompatible: choose a smaller supported target model or switch to independent small LM drafters while preserving the same matrix and selection-policy analysis.
- Specialization does not beat `eagle3-all`: report it as evidence that all-domain EAGLE3 is sufficient in this setting and focus selection experiments on robustness/domain shift rather than peak in-domain gains.
