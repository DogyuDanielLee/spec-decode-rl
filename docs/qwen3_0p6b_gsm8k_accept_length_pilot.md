# Qwen3-0.6B GSM8K Accept-Length Pilot

Date: 2026-05-17

## Goal

Run a slightly larger local pilot than the 5-step smoke test and check whether EAGLE3 training improves average accept length on held-out GSM8K examples.

Average accept length is defined as:

```text
generated output tokens / target verification steps
```

All speculative runs use 4 draft tokens.

## Setup

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB VRAM.
- Target model: `Qwen/Qwen3-0.6B`.
- Drafter method: EAGLE3 through SpecForge.
- Dataset: GSM8K train split, first 1000 examples after deterministic shuffle.
- Train split: 900 examples.
- Eval split: 100 held-out examples.
- Training max sequence length: 512.
- Eval max new tokens: 128.
- Eval server: SGLang, `--speculative-num-draft-tokens 4`, batch size 1.

Data files:

- `reference/code/SpecForge/cache/dataset/rl_project/gsm8k_1k/gsm8k_train_900.jsonl`
- `reference/code/SpecForge/cache/dataset/rl_project/gsm8k_1k/gsm8k_eval_100.jsonl`

Checkpoints:

- Step-1 reference: `reference/code/SpecForge/outputs/rl_project/qwen3-0.6b-eagle3-gsm8k-step1/epoch_0_step_1`
- Step-900 pilot: `reference/code/SpecForge/outputs/rl_project/qwen3-0.6b-eagle3-gsm8k-900/epoch_0_step_900`

## Result

| Drafter checkpoint | Eval examples | Completion tokens | Target verify steps | Average accept length | Output throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| Step 1 | 100 | 12800 | 12691 | 1.0086 | 69.35 tok/s |
| Step 900 | 100 | 12800 | 10496 | 1.2195 | 90.41 tok/s |

Saved result JSON:

- `reference/code/SpecForge/results/rl_project/qwen3_0p6b_gsm8k_step1_accept_length.json`
- `reference/code/SpecForge/results/rl_project/qwen3_0p6b_gsm8k_step900_accept_length.json`

## Interpretation

The 900-step GSM8K-trained EAGLE3 drafter improved average accept length from 1.0086 to 1.2195 on the held-out GSM8K eval set. This is a useful milestone result: the local RTX 4060 setup can train a small EAGLE3 drafter, serve it with SGLang, and observe a measurable accept-length improvement.

This is still only a pilot. It does not yet show domain specialization because it compares two training stages of one GSM8K drafter, not a matrix across GSM8K, MMLU, ShareGPT, and code/translation drafters.

## Notes

- `Qwen/Qwen3-1.7B` target EAGLE3 training was attempted first but ran out of memory on the 8GB GPU during backward pass.
- `google/gemma-3-1b-it` was not usable without gated Hugging Face access.
- For this machine, `Qwen/Qwen3-0.6B` is the practical local target for repeatable smoke and pilot runs.
