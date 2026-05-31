# SpecForge Smoke Test

This smoke test trains an EAGLE3 draft model for a few steps on a tiny local dataset.

Target model:

- `Qwen/Qwen3-0.6B`

Why this target:

- It is small enough to plausibly fit on the local RTX 4060 8GB GPU.
- SpecForge already has Qwen3 support.
- `train_eagle3.py` can auto-generate an EAGLE3 draft config from the target model.

Commands:

```bash
python src/prepare_smoke_dataset.py

cd reference/code/SpecForge
python -m venv .venv
source .venv/bin/activate
pip install -e .

cd ../../..
bash src/run_specforge_smoke_train.sh
```

Expected evidence for the milestone:

- `scripts/train_eagle3.py` starts successfully.
- The target model and draft model initialize.
- Training reaches at least one optimization step with finite loss.
- A checkpoint is written under `reference/code/SpecForge/outputs/rl_project/smoke-qwen3-0.6b-eagle3`.

The smoke test is intentionally not a useful drafter. It only validates that the local SpecForge EAGLE3 training path can run end-to-end.
