#!/usr/bin/env python3
"""Prepare a 900/100 GSM8K split in SpecForge conversation JSONL format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def to_conversation(row: dict, idx: int) -> dict:
    return {
        "id": f"gsm8k_{idx:04d}",
        "conversations": [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".artifacts/rl_project/data/gsm8k_1k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=900)
    parser.add_argument("--eval-size", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = args.train_size + args.eval_size
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=args.seed).select(range(total))

    train_path = output_dir / "gsm8k_train_900.jsonl"
    eval_path = output_dir / "gsm8k_eval_100.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds.select(range(args.train_size))):
            f.write(json.dumps(to_conversation(row, i), ensure_ascii=False) + "\n")

    with eval_path.open("w", encoding="utf-8") as f:
        for j, row in enumerate(ds.select(range(args.train_size, total))):
            f.write(json.dumps(to_conversation(row, args.train_size + j), ensure_ascii=False) + "\n")

    manifest = {
        "dataset": "openai/gsm8k/main/train",
        "seed": args.seed,
        "train_size": args.train_size,
        "eval_size": args.eval_size,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
