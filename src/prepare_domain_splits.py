#!/usr/bin/env python3
"""Prepare GSM8K/MMLU/ShareGPT train/eval splits for EAGLE3 experiments."""

from __future__ import annotations

import argparse
import html
import json
import random
import re
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


CHOICE_LABELS = ["A", "B", "C", "D"]
TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def gsm8k_row(row: dict, idx: int, split: str) -> dict:
    return {
        "id": f"gsm8k_{split}_{idx:05d}",
        "domain": "gsm8k",
        "conversations": [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ],
    }


def mmlu_prompt(row: dict) -> str:
    choices = "\n".join(f"{label}. {choice}" for label, choice in zip(CHOICE_LABELS, row["choices"]))
    return (
        f"Subject: {row['subject']}\n\n"
        f"Question: {row['question']}\n\n"
        f"Choices:\n{choices}\n\n"
        "Answer with the correct option and a brief explanation."
    )


def mmlu_row(row: dict, idx: int, split: str) -> dict:
    answer_idx = int(row["answer"])
    answer_label = CHOICE_LABELS[answer_idx]
    answer_text = row["choices"][answer_idx]
    return {
        "id": f"mmlu_{split}_{idx:05d}",
        "domain": "mmlu",
        "subject": row["subject"],
        "conversations": [
            {"role": "user", "content": mmlu_prompt(row)},
            {"role": "assistant", "content": f"The correct answer is {answer_label}. {answer_text}"},
        ],
    }


def sharegpt_row(row: dict, idx: int, split: str) -> dict | None:
    turns = []
    for msg in row.get("conversations", []):
        role = msg.get("from")
        content = clean_text(str(msg.get("value", "")))
        if not content:
            continue
        if role == "human":
            turns.append({"role": "user", "content": content})
        elif role == "gpt":
            turns.append({"role": "assistant", "content": content})

    if len(turns) < 2 or turns[0]["role"] != "user" or turns[1]["role"] != "assistant":
        return None

    # Use the first exchange to keep lengths stable on the 8GB local GPU.
    return {
        "id": f"sharegpt_{split}_{idx:05d}",
        "domain": "sharegpt",
        "source_id": row.get("id"),
        "conversations": turns[:2],
    }


def prepare_gsm8k(output_dir: Path, n_train: int, n_eval: int, seed: int) -> tuple[list[dict], list[dict]]:
    total = n_train + n_eval
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=seed).select(range(total))
    train = [gsm8k_row(row, i, "train") for i, row in enumerate(ds.select(range(n_train)))]
    eval_rows = [
        gsm8k_row(row, i, "eval")
        for i, row in enumerate(ds.select(range(n_train, total)))
    ]
    write_jsonl(output_dir / "gsm8k_train.jsonl", train)
    write_jsonl(output_dir / "gsm8k_eval.jsonl", eval_rows)
    return train, eval_rows


def prepare_mmlu(output_dir: Path, n_train: int, n_eval: int, seed: int) -> tuple[list[dict], list[dict]]:
    train_ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    eval_ds = load_dataset("cais/mmlu", "all", split="test")
    train_ds = train_ds.shuffle(seed=seed).select(range(n_train))
    eval_ds = eval_ds.shuffle(seed=seed + 1).select(range(n_eval))
    train = [mmlu_row(row, i, "train") for i, row in enumerate(train_ds)]
    eval_rows = [mmlu_row(row, i, "eval") for i, row in enumerate(eval_ds)]
    write_jsonl(output_dir / "mmlu_train.jsonl", train)
    write_jsonl(output_dir / "mmlu_eval.jsonl", eval_rows)
    return train, eval_rows


def prepare_sharegpt(output_dir: Path, n_train: int, n_eval: int, seed: int) -> tuple[list[dict], list[dict]]:
    total = n_train + n_eval
    ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    ds = ds.shuffle(seed=seed)

    rows: list[dict] = []
    for raw in ds:
        row = sharegpt_row(raw, len(rows), "all")
        if row is None:
            continue
        rows.append(row)
        if len(rows) >= total:
            break

    if len(rows) < total:
        raise RuntimeError(f"Only found {len(rows)} valid ShareGPT rows, need {total}")

    train = []
    for i, row in enumerate(rows[:n_train]):
        row = dict(row)
        row["id"] = f"sharegpt_train_{i:05d}"
        train.append(row)
    eval_rows = []
    for i, row in enumerate(rows[n_train:total]):
        row = dict(row)
        row["id"] = f"sharegpt_eval_{i:05d}"
        eval_rows.append(row)

    write_jsonl(output_dir / "sharegpt_train.jsonl", train)
    write_jsonl(output_dir / "sharegpt_eval.jsonl", eval_rows)
    return train, eval_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".artifacts/rl_project/data/domain_n1500_m150")
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--eval-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gsm_train, gsm_eval = prepare_gsm8k(output_dir, args.train_size, args.eval_size, args.seed)
    mmlu_train, mmlu_eval = prepare_mmlu(output_dir, args.train_size, args.eval_size, args.seed)
    share_train, share_eval = prepare_sharegpt(output_dir, args.train_size, args.eval_size, args.seed)

    rng = random.Random(args.seed)
    all_train = [*gsm_train, *mmlu_train, *share_train]
    rng.shuffle(all_train)
    mixed_eval = [*gsm_eval, *mmlu_eval, *share_eval]
    rng.shuffle(mixed_eval)

    write_jsonl(output_dir / "all_train.jsonl", all_train)
    write_jsonl(output_dir / "mixed_eval.jsonl", mixed_eval)

    manifest = {
        "seed": args.seed,
        "train_size_per_domain": args.train_size,
        "eval_size_per_domain": args.eval_size,
        "datasets": {
            "gsm8k": "openai/gsm8k main train",
            "mmlu_train": "cais/mmlu all auxiliary_train",
            "mmlu_eval": "cais/mmlu all test",
            "sharegpt": "Aeala/ShareGPT_Vicuna_unfiltered train",
        },
        "files": {
            "gsm8k_train": str(output_dir / "gsm8k_train.jsonl"),
            "gsm8k_eval": str(output_dir / "gsm8k_eval.jsonl"),
            "mmlu_train": str(output_dir / "mmlu_train.jsonl"),
            "mmlu_eval": str(output_dir / "mmlu_eval.jsonl"),
            "sharegpt_train": str(output_dir / "sharegpt_train.jsonl"),
            "sharegpt_eval": str(output_dir / "sharegpt_eval.jsonl"),
            "all_train": str(output_dir / "all_train.jsonl"),
            "mixed_eval": str(output_dir / "mixed_eval.jsonl"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
