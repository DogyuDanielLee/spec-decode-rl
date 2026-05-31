#!/usr/bin/env python3
"""Create a tiny SpecForge conversation-format dataset for EAGLE3 smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SAMPLES = [
    {
        "id": "math_000",
        "conversations": [
            {"role": "user", "content": "If Mina has 3 apples and buys 4 more, how many apples does she have?"},
            {"role": "assistant", "content": "Mina has 3 + 4 = 7 apples."},
        ],
    },
    {
        "id": "math_001",
        "conversations": [
            {"role": "user", "content": "A box has 5 red balls and 2 blue balls. How many balls are in the box?"},
            {"role": "assistant", "content": "There are 5 + 2 = 7 balls in the box."},
        ],
    },
    {
        "id": "qa_000",
        "conversations": [
            {"role": "user", "content": "What is the capital city of France?"},
            {"role": "assistant", "content": "The capital city of France is Paris."},
        ],
    },
    {
        "id": "qa_001",
        "conversations": [
            {"role": "user", "content": "What gas do plants absorb from the air during photosynthesis?"},
            {"role": "assistant", "content": "Plants absorb carbon dioxide during photosynthesis."},
        ],
    },
    {
        "id": "chat_000",
        "conversations": [
            {"role": "user", "content": "Write a short polite email asking to reschedule a meeting."},
            {
                "role": "assistant",
                "content": "Subject: Request to Reschedule\n\nHello, could we please reschedule our meeting to another convenient time? Thank you.",
            },
        ],
    },
    {
        "id": "code_000",
        "conversations": [
            {"role": "user", "content": "Write a Python function that returns the square of a number."},
            {"role": "assistant", "content": "def square(x):\n    return x * x"},
        ],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=".artifacts/rl_project/data/smoke_train.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in SAMPLES:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(SAMPLES)} samples to {output}")


if __name__ == "__main__":
    main()
