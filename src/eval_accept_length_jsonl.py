#!/usr/bin/env python3
"""Evaluate SGLang accept length on a conversation JSONL file."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sglang as sgl
from sglang import set_default_backend
from sglang.test.test_utils import select_sglang_backend


@sgl.function
def answer_question(s, question, max_new_tokens):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=max_new_tokens))


def read_questions(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            user_messages = [m for m in row["conversations"] if m["role"] == "user"]
            if not user_messages:
                continue
            rows.append(
                {
                    "id": row["id"],
                    "domain": row.get("domain", "unknown"),
                    "question": user_messages[0]["content"],
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--host", default="http://localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    backend_args = argparse.Namespace(host=args.host, port=args.port, backend="srt-no-parallel")
    set_default_backend(select_sglang_backend(backend_args))

    questions = read_questions(Path(args.input), args.limit)
    payload = [{"question": q["question"], "max_new_tokens": args.max_new_tokens} for q in questions]

    tic = time.perf_counter()
    states = answer_question.run_batch(
        payload,
        temperature=0,
        max_new_tokens=args.max_new_tokens,
        num_threads=args.batch_size,
        progress_bar=True,
    )
    latency = time.perf_counter() - tic

    total_completion = 0
    total_verify = 0
    per_item = []
    for q, state in zip(questions, states):
        meta = state.get_meta_info("answer")
        completion = int(meta.get("completion_tokens", 0))
        verify = int(meta.get("spec_verify_ct", 0))
        total_completion += completion
        total_verify += verify
        per_item.append(
            {
                "id": q["id"],
                "domain": q["domain"],
                "completion_tokens": completion,
                "spec_verify_ct": verify,
                "accept_length": (completion / verify if verify else 1.0),
            }
        )

    result = {
        "input": args.input,
        "label": args.label,
        "num_questions": len(questions),
        "latency_sec": latency,
        "total_completion_tokens": total_completion,
        "total_spec_verify_ct": total_verify,
        "average_accept_length": total_completion / total_verify if total_verify else 1.0,
        "output_throughput": total_completion / latency if latency > 0 else 0.0,
        "items": per_item,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
