#!/usr/bin/env python3
"""Summarize drafter accept-length results into a matrix and reward table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", nargs=2, metavar=("LABEL", "JSON"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    per_label_items: dict[str, dict[str, dict]] = {}
    for label, path_str in args.result:
        labels.append(label)
        result = load_result(Path(path_str))
        per_label_items[label] = {item["id"]: item for item in result["items"]}

    common_ids = sorted(set.intersection(*(set(items) for items in per_label_items.values())))
    if not common_ids:
        raise SystemExit("no common request ids across result files")

    domains = sorted({per_label_items[labels[0]][request_id].get("domain", "unknown") for request_id in common_ids})

    matrix_rows = []
    for domain in domains:
        row = {"domain": domain}
        ids = [request_id for request_id in common_ids if per_label_items[labels[0]][request_id].get("domain", "unknown") == domain]
        for label in labels:
            row[label] = mean([float(per_label_items[label][request_id]["accept_length"]) for request_id in ids])
        matrix_rows.append(row)

    overall = {"domain": "overall"}
    for label in labels:
        overall[label] = mean([float(per_label_items[label][request_id]["accept_length"]) for request_id in common_ids])
    matrix_rows.append(overall)

    matrix_csv = output_dir / "domain_accept_length_matrix.csv"
    with matrix_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", *labels])
        writer.writeheader()
        writer.writerows(matrix_rows)

    matrix_md = output_dir / "domain_accept_length_matrix.md"
    lines = ["| Domain | " + " | ".join(labels) + " |", "|---" + "|---" * len(labels) + "|"]
    for row in matrix_rows:
        lines.append("| " + row["domain"] + " | " + " | ".join(f"{row[label]:.4f}" for label in labels) + " |")
    matrix_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reward_csv = output_dir / "per_request_reward_table.csv"
    reward_fields = ["request_id", "domain", *labels, "best_drafter", "best_accept_length"]
    with reward_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reward_fields)
        writer.writeheader()
        for request_id in common_ids:
            domain = per_label_items[labels[0]][request_id].get("domain", "unknown")
            rewards = {label: float(per_label_items[label][request_id]["accept_length"]) for label in labels}
            best_label = max(labels, key=lambda label: rewards[label])
            writer.writerow(
                {
                    "request_id": request_id,
                    "domain": domain,
                    **rewards,
                    "best_drafter": best_label,
                    "best_accept_length": rewards[best_label],
                }
            )

    wins: dict[str, dict[str, int]] = {domain: defaultdict(int) for domain in [*domains, "overall"]}
    for request_id in common_ids:
        domain = per_label_items[labels[0]][request_id].get("domain", "unknown")
        rewards = {label: float(per_label_items[label][request_id]["accept_length"]) for label in labels}
        best_label = max(labels, key=lambda label: rewards[label])
        wins[domain][best_label] += 1
        wins["overall"][best_label] += 1

    summary = {
        "num_requests": len(common_ids),
        "labels": labels,
        "matrix": matrix_rows,
        "win_counts": {domain: dict(counts) for domain, counts in wins.items()},
        "outputs": {
            "matrix_csv": str(matrix_csv),
            "matrix_md": str(matrix_md),
            "reward_csv": str(reward_csv),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
