#!/usr/bin/env python3
"""Merge the V1-V15 inventory with every V16 split input."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    for key in ("query_id", "_id", "id", "qid", "question_id"):
        if row.get(key) is not None:
            return str(row[key])
    return ""


def norm_question(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("question", "")).strip().lower().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-inventory", type=Path, required=True)
    parser.add_argument("--v16-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("used_query_inventory.json"))
    args = parser.parse_args()

    base = json.loads(args.base_inventory.read_text(encoding="utf-8"))
    used_ids: dict[str, set[str]] = defaultdict(set)
    used_questions: dict[str, set[str]] = defaultdict(set)
    for dataset, values in base.get("used_query_ids_by_dataset", {}).items():
        used_ids[dataset].update(map(str, values))
    for dataset, values in base.get("used_normalized_questions_by_dataset", {}).items():
        used_questions[dataset].update(map(str, values))

    v16_counts: dict[str, dict[str, int]] = {}
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        before_ids, before_questions = len(used_ids[dataset]), len(used_questions[dataset])
        paths = [args.v16_data_root / dataset / f"{split}.jsonl" for split in ("train", "development", "calibration")]
        paths.append(args.v16_data_root / dataset / "final_test_inputs.jsonl")
        for path in paths:
            for row in iter_rows(path):
                if qid(row):
                    used_ids[dataset].add(qid(row))
                if norm_question(row):
                    used_questions[dataset].add(norm_question(row))
        v16_counts[dataset] = {
            "new_ids": len(used_ids[dataset]) - before_ids,
            "new_questions": len(used_questions[dataset]) - before_questions,
        }

    payload = {
        "schema_version": 2,
        "experiment": "V7-HP-PAPER-v17-federated-action-rag",
        "base_inventory": str(args.base_inventory.resolve()),
        "v16_data_root": str(args.v16_data_root.resolve()),
        "v16_additions": v16_counts,
        "used_query_ids_by_dataset": {key: sorted(values) for key, values in used_ids.items()},
        "used_normalized_questions_by_dataset": {key: sorted(values) for key, values in used_questions.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = args.output.with_suffix(".md")
    lines = ["# V17 Used Query Inventory", "", "V1-V15 inventory plus every V16 train/development/calibration/final input.", "", "| Dataset | Used IDs | Used normalized questions | V16 new IDs |", "|---|---:|---:|---:|"]
    for dataset in sorted(set(used_ids) | set(used_questions)):
        lines.append(f"| {dataset} | {len(used_ids[dataset]):,} | {len(used_questions[dataset]):,} | {v16_counts.get(dataset, {}).get('new_ids', 0):,} |")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output), "v16_additions": v16_counts}, indent=2))


if __name__ == "__main__":
    main()
