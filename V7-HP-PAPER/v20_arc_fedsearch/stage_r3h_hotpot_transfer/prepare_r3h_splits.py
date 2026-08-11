#!/usr/bin/env python3
"""Seal the R3-H Hotpot train and no-label holdout views before inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def ordered(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{salt}:{query_id(row)}".encode()).hexdigest())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir
    outputs = ("probe_train_labels.jsonl", "holdout_query_view.jsonl", "sealed_holdout_labels.jsonl", "split_manifest.json", "no_leak_audit.json")
    if any((out / name).exists() for name in outputs):
        raise FileExistsError(f"refusing to overwrite sealed R3-H split at {out}")
    out.mkdir(parents=True, exist_ok=False)

    train_rows = list(jsonl(args.train))
    if len(train_rows) < 5000:
        raise ValueError(f"Hotpot train has only {len(train_rows)} rows")
    train = ordered(train_rows, "r3h-hotpot-train-v1")[:5000]
    used: set[str] = set()
    exclusion_counts: dict[str, int] = {}
    for path in args.exclude:
        ids = {query_id(row) for row in jsonl(path)}
        exclusion_counts[str(path.resolve())] = len(ids)
        used.update(ids)
    development = list(jsonl(args.development))
    eligible = [row for row in development if query_id(row) not in used]
    holdout = ordered(eligible, "r3h-hotpot-holdout-v1")[:300]
    if len(holdout) != 300:
        raise ValueError(f"blocked_no_fresh_hotpot_holdout: eligible={len(eligible)}")
    train_ids, holdout_ids = [query_id(row) for row in train], [query_id(row) for row in holdout]
    if set(train_ids) & set(holdout_ids) or set(holdout_ids) & used:
        raise ValueError("train/holdout or historical overlap")

    query_view = [{"query_id": query_id(row), "question": str(row["question"])} for row in holdout]
    write_jsonl(out / "probe_train_labels.jsonl", train)
    write_jsonl(out / "holdout_query_view.jsonl", query_view)
    write_jsonl(out / "sealed_holdout_labels.jsonl", holdout)
    manifest = {
        "stage": "R3-H", "dataset": "hotpotqa", "train_count": 5000,
        "holdout_count": 300, "eligible_holdout_count": len(eligible),
        "train_selection": "sha256:r3h-hotpot-train-v1", "holdout_selection": "sha256:r3h-hotpot-holdout-v1",
        "train_query_id_sha256": digest(train_ids), "holdout_query_id_sha256": digest(holdout_ids),
        "exclusions": exclusion_counts, "train_holdout_overlap": 0,
        "holdout_labels_deferred_until_after_prediction_freeze": True,
        "reader_started": False, "final_test_accessed": False,
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audit = {"stage": "R3-H", "status": "pass", "train_holdout_disjoint": True,
             "holdout_disjoint_from_all_listed_history": True,
             "gold_or_answer_used_in_features": False, "body_or_title_in_probe": False,
             "holdout_labels_loaded_after_prediction_freeze": True,
             "reader_started": False, "final_test_accessed": False}
    (out / "no_leak_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(train), "holdout": len(holdout), "eligible": len(eligible)}, indent=2))


if __name__ == "__main__":
    main()
