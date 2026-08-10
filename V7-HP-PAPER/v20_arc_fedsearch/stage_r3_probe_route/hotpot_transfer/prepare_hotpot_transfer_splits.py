#!/usr/bin/env python3
"""Seal Hotpot R3-T train and fresh holdout without inspecting holdout labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--prior-used", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in ("probe_train.jsonl", "probe_holdout.jsonl", "hotpot_transfer_results")):
        raise AssertionError("Hotpot transfer split/output already materialized")
    train = list(rows(args.train))
    development = list(rows(args.development))
    used: set[str] = set()
    for path in args.prior_used:
        used.update(qid(row) for row in rows(path))
    available = [row for row in development if qid(row) not in used]
    # Fixed source-order tail: never optimize the holdout by observed labels.
    holdout = available[-300:]
    if len(train) != 5000 or len(holdout) != 300:
        raise AssertionError(f"unexpected train/holdout counts {len(train)}/{len(holdout)}")
    train_ids, holdout_ids = [qid(row) for row in train], [qid(row) for row in holdout]
    if len(set(train_ids)) != len(train_ids) or len(set(holdout_ids)) != len(holdout_ids):
        raise AssertionError("duplicate query IDs")
    if set(train_ids) & set(holdout_ids):
        raise AssertionError("train/holdout overlap")
    if set(holdout_ids) & used:
        raise AssertionError("fresh holdout overlaps prior M0/V19 development")
    write_jsonl(args.output_dir / "probe_train.jsonl", train)
    write_jsonl(args.output_dir / "probe_holdout.jsonl", holdout)
    manifest = {
        "stage": "R3-T_frozen_hotpot_transfer",
        "dataset": "hotpotqa",
        "train": {"source": str(args.train.resolve()), "count": len(train_ids), "query_id_sha256": digest(train_ids)},
        "holdout": {"source": str(args.development.resolve()), "selection": "source_order_tail_after_prior_used_exclusion", "count": len(holdout_ids), "query_id_sha256": digest(holdout_ids)},
        "prior_used_sources": [str(path.resolve()) for path in args.prior_used],
        "prior_used_query_count": len(used),
        "no_feature_or_hyperparameter_selection_on_hotpot_holdout": True,
        "reader_started": False,
        "final_test_accessed": False,
    }
    (args.output_dir / "transfer_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "no_leak_audit.json").write_text(json.dumps({
        "dataset": "hotpotqa", "status": "pass", "train_holdout_disjoint": True,
        "holdout_disjoint_from_prior_m0_v19_development": True,
        "holdout_labels_deferred_until_post_ranking_metrics": True,
        "reader_started": False, "final_test_accessed": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(train), "holdout": len(holdout), "excluded_prior_ids": len(used)}, indent=2))


if __name__ == "__main__":
    main()
