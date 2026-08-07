#!/usr/bin/env python3
"""Seal R3 Probe-Train and the previously unopened fresh Probe-Holdout."""
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


def copy_jsonl(source: Path, target: Path) -> list[str]:
    values = list(rows(source))
    with target.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    return [qid(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--r2-root", type=Path, required=True)
    parser.add_argument("--r2a6-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_source = args.r2_root / "protocol" / "router_train.jsonl"
    holdout_source = args.r2a6_root / "protocol" / args.dataset / "recovery_holdout.jsonl"
    # R2-A.6 was stopped on Recovery-Dev. A presealed holdout is acceptable only
    # if no result artifact was ever materialized from it.
    prior_holdout_outputs = list((args.r2a6_root / "candidate_generation" / "recovery_holdout").glob("**/*"))
    if prior_holdout_outputs:
        raise AssertionError("R2-A.6 recovery holdout was previously materialized")
    train_path, holdout_path = args.output_dir / "probe_train.jsonl", args.output_dir / "probe_holdout.jsonl"
    train_ids = copy_jsonl(train_source, train_path)
    holdout_ids = copy_jsonl(holdout_source, holdout_path)
    if len(train_ids) != 5000 or len(holdout_ids) != 300:
        raise AssertionError(f"unexpected counts train={len(train_ids)} holdout={len(holdout_ids)}")
    if set(train_ids) & set(holdout_ids):
        raise AssertionError("Probe-Train and Probe-Holdout overlap")
    manifest = {
        "stage": "R3_lightweight_probe_ranker",
        "dataset": args.dataset,
        "probe_train": {"source": str(train_source.resolve()), "count": len(train_ids), "query_id_sha256": digest(train_ids)},
        "probe_holdout": {"source": str(holdout_source.resolve()), "count": len(holdout_ids), "query_id_sha256": digest(holdout_ids)},
        "recovery_holdout_previously_materialized": False,
        "model_contract": {
            "candidate_L": 8,
            "probe_schema": "r3-fixed-f32-v1",
            "model": "logistic_regression_only",
            "objective": "multi_label_bce_equivalent_client_log_loss",
            "selection": "independent_top3",
            "hard_negatives": "P0_top8_non_support_clients",
            "no_set_aware_selector": True,
        },
        "reader_started": False,
        "final_test_accessed": False,
    }
    (args.output_dir / "ranker_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "no_leak_audit.json").write_text(json.dumps({
        "dataset": args.dataset,
        "status": "pass",
        "train_holdout_disjoint": True,
        "prior_holdout_result_artifacts": 0,
        "training_labels_used_only_in_probe_train": True,
        "holdout_labels_deferred_until_post_ranking_metrics": True,
        "reader_started": False,
        "final_test_accessed": False,
    }, indent=2) + "\n")
    print(json.dumps({"dataset": args.dataset, "train": len(train_ids), "holdout": len(holdout_ids), "overlap": 0}, indent=2))


if __name__ == "__main__":
    main()
