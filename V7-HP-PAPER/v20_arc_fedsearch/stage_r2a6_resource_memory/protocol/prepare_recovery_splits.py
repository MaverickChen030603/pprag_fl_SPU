#!/usr/bin/env python3
"""Freeze fresh Recovery-Dev and Recovery-Holdout before REMP evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DEV_START = 100
DEV_N = 100
HOLDOUT_START = 0
HOLDOUT_N = 300


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def write_jsonl(path: Path, selected: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def manifest_entry(source: Path, start: int, selected: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    ids = [query_id(row) for row in selected]
    return {
        "source_path": str(source.resolve()),
        "start_index_zero_based": start,
        "count": len(selected),
        "query_ids": ids,
        "query_id_sha256": digest(ids),
        "output_path": str(output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--r2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = args.r2_root / "protocol"
    router_dev = list(rows(protocol / "router_dev.jsonl"))
    router_holdout = list(rows(protocol / "router_holdout.jsonl"))
    prior_smoke = list(rows(protocol / "router_dev_smoke100.jsonl"))
    recovery_dev = router_dev[DEV_START : DEV_START + DEV_N]
    recovery_holdout = router_holdout[HOLDOUT_START : HOLDOUT_START + HOLDOUT_N]
    if len(recovery_dev) != DEV_N or len(recovery_holdout) != HOLDOUT_N:
        raise ValueError("insufficient fresh rows for the preregistered recovery split")

    used_ids = {query_id(row) for row in prior_smoke}
    dev_ids = [query_id(row) for row in recovery_dev]
    holdout_ids = [query_id(row) for row in recovery_holdout]
    if used_ids.intersection(dev_ids):
        raise AssertionError("Recovery-Dev overlaps R2-A/R2-A.5 smoke queries")
    if set(dev_ids).intersection(holdout_ids):
        raise AssertionError("Recovery-Dev overlaps Recovery-Holdout")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev_path = args.output_dir / "recovery_dev.jsonl"
    holdout_path = args.output_dir / "recovery_holdout.jsonl"
    write_jsonl(dev_path, recovery_dev)
    write_jsonl(holdout_path, recovery_holdout)
    payload = {
        "stage": "R2-A.6_REMP",
        "dataset": args.dataset,
        "created_before_profile_or_result_run": True,
        "final_test_accessed": False,
        "reader_started": False,
        "existing_r2a_smoke_query_id_sha256": digest([query_id(row) for row in prior_smoke]),
        "recovery_dev": manifest_entry(protocol / "router_dev.jsonl", DEV_START, recovery_dev, dev_path),
        "recovery_holdout": manifest_entry(protocol / "router_holdout.jsonl", HOLDOUT_START, recovery_holdout, holdout_path),
        "selection_rule": "Recovery-Dev chooses one strategy and one shared pooling formula once; holdout is not read until the Dev gate passes.",
    }
    (args.output_dir / "recovery_split_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"dataset": args.dataset, "recovery_dev": DEV_N, "recovery_holdout": HOLDOUT_N, "overlap": 0}, indent=2))


if __name__ == "__main__":
    main()
