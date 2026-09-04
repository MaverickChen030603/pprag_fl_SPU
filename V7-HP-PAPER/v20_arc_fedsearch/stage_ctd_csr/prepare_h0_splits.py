#!/usr/bin/env python3
"""Freeze H0 fresh holdouts without reading support labels or final test data."""
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


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", default="ctd-csr-h0-v1")
    args = parser.parse_args()

    source = list(jsonl(args.source))
    excluded_by_path: dict[str, set[str]] = {}
    excluded: set[str] = set()
    for path in args.exclude:
        values = {query_id(row) for row in jsonl(path)}
        excluded_by_path[str(path)] = values
        excluded.update(values)
    eligible = [row for row in source if query_id(row) not in excluded]
    ranked = sorted(
        eligible,
        key=lambda row: hashlib.sha256(f"{args.seed}:{query_id(row)}".encode()).hexdigest(),
    )
    selected = ranked[: args.count]
    if len(selected) < args.count:
        raise ValueError(f"only {len(selected)} eligible rows, need {args.count}")
    selected_ids = [query_id(row) for row in selected]
    if len(set(selected_ids)) != len(selected_ids):
        raise AssertionError("duplicate query IDs in H0 selection")
    overlaps = {path: len(set(selected_ids) & values) for path, values in excluded_by_path.items()}
    if any(overlaps.values()):
        raise AssertionError(f"H0 overlap audit failed: {overlaps}")

    split_path = args.output_dir / "fresh_router_holdout.jsonl"
    write_jsonl(split_path, selected)
    manifest = {
        "stage": "CTD-CSR-H0",
        "dataset": args.dataset,
        "source_split": "calibration",
        "source_path": str(args.source.resolve()),
        "source_rows": len(source),
        "excluded_rows_by_artifact": {path: len(values) for path, values in excluded_by_path.items()},
        "eligible_rows": len(eligible),
        "sampling": "ascending_sha256(seed + ':' + query_id)",
        "seed": args.seed,
        "count": len(selected),
        "query_ids": selected_ids,
        "query_id_sha256": digest(selected_ids),
        "overlap_count_by_artifact": overlaps,
        "reader_started": False,
        "final_test_accessed": False,
    }
    (args.output_dir / "h0_split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"dataset": args.dataset, "count": len(selected), "eligible": len(eligible), "overlap": 0}, indent=2))


if __name__ == "__main__":
    main()
