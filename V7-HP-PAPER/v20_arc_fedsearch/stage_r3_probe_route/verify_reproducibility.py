#!/usr/bin/env python3
"""Verify deterministic R3 decisions while excluding wall-clock measurements."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


TIMING_FIELDS = {"local_retrieval_latency_ms", "probe_latency_ms", "deep_retrieval_latency_ms", "mean_probe_latency_ms", "mean_deep_retrieval_latency_ms"}
RELATIVE_FILES = (
    "probe_features/per_query_client_probe.jsonl",
    "probe_features/feature_discrimination.csv",
    "probe_oracle/probe_upper_bound.csv",
    "probe_oracle/probe_upper_bound_per_query.csv",
    "label_free_baselines/per_query_results.csv",
    "label_free_baselines/main_results.csv",
    "reports/probe_route_go_no_go.json",
    "protocol_no_leak_audit.json",
)


def canonical_jsonl(path: Path) -> str:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            values.append({key: item for key, item in value.items() if key not in TIMING_FIELDS})
    return "\n".join(json.dumps(value, sort_keys=True, separators=(",", ":")) for value in values) + "\n"


def canonical_csv(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = [column for column in (reader.fieldnames or []) if column not in TIMING_FIELDS]
        values = [{column: row[column] for column in columns} for row in reader]
    return "\n".join(json.dumps(value, sort_keys=True, separators=(",", ":")) for value in values) + "\n"


def canonical(path: Path) -> str:
    if path.suffix == ".jsonl":
        return canonical_jsonl(path)
    if path.suffix == ".csv":
        return canonical_csv(path)
    return json.dumps(json.loads(path.read_text(encoding="utf-8")), sort_keys=True, separators=(",", ":")) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run1", type=Path, required=True)
    parser.add_argument("--run2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    digests: dict[str, str] = {}
    for relative in RELATIVE_FILES:
        left, right = args.run1 / relative, args.run2 / relative
        if not left.exists() or not right.exists():
            raise FileNotFoundError(f"missing replay artifact: {relative}")
        left_data, right_data = canonical(left), canonical(right)
        if left_data != right_data:
            raise AssertionError(f"semantic replay mismatch: {relative}")
        digests[relative] = hashlib.sha256(left_data.encode()).hexdigest()
    result: dict[str, Any] = {
        "dataset": args.dataset,
        "run1_run2_semantically_identical_after_excluding_wall_clock_timings": True,
        "deterministic_artifact_sha256": digests,
        "timing_fields_excluded_from_exact_comparison": sorted(TIMING_FIELDS),
        "reader_started": False,
        "final_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
