#!/usr/bin/env python3
"""Freeze the untouched third Router-Dev hundred for Stage R3."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PROBE_DEV_START = 200
PROBE_DEV_N = 100


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--r2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = args.r2_root / "protocol"
    router_dev = list(rows(protocol / "router_dev.jsonl"))
    smoke = list(rows(protocol / "router_dev_smoke100.jsonl"))
    recovery = list(
        rows(
            args.r2_root.parent.parent
            / "stage_r2a6_resource_memory"
            / "protocol"
            / args.dataset
            / "recovery_dev.jsonl"
        )
    )
    selected = router_dev[PROBE_DEV_START : PROBE_DEV_START + PROBE_DEV_N]
    if len(selected) != PROBE_DEV_N:
        raise ValueError(f"Probe-Dev has {len(selected)} rows, expected {PROBE_DEV_N}")

    selected_ids = [query_id(row) for row in selected]
    overlaps = {
        "r2a_smoke100": sorted(set(selected_ids) & {query_id(row) for row in smoke}),
        "r2a6_recovery_dev": sorted(set(selected_ids) & {query_id(row) for row in recovery}),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Probe-Dev overlap: {overlaps}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.output_dir / "probe_dev.jsonl"
    write_jsonl(split_path, selected)
    manifest = {
        "stage": "R3_ProbeRoute_FedRAG",
        "dataset": args.dataset,
        "created_before_probe_feature_or_result_run": True,
        "source": str((protocol / "router_dev.jsonl").resolve()),
        "start_index_zero_based": PROBE_DEV_START,
        "count": PROBE_DEV_N,
        "query_ids": selected_ids,
        "query_id_sha256": digest(selected_ids),
        "overlap_with_prior_decision_splits": {name: len(ids) for name, ids in overlaps.items()},
        "reader_started": False,
        "final_test_accessed": False,
        "gold_fields_permitted_only_for_offline_metrics": True,
    }
    (args.output_dir / "probe_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"dataset": args.dataset, "probe_dev": len(selected), "overlap": 0}, indent=2))


if __name__ == "__main__":
    main()
