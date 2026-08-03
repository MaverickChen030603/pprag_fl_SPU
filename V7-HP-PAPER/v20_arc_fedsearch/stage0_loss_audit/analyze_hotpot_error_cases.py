#!/usr/bin/env python3
"""Post-hoc HotpotQA loss attribution for the frozen V20 M0-Confirm matrix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(value: str) -> int:
    return int(float(value))


def dump_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    details = load_csv(args.matrix_dir / "per_query_allocation_merge.csv")
    a0 = {row["query_id"]: row for row in details if row["allocation"] == "A0_equal_5_5_5"}
    a1 = {row["query_id"]: row for row in details if row["allocation"] == "A1_confidence_proportional"}
    if set(a0) != set(a1):
        raise ValueError("A0/A1 query IDs do not align")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    venn: dict[tuple[int, int], list[str]] = defaultdict(list)
    rescues: list[dict[str, Any]] = []
    residual: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    raw_failure_clients: Counter[str] = Counter()
    for qid in sorted(a0):
        row0, row1 = a0[qid], a1[qid]
        a0_success = int(set(json.loads(row0["percentile_top10"])) >= set())
        # Complete-support status is recovered from the calibration flag plus
        # raw support-loss state without rerunning any method or reader.
        raw_success = not bool(as_int(row0["support_lost_by_raw_merge"])) if as_int(row0["complete_transmitted15"]) else False
        a0_success = int(raw_success or as_int(row0["support_rescued_by_calibration"]))
        raw_success_a1 = not bool(as_int(row1["support_lost_by_raw_merge"])) if as_int(row1["complete_transmitted15"]) else False
        a1_success = int(raw_success_a1 or as_int(row1["support_rescued_by_calibration"]))
        venn[(a0_success, a1_success)].append(qid)

        base = {
            "query_id": qid,
            "selected_clients": json.loads(row0["selected_clients"]),
            "gold_clients_offline_audit_only": json.loads(row0["gold_clients"]),
            "support_local_ranks": json.loads(row0["support_local_ranks"]),
            "support_1_local_rank": row0.get("support_1_local_rank") or None,
            "support_2_local_rank": row0.get("support_2_local_rank") or None,
            "worst_support_rank": row0.get("worst_support_rank") or None,
            "complete_local5": as_int(row0["complete_local5"]),
            "complete_local10": as_int(row0["complete_local10"]),
            "complete_transmitted15": as_int(row0["complete_transmitted15"]),
            "raw_top10": json.loads(row0["raw_top10"]),
            "percentile_top10": json.loads(row0["percentile_top10"]),
        }
        if as_int(row0["support_rescued_by_calibration"]):
            rescues.append(base | {"event": "A0_rank_percentile_rescue"})
        if base["complete_local10"] and not a0_success:
            category = "allocation_loss" if not base["complete_transmitted15"] else "rank_calibration_residual"
            categories[category] += 1
            residual.append(base | {"failure_category": category})
        if as_int(row0["support_lost_by_raw_merge"]):
            for value in json.loads(row0["support_local_ranks"]).values():
                if value is not None:
                    raw_failure_clients[value.split(":", 1)[0]] += 1

    labels = {
        (1, 1): "both_A0_percentile_and_A1_percentile",
        (1, 0): "A0_percentile_only",
        (0, 1): "A1_percentile_only",
        (0, 0): "neither",
    }
    venn_rows = [{"membership": labels[key], "query_count": len(value), "query_ids": json.dumps(value)} for key, value in sorted(venn.items(), reverse=True)]
    with (args.output_dir / "a0_a1_success_venn.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(venn_rows[0]))
        writer.writeheader()
        writer.writerows(venn_rows)
    dump_jsonl(args.output_dir / "percentile_rescue_cases.jsonl", rescues)
    dump_jsonl(args.output_dir / "remaining_merge_failures.jsonl", residual)
    summary = ([{"summary_type": "remaining_failure_category", "bucket": key, "count": value} for key, value in sorted(categories.items())] +
               [{"summary_type": "raw_merge_failure_support_client", "bucket": key, "count": value} for key, value in sorted(raw_failure_clients.items())] +
               [{"summary_type": "calibration", "bucket": "rescued", "count": len(rescues)},
                {"summary_type": "calibration", "bucket": "remaining_local10_to_percentile_failures", "count": len(residual)}])
    with (args.output_dir / "failure_category_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["summary_type", "bucket", "count"])
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"rescue_cases": len(rescues), "remaining_local10_to_percentile_failures": len(residual), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
