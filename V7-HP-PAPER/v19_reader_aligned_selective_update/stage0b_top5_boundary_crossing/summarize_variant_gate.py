#!/usr/bin/env python3
"""Summarize Stage 0B variant retrieval gate from boundary audit rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def truthy(value: Any) -> bool:
    return str(value).lower() == "true"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-dir", type=Path, required=True)
    parser.add_argument("--baseline-support-recall-at5", type=float, default=0.71)
    parser.add_argument("--baseline-complete-support-at5", type=float, default=0.44)
    args = parser.parse_args()

    rows = list(csv.DictReader((args.variant_dir / "boundary_audit/pc1_per_query_categories.csv").open(encoding="utf-8")))
    train = json.loads((args.variant_dir / "training_log.json").read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "queries": len(rows),
        "top5_changed_rate": sum(truthy(row["top5_changed"]) for row in rows) / len(rows),
        "top10_changed_rate": sum(truthy(row["top10_changed"]) for row in rows) / len(rows),
        "top20_changed_rate": sum(truthy(row["top20_changed"]) for row in rows) / len(rows),
        "support_rank_improved_queries": sum(truthy(row["support_moved_toward_top5"]) for row in rows),
        "support_rank_worsened_queries": sum(truthy(row["support_moved_away"]) for row in rows),
        "useful_top5_change_count": sum(row["category"] == "E_top5_beneficial_swap" for row in rows),
        "harmful_top5_change_count": sum(row["category"] == "F_top5_harmful_swap" for row in rows),
        "complete_support_gain_count": sum(
            (not truthy(row["baseline_complete_support_at5"])) and truthy(row["pc1_complete_support_at5"])
            for row in rows
        ),
        "complete_support_loss_count": sum(
            truthy(row["baseline_complete_support_at5"]) and (not truthy(row["pc1_complete_support_at5"]))
            for row in rows
        ),
        "boundary_conversion_count": sum(truthy(row["support_entered_top5"]) for row in rows),
    }
    for key in ["loss_first", "loss_last", "adapter_bytes", "support_recall_at_5", "complete_support_at_5"]:
        summary[key] = train[key]
    summary["support_recall_delta_at_5"] = summary["support_recall_at_5"] - args.baseline_support_recall_at5
    summary["complete_support_delta_at_5"] = summary["complete_support_at_5"] - args.baseline_complete_support_at5
    summary["gate_pass"] = bool(
        summary["top5_changed_rate"] >= 0.05
        and summary["useful_top5_change_count"] > summary["harmful_top5_change_count"]
        and summary["support_rank_improved_queries"] > summary["support_rank_worsened_queries"]
        and summary["complete_support_gain_count"] > summary["complete_support_loss_count"]
        and (
            summary["support_recall_delta_at_5"] >= 0.01
            or summary["complete_support_delta_at_5"] >= 0.02
            or summary["boundary_conversion_count"] > 0
        )
    )
    (args.variant_dir / "pc2a_gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.variant_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    report = [
        "# PC-2A N=100 Retrieval Gate Summary",
        "",
        f"Status: {'PASS' if summary['gate_pass'] else 'FAIL'}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        report.append(f"| {key} | {value} |")
    report += ["", "Decision: Reader remains disabled unless the pre-registered Stage 0B retrieval gate passes."]
    (args.variant_dir / "pc2a_retrieval_gate.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
