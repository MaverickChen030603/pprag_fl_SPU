#!/usr/bin/env python3
"""Combine frozen R3 Probe-Dev audits without opening any Holdout or reader."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, Any], name: str) -> float:
    return float(row[name])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    args = parser.parse_args()
    records = []
    all_pass = True
    for dataset in ("2wikimultihopqa", "musique"):
        root = args.stage_root / "label_free_baselines" / dataset
        summary = csv_rows(root / "main_results.csv")
        per_query = csv_rows(root / "per_query_results.csv")
        reference = next(row for row in summary if row["candidate_L"] == "8" and row["method"] == "P0_static_single_centroid")
        alternatives = [row for row in summary if row["candidate_L"] == "8" and row["method"] != "P0_static_single_centroid"]
        best = max(alternatives, key=lambda row: number(row, "complete_client_set_recall_at_3"))
        reference_q = {row["query_id"]: row for row in per_query if row["candidate_L"] == "8" and row["method"] == "P0_static_single_centroid"}
        best_q = {row["query_id"]: row for row in per_query if row["candidate_L"] == "8" and row["method"] == best["method"]}
        rescue = sum(number(best_q[key], "complete_client_set_recall_at_3") > number(reference_q[key], "complete_client_set_recall_at_3") for key in reference_q)
        harm = sum(number(best_q[key], "complete_client_set_recall_at_3") < number(reference_q[key], "complete_client_set_recall_at_3") for key in reference_q)
        passed = (
            number(best, "coverage_minus_P0") >= 0.05
            and number(best, "local_complete_at_10") >= number(reference, "local_complete_at_10") + 0.03
            and rescue > harm
        )
        discrimination = csv_rows(args.stage_root / "probe_features" / dataset / "feature_discrimination.csv")
        static = next(row for row in discrimination if row["feature"] == "static_score")
        best_feature = max(discrimination, key=lambda row: number(row, "auprc"))
        records.append({
            "dataset": dataset,
            "best_label_free_method": best["method"],
            "p0_complete_client_set_recall_at_3": reference["complete_client_set_recall_at_3"],
            "best_complete_client_set_recall_at_3": best["complete_client_set_recall_at_3"],
            "coverage_gain": best["coverage_minus_P0"],
            "coverage_gain_ci95_low": best["coverage_minus_P0_ci95_low"],
            "coverage_gain_ci95_high": best["coverage_minus_P0_ci95_high"],
            "p0_local_complete_at_10": reference["local_complete_at_10"],
            "best_local_complete_at_10": best["local_complete_at_10"],
            "rescue_queries": rescue,
            "harm_queries": harm,
            "static_score_auprc": static["auprc"],
            "best_probe_feature": best_feature["feature"],
            "best_probe_feature_auprc": best_feature["auprc"],
            "feature_beats_static_auprc": number(best_feature, "auprc") > number(static, "auprc"),
            "label_free_gate_passed": passed,
            "reader_started": False,
            "final_test_accessed": False,
        })
        all_pass = all_pass and passed
    args.stage_root.joinpath("reports").mkdir(parents=True, exist_ok=True)
    with (args.stage_root / "reports" / "probe_dev_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    status = "query_time_probe_signal_confirmed" if all_pass else "query_time_probe_failed"
    decision = {
        "stage": "R3_ProbeRoute_FedRAG_Probe_Dev",
        "status": status,
        "label_free_gate_passed_on_both_datasets": all_pass,
        "next_step": "lightweight_supervised_probe_ranker" if all_pass else "stop_router_method_line_and_write_bottleneck_audit",
        "reader_start_decision": "blocked_before_fresh_holdout",
        "final_test_accessed": False,
        "datasets": records,
    }
    (args.stage_root / "reports" / "probe_route_go_no_go.md").write_text(
        "# R3 ProbeRoute-FedRAG: Probe-Dev Go/No-Go\n\n"
        f"Final state: `{status}`. Reader remains `blocked_before_fresh_holdout`; final test remains sealed.\n\n"
        "The report is generated from the untouched Probe-Dev split. A supervised probe ranker is prohibited unless both data sets pass the preregistered label-free gate.\n",
        encoding="utf-8",
    )
    (args.stage_root / "reports" / "next_method_decision.md").write_text(
        f"# Next Method Decision\n\nStatus: `{status}`. Next action: `{decision['next_step']}`.\n", encoding="utf-8"
    )
    (args.stage_root / "reports" / "reader_start_decision.json").write_text(
        json.dumps({"status": "blocked_before_fresh_holdout", "reader_started": False}, indent=2) + "\n", encoding="utf-8"
    )
    (args.stage_root / "reports" / "probe_route_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
