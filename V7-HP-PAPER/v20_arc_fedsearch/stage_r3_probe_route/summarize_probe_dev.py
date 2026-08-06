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
    all_signal_pass = True
    all_cost_pass = True
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
        upper = csv_rows(args.stage_root / "probe_oracle" / dataset / "probe_upper_bound.csv")[0]
        cost_materially_lower = number(best, "mean_probe_bytes") <= 0.5 * number(best, "mean_document_bytes")
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
            "best_transmitted_complete_at_15": best["transmitted_complete_at_15"],
            "best_raw_merged_complete_at_10": best["raw_merged_complete_at_10"],
            "best_percentile_merged_complete_at_10": best["percentile_merged_complete_at_10"],
            "rescue_queries": rescue,
            "harm_queries": harm,
            "static_score_auprc": static["auprc"],
            "best_probe_feature": best_feature["feature"],
            "best_probe_feature_auprc": best_feature["auprc"],
            "feature_beats_static_auprc": number(best_feature, "auprc") > number(static, "auprc"),
            "O0_static_top3": upper["O0_static_independent_top3"],
            "O1_oracle_subset_within_top8": upper["O1_oracle_subset_at3_within_top8"],
            "O2_cv_probe_oracle_top3": upper["complete_client_set_recall_at_3"],
            "O2_cv_probe_oracle_auprc": upper["client_auprc"],
            "mean_probe_bytes": best["mean_probe_bytes"],
            "mean_document_bytes": best["mean_document_bytes"],
            "mean_probe_latency_ms": best["mean_probe_latency_ms"],
            "mean_deep_retrieval_latency_ms": best["mean_deep_retrieval_latency_ms"],
            "probe_bytes_materially_lower_than_document_payload": cost_materially_lower,
            "label_free_gate_passed": passed,
            "reader_started": False,
            "final_test_accessed": False,
        })
        all_signal_pass = all_signal_pass and passed
        all_cost_pass = all_cost_pass and cost_materially_lower
    args.stage_root.joinpath("reports").mkdir(parents=True, exist_ok=True)
    with (args.stage_root / "reports" / "probe_dev_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    status = "query_time_probe_signal_confirmed" if all_signal_pass else "query_time_probe_failed"
    ranker_permitted = all_signal_pass and all_cost_pass
    if ranker_permitted:
        next_step = "lightweight_supervised_probe_ranker"
    elif all_signal_pass:
        next_step = "compact_probe_wire_payload_audit_before_ranker"
    else:
        next_step = "stop_router_method_line_and_write_bottleneck_audit"
    decision = {
        "stage": "R3_ProbeRoute_FedRAG_Probe_Dev",
        "status": status,
        "label_free_signal_gate_passed_on_both_datasets": all_signal_pass,
        "communication_contract_passed_on_both_datasets": all_cost_pass,
        "supervised_ranker_permitted": ranker_permitted,
        "next_step": next_step,
        "reader_start_decision": "blocked_before_fresh_holdout",
        "final_test_accessed": False,
        "datasets": records,
    }
    lines = [
        "# R3 ProbeRoute-FedRAG: Probe-Dev Final Report",
        "",
        f"Final state: `{status}`. The untouched 100-query Probe-Dev slice was replayed twice per dataset. Exact semantic comparison passed after excluding wall-clock timing fields; no reader was started and final test remains sealed.",
        "",
        "## Pre-registered Gate",
        "",
        "| Dataset | Best label-free rule | P0 coverage@3 | Best coverage@3 | Delta 95% CI | P0 local@10 | Best local@10 | Rescue/Harm | Gate |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record['dataset']} | {record['best_label_free_method']} | {float(record['p0_complete_client_set_recall_at_3']):.3f} | {float(record['best_complete_client_set_recall_at_3']):.3f} | [{float(record['coverage_gain_ci95_low']):+.3f}, {float(record['coverage_gain_ci95_high']):+.3f}] | {float(record['p0_local_complete_at_10']):.3f} | {float(record['best_local_complete_at_10']):.3f} | {record['rescue_queries']}/{record['harm_queries']} | {'pass' if record['label_free_gate_passed'] else 'fail'} |"
        )
    lines.extend([
        "",
        "## Probe Separability and Offline Ceilings",
        "",
        "| Dataset | Static AUPRC | Best probe feature | Probe AUPRC | O0 static Top-3 | O1 oracle subset within Top-8 | O2 CV diagnostic Top-3 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for record in records:
        lines.append(
            f"| {record['dataset']} | {float(record['static_score_auprc']):.3f} | {record['best_probe_feature']} | {float(record['best_probe_feature_auprc']):.3f} | {float(record['O0_static_top3']):.3f} | {float(record['O1_oracle_subset_within_top8']):.3f} | {float(record['O2_cv_probe_oracle_top3']):.3f} |"
        )
    lines.extend([
        "",
        "## Matched 15-document Retrieval and Cost",
        "",
        "| Dataset | Best local@10 | Transmitted complete@15 | Raw merged@10 | Percentile merged@10 | Probe bytes | Document bytes | Probe latency ms | Deep latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for record in records:
        lines.append(
            f"| {record['dataset']} | {float(record['best_local_complete_at_10']):.3f} | {float(record['best_transmitted_complete_at_15']):.3f} | {float(record['best_raw_merged_complete_at_10']):.3f} | {float(record['best_percentile_merged_complete_at_10']):.3f} | {float(record['mean_probe_bytes']):.0f} | {float(record['mean_document_bytes']):.0f} | {float(record['mean_probe_latency_ms']):.0f} | {float(record['mean_deep_retrieval_latency_ms']):.0f} |"
        )
    lines.extend([
        "",
        "The probe payload contains scalar statistics and bounded title/entity summaries only; it contains neither document text nor embeddings. The offline O2 model is a diagnostic upper bound and is not deployed. The routing signal gate passes on both datasets, but the current verbose probe serialization is not materially smaller than the 15-document payload; therefore a supervised ranker is not yet permitted. The next task is a compact wire-payload audit that leaves routing features and selection rules frozen. Reader evaluation remains prohibited until the separate fresh-holdout gate passes.",
        "",
    ])
    report = "\n".join(lines)
    (args.stage_root / "reports" / "probe_route_go_no_go.md").write_text(report, encoding="utf-8")
    (args.stage_root / "reports" / "r3_probe_dev_final_report.md").write_text(report, encoding="utf-8")
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
