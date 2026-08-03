#!/usr/bin/env python3
"""Write the pre-registered V20 M0-Confirm cross-dataset decision artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(value: str | None) -> int:
    return int(float(value or "0"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cell(matrix: list[dict[str, str]], allocation: str, merge: str) -> dict[str, str]:
    return next(row for row in matrix if row["allocation"] == allocation and row["merge"] == merge)


def summarize_dataset(name: str, directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    run1, run2 = directory / "run1", directory / "run2"
    stage = load_csv(run1 / "per_query_stage_metrics.csv")
    details = load_csv(run1 / "per_query_allocation_merge.csv")
    matrix = load_csv(run1 / "allocation_merge_matrix.csv")
    if not stage or len(stage) != 300:
        raise ValueError(f"{name}: expected exactly 300 frozen queries")
    a0 = {row["query_id"]: row for row in details if row["allocation"] == "A0_equal_5_5_5"}
    a1 = {row["query_id"]: row for row in details if row["allocation"] == "A1_confidence_proportional"}
    if set(a0) != set(a1) or set(a0) != {row["query_id"] for row in stage}:
        raise ValueError(f"{name}: A0/A1/stage query alignment failure")
    raw_a0, pct_a0 = cell(matrix, "A0_equal_5_5_5", "M0_raw"), cell(matrix, "A0_equal_5_5_5", "M1_rank_percentile")
    raw_a1, pct_a1 = cell(matrix, "A1_confidence_proportional", "M0_raw"), cell(matrix, "A1_confidence_proportional", "M1_rank_percentile")
    local5 = mean(as_int(row["selected_client_local_complete_at_5"]) for row in stage)
    local10 = mean(as_int(row["selected_client_local_complete_at_10"]) for row in stage)
    coverage = mean(
        int(set(json.loads(row["gold_clients_offline_audit_only"])).issubset(set(json.loads(row["selected_clients"]))))
        for row in stage
    )
    per_query: list[dict[str, Any]] = []
    for row in stage:
        qid = row["query_id"]
        per_query.append({
            "dataset": name,
            "query_id": qid,
            "selected_clients": json.loads(row["selected_clients"]),
            "gold_clients_offline_audit_only": json.loads(row["gold_clients_offline_audit_only"]),
            "support_local_ranks": json.loads(row["support_local_ranks"]),
            "support_1_local_rank": row.get("support_1_local_rank") or None,
            "support_2_local_rank": row.get("support_2_local_rank") or None,
            "worst_support_rank": row.get("worst_support_rank") or None,
            "complete_local5": as_int(row["selected_client_local_complete_at_5"]),
            "complete_local10": as_int(row["selected_client_local_complete_at_10"]),
            "A0": {
                "transmitted_doc_ids": json.loads(a0[qid]["transmitted_doc_ids"]),
                "complete_transmitted15": as_int(a0[qid]["complete_transmitted15"]),
                "raw_top10": json.loads(a0[qid]["raw_top10"]),
                "percentile_top10": json.loads(a0[qid]["percentile_top10"]),
                "support_lost_by_allocation": as_int(a0[qid]["support_lost_by_allocation"]),
                "support_lost_by_raw_merge": as_int(a0[qid]["support_lost_by_raw_merge"]),
                "support_rescued_by_calibration": as_int(a0[qid]["support_rescued_by_calibration"]),
                "support_harmed_by_calibration": as_int(a0[qid]["support_harmed_by_calibration"]),
            },
            "A1": {
                "complete_transmitted15": as_int(a1[qid]["complete_transmitted15"]),
                "raw_top10": json.loads(a1[qid]["raw_top10"]),
                "percentile_top10": json.loads(a1[qid]["percentile_top10"]),
            },
        })
    matrix_identical = (run1 / "allocation_merge_matrix.csv").read_bytes() == (run2 / "allocation_merge_matrix.csv").read_bytes()
    per_query_identical = (run1 / "per_query_allocation_merge.csv").read_bytes() == (run2 / "per_query_allocation_merge.csv").read_bytes()
    result = {
        "dataset": name,
        "queries": len(stage),
        "client_coverage_at_3": coverage,
        "local_complete_at_5": local5,
        "local_complete_at_10": local10,
        "depth_rescue_count": sum(as_int(row["selected_client_local_complete_at_10"]) and not as_int(row["selected_client_local_complete_at_5"]) for row in stage),
        "A0_transmitted_complete_at_15": float(raw_a0["complete_transmitted_at_15"]),
        "A0_raw_complete_at_10": float(raw_a0["complete_merged_at_10"]),
        "A0_rank_percentile_complete_at_10": float(pct_a0["complete_merged_at_10"]),
        "A0_percentile_minus_raw": float(pct_a0["complete_merged_at_10"]) - float(raw_a0["complete_merged_at_10"]),
        "A1_transmitted_complete_at_15": float(raw_a1["complete_transmitted_at_15"]),
        "A1_raw_complete_at_10": float(raw_a1["complete_merged_at_10"]),
        "A1_rank_percentile_complete_at_10": float(pct_a1["complete_merged_at_10"]),
        "percentile_rescue_count": sum(as_int(row["support_rescued_by_calibration"]) for row in a0.values()),
        "percentile_harm_count": sum(as_int(row["support_harmed_by_calibration"]) for row in a0.values()),
        "mean_clients_contacted": float(raw_a0["mean_clients_contacted"]),
        "mean_documents_transmitted": float(raw_a0["mean_documents_transmitted"]),
        "reader_started": False,
        "formal_client_set": "frozen_Bc3",
        "byte_identical_matrix": matrix_identical,
        "byte_identical_per_query": per_query_identical,
    }
    repro = {
        "dataset": name,
        "run1_matrix_sha256": sha256(run1 / "allocation_merge_matrix.csv"),
        "run2_matrix_sha256": sha256(run2 / "allocation_merge_matrix.csv"),
        "run1_per_query_sha256": sha256(run1 / "per_query_allocation_merge.csv"),
        "run2_per_query_sha256": sha256(run2 / "per_query_allocation_merge.csv"),
        "byte_identical_matrix": matrix_identical,
        "byte_identical_per_query": per_query_identical,
    }
    return result, per_query, repro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpot-dir", type=Path, required=True)
    parser.add_argument("--two-wiki-dir", type=Path, required=True)
    parser.add_argument("--musique-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    args = parser.parse_args()
    specs = (("hotpotqa", args.hotpot_dir), ("2wikimultihopqa", args.two_wiki_dir), ("musique", args.musique_dir))
    results, per_query, reproducibility = [], [], []
    for name, directory in specs:
        result, queries, repro = summarize_dataset(name, directory)
        results.append(result)
        per_query.extend(queries)
        reproducibility.append(repro)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "main_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (args.output_root / "per_query_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_query:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_root / "reproducibility.json").write_text(json.dumps(reproducibility, indent=2) + "\n", encoding="utf-8")

    passed = [row for row in results if row["A0_percentile_minus_raw"] >= 0.05 and row["percentile_rescue_count"] > row["percentile_harm_count"] and row["local_complete_at_10"] > row["local_complete_at_5"] and row["mean_documents_transmitted"] <= 15 and row["mean_clients_contacted"] <= 3 and row["formal_client_set"] == "frozen_Bc3" and row["byte_identical_matrix"] and row["byte_identical_per_query"]]
    allocation_wins = [row for row in results if row["A1_rank_percentile_complete_at_10"] > row["A0_rank_percentile_complete_at_10"]]
    if len(passed) >= 2:
        status = "multidataset_calibrated_merge_signal_confirmed"
        next_action = "freeze A0 plus rank-percentile and permit the pre-registered frozen-reader evaluation on all N=300 queries."
    elif len(allocation_wins) >= 2:
        status = "allocation_and_calibration_signal"
        next_action = "compare only pre-specified simple allocation rules; keep reader forbidden."
    elif next(row for row in results if row["dataset"] == "hotpotqa") in passed:
        low_coverage = [row for row in results if row["dataset"] != "hotpotqa" and row["client_coverage_at_3"] < 0.5]
        status = "routing_residual_reconfirmed" if low_coverage else "hotpot_specific_merge_signal"
        next_action = "audit frozen Bc=3 routing coverage before any reader evaluation."
    else:
        status = "v20_retrieval_method_not_generalized"
        next_action = "stop before reader evaluation and retain the result as a negative generalization finding."
    reader_permitted = status == "multidataset_calibrated_merge_signal_confirmed"
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    reader_decision = {"status": status, "reader_started": False, "reader_forbidden_during_M0_confirm": True, "reader_start_permitted_after_gate": reader_permitted, "required_readers_if_permitted": ["FLAN", "UnifiedQA"], "sample_per_reader": 300}
    (args.reports_dir / "reader_start_decision.json").write_text(json.dumps(reader_decision, indent=2) + "\n", encoding="utf-8")
    (args.reports_dir / "next_module_decision.md").write_text(f"# V20 Next-Module Decision\n\n- Final state: `{status}`\n- Reader status: `{'permitted, not started' if reader_permitted else 'forbidden'}`\n- Next action: {next_action}\n", encoding="utf-8")
    table = "\n".join(f"| {row['dataset']} | {row['client_coverage_at_3']:.3f} | {row['local_complete_at_5']:.3f} | {row['local_complete_at_10']:.3f} | {row['A0_raw_complete_at_10']:.3f} | {row['A0_rank_percentile_complete_at_10']:.3f} | {row['A0_percentile_minus_raw']:+.3f} | {row['percentile_rescue_count']} | {row['percentile_harm_count']} |" for row in results)
    report = f"# V20 M0-Confirm: Cross-Dataset Frozen Retrieval\n\nAll rows use the frozen inherited topic route (`Bc=3`), local depth 10, 15 transmitted documents, global top-10, A0 equal allocation, and label-free rank-percentile merge. Reader execution was forbidden and did not occur.\n\n| Dataset | Coverage@3 | Local@5 | Local@10 | A0 raw@10 | A0 percentile@10 | Delta | Rescue | Harm |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n{table}\n\n## Decision\n\nFinal state: `{status}`. {next_action}\n\nReproducibility: both runs were required to be byte-identical for matrix and per-query artifacts.\n"
    (args.reports_dir / "multidataset_retrieval_go_no_go.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": status, "reader_start_permitted_after_gate": reader_permitted, "passed_datasets": [row["dataset"] for row in passed]}, indent=2))


if __name__ == "__main__":
    main()
