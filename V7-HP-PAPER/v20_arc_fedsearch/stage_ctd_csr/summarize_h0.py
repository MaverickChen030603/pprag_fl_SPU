#!/usr/bin/env python3
"""Apply the cross-dataset H0 gate after both dataset runs are frozen."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = ("2wikimultihopqa", "musique")


def summary_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["method"] == "REMP_rrf_p0_dense_lexical" and row["L"] == "5"]
    if len(matches) != 1:
        raise ValueError(f"expected one REM-P L=5 row in {path}")
    return matches[0]


def candidate_outputs_are_reproducible(repro: dict[str, object]) -> bool:
    """Treat wall-clock latency as telemetry, rather than a candidate output."""
    if "byte_identical_per_query_excluding_latency" in repro:
        keys = (
            "byte_identical_rankings",
            "byte_identical_query_views",
            "byte_identical_per_query_excluding_latency",
            "byte_identical_summary_excluding_latency",
        )
    else:
        keys = ("byte_identical_rankings", "byte_identical_per_query", "byte_identical_summary")
    return all(repro.get(key) is True for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--attempt", default="full")
    args = parser.parse_args()
    rows: dict[str, dict[str, str]] = {}
    reproducible: dict[str, bool] = {}
    for dataset in DATASETS:
        root = args.stage_root / "h0" / dataset
        if args.attempt != "full":
            root = root / args.attempt
        rows[dataset] = summary_row(root / "run1" / "evaluation" / "candidate_summary.csv")
        repro = json.loads((root / "reports" / "reproducibility.json").read_text(encoding="utf-8"))
        reproducible[dataset] = candidate_outputs_are_reproducible(repro)
    deltas = {dataset: float(row["delta_vs_p0_complete"]) for dataset, row in rows.items()}
    rescue_harm = {dataset: (int(row["rescue_vs_p0"]), int(row["harm_vs_p0"])) for dataset, row in rows.items()}
    non_decreasing = all(delta >= 0.0 for delta in deltas.values())
    magnitude = max(deltas.values()) >= 0.05 and min(deltas.values()) >= 0.03
    rescue_over_harm = all(rescue > harm for rescue, harm in rescue_harm.values())
    reproducible_all = all(reproducible.values())
    passed = non_decreasing and magnitude and rescue_over_harm and reproducible_all
    payload = {
        "stage": "CTD-CSR-H0",
        "attempt": args.attempt,
        "h0_passed": passed,
        "status": "h0_passed_proceed_ct0" if passed else "remp_not_generalized",
        "gate": {
            "non_decreasing_complete_at5_both": non_decreasing,
            "one_delta_at_least_005_other_at_least_003": magnitude,
            "rescue_greater_than_harm_both": rescue_over_harm,
            "byte_identical_both": reproducible_all,
        },
        "per_dataset": {
            dataset: {
                "remp_complete_at5": float(rows[dataset]["candidate_complete_client_set_recall_at_L"]),
                "delta_vs_p0_complete_at5": deltas[dataset],
                "rescue_vs_p0": rescue_harm[dataset][0],
                "harm_vs_p0": rescue_harm[dataset][1],
                "byte_identical": reproducible[dataset],
            }
            for dataset in DATASETS
        },
        "reader_started": False,
        "final_test_accessed": False,
    }
    reports = args.stage_root / "h0" / "reports"
    if args.attempt != "full":
        reports = args.stage_root / "h0" / args.attempt / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "h0_cross_dataset_gate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
