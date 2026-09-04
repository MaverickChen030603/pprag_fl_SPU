#!/usr/bin/env python3
"""Fit and apply an independent per-query empirical risk gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gate_common import apply_threshold, observed_risk, read_jsonl, threshold_grid, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--risk-target", type=float, default=0.05)
    args = parser.parse_args()
    calibration = read_jsonl(args.calibration)
    feasible = []
    for utility, harm in threshold_grid(calibration):
        decisions = apply_threshold(calibration, utility, harm)
        risk = observed_risk(decisions)
        if risk["selected"] and risk["answer_drop_rate"] <= args.risk_target and risk["joint_drop_rate"] <= args.risk_target and risk["mean_answer_delta"] >= 0.0:
            feasible.append((risk["coverage"], risk["mean_joint_delta"], utility, harm, risk))
    if not feasible:
        utility, harm, risk = float("inf"), 0.0, observed_risk([])
        status = "fallback_only_no_feasible_threshold"
    else:
        _, _, utility, harm, risk = max(feasible)
        status = "complete"
    inference = apply_threshold(read_jsonl(args.inference), utility, harm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "per_query_gate_results.jsonl", inference)
    summary = {"status": status, "gate": "empirical_per_query", "risk_target": args.risk_target, "utility_threshold": utility, "harm_threshold": harm, "calibration": risk, "inference_queries": len(inference), "inference_coverage": sum(row["selected"] for row in inference) / len(inference) if inference else 0.0}
    (args.output_dir / "empirical_gate.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

