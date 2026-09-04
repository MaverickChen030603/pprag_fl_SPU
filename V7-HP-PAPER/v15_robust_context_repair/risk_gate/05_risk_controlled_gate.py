#!/usr/bin/env python3
"""Bonferroni Learn-then-Test calibration for independent per-query gating."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import beta

from gate_common import apply_threshold, observed_risk, read_jsonl, threshold_grid, write_jsonl


def binomial_upper(errors: int, total: int, alpha: float) -> float:
    if total == 0 or errors == total:
        return 1.0
    return float(beta.ppf(1.0 - alpha, errors + 1, total - errors))


def mean_lower(mean: float, total: int, alpha: float) -> float:
    # Answer deltas are bounded to [-1, 1]; one-sided Hoeffding bound.
    return mean - math.sqrt(2.0 * math.log(1.0 / alpha) / max(1, total))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--risk-target", type=float, choices=(0.04, 0.05, 0.08), default=0.05)
    parser.add_argument("--family-alpha", type=float, default=0.05)
    args = parser.parse_args()
    calibration = read_jsonl(args.calibration)
    grid = threshold_grid(calibration)
    per_test_alpha = args.family_alpha / max(1, 3 * len(grid))
    candidates = []
    curve = []
    for utility, harm in grid:
        decisions = apply_threshold(calibration, utility, harm)
        risk = observed_risk(decisions)
        selected = [row for row in decisions if row["selected"]]
        n = len(selected)
        answer_errors = sum(int(row["answer_drop"]) for row in selected)
        joint_errors = sum(int(row["joint_drop"]) for row in selected)
        answer_ucb = binomial_upper(answer_errors, n, per_test_alpha)
        joint_ucb = binomial_upper(joint_errors, n, per_test_alpha)
        answer_mean_lcb = mean_lower(risk["mean_answer_delta"], n, per_test_alpha)
        valid = n > 0 and answer_ucb <= args.risk_target and joint_ucb <= args.risk_target and answer_mean_lcb >= 0.0
        row = {"utility_threshold": utility, "harm_threshold": harm, **risk, "answer_drop_ucb": answer_ucb, "joint_drop_ucb": joint_ucb, "answer_delta_lcb": answer_mean_lcb, "passes": int(valid)}
        curve.append(row)
        if valid:
            candidates.append((risk["coverage"], risk["mean_joint_delta"], utility, harm, row))
    if candidates:
        _, _, utility, harm, calibration_result = max(candidates)
        status = "complete"
    else:
        utility, harm, calibration_result = float("inf"), 0.0, {"coverage": 0.0, "selected": 0}
        status = "fallback_only_no_certified_threshold"
    inference = apply_threshold(read_jsonl(args.inference), utility, harm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "per_query_gate_results.jsonl", inference)
    write_jsonl(args.output_dir / "risk_coverage_curves.jsonl", curve)
    summary = {"status": status, "gate": "bonferroni_learn_then_test", "risk_target": args.risk_target, "family_alpha": args.family_alpha, "tested_thresholds": len(grid), "per_test_alpha": per_test_alpha, "utility_threshold": utility, "harm_threshold": harm, "calibration": calibration_result, "inference_queries": len(inference), "inference_coverage": sum(row["selected"] for row in inference) / len(inference) if inference else 0.0}
    (args.output_dir / "risk_controlled_gate.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

