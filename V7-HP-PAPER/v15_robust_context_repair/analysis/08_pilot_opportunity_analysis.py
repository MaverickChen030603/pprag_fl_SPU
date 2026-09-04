#!/usr/bin/env python3
"""Summarize reader-positive opportunity in a frozen V15 action pool."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


METRICS = ("answer_f1", "sp_f1", "joint_f1")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reader_summary(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["query_id"])].append(row)
    baseline_values = {metric: [] for metric in METRICS}
    oracle_values = {metric: [] for metric in METRICS}
    oracle_deltas = {metric: [] for metric in METRICS}
    positive_joint = []
    nonbaseline_joint_drops = []
    for actions in groups.values():
        baseline = next(row for row in actions if row["is_baseline"])
        best_joint = max(actions, key=lambda row: float(row["joint_f1"]))
        for metric in METRICS:
            baseline_values[metric].append(float(baseline[metric]))
            oracle_values[metric].append(float(best_joint[metric]))
            oracle_deltas[metric].append(float(best_joint[metric]) - float(baseline[metric]))
        positive_joint.append(float(best_joint["joint_f1"]) > float(baseline["joint_f1"]) + 1e-9)
        nonbaseline_joint_drops.extend(
            float(row["joint_f1"]) < float(baseline["joint_f1"]) - 1e-9
            for row in actions if not row["is_baseline"]
        )
    return {
        "queries": len(groups),
        "actions_per_query": sorted({len(actions) for actions in groups.values()}),
        "baseline": {metric: float(np.mean(values)) for metric, values in baseline_values.items()},
        "joint_oracle": {metric: float(np.mean(values)) for metric, values in oracle_values.items()},
        "joint_oracle_delta": {metric: float(np.mean(values)) for metric, values in oracle_deltas.items()},
        "positive_joint_opportunity_rate": float(np.mean(positive_joint)),
        "nonbaseline_joint_drop_prevalence": float(np.mean(nonbaseline_joint_drops)),
    }


def robust_oracle(reader_rows: dict[str, list[dict]]) -> dict:
    by_reader: dict[str, dict[tuple[str, str], dict]] = {}
    for reader, rows in reader_rows.items():
        by_reader[reader] = {(str(row["query_id"]), str(row["action_id"])): row for row in rows}
    readers = list(reader_rows)
    common = set.intersection(*(set(values) for values in by_reader.values()))
    groups: dict[str, list[str]] = defaultdict(list)
    for query_id, action_id in common:
        groups[query_id].append(action_id)

    results = {}
    for beta in (0.0, 0.25, 0.5, 1.0):
        chosen_deltas = []
        intervention = []
        for query_id, action_ids in groups.items():
            candidates = []
            for action_id in action_ids:
                deltas = np.asarray([float(by_reader[reader][(query_id, action_id)]["joint_delta"]) for reader in readers])
                is_baseline = bool(by_reader[readers[0]][(query_id, action_id)]["is_baseline"])
                candidates.append((float(deltas.mean() - beta * deltas.std()), is_baseline, action_id, deltas))
            _, _, action_id, deltas = max(candidates, key=lambda item: (item[0], item[1]))
            chosen_deltas.append(deltas)
            intervention.append(not by_reader[readers[0]][(query_id, action_id)]["is_baseline"])
        values = np.asarray(chosen_deltas)
        results[f"beta_{beta:g}"] = {
            "queries": len(groups),
            "intervention_rate": float(np.mean(intervention)),
            "mean_reader_joint_delta": float(values.mean(axis=1).mean()),
            "minimum_reader_joint_delta": float(values.min(axis=1).mean()),
            "both_readers_positive_rate": float((values > 1e-9).all(axis=1).mean()),
            "any_reader_harm_rate": float((values < -1e-9).any(axis=1).mean()),
        }

    ordered = sorted(common)
    left = np.asarray([float(by_reader[readers[0]][key]["joint_delta"]) for key in ordered])
    right = np.asarray([float(by_reader[readers[1]][key]["joint_delta"]) for key in ordered])
    active = (np.abs(left) > 1e-9) | (np.abs(right) > 1e-9)
    agreement = {
        "spearman": float(spearmanr(left, right).statistic),
        "pearson": float(pearsonr(left, right).statistic),
        "nonzero_sign_disagreement_rate": float(((left * right < 0) & active).sum() / max(1, active.sum())),
    }
    return {"readers": readers, "agreement": agreement, "robust_oracle": results}


def render(report: dict) -> str:
    lines = ["# V15 HotpotQA 100-Query Pilot Opportunity", ""]
    for reader, values in report["per_reader"].items():
        lines.extend(
            [
                f"## {reader}",
                "",
                f"- Baseline Answer/SP/Joint F1: {values['baseline']['answer_f1']:.4f} / {values['baseline']['sp_f1']:.4f} / {values['baseline']['joint_f1']:.4f}",
                f"- Joint-oracle Answer/SP/Joint delta: {values['joint_oracle_delta']['answer_f1']:+.4f} / {values['joint_oracle_delta']['sp_f1']:+.4f} / {values['joint_oracle_delta']['joint_f1']:+.4f}",
                f"- Queries with positive Joint opportunity: {values['positive_joint_opportunity_rate']:.2%}",
                f"- Non-baseline actions causing Joint drop: {values['nonbaseline_joint_drop_prevalence']:.2%}",
                "",
            ]
        )
    lines.extend(
        [
            "## Same-Action Robust Oracle",
            "",
            "| Beta | Intervention | Mean-reader Joint delta | Min-reader Joint delta | Both positive | Any harm |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, values in report["cross_reader"]["robust_oracle"].items():
        lines.append(
            f"| {name.replace('beta_', '', 1)} | {values['intervention_rate']:.4f} | {values['mean_reader_joint_delta']:+.4f} | "
            f"{values['minimum_reader_joint_delta']:+.4f} | {values['both_readers_positive_rate']:.4f} | {values['any_reader_harm_rate']:.4f} |"
        )
    lines.extend(["", "> All oracle values are retrospective action-set upper bounds and are not deployable selector results.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", action="append", nargs=2, metavar=("NAME", "JSONL"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    reader_rows = {name: read_jsonl(Path(path)) for name, path in args.reader}
    report = {
        "status": "complete",
        "per_reader": {name: reader_summary(rows) for name, rows in reader_rows.items()},
        "cross_reader": robust_oracle(reader_rows),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render(report), encoding="utf-8")
    print(json.dumps({"status": "complete", "readers": list(reader_rows)}, indent=2))


if __name__ == "__main__":
    main()
