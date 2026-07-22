#!/usr/bin/env python3
"""Summarize reader-evaluated V16 action trajectories and enforce Go/No-Go 1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRICS = ("answer_f1", "sp_f1", "joint_f1")
EPSILONS = (0.0, 0.01, 0.02)


def read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            if path.suffix == ".jsonl":
                rows.extend(json.loads(line) for line in handle if line.strip())
            else:
                rows.extend(csv.DictReader(handle))
    return rows


def bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draws.append(statistics.fmean(values[rng.randrange(len(values))] for _ in values))
    draws.sort()
    return draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]


def best(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    return max(rows, key=lambda row: (float(row[metric]), str(row.get("trajectory_id", row.get("action_id", "")))), default=None)


def summarize_query(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baselines = [row for row in rows if int(row.get("depth", 0)) == 0 or bool(row.get("is_baseline", False))]
    if not baselines:
        raise ValueError(f"query {rows[0].get('query_id')} has no T=0 baseline")
    baseline = baselines[0]
    singles = [row for row in rows if int(row.get("depth", 0)) == 1 and not bool(row.get("is_baseline", False))]
    composed = [row for row in rows if 2 <= int(row.get("depth", 0)) <= 3]
    result: dict[str, Any] = {
        "dataset": rows[0].get("dataset", "unknown"),
        "reader": rows[0].get("reader", "unknown"),
        "query_id": rows[0].get("query_id"),
        "hop_count": rows[0].get("hop_count", "unknown"),
        "question_type": rows[0].get("question_type", rows[0].get("type", "unknown")),
        "single_candidate_count": len(singles),
        "composed_candidate_count": len(composed),
    }
    for metric in METRICS:
        base_value = float(baseline[metric])
        single = best(singles, metric)
        composition = best(composed, metric)
        single_delta = (float(single[metric]) - base_value) if single else 0.0
        composed_delta = (float(composition[metric]) - base_value) if composition else 0.0
        strict_synergy = composed_delta - single_delta
        prefix = metric[:-3] if metric.endswith("_f1") else metric
        result.update({
            f"baseline_{metric}": base_value,
            f"best_single_delta_{prefix}": single_delta,
            f"best_composed_delta_{prefix}": composed_delta,
            f"strict_synergy_{prefix}": strict_synergy,
            f"composition_only_positive_{prefix}": int(single_delta <= 0.0 < composed_delta),
            f"best_single_id_{prefix}": "" if single is None else single.get("trajectory_id", single.get("action_id", "")),
            f"best_composed_id_{prefix}": "" if composition is None else composition.get("trajectory_id", composition.get("action_id", "")),
            f"best_composed_depth_{prefix}": "" if composition is None else int(composition.get("depth", 0)),
        })
        for epsilon in EPSILONS:
            label = str(epsilon).replace(".", "p")
            result[f"synergistic_{prefix}_eps_{label}"] = int(strict_synergy > epsilon)
    return result


def aggregate(per_query: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_query:
        grouped[(str(row["dataset"]), str(row["reader"]))].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, reader), rows in sorted(grouped.items()):
        for metric in ("answer", "sp", "joint"):
            strict = [float(row[f"strict_synergy_{metric}"]) for row in rows]
            low, high = bootstrap_mean_ci(strict, bootstrap_samples, seed)
            result: dict[str, Any] = {
                "dataset": dataset,
                "reader": reader,
                "metric": metric,
                "queries": len(rows),
                "positive_single_rate": statistics.fmean(float(row[f"best_single_delta_{metric}"]) > 0 for row in rows),
                "positive_composed_rate": statistics.fmean(float(row[f"best_composed_delta_{metric}"]) > 0 for row in rows),
                "composition_only_positive_rate": statistics.fmean(int(row[f"composition_only_positive_{metric}"]) for row in rows),
                "mean_strict_synergy": statistics.fmean(strict),
                "median_strict_synergy": statistics.median(strict),
                "strict_synergy_ci_low": low,
                "strict_synergy_ci_high": high,
            }
            for epsilon in EPSILONS:
                label = str(epsilon).replace(".", "p")
                result[f"synergy_rate_eps_{label}"] = statistics.fmean(int(row[f"synergistic_{metric}_eps_{label}"]) for row in rows)
            output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decision(summary: list[dict[str, Any]]) -> dict[str, Any]:
    joint = [row for row in summary if row["metric"] == "joint"]
    per_reader = []
    for row in joint:
        passed = (
            float(row["composition_only_positive_rate"]) >= 0.10
            and float(row["mean_strict_synergy"]) > 0.0
            and float(row["strict_synergy_ci_low"]) > 0.0
        )
        per_reader.append({**row, "passes_reader_level": passed})
    dataset_pass = {
        dataset: any(row["passes_reader_level"] for row in per_reader if row["dataset"] == dataset)
        for dataset in sorted({str(row["dataset"]) for row in per_reader})
    }
    return {
        "status": "continue_composition" if sum(dataset_pass.values()) >= 2 else "hold_or_redirect",
        "dataset_pass": dataset_pass,
        "reader_level": per_reader,
        "rule": "Checkpoint 1 requires at least two datasets with composition-only positive rate >= 10%, positive mean StrictSynJoint, and paired bootstrap CI lower bound > 0.",
    }


def render_report(summary: list[dict[str, Any]], verdict: dict[str, Any]) -> str:
    lines = [
        "# V16 Oracle Single vs Composed",
        "",
        "Results below use reader-evaluated contexts. Strict synergy is the best depth-2/3 delta minus the best delta over all evaluated depth-1 actions for the same query.",
        "",
        "| Dataset | Reader | N | Single positive | Composed positive | Composition-only | Mean StrictSyn Joint | 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["metric"] != "joint":
            continue
        lines.append(
            f"| {row['dataset']} | {row['reader']} | {row['queries']} | {row['positive_single_rate']:.3f} | "
            f"{row['positive_composed_rate']:.3f} | {row['composition_only_positive_rate']:.3f} | "
            f"{row['mean_strict_synergy']:+.4f} | [{row['strict_synergy_ci_low']:+.4f}, {row['strict_synergy_ci_high']:+.4f}] |"
        )
    lines += ["", "## Decision", "", f"**{verdict['status']}**", "", verdict["rule"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_rows(args.input):
        grouped[(str(row.get("dataset", "unknown")), str(row.get("reader", "unknown")), str(row["query_id"]))].append(row)
    per_query = [summarize_query(rows) for rows in grouped.values()]
    summary = aggregate(per_query, args.bootstrap_samples, args.seed)
    verdict = decision(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "oracle_action_results.csv", per_query)
    write_csv(args.output_dir / "oracle_synergy_statistics.csv", summary)
    report = render_report(summary, verdict)
    (args.output_dir / "oracle_single_vs_composed.md").write_text(report, encoding="utf-8")
    (args.output_dir / "composition_go_no_go_1.md").write_text(
        "# V16 Composition Go/No-Go 1\n\n" + f"Status: **{verdict['status']}**\n\n" + verdict["rule"] + "\n\n```json\n" + json.dumps(verdict, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
