#!/usr/bin/env python3
"""Aggregate fresh REMP results, enforce the preregistered Dev/Holdout gates."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATASETS = ("2wikimultihopqa", "musique")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        all_strata = ["overall"] + json.loads(row["strata"])
        for stratum in all_strata:
            grouped[(row["dataset"], row["method"], int(row["K"]), stratum)].append(row)
    summary = []
    for (dataset, method, cutoff, stratum), values in sorted(grouped.items()):
        summary.append({
            "dataset": dataset,
            "method": method,
            "K": cutoff,
            "stratum": stratum,
            "queries": len(values),
            "gold_client_recall": float(np.mean([float(row["gold_client_recall"]) for row in values])),
            "complete_client_set_recall": float(np.mean([float(row["complete_client_set_recall"]) for row in values])),
            "candidate_absence_loss": float(np.mean([float(row["candidate_absence_loss_at_8"]) for row in values if row["K"] == "8"])) if cutoff == 8 else "",
            "reader_started": False,
        })
    return summary


def lookup(summary: list[dict[str, Any]], dataset: str, method: str, cutoff: int, metric: str) -> float:
    matches = [row for row in summary if row["dataset"] == dataset and row["method"] == method and row["K"] == cutoff and row["stratum"] == "overall"]
    if len(matches) != 1:
        raise ValueError(f"missing aggregate {dataset} {method} K{cutoff}")
    return float(matches[0][metric])


def profile_costs(profile_roots: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    values: list[dict[str, Any]] = []
    means: dict[tuple[str, str], float] = {}
    method_bytes: dict[tuple[str, str], float] = {}
    for dataset, root in profile_roots.items():
        rows = read_csv(root / "memory_statistics.csv")
        for row in rows:
            values.append({"dataset": dataset, **row})
        for variant in sorted({row["variant"] for row in rows}):
            subset = [row for row in rows if row["variant"] == variant]
            means[(dataset, variant)] = float(np.mean([float(row["bytes_per_client"]) for row in subset]))
        # Inherited controls only retain vector prototypes.  Their cost is
        # included explicitly so the REMP quality/storage chart is not an
        # asymmetric text-only comparison.
        prototype_bytes = 768 * np.dtype(np.float32).itemsize
        best_p = 8 if dataset == "2wikimultihopqa" else 16
        for method, units, total_bytes in (
            ("B0_single_centroid", 1, prototype_bytes),
            ("B1_kmeans_multi_prototype", best_p, best_p * prototype_bytes),
        ):
            method_bytes[(dataset, method)] = float(total_bytes)
            values.append({
                "dataset": dataset,
                "variant": method,
                "client_id": "mean_control",
                "units_per_client": units,
                "tokens_per_client": 0,
                "text_bytes_per_client": 0,
                "selected_embedding_bytes_per_client": total_bytes,
                "bytes_per_client": total_bytes,
                "profile_construction_seconds": "inherited_control",
                "peak_rss_bytes": "",
                "title_units": 0,
                "entity_units": 0,
                "snippet_units": 0,
                "relation_units": 0,
            })
    return values, means, method_bytes


def bootstrap_paired(method: list[float], baseline: list[float], seed: int = 20260806, draws: int = 10000) -> tuple[float, float, float]:
    delta = np.asarray(method, dtype=float) - np.asarray(baseline, dtype=float)
    rng = np.random.default_rng(seed)
    samples = [float(delta[rng.integers(0, len(delta), len(delta))].mean()) for _ in range(draws)]
    return float(delta.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def selected_key(method: str) -> tuple[str, str] | None:
    if "__" not in method:
        return None
    variant, pooling = method.split("__", 1)
    return variant, pooling


def dev_gate(summary: list[dict[str, Any]], bytes_by_variant: dict[tuple[str, str], float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = sorted({row["method"] for row in summary if row["method"].startswith("R")})
    audit: list[dict[str, Any]] = []
    eligible: list[tuple[tuple[float, float, float, str], dict[str, Any]]] = []
    for method in candidates:
        variant, pooling = selected_key(method) or ("", "")
        per_dataset = {}
        for dataset in DATASETS:
            b0_c5 = lookup(summary, dataset, "B0_single_centroid", 5, "complete_client_set_recall")
            r_c5 = lookup(summary, dataset, method, 5, "complete_client_set_recall")
            b0_g5 = lookup(summary, dataset, "B0_single_centroid", 5, "gold_client_recall")
            r_g5 = lookup(summary, dataset, method, 5, "gold_client_recall")
            b0_c8 = lookup(summary, dataset, "B0_single_centroid", 8, "complete_client_set_recall")
            r_c8 = lookup(summary, dataset, method, 8, "complete_client_set_recall")
            per_dataset[dataset] = {"complete_at5_delta": r_c5 - b0_c5, "gold_at5_delta": r_g5 - b0_g5, "complete_at8_delta": r_c8 - b0_c8}
        passes = all(item["complete_at5_delta"] >= 0.05 and item["gold_at5_delta"] >= 0 and item["complete_at8_delta"] >= -0.01 and min(item.values()) >= -0.02 for item in per_dataset.values())
        average_bytes = float(np.mean([bytes_by_variant[(dataset, variant)] for dataset in DATASETS]))
        item = {"method": method, "variant": variant, "pooling": pooling, "per_dataset": per_dataset, "average_bytes_per_client": average_bytes, "eligible": passes}
        audit.append(item)
        if passes:
            key = (min(value["complete_at5_delta"] for value in per_dataset.values()), np.mean([value["complete_at5_delta"] for value in per_dataset.values()]), -average_bytes, method)
            eligible.append((key, item))
    if eligible:
        _, chosen = max(eligible, key=lambda item: item[0])
        return {"status": "recovery_dev_passed_holdout_required", "selected": chosen, "reader_start_decision": "blocked_before_reader"}, audit
    dataset_hits = {dataset: any(item["per_dataset"][dataset]["complete_at5_delta"] >= 0.05 for item in audit) for dataset in DATASETS}
    status = "dataset_specific_memory_signal" if sum(dataset_hits.values()) == 1 else "resource_memory_recovery_failed"
    return {"status": status, "selected": None, "dataset_plus5pp": dataset_hits, "reader_start_decision": "blocked_before_reader"}, audit


def combine_rank_records(input_roots: dict[str, Path], output: Path) -> None:
    with output.open("w", encoding="utf-8") as handle:
        for dataset in DATASETS:
            for row in read_jsonl(input_roots[dataset] / "per_query_client_ranks.jsonl"):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def holdout_gate(rows: list[dict[str, str]], method: str) -> dict[str, Any]:
    outcome: dict[str, Any] = {"selected_method": method, "per_dataset": {}, "reader_start_decision": "blocked_before_reader"}
    passes: list[bool] = []
    for dataset in DATASETS:
        selected = {cutoff: [row for row in rows if row["dataset"] == dataset and row["method"] == method and int(row["K"]) == cutoff] for cutoff in (3, 5, 8)}
        baseline = {cutoff: [row for row in rows if row["dataset"] == dataset and row["method"] == "B0_single_centroid" and int(row["K"]) == cutoff] for cutoff in (3, 5, 8)}
        selected_k5 = {row["query_id"]: float(row["complete_client_set_recall"]) for row in selected[5]}
        baseline_k5 = {row["query_id"]: float(row["complete_client_set_recall"]) for row in baseline[5]}
        if set(selected_k5) != set(baseline_k5):
            raise ValueError("paired holdout IDs do not match")
        ids = sorted(selected_k5)
        delta, lower, upper = bootstrap_paired([selected_k5[q] for q in ids], [baseline_k5[q] for q in ids])
        selected_absence = np.mean([float(row["candidate_absence_loss_at_8"]) for row in selected[8]])
        baseline_absence = np.mean([float(row["candidate_absence_loss_at_8"]) for row in baseline[8]])
        rescue = [q for q in ids if baseline_k5[q] == 0 and selected_k5[q] == 1]
        selected_gold = {row["query_id"]: json.loads(row["gold_clients_offline_only"]) for row in selected[5]}
        rescue_client_counts = Counter(client for query in rescue for client in selected_gold[query])
        max_share = (max(rescue_client_counts.values()) / max(1, sum(rescue_client_counts.values())))
        outcome["per_dataset"][dataset] = {
            "complete_set_at5_delta": delta,
            "paired_bootstrap_ci95": [lower, upper],
            "candidate_absence_loss_at8_delta": float(selected_absence - baseline_absence),
            "rescue_queries": len(rescue),
            "largest_selected_client_share_among_rescues": max_share,
        }
        passes.append(delta >= 0.05 and selected_absence <= baseline_absence and max_share < 0.5)
    per = outcome["per_dataset"]
    ci_ok = any(value["paired_bootstrap_ci95"][0] > 0 for value in per.values())
    direction_ok = all(value["complete_set_at5_delta"] >= 0 for value in per.values())
    if all(passes) and ci_ok and direction_ok:
        outcome["status"] = "resource_memory_profile_confirmed"
        outcome["next_method"] = "ready_for_r2b"
    else:
        one_pass = sum(value["complete_set_at5_delta"] >= 0.05 for value in per.values()) == 1
        outcome["status"] = "dataset_specific_memory_signal" if one_pass else "resource_memory_development_overfit"
        outcome["next_method"] = "stop_mars_route"
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("dev", "holdout"), required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True, help="contains one subdirectory per dataset")
    parser.add_argument("--profile-root", type=Path, required=True, help="contains one subdirectory per dataset")
    parser.add_argument("--selected-method")
    args = parser.parse_args()

    input_roots = {dataset: args.run_root / dataset for dataset in DATASETS}
    profile_roots = {dataset: args.profile_root / dataset for dataset in DATASETS}
    raw = [row for dataset in DATASETS for row in read_csv(input_roots[dataset] / "recovery_results_per_query.csv")]
    summary = aggregate(raw)
    costs, bytes_by_variant, bytes_by_method = profile_costs(profile_roots)
    suffix = "recovery_dev" if args.phase == "dev" else "holdout"
    candidate_root = args.stage_root / "candidate_generation"
    write_csv(candidate_root / f"{suffix}_results.csv", summary)
    combined_path = candidate_root / ("per_query_client_ranks.jsonl" if args.phase == "dev" else "holdout_per_query_client_ranks.jsonl")
    combine_rank_records(input_roots, combined_path)
    write_csv(args.stage_root / "efficiency" / "profile_costs.csv", costs)
    pareto = []
    for row in summary:
        if row["K"] != 5 or row["stratum"] != "overall":
            continue
        if row["method"].startswith("R"):
            variant, pooling = selected_key(row["method"]) or ("", "")
            bytes_per_client = bytes_by_variant[(row["dataset"], variant)]
        else:
            variant, pooling = row["method"], "inherited"
            bytes_per_client = bytes_by_method[(row["dataset"], row["method"])]
        pareto.append({"dataset": row["dataset"], "method": row["method"], "variant": variant, "pooling": pooling, "bytes_per_client": bytes_per_client, "complete_set_recall_at5": row["complete_client_set_recall"], "gold_client_recall_at5": row["gold_client_recall"]})
    write_csv(args.stage_root / "efficiency" / "quality_storage_pareto.csv", pareto)

    if args.phase == "dev":
        decision, audit = dev_gate(summary, bytes_by_variant)
        decision["eligibility_audit"] = audit
        (args.stage_root / "reports" / "next_method_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        lines = ["# R2-A.6 Recovery-Dev Go/No-Go", "", f"Status: `{decision['status']}`.", "", "| Method | 2Wiki Delta@5 | MuSiQue Delta@5 | Eligible |", "| --- | ---: | ---: | --- |"]
        for item in audit:
            lines.append(f"| {item['method']} | {item['per_dataset']['2wikimultihopqa']['complete_at5_delta']:+.3f} | {item['per_dataset']['musique']['complete_at5_delta']:+.3f} | {item['eligible']} |")
        (args.stage_root / "reports" / "recovery_go_no_go.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        if not args.selected_method:
            raise ValueError("--selected-method is required for holdout aggregation")
        decision = holdout_gate(raw, args.selected_method)
        (args.stage_root / "reports" / "next_method_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.stage_root / "holdout" / "bootstrap_results.csv").write_text("dataset,delta,ci95_lower,ci95_upper\n" + "\n".join(f"{dataset},{values['complete_set_at5_delta']},{values['paired_bootstrap_ci95'][0]},{values['paired_bootstrap_ci95'][1]}" for dataset, values in decision["per_dataset"].items()) + "\n", encoding="utf-8")
        (args.stage_root / "reports" / "recovery_go_no_go.md").write_text(f"# R2-A.6 Recovery-Holdout Go/No-Go\n\nStatus: `{decision['status']}`.\nNext method: `{decision['next_method']}`.\n", encoding="utf-8")
    reader = {"status": "blocked_before_reader", "reader_started": False, "reason": "R2-A.6 candidate-generation gate never authorizes reader evaluation."}
    (args.stage_root / "reports" / "reader_start_decision.json").write_text(json.dumps(reader, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
