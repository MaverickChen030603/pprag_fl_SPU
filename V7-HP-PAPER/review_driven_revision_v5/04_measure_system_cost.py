#!/usr/bin/env python3
"""Separate offline development cost, online cost, and exact intervention effects."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from v5_common import HERE, V4, config, percentile, read_json, read_jsonl, write_json, write_text


OUT = HERE / "outputs" / "cost"
FLAN = "/home/iiserver31/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
SYSTEMS = ("frozen_top5_baseline", "full_v4", "lite_method", "recomp_top1", "recomp_budgetmatched")


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "rows": "[NEEDS SOURCE FILE]", "bytes": 0}
    rows = sum(1 for line in path.open("r", encoding="utf-8") if line.strip()) if path.suffix == ".jsonl" else 1
    return {"path": str(path), "exists": True, "rows": rows, "bytes": path.stat().st_size}


def exact_intervention_effects() -> dict[str, Any]:
    metrics_path = V4 / "outputs/scaleup/official_metrics/flan_per_query.jsonl"
    selection_path = V4 / "outputs/scaleup/frozen_selector_selections_3000.jsonl"
    rows = read_jsonl(metrics_path)
    selections = {str(row["query_id"]): bool(row["selected"]) for row in read_jsonl(selection_path)}
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["query_id"])][str(row["method"])] = row
    metrics = ("answer_f1", "sp_f1", "joint_f1")
    per_query = []
    for query_id, values in grouped.items():
        if set(values) != {"baseline", "v4_selected"}:
            raise AssertionError(f"Incomplete paired metrics for {query_id}")
        per_query.append(
            {
                "query_id": query_id,
                "selected": selections[query_id],
                **{
                    f"{metric}_delta": float(values["v4_selected"][metric]) - float(values["baseline"][metric])
                    for metric in metrics
                },
            }
        )
    selected = [row for row in per_query if row["selected"]]
    fallback = [row for row in per_query if not row["selected"]]

    def summary(values: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(values),
            **{f"{metric}_delta": mean(float(row[f"{metric}_delta"]) for row in values) for metric in metrics},
            **{
                f"{metric}_gain_per_100_queries": 100.0 * mean(float(row[f"{metric}_delta"]) for row in values)
                for metric in metrics
            },
        }

    payload = {
        "status": "complete",
        "paired_query_count": len(per_query),
        "coverage": len(selected) / len(per_query),
        "overall_population": summary(per_query),
        "selected_interventions": summary(selected),
        "fallback_queries": summary(fallback),
        "calculation": "direct paired per-query deltas; no aggregate-delta/coverage division",
    }
    write_json(OUT / "selected_query_effects.json", payload)
    return payload


def offline_cost() -> dict[str, Any]:
    full_actions = file_info(V4 / "outputs/generated_actions/v4_outer_test_actions.jsonl")
    full_outcomes = file_info(V4 / "outputs/action_outcomes/v4_action_outputs.jsonl")
    lite_actions = file_info(HERE / "outputs/lite_model/lite_actions_development.jsonl")
    lite_outcomes = file_info(HERE / "outputs/lite_model/lite_action_outcomes.jsonl")
    manifest = read_json(V4 / "outputs/action_outcomes/reader_environment_manifest.json", {})
    payload = {
        "status": "partial_measurement",
        "scope": "offline supervised construction only; excluded from online deployment latency",
        "full_v4": {
            "generated_action_rows": full_actions["rows"],
            "reader_outcome_evaluations": full_outcomes["rows"],
            "action_label_storage_bytes": full_actions["bytes"] + full_outcomes["bytes"],
            "reader_model": manifest.get("model_path", "[NEEDS SOURCE FILE]"),
            "generator_training_seconds": "[NEEDS MEASUREMENT]",
            "selector_training_seconds": "[NEEDS MEASUREMENT]",
            "support_predictor_training_seconds": "[NEEDS MEASUREMENT]",
            "total_gpu_hours": "[NOT AVAILABLE]",
        },
        "lite": {
            "generated_action_rows": lite_actions["rows"],
            "reader_outcome_evaluations_total_available": lite_outcomes["rows"],
            "new_unique_reader_evaluations": read_json(HERE / "outputs/lite_model/lite_generator_audit.json", {}).get("n_pending_contexts", "[NEEDS MEASUREMENT]"),
            "action_label_storage_bytes": lite_actions["bytes"] + lite_outcomes["bytes"],
            "generator_training_seconds": "[NEEDS MEASUREMENT]",
            "selector_training_seconds": "[NEEDS MEASUREMENT]",
            "support_predictor_training_seconds": "[NEEDS MEASUREMENT]",
            "total_gpu_hours": "[NOT AVAILABLE]",
        },
        "interpretation": "Reader evaluations for candidate actions are offline supervision. At deployment, each method executes the final answer reader once per query.",
    }
    write_json(OUT / "offline_training_cost.json", payload)
    return payload


def context_for_system(system: str) -> list[dict[str, Any]]:
    full_actions = {
        str(row["action_id"]): row for row in read_jsonl(V4 / "outputs/generated_actions/v4_outer_test_actions.jsonl")
    }
    if system == "frozen_top5_baseline":
        return [row for row in full_actions.values() if row["action_family"] == "fallback"]
    if system == "full_v4":
        selections = read_jsonl(V4 / "outputs/nested_selector/v4_nested_per_query.jsonl")
        return [full_actions[str(row["action_id"])] for row in selections]
    if system == "lite_method":
        decision = read_json(HERE / "outputs/lite_model/lite_architecture_decision.json", {})
        variant = decision.get("selected_variant")
        if not variant:
            raise RuntimeError("Lite architecture has not been frozen")
        actions = {str(row["action_id"]): row for row in read_jsonl(HERE / "outputs/lite_model/lite_actions_development.jsonl")}
        selections = [row for row in read_jsonl(HERE / "outputs/lite_model/lite_nested_per_query.jsonl") if row["variant"] == variant]
        return [actions[str(row["action_id"])] for row in selections]
    recomp_method = "recomp_top1" if system == "recomp_top1" else "recomp_budget_660"
    return [
        {
            "query_id": row["query_id"],
            "question": row["question"],
            "context_docs": [
                {"title": sentence["title"], "text": sentence["text"]} for sentence in row["sentences"]
            ],
        }
        for row in read_jsonl(HERE / "outputs/recomp/contexts_development.jsonl")
        if row["method"] == recomp_method
    ]


def benchmark_reader(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    rows = context_for_system(args.system)[: args.sample_size]
    tokenizer = AutoTokenizer.from_pretrained(args.reader_model, local_files_only=True, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.reader_model, local_files_only=True, torch_dtype=torch.float16
    ).to(args.device)
    model.eval()
    device = torch.device(args.device)
    latencies, token_counts = [], []
    peak_memory = 0
    for index, row in enumerate(rows):
        context = "\n".join(
            f"[{rank}] {doc['title']}: {doc['text']}" for rank, doc in enumerate(row["context_docs"], 1)
        )
        prompt_text = (
            "Answer the question using only the context. Return a short answer.\n\n"
            f"Question: {row['question']}\n\nContext:\n{context[:3200]}\n\nAnswer:"
        )
        encoded = tokenizer(prompt_text, truncation=True, max_length=1024, return_tensors="pt").to(device)
        token_counts.append(len(tokenizer.encode(context[:3200], add_special_tokens=True)))
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        torch.cuda.synchronize(device)
        latency = time.perf_counter() - started
        if index >= args.warmup:
            latencies.append(latency)
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated(device)))
    payload = {
        "status": "complete",
        "system": args.system,
        "sample_size": len(latencies),
        "warmup": args.warmup,
        "reader_calls_per_query": 1,
        "reader_latency_seconds": {
            "mean": mean(latencies),
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
        "reader_throughput_queries_per_second": 1.0 / mean(latencies),
        "peak_gpu_memory_bytes": peak_memory,
        "context_tokens_mean": mean(token_counts[args.warmup :]),
        "context_generator_latency": "[NEEDS MEASUREMENT]",
        "encoder_calls_per_query": 0 if args.system == "frozen_top5_baseline" else "see static call audit",
        "cross_encoder_calls_per_query": 10 if args.system == "full_v4" else 0,
        "pair_scoring_calls_per_query": 10 if args.system in ("full_v4", "lite_method") else 0,
    }
    write_json(OUT / f"runtime_benchmark_{args.system}.json", payload)
    print(json.dumps(payload, indent=2))


def summarize_runtime() -> dict[str, Any]:
    cfg = config()
    systems = {}
    for system in SYSTEMS:
        measured = read_json(OUT / f"runtime_benchmark_{system}.json")
        if measured:
            systems[system] = measured
        else:
            systems[system] = {
                "status": "[NEEDS MEASUREMENT]",
                "reader_calls_per_query": 1,
                "cross_encoder_calls_per_query": 10 if system == "full_v4" else 0,
                "encoder_calls_per_query": 1 if system in ("full_v4", "lite_method", "recomp_top1", "recomp_budgetmatched") else 0,
                "pair_scoring_calls_per_query": 10 if system in ("full_v4", "lite_method") else 0,
            }
    payload = {
        "status": "reader_only_complete" if all(row.get("status") == "complete" for row in systems.values()) else "partial_measurement",
        "systems": systems,
        "all_final_reader_benchmarks_complete": all(row.get("status") == "complete" for row in systems.values()),
        "online_end_to_end_cost_measured": False,
        "missing_online_component": "context generator / encoder latency",
        "deployment_reader_calls_per_query": 1,
        "candidate_action_reader_calls_at_inference": 0,
        "lite_minimum_reduction_target": cfg["lite"]["minimum_online_latency_reduction"],
    }
    write_json(OUT / "runtime_benchmark.json", payload)
    return payload


def quality_cost_plot(runtime: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    lite = read_json(HERE / "outputs/lite_model/lite_nested_metrics.json", {})
    recomp = read_json(HERE / "outputs/recomp/recomp_budget_matched_metrics.json", {})
    selected_variant = read_json(HERE / "outputs/lite_model/lite_architecture_decision.json", {}).get(
        "selected_variant", "lite_lexical_pair"
    )
    quality = {
        "frozen_top5_baseline": lite.get("metrics", {}).get("baseline", {}).get("joint_f1"),
        "full_v4": lite.get("metrics", {}).get("full_v4", {}).get("joint_f1"),
        "lite_method": lite.get("metrics", {}).get(selected_variant, {}).get("joint_f1"),
        "recomp_top1": recomp.get("metrics", {}).get("recomp_top1", {}).get("joint_f1"),
        "recomp_budgetmatched": recomp.get("metrics", {}).get("recomp_budget_660", {}).get("joint_f1"),
    }
    points = []
    for system, row in runtime["systems"].items():
        latency = row.get("reader_latency_seconds", {}).get("mean")
        joint = quality.get(system)
        if isinstance(latency, (int, float)) and isinstance(joint, (int, float)):
            points.append((system, float(latency), float(joint)))
    if not points:
        return
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for system, latency, joint in points:
        axis.scatter(latency, joint, s=55)
        axis.annotate(system.replace("_", " "), (latency, joint), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Final reader latency per query (seconds; generator excluded)")
    axis.set_ylabel("Development Joint F1")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output = HERE / "outputs/figures/quality_cost_tradeoff.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def fmt(value: Any, digits: int = 4) -> str:
    return value if isinstance(value, str) else f"{float(value):.{digits}f}"


def report() -> None:
    offline = offline_cost()
    effects = exact_intervention_effects()
    runtime = summarize_runtime()
    quality_cost_plot(runtime)
    lines = [
        "# Computational Cost Report",
        "",
        "## Deployment Answer",
        "",
        "At inference time the system executes the answer reader **once per query**, on the final selected context. Reader outcomes for candidate actions are generated only during offline supervised development and are not deployment-time reader calls.",
        "",
        "## Exact Paired Effects on the Frozen 3,000 Queries",
        "",
        "| Population | N | Coverage | Answer F1 delta | SP F1 delta | Joint F1 delta | Answer gain / 100 | Joint gain / 100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("All queries", "overall_population"), ("Selected interventions", "selected_interventions"), ("Fallback", "fallback_queries")):
        row = effects[key]
        coverage = effects["coverage"] if key == "selected_interventions" else (1.0 if key == "overall_population" else 1.0 - effects["coverage"])
        lines.append(f"| {label} | {row['n']} | {coverage:.4f} | {row['answer_f1_delta']:+.4f} | {row['sp_f1_delta']:+.4f} | {row['joint_f1_delta']:+.4f} | {row['answer_f1_gain_per_100_queries']:+.2f} | {row['joint_f1_gain_per_100_queries']:+.2f} |")
    lines.extend(["", "## Final Reader Runtime (Generator Excluded)", "", "The values below are measured after a context has been constructed. End-to-end online latency remains `[NEEDS MEASUREMENT]` because no comparable generator/encoder timing harness is available for every system.", "", "| System | Reader mean | Reader P50 | Reader P95 | Generator | End-to-end total | GPU memory | Reader calls | Cross-encoder calls | Context tokens |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for system, row in runtime["systems"].items():
        latency = row.get("reader_latency_seconds", {})
        lines.append(f"| {system} | {fmt(latency.get('mean', '[NEEDS MEASUREMENT]'))} | {fmt(latency.get('p50', '[NEEDS MEASUREMENT]'))} | {fmt(latency.get('p95', '[NEEDS MEASUREMENT]'))} | {row.get('context_generator_latency', '[NEEDS MEASUREMENT]')} | [NEEDS MEASUREMENT] | {fmt(row.get('peak_gpu_memory_bytes', '[NEEDS MEASUREMENT]'), 0)} | {row.get('reader_calls_per_query', 1)} | {row.get('cross_encoder_calls_per_query', 0)} | {fmt(row.get('context_tokens_mean', '[NEEDS MEASUREMENT]'), 1)} |")
    lines.extend(["", "## Offline Development", "", f"- Full V4 action reader evaluations: `{offline['full_v4']['reader_outcome_evaluations']}`", f"- Lite new unique reader evaluations: `{offline['lite']['new_unique_reader_evaluations']}`", f"- Full V4 action-label storage: `{offline['full_v4']['action_label_storage_bytes']}` bytes", f"- Total historical GPU hours: `{offline['full_v4']['total_gpu_hours']}`", "- Generator/selector/support training wall times: `[NEEDS MEASUREMENT]` unless recovered from an explicit timing manifest.", "", "Offline action labeling is expensive but amortized; it must not be described as online latency. The deployment scope is bounded post-retrieval pools and moderate-latency or offline QA, not streaming web-scale RAG."])
    text = "\n".join(lines)
    write_text(HERE / "reports/computational_cost_report.md", text)
    write_text(HERE / "computational_cost_report.md", text)
    write_text(HERE / "outputs/tables/complexity_and_latency.md", "\n".join(lines[lines.index("## Final Reader Runtime (Generator Excluded)") :]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["analyze", "benchmark-reader", "report"])
    parser.add_argument("--system", choices=SYSTEMS, default="frozen_top5_baseline")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reader-model", default=os.environ.get("V4_FLAN_T5_LARGE", FLAN))
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "analyze":
        print(json.dumps({"offline": offline_cost(), "effects": exact_intervention_effects()}, ensure_ascii=False, indent=2))
    elif args.stage == "benchmark-reader":
        benchmark_reader(args)
    else:
        report()


if __name__ == "__main__":
    main()
