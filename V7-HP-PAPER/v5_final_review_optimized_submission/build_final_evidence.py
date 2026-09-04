#!/usr/bin/env python3
"""Assemble final frozen evidence tables, distributions, and cost figures."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
V4 = PROJECT / "V7-HP-PAPER/opportunity_aware_semantic_generation_v4"
V5 = PROJECT / "V7-HP-PAPER/review_driven_revision_v5"
OUT = HERE / "outputs"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def fmt_p(value: float) -> str:
    return "<0.0002" if value == 0 else f"{value:.4f}"


def fmt_ci(row: dict[str, Any], v4: bool = False) -> str:
    low = row["ci95_low"] if v4 else row["ci_low"]
    high = row["ci95_high"] if v4 else row["ci_high"]
    return f"[{low:+.4f}, {high:+.4f}]"


def method_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        result[str(row["method"])][str(row["query_id"])] = row
    return result


def selected_distribution(
    rows: list[dict[str, Any]],
    baseline_method: str,
    selected_method: str,
    selected_ids: set[str],
) -> dict[str, Any]:
    indexed = method_index(rows)
    baseline = indexed[baseline_method]
    selected = indexed[selected_method]
    query_ids = sorted(set(baseline) & set(selected))
    if set(query_ids) != set(baseline) or set(query_ids) != set(selected):
        raise AssertionError("Unpaired per-query outcomes")
    eps = 1e-12
    payload: dict[str, Any] = {
        "n_queries": len(query_ids),
        "selected_n": len(selected_ids),
        "fallback_n": len(query_ids) - len(selected_ids),
        "coverage": len(selected_ids) / len(query_ids),
        "label": "descriptive gains conditional on policy-selected interventions",
        "metrics": {},
    }
    for metric in ("answer_f1", "sp_f1", "joint_f1"):
        all_deltas = [float(selected[q][metric]) - float(baseline[q][metric]) for q in query_ids]
        deltas = [float(selected[q][metric]) - float(baseline[q][metric]) for q in query_ids if q in selected_ids]
        fallback = [float(selected[q][metric]) - float(baseline[q][metric]) for q in query_ids if q not in selected_ids]
        if any(abs(value) > eps for value in fallback):
            raise AssertionError(f"Fallback changed {metric}")
        wins = sum(value > eps for value in deltas)
        losses = sum(value < -eps for value in deltas)
        ties = len(deltas) - wins - losses
        payload["metrics"][metric] = {
            "population_mean_delta": mean(all_deltas),
            "selected_mean_delta": mean(deltas),
            "selected_median_delta": median(deltas),
            "selected_q25_delta": percentile(deltas, 0.25),
            "selected_q75_delta": percentile(deltas, 0.75),
            "selected_wins": wins,
            "selected_losses": losses,
            "selected_ties": ties,
            "selected_win_rate": wins / len(deltas),
            "selected_drop_rate": losses / len(deltas),
            "gain_per_100_interventions": 100 * mean(deltas),
            "gain_per_100_population_queries": 100 * mean(all_deltas),
            "fallback_all_exactly_zero": True,
        }
    return payload


def build_holdouts() -> dict[str, Any]:
    original = load_json(V4 / "outputs/scaleup/scaleup_summary.json")
    revision = load_json(V5 / "outputs/lite_model/lite_holdout_metrics.json")
    flan = original["official_dual_reader"]["readers"]["flan"]
    original_selected = load_json(V5 / "outputs/cost/selected_query_effects.json")
    original_rows = load_jsonl(V4 / "outputs/scaleup/official_metrics/flan_per_query.jsonl")
    original_selection_rows = load_jsonl(V4 / "outputs/scaleup/frozen_selector_selections_3000.jsonl")
    original_selected_ids = {str(row["query_id"]) for row in original_selection_rows if row["selected"]}
    original_distribution = selected_distribution(original_rows, "baseline", "v4_selected", original_selected_ids)
    rows = [
        {
            "split": "Original frozen holdout",
            "n": 3000,
            "coverage": original_selected["coverage"],
            "selected_n": original_selected["selected_interventions"]["n"],
            "baseline": flan["metrics"]["baseline"],
            "full": flan["metrics"]["v4_selected"],
            "statistics": flan["significance"],
            "answer_drop_rate_population": flan["answer_drop_rate"],
            "answer_drop_rate_selected": original_distribution["metrics"]["answer_f1"]["selected_drop_rate"],
            "statistics_schema": "v4",
        },
        {
            "split": "Untouched revision holdout",
            "n": 3405,
            "coverage": revision["selected_query_effects"]["full_v4"]["coverage"],
            "selected_n": revision["selected_query_effects"]["full_v4"]["n"],
            "baseline": revision["metrics"]["frozen_top5_baseline"],
            "full": revision["metrics"]["full_v4"],
            "statistics": revision["comparisons"]["full_v4_vs_frozen_top5_baseline"],
            "answer_drop_rate_population": None,
            "answer_drop_rate_selected": revision["selected_query_effects"]["full_v4"]["answer_drop_rate"],
            "statistics_schema": "v5",
        },
    ]
    payload = {
        "status": "complete",
        "claim_scope": "two disjoint same-source confirmations; no pooled significance claim",
        "development_n": 1000,
        "development_overlap": False,
        "full_frozen_before_both_holdouts": True,
        "revision_outcomes_unread_before_lite_architecture_freeze": True,
        "rows": rows,
    }
    write_json(OUT / "tables/two_frozen_same_source_holdouts.json", payload)
    with (OUT / "tables/two_frozen_same_source_holdouts.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["split", "n", "selected_n", "coverage", "baseline_answer_f1", "full_answer_f1", "answer_delta", "answer_ci", "answer_p", "sp_delta", "sp_ci", "sp_p", "joint_delta", "joint_ci", "joint_p", "selected_answer_drop_rate"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            v4 = row["statistics_schema"] == "v4"
            stats = row["statistics"]
            writer.writerow({
                "split": row["split"], "n": row["n"], "selected_n": row["selected_n"], "coverage": row["coverage"],
                "baseline_answer_f1": row["baseline"]["answer_f1"], "full_answer_f1": row["full"]["answer_f1"],
                "answer_delta": stats["answer_f1"]["mean" if v4 else "delta"], "answer_ci": fmt_ci(stats["answer_f1"], v4), "answer_p": fmt_p(stats["answer_f1"]["p_value"]),
                "sp_delta": stats["sp_f1"]["mean" if v4 else "delta"], "sp_ci": fmt_ci(stats["sp_f1"], v4), "sp_p": fmt_p(stats["sp_f1"]["p_value"]),
                "joint_delta": stats["joint_f1"]["mean" if v4 else "delta"], "joint_ci": fmt_ci(stats["joint_f1"], v4), "joint_p": fmt_p(stats["joint_f1"]["p_value"]),
                "selected_answer_drop_rate": row["answer_drop_rate_selected"],
            })
    lines = [
        "# Two Frozen Same-Source Holdouts", "",
        "| Split | N | Coverage | Baseline Answer F1 | Full Answer F1 | Delta | Answer 95% CI / p | SP F1 delta | SP 95% CI / p | Joint F1 delta | Joint 95% CI / p | Answer-drop rate |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|---:|",
    ]
    for row in rows:
        v4 = row["statistics_schema"] == "v4"
        stats = row["statistics"]
        value = lambda metric: stats[metric]["mean" if v4 else "delta"]
        answer_drop = row["answer_drop_rate_selected"]
        answer_drop_text = f"{answer_drop:.2%} selected"
        lines.append(
            f"| {row['split']} | {row['n']} | {row['coverage']:.1%} ({row['selected_n']}) | {row['baseline']['answer_f1']:.4f} | {row['full']['answer_f1']:.4f} | {value('answer_f1'):+.4f} | {fmt_ci(stats['answer_f1'], v4)} / {fmt_p(stats['answer_f1']['p_value'])} | {value('sp_f1'):+.4f} | {fmt_ci(stats['sp_f1'], v4)} / {fmt_p(stats['sp_f1']['p_value'])} | {value('joint_f1'):+.4f} | {fmt_ci(stats['joint_f1'], v4)} / {fmt_p(stats['joint_f1']['p_value'])} | {answer_drop_text} |"
        )
    lines += ["", "Both holdouts are disjoint from the 1,000-query development set. Full was frozen before either holdout was run. The revision holdout outcomes were not read before the Lite architecture was frozen; that holdout serves both as the independent Lite non-inferiority test and a Full replication. Both sets remain same-source HotpotQA evidence, and no pooled significance claim is made."]
    write_text(OUT / "tables/two_frozen_same_source_holdouts.md", "\n".join(lines))
    return payload


def build_selected_effects() -> dict[str, Any]:
    original_rows = load_jsonl(V4 / "outputs/scaleup/official_metrics/flan_per_query.jsonl")
    original_selections = load_jsonl(V4 / "outputs/scaleup/frozen_selector_selections_3000.jsonl")
    original_ids = {str(row["query_id"]) for row in original_selections if row["selected"]}
    revision_root = V5 / "outputs/lite_model/revision_holdout"
    revision_rows = load_jsonl(revision_root / "official_per_query_3405.jsonl")
    revision_selections = load_jsonl(revision_root / "full_v4_selections_3405.jsonl")
    revision_ids = {str(row["query_id"]) for row in revision_selections if row["selected"]}
    payload = {
        "status": "complete",
        "claim_boundary": "These are descriptive gains conditional on policy-selected interventions, not causal effects or expected gains for arbitrary queries.",
        "original_holdout_3000": selected_distribution(original_rows, "baseline", "v4_selected", original_ids),
        "revision_holdout_3405": selected_distribution(revision_rows, "frozen_top5_baseline", "full_v4", revision_ids),
    }
    write_json(OUT / "selected_effect/selected_effect_distribution.json", payload)
    original = payload["original_holdout_3000"]
    lines = [
        "# Selected-Policy Effect", "",
        "These are **descriptive gains conditional on policy-selected interventions**. They are not causal treatment effects, effects on arbitrary queries, or effects on all improvable queries.", "",
        f"Population effect (N={original['n_queries']}): Answer/SP/Joint F1 {original['metrics']['answer_f1']['population_mean_delta']:+.4f}/{original['metrics']['sp_f1']['population_mean_delta']:+.4f}/{original['metrics']['joint_f1']['population_mean_delta']:+.4f}.",
        f"The policy intervenes on {original['selected_n']}/{original['n_queries']} queries ({original['coverage']:.1%}); {original['fallback_n']} fallback queries have exactly zero context and metric change.", "",
        "| Metric | Mean | Median | Q25 | Q75 | Wins | Losses | Ties | Drop rate | Gain / 100 interventions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, label in (("answer_f1", "Answer F1"), ("sp_f1", "SP F1"), ("joint_f1", "Joint F1")):
        row = original["metrics"][metric]
        lines.append(f"| {label} | {row['selected_mean_delta']:+.4f} | {row['selected_median_delta']:+.4f} | {row['selected_q25_delta']:+.4f} | {row['selected_q75_delta']:+.4f} | {row['selected_wins']} | {row['selected_losses']} | {row['selected_ties']} | {row['selected_drop_rate']:.2%} | {row['gain_per_100_interventions']:+.2f} |")
    write_text(OUT / "tables/selected_policy_effect.md", "\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    indexed = method_index(original_rows)
    for axis, metric, title in zip(axes, ("answer_f1", "joint_f1"), ("Answer F1", "Joint F1")):
        deltas = [indexed["v4_selected"][qid][metric] - indexed["baseline"][qid][metric] for qid in original_ids]
        axis.hist(deltas, bins=31, color="#315f78", edgecolor="white", linewidth=0.4)
        axis.axvline(0, color="#a3392f", linewidth=1)
        axis.axvline(mean(deltas), color="#de8f05", linewidth=1.5, label=f"mean {mean(deltas):+.3f}")
        axis.set_title(title)
        axis.set_xlabel("Paired delta on selected interventions")
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Queries")
    fig.suptitle("Policy-selected effects are heterogeneous and descriptive")
    fig.tight_layout()
    fig.savefig(OUT / "figures/selected_delta_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    return payload


def build_recomp_and_transfer() -> None:
    recomp = load_json(V5 / "outputs/recomp/recomp_holdout_metrics.json")
    comparison = recomp["vs_baseline_significance"]["recomp_budget_660"]
    payload = {
        "status": "complete",
        "label": "official-compressor implementation under reader and budget adaptation",
        "protocol": recomp["protocol"],
        "metrics": {name: recomp["metrics"][name] for name in ("frozen_top5_baseline", "baseline_truncated_660", "recomp_budget_660", "full_v4")},
        "recomp_660_vs_baseline": comparison,
        "claim": "Under an approximately matched context budget and a standardized FLAN reader, RECOMP-style sentence packing does not improve the frozen multi-hop baseline, whereas Full context actions retain a positive same-source effect.",
        "boundary": "Matched tokens do not imply an identical structural action space; RECOMP and Full optimize different context-construction objectives.",
    }
    write_json(OUT / "audits/recomp_fairness.json", payload)
    lines = [
        "# Budget-Matched Compression", "",
        "The comparison uses an official-compressor implementation under reader and budget adaptation. It is not presented as an end-to-end reproduction of the original system.", "",
        "| System | Tokens | Documents represented | Answer F1 | SP F1 | Joint F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"frozen_top5_baseline": "Frozen Top-5", "baseline_truncated_660": "Baseline-Truncated-660", "recomp_budget_660": "RECOMP-660", "full_v4": "Full"}
    for name in labels:
        row = payload["metrics"][name]
        lines.append(f"| {labels[name]} | {row['context_tokens']:.1f} | {row['represented_documents']:.3f} | {row['answer_f1']:.4f} | {row['sp_f1']:.4f} | {row['joint_f1']:.4f} |")
    sig = comparison["joint_f1"]
    lines += ["", f"On the frozen 3,000-query holdout, RECOMP-660 changes Joint F1 by {sig['delta']:+.4f} (95% CI {fmt_ci(sig)}, p={fmt_p(sig['p_value'])}) relative to Frozen Top-5. Under an approximately matched budget and a standardized FLAN reader, sentence packing does not improve the frozen multi-hop baseline, while Full retains a positive same-source effect. The objectives and structural action spaces are not identical."]
    write_text(OUT / "tables/recomp_budget_matched_final.md", "\n".join(lines))

    transfer = load_json(V5 / "outputs/2wiki_calibration/calibration_results.json")
    summary_rows = []
    for k in ("16", "32", "64", "128"):
        for method, row in transfer["summary"][k].items():
            summary_rows.append({"k": int(k), "method": method, **row})
    best = min(summary_rows, key=lambda row: row["answer_drop_rate_mean"])
    if abs(best["answer_drop_rate_mean"] - 0.051016956806430494) > 1e-12:
        raise AssertionError("Frozen 2Wiki best calibration changed")
    external = {
        "status": "complete",
        "zero_shot": transfer["zero_shot"],
        "few_shot_rows": summary_rows,
        "best_calibrated": best,
        "target_answer_drop_rate": 0.04,
        "gate_passed": False,
        "claim": "Few-shot calibration partially reduces answer-drop risk but does not recover the in-domain safety level.",
    }
    write_json(OUT / "audits/external_transfer.json", external)
    with (OUT / "tables/2wiki_calibration_full.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["k", "method", "seeds", "coverage", "answer_drop_rate", "answer_f1", "sp_f1", "joint_f1", "ece", "brier"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({"k": row["k"], "method": row["method"], "seeds": row["seeds"], "coverage": row["coverage_mean"], "answer_drop_rate": row["answer_drop_rate_mean"], "answer_f1": row["answer_f1_mean"], "sp_f1": row["sp_f1_mean"], "joint_f1": row["joint_f1_mean"], "ece": row["ece_mean"], "brier": row["brier_mean"]})


def build_cost() -> dict[str, Any]:
    cost = load_json(OUT / "cost/frozen_end_to_end_latency.json")
    if not (cost["same_query_fingerprint"] and cost["all_reader_calls_per_query_one"] and cost["all_frozen_context_match"]):
        raise AssertionError("Cost protocol mismatch")
    lite = load_json(V5 / "outputs/lite_model/lite_nested_metrics.json")["metrics"]
    recomp = load_json(V5 / "outputs/recomp/recomp_budget_matched_metrics.json")["metrics"]
    quality = {
        "frozen_top5_baseline": recomp["frozen_top5_baseline"],
        "full_v4": recomp["full_v4"],
        "lite_lexical_pair": lite["lite_lexical_pair"],
        "baseline_truncated_660": recomp["baseline_truncated_660"],
        "recomp_top1": recomp["recomp_top1"],
        "recomp_budgetmatched": recomp["recomp_budget_660"],
    }
    fields = ["system", "answer_f1", "sp_f1", "joint_f1", "joint_delta_vs_baseline", "generator_mean_ms", "selector_mean_ms", "reader_mean_ms", "total_mean_ms", "total_median_ms", "total_p95_ms", "throughput_qps", "peak_gpu_memory_bytes", "encoder_calls_per_query", "cross_encoder_scores_per_query", "pairs_scored_per_query", "reader_calls_per_query", "context_tokens_mean"]
    rows = []
    base_joint = quality["frozen_top5_baseline"]["joint_f1"]
    for system, row in cost["systems"].items():
        q = quality[system]
        value = {
            "system": system, "answer_f1": q["answer_f1"], "sp_f1": q["sp_f1"], "joint_f1": q["joint_f1"], "joint_delta_vs_baseline": q["joint_f1"] - base_joint,
            "generator_mean_ms": 1000 * row["generator_only_latency"]["mean_seconds"], "selector_mean_ms": 1000 * row["selector_only_latency"]["mean_seconds"], "reader_mean_ms": 1000 * row["reader_only_latency"]["mean_seconds"],
            "total_mean_ms": 1000 * row["end_to_end_post_retrieval_latency"]["mean_seconds"], "total_median_ms": 1000 * row["end_to_end_post_retrieval_latency"]["median_seconds"], "total_p95_ms": 1000 * row["end_to_end_post_retrieval_latency"]["p95_seconds"],
            "throughput_qps": row["throughput_queries_per_second"], "peak_gpu_memory_bytes": row["peak_gpu_memory_bytes"], "encoder_calls_per_query": row["encoder_calls_per_query"], "cross_encoder_scores_per_query": row["cross_encoder_document_scores_per_query"], "pairs_scored_per_query": row["pairs_scored_per_query"], "reader_calls_per_query": row["final_reader_calls_per_query"], "context_tokens_mean": row["context_tokens_per_query"]["mean"],
        }
        rows.append(value)
    with (OUT / "cost/quality_cost_frontier.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    labels = {"frozen_top5_baseline": "Top-5", "full_v4": "Full", "lite_lexical_pair": "Lite", "baseline_truncated_660": "Truncated 660", "recomp_top1": "RECOMP Top1", "recomp_budgetmatched": "RECOMP 660"}
    fig, axis = plt.subplots(figsize=(8.5, 4.1))
    systems = [row["system"] for row in rows]
    reader = [row["reader_mean_ms"] for row in rows]
    generator = [row["generator_mean_ms"] for row in rows]
    selector = [row["selector_mean_ms"] for row in rows]
    other = [max(0, row["total_mean_ms"] - row["reader_mean_ms"] - row["generator_mean_ms"] - row["selector_mean_ms"]) for row in rows]
    x = list(range(len(rows)))
    axis.bar(x, generator, label="Generator", color="#4c78a8")
    axis.bar(x, selector, bottom=generator, label="Selector", color="#f58518")
    lower = [a + b for a, b in zip(generator, selector)]
    axis.bar(x, other, bottom=lower, label="Other online", color="#bab0ac")
    lower = [a + b for a, b in zip(lower, other)]
    axis.bar(x, reader, bottom=lower, label="Reader", color="#54a24b")
    axis.set_xticks(x, [labels[name] for name in systems], rotation=18, ha="right")
    axis.set_ylabel("Mean latency (ms/query)")
    axis.set_title("Measured end-to-end post-retrieval latency (50 warmup, 500 measured)")
    axis.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures/end_to_end_latency_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    offsets = {
        "frozen_top5_baseline": (4, -14),
        "lite_lexical_pair": (4, 8),
        "baseline_truncated_660": (4, 5),
        "recomp_top1": (4, 5),
        "recomp_budgetmatched": (4, 5),
        "full_v4": (4, 5),
    }
    for row in rows:
        axis.scatter(row["total_mean_ms"], row["joint_f1"], s=55)
        axis.annotate(labels[row["system"]], (row["total_mean_ms"], row["joint_f1"]), xytext=offsets[row["system"]], textcoords="offset points", fontsize=8)
    axis.set_xlabel("End-to-end post-retrieval mean latency (ms/query)")
    axis.set_ylabel("Development Joint F1")
    axis.set_title("Quality-cost frontier under a shared frozen protocol")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "figures/quality_cost_frontier.pdf", bbox_inches="tight")
    plt.close(fig)

    full = next(row for row in rows if row["system"] == "full_v4")
    baseline = next(row for row in rows if row["system"] == "frozen_top5_baseline")
    lite_row = next(row for row in rows if row["system"] == "lite_lexical_pair")
    lines = [
        "# Final Frozen Online Cost Report", "",
        f"All systems use the same {cost['environment']['gpu']} GPU on one host, batch size 1, the same first 550 frozen development queries (50 warmup and 500 measured), CUDA synchronization around every timed block, and one FLAN-T5-large reader call per query. Model loading is excluded. Online generation is recomputed, while each final context is checked against the frozen artifact; all match rates are 100%.", "",
        "| System | Generator ms | Selector ms | Reader ms | Total mean ms | Total median ms | P95 total ms | Throughput q/s | Peak GPU GiB | Encoders | Cross scores | Pairs | Reader calls | Tokens | Dev Joint delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {labels[row['system']]} | {row['generator_mean_ms']:.2f} | {row['selector_mean_ms']:.2f} | {row['reader_mean_ms']:.2f} | {row['total_mean_ms']:.2f} | {row['total_median_ms']:.2f} | {row['total_p95_ms']:.2f} | {row['throughput_qps']:.2f} | {row['peak_gpu_memory_bytes'] / 2**30:.2f} | {row['encoder_calls_per_query']} | {row['cross_encoder_scores_per_query']} | {row['pairs_scored_per_query']} | {row['reader_calls_per_query']} | {row['context_tokens_mean']:.1f} | {row['joint_delta_vs_baseline']:+.4f} |")
    full_overhead = full["total_mean_ms"] - baseline["total_mean_ms"]
    lite_saving = full["total_mean_ms"] - lite_row["total_mean_ms"]
    lines += [
        "", "## Full Component Breakdown", "",
        "| Component | Mean ms | Median ms | P95 ms |",
        "|---|---:|---:|---:|",
    ]
    for name, value in cost["systems"]["full_v4"]["component_latency"].items():
        if name != "end_to_end_post_retrieval":
            lines.append(f"| {name.replace('_', ' ')} | {1000 * value['mean_seconds']:.3f} | {1000 * value['median_seconds']:.3f} | {1000 * value['p95_seconds']:.3f} |")
    lines += [
        "", "## Interpretation", "",
        f"Full adds {full_overhead:+.2f} ms/query relative to Frozen Top-5 under this post-retrieval benchmark. Lite removes the semantic encoders and saves {lite_saving:.2f} ms/query relative to Full, but it failed the independent 0.002 Joint-F1 non-inferiority test and therefore is not the main method.",
        "The reader is not evaluated once per candidate action: every system invokes it exactly once after final context construction. This fact alone is not an efficiency claim; the table reports total online latency.",
        "Offline costs include action-outcome labeling and model training. Historical offline GPU-hour totals were not recorded and are therefore unavailable.",
        "The deployment claim is limited to selective context construction over a bounded post-retrieval candidate pool; open-domain indexing and streaming scalability were not evaluated.",
    ]
    write_text(HERE / "reports/final_cost_report.md", "\n".join(lines))
    write_text(HERE / "final_cost_report.md", "\n".join(lines))
    return {"status": "complete", "rows": rows, "full_overhead_ms": full_overhead, "lite_saving_vs_full_ms": lite_saving}


def main() -> None:
    for path in (OUT / "tables", OUT / "selected_effect", OUT / "figures", OUT / "audits", HERE / "reports"):
        path.mkdir(parents=True, exist_ok=True)
    holdouts = build_holdouts()
    selected = build_selected_effects()
    build_recomp_and_transfer()
    cost = build_cost()
    write_json(OUT / "audits/final_evidence_manifest.json", {"status": "complete", "holdouts": holdouts, "selected_effects": selected, "cost": cost})
    print(json.dumps({"status": "complete", "outputs": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
