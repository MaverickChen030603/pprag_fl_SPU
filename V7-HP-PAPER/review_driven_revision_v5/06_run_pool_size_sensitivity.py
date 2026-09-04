#!/usr/bin/env python3
"""Audit bounded candidate-pool scope and top-L pair-pruning complexity."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

from v5_common import HERE, V4, config, read_jsonl, write_json, write_text


OUT = HERE / "outputs" / "pool_sensitivity"


def main() -> None:
    cfg = config()["pool_sensitivity"]
    rows = read_jsonl(V4 / "outputs/scaleup/frozen_baseline_contexts_3000.jsonl")
    requested = [int(value) for value in cfg["requested_pool_sizes"]]
    top_l = int(cfg["top_l_documents_for_pairing"])
    availability = {size: sum(len(row["all_docs"]) >= size for row in rows) for size in requested}
    fixed_subset_n = min(availability.values())
    sys.path.insert(0, str(V4))
    import v4_common

    results = []
    for size in requested:
        eligible = [row for row in rows if len(row["all_docs"]) >= size]
        pair_before = math.comb(size, 2)
        pair_after = math.comb(min(size, top_l), 2)
        if size == 10 and eligible:
            latencies, densities = [], []
            for row in eligible[:500]:
                docs = row["all_docs"][:size]
                baseline_ids = [doc_id for doc_id in row["baseline_doc_ids"] if doc_id in {doc["doc_id"] for doc in docs}]
                started = time.perf_counter()
                features = v4_common.lexical_doc_features(row["question"], docs, baseline_ids)
                latencies.append(time.perf_counter() - started)
                densities.append(mean(float(value["bridge_entity_match"] > 0) for value in features.values()))
            latency: Any = mean(latencies)
            positive_density: Any = mean(densities)
            opportunity: Any = "[NOT AVAILABLE: requires new reader outcomes]"
            answer_joint: Any = "[NOT AVAILABLE: only the frozen size-10 retrieval pool has reader outcomes]"
        else:
            latency = "[NOT AVAILABLE]"
            positive_density = "[NOT AVAILABLE]"
            opportunity = "[NOT AVAILABLE]"
            answer_joint = "[NOT AVAILABLE]"
        results.append(
            {
                "requested_pool_size": size,
                "eligible_queries_in_frozen_3000": len(eligible),
                "eligible_fraction": len(eligible) / len(rows),
                "common_fixed_subset_size_across_10_20_50_100": fixed_subset_n,
                "pair_count_before_pruning": pair_before,
                "pair_count_after_top_l_pruning": pair_after,
                "top_l": top_l,
                "lexical_generator_latency_seconds": latency,
                "bridge_positive_density_proxy": positive_density,
                "opportunity_coverage": opportunity,
                "answer_joint_result": answer_joint,
                "online_memory": "[NEEDS MEASUREMENT]" if size == 10 else "[NOT AVAILABLE]",
            }
        )
    payload = {
        "status": "scope_limited",
        "source": "same-source HotpotQA distractor frozen 3,000 contexts",
        "n_queries": len(rows),
        "results": results,
        "fixed_subset_protocol_feasible": fixed_subset_n >= 100,
        "reason": "The official per-query distractor pool is approximately 10 documents; no common fixed subset exposes 20/50/100 real candidate documents without changing the retrieval corpus.",
        "synthetic_cross_query_pooling_used": False,
        "open_domain_claim_allowed": False,
        "scope_statement": "The method targets reader-facing organization over a bounded retrieved pool; corpus-scale retrieval and streaming index maintenance are outside its scope.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "pool_size_results.json", payload)
    try:
        import matplotlib.pyplot as plt

        figures = HERE / "outputs" / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(5.4, 3.5))
        axis.plot(requested, [row["pair_count_before_pruning"] for row in results], marker="o", label="All pairs")
        axis.plot(requested, [row["pair_count_after_top_l_pruning"] for row in results], marker="s", label=f"Top-L (L={top_l})")
        axis.set_xlabel("Candidate pool size")
        axis.set_ylabel("Pair-scoring calls")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures / "pool_size_quality_latency.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        write_text(HERE / "outputs/figures/pool_size_plot_error.txt", str(exc))
    report = """# Candidate-Pool Sensitivity and Scope

The frozen HotpotQA distractor artifact does not provide a common 20/50/100-document per-query pool. Expanding it by mixing documents from other queries would change the retrieval problem and would not be a valid sensitivity analysis. We therefore report the observed pool availability, the measured size-10 lexical cost, and the exact pair-count bound under top-L pruning; unavailable quality cells are marked rather than imputed.

| Requested pool | Eligible frozen queries | All pairs | Pairs after top-L | Quality result |
|---:|---:|---:|---:|---|
"""
    for row in results:
        report += f"| {row['requested_pool_size']} | {row['eligible_queries_in_frozen_3000']} | {row['pair_count_before_pruning']} | {row['pair_count_after_top_l_pruning']} | {row['answer_joint_result']} |\n"
    report += "\n**Scope:** The method targets reader-facing organization over a bounded retrieved pool; corpus-scale retrieval and streaming index maintenance are outside its scope.\n"
    write_text(HERE / "reports/pool_size_sensitivity_report.md", report)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
