#!/usr/bin/env python3
"""Run the fixed-route M0/M1 allocation x label-free merge matrix.

All-client local depth is used only to materialize candidates.  Every
non-oracle row evaluates exactly the frozen Bc=3 client list inherited from
V17.  Gold support is read only after candidate generation for offline metrics
and the explicitly named A5 oracle-allocation upper analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterable


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def norm(value: str) -> str:
    return " ".join(str(value).lower().split())


def doc_id(dataset: str, title: str, text: str = "") -> str:
    identity = norm(title) if dataset != "musique" else norm(title) + "\n" + norm(text)
    return f"{dataset}:{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:20]}"


def support_ids(row: dict[str, Any], dataset: str) -> set[str]:
    if dataset == "musique":
        return {
            doc_id(dataset, item.get("title", ""), item.get("paragraph_text", ""))
            for item in row.get("paragraphs", [])
            if item.get("is_supporting", item.get("is_support", False))
        }
    facts = row.get("supporting_facts", [])
    titles = facts.get("title", []) if isinstance(facts, dict) else [item[0] for item in facts if item]
    return {doc_id(dataset, title) for title in titles}


def assignments(path: Path) -> dict[str, int]:
    return {str(row["doc_id"]): int(row["client_id"]) for row in rows(path)}


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    if not values:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def complete(support: set[str], docs: list[dict[str, Any]]) -> int:
    return int(bool(support) and support.issubset({str(doc["doc_id"]) for doc in docs}))


def recall(support: set[str], docs: list[dict[str, Any]]) -> float:
    return len(support & {str(doc["doc_id"]) for doc in docs}) / max(1, len(support))


def groups(pool: dict[str, Any], selected: list[int]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {client: [] for client in selected}
    for doc in pool["pool"]:
        client = int(doc["client_id"])
        if client in result:
            result[client].append(doc)
    for values in result.values():
        values.sort(key=lambda doc: (int(doc.get("local_rank", 10**6)), -float(doc.get("local_hybrid_score", doc.get("hybrid_score", 0.0))), str(doc["doc_id"])))
    return result


def confidence_order(selected: list[int], inherited: dict[str, Any]) -> list[int]:
    scores = {int(key): float(value) for key, value in inherited.get("client_scores", {}).items()}
    return sorted(selected, key=lambda client: (-scores.get(client, float("-inf")), client))


def allocation_equal(local: dict[int, list[dict[str, Any]]], ordered: list[int]) -> list[dict[str, Any]]:
    return [doc for client in ordered for doc in local[client][:5]]


def allocation_depths(local: dict[int, list[dict[str, Any]]], ordered: list[int], depths: list[int]) -> list[dict[str, Any]]:
    return [doc for client, depth in zip(ordered, depths) for doc in local[client][:depth]]


def allocation_proportional(local: dict[int, list[dict[str, Any]]], selected: list[int], inherited: dict[str, Any]) -> list[dict[str, Any]]:
    """Allocate 15 docs with a two-document minimum and source-score shares."""
    scores = {int(key): float(value) for key, value in inherited.get("client_scores", {}).items()}
    raw = [scores.get(client, 0.0) for client in selected]
    floor = min(raw) if raw else 0.0
    weights = [max(value - floor, 1e-6) for value in raw]
    remaining = 9
    expected = [2.0 + remaining * weight / sum(weights) for weight in weights]
    depths = [min(10, int(value)) for value in expected]
    # Largest-remainder allocation with a hard local-depth-10 cap.  A highly
    # confident source may saturate; residual budget is then redistributed.
    while sum(depths) < 15:
        candidates = [index for index in range(len(selected)) if depths[index] < 10]
        if not candidates:
            raise AssertionError("cannot allocate 15 documents within local depth cap")
        index = max(candidates, key=lambda value: (expected[value] - depths[value], weights[value], -selected[value]))
        depths[index] += 1
    return allocation_depths(local, selected, depths)


def allocation_round_robin(local: dict[int, list[dict[str, Any]]], ordered: list[int]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rank in range(5):
        output.extend(local[client][rank : rank + 1] for client in ordered)
    return [item for group in output for item in (group if isinstance(group, list) else [group])]


def allocation_oracle(local: dict[int, list[dict[str, Any]]], ordered: list[int], support: set[str]) -> list[dict[str, Any]]:
    """Gold-only upper analysis; never a deployable allocation."""
    candidates = [doc for client in ordered for doc in local[client]]
    positives = [doc for doc in candidates if str(doc["doc_id"]) in support]
    remainder = [doc for doc in candidates if str(doc["doc_id"]) not in support]
    remainder.sort(key=lambda doc: (-float(doc.get("local_hybrid_score", doc.get("hybrid_score", 0.0))), str(doc["doc_id"])))
    return (positives + remainder)[:15]


def merge_raw(docs: list[dict[str, Any]], _local: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return sorted(docs, key=lambda doc: (-float(doc.get("local_hybrid_score", doc.get("hybrid_score", 0.0))), -float(doc.get("dense_score", 0.0)), str(doc["doc_id"])))


def merge_percentile(docs: list[dict[str, Any]], local: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    def score(doc: dict[str, Any]) -> tuple[float, float, str]:
        values = local[int(doc["client_id"])]
        return (1.0 - int(doc["local_rank"]) / max(1, len(values)), float(doc.get("dense_score", 0.0)), str(doc["doc_id"]))
    return [item[-1] for item in sorted(((*score(doc), doc) for doc in docs), reverse=True)]


def merge_rrf(docs: list[dict[str, Any]], _local: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return sorted(docs, key=lambda doc: (-(1.0 / (61 + int(doc["local_rank"]))), -float(doc.get("dense_score", 0.0)), str(doc["doc_id"])))


def merge_zscore(docs: list[dict[str, Any]], local: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    stats: dict[int, tuple[float, float]] = {}
    for client, values in local.items():
        scores = [float(doc.get("local_hybrid_score", doc.get("hybrid_score", 0.0))) for doc in values]
        stats[client] = (mean(scores), max(pstdev(scores), 1e-8))
    return sorted(docs, key=lambda doc: (-(float(doc.get("local_hybrid_score", doc.get("hybrid_score", 0.0))) - stats[int(doc["client_id"])][0]) / stats[int(doc["client_id"])][1], -float(doc.get("dense_score", 0.0)), str(doc["doc_id"])))


def merge_cdf(docs: list[dict[str, Any]], local: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    def score(doc: dict[str, Any]) -> tuple[float, float, str]:
        values = local[int(doc["client_id"])]
        raw = float(doc.get("local_hybrid_score", doc.get("hybrid_score", 0.0)))
        cdf = sum(float(item.get("local_hybrid_score", item.get("hybrid_score", 0.0))) <= raw for item in values) / max(1, len(values))
        return (cdf, float(doc.get("dense_score", 0.0)), str(doc["doc_id"]))
    return [item[-1] for item in sorted(((*score(doc), doc) for doc in docs), reverse=True)]


MERGERS: dict[str, Callable[[list[dict[str, Any]], dict[int, list[dict[str, Any]]]], list[dict[str, Any]]]] = {
    "M0_raw": merge_raw,
    "M1_rank_percentile": merge_percentile,
    "M2_rrf": merge_rrf,
    "M3_zscore": merge_zscore,
    "M4_empirical_cdf": merge_cdf,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--depth-pool", type=Path, required=True)
    parser.add_argument("--inherited-pool", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = {query_id(row): row for row in rows(args.split)}
    depth_pool = {str(row["query_id"]): row for row in rows(args.depth_pool)}
    inherited = {str(row["query_id"]): row for row in rows(args.inherited_pool)}
    client_for_doc = assignments(args.assignment)
    ids = [qid for qid in depth_pool if qid in data and qid in inherited]
    if not ids:
        raise ValueError("no aligned queries")

    stage_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    for qid in ids:
        support = support_ids(data[qid], args.dataset)
        gold_clients = sorted({client_for_doc[doc] for doc in support if doc in client_for_doc})
        selected = [int(value) for value in inherited[qid]["selected_clients"]]
        if len(selected) != 3:
            raise ValueError(f"{qid}: inherited route is not frozen Bc=3")
        local = groups(depth_pool[qid], selected)
        if any(len(local[client]) < 10 for client in selected):
            raise ValueError(f"{qid}: missing local depth-10 candidates for a selected client")
        ordered = confidence_order(selected, inherited[qid])
        local5 = [doc for client in selected for doc in local[client][:5]]
        local10 = [doc for client in selected for doc in local[client][:10]]
        rank_map = {str(doc["doc_id"]): f"{doc['client_id']}:{doc['local_rank']}" for doc in local10}
        ordered_support = sorted(support)
        support_rank_values = [rank_map.get(doc) for doc in ordered_support]
        support_rank_numbers = [int(value.rsplit(":", 1)[1]) for value in support_rank_values if value is not None]
        support_1_rank = support_rank_values[0] if len(support_rank_values) >= 1 else None
        support_2_rank = support_rank_values[1] if len(support_rank_values) >= 2 else None
        worst_support_rank = max(support_rank_numbers) if support_rank_numbers else None

        allocations = {
            "A0_equal_5_5_5": allocation_equal(local, selected),
            "A1_confidence_proportional": allocation_proportional(local, selected, inherited[qid]),
            "A2_8_5_2": allocation_depths(local, ordered, [8, 5, 2]),
            "A3_7_4_4": allocation_depths(local, ordered, [7, 4, 4]),
            "A4_round_robin_rank": allocation_round_robin(local, selected),
            "A5_oracle_analysis": allocation_oracle(local, ordered, support),
        }
        if any(len(docs) != 15 for docs in allocations.values()):
            raise AssertionError(f"{qid}: allocation does not meet 15 document budget")

        stage_rows.append({
            "query_id": qid,
            "selected_clients": json.dumps(selected),
            "selected_client_set_is_frozen_bc3": 1,
            "candidate_materialization": "query_all_diagnostic_only",
            "gold_clients_offline_audit_only": json.dumps(gold_clients),
            "support_local_ranks": json.dumps({doc: rank_map.get(doc) for doc in ordered_support}),
            "support_1_local_rank": support_1_rank,
            "support_2_local_rank": support_2_rank,
            "worst_support_rank": worst_support_rank,
            "selected_client_local_complete_at_5": complete(support, local5),
            "selected_client_local_complete_at_10": complete(support, local10),
        })

        by_allocation: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for allocation, docs in allocations.items():
            by_allocation[allocation] = {}
            transmitted_complete = complete(support, docs)
            for merge_name, merge_fn in MERGERS.items():
                ranked = merge_fn(docs, local)
                by_allocation[allocation][merge_name] = ranked
                matrix_rows.append({
                    "query_id": qid,
                    "allocation": allocation,
                    "merge": merge_name,
                    "is_oracle_allocation": int(allocation == "A5_oracle_analysis"),
                    "clients_contacted": 3,
                    "documents_transmitted": len(docs),
                    "complete_transmitted_at_15": transmitted_complete,
                    "complete_merged_at_10": complete(support, ranked[:10]),
                    "support_recall_merged_at_10": recall(support, ranked[:10]),
                    "top10_doc_ids": json.dumps([str(doc["doc_id"]) for doc in ranked[:10]]),
                })
            raw = by_allocation[allocation]["M0_raw"][:10]
            percentile = by_allocation[allocation]["M1_rank_percentile"][:10]
            detail_rows.append({
                "query_id": qid,
                "allocation": allocation,
                "selected_clients": json.dumps(selected),
                "gold_clients": json.dumps(gold_clients),
                "support_local_ranks": json.dumps({doc: rank_map.get(doc) for doc in ordered_support}),
                "support_1_local_rank": support_1_rank,
                "support_2_local_rank": support_2_rank,
                "worst_support_rank": worst_support_rank,
                "complete_local5": complete(support, local5),
                "complete_local10": complete(support, local10),
                "transmitted_doc_ids": json.dumps([str(doc["doc_id"]) for doc in docs]),
                "complete_transmitted15": transmitted_complete,
                "raw_top10": json.dumps([str(doc["doc_id"]) for doc in raw]),
                "percentile_top10": json.dumps([str(doc["doc_id"]) for doc in percentile]),
                "support_lost_by_allocation": int(complete(support, local10) and not transmitted_complete),
                "support_lost_by_raw_merge": int(transmitted_complete and not complete(support, raw)),
                "support_rescued_by_calibration": int(not complete(support, raw) and complete(support, percentile)),
                "support_harmed_by_calibration": int(complete(support, raw) and not complete(support, percentile)),
            })
            allocation_rows.append({
                "query_id": qid,
                "allocation": allocation,
                "is_oracle_allocation": int(allocation == "A5_oracle_analysis"),
                "complete_local10": complete(support, local10),
                "complete_transmitted_at_15": transmitted_complete,
                "support_lost_by_allocation": int(complete(support, local10) and not transmitted_complete),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_query_stage_metrics.csv", stage_rows)
    write_csv(args.output_dir / "per_query_allocation_merge.csv", detail_rows)
    write_csv(args.output_dir / "allocation_results_per_query.csv", allocation_rows)
    write_csv(args.output_dir / "allocation_merge_per_query.csv", matrix_rows)

    summary: list[dict[str, Any]] = []
    for allocation in sorted({row["allocation"] for row in matrix_rows}):
        for merge in MERGERS:
            values = [row for row in matrix_rows if row["allocation"] == allocation and row["merge"] == merge]
            summary.append({
                "allocation": allocation,
                "merge": merge,
                "is_oracle_allocation": int(allocation == "A5_oracle_analysis"),
                "queries": len(values),
                "mean_clients_contacted": sum(float(row["clients_contacted"]) for row in values) / len(values),
                "mean_documents_transmitted": sum(float(row["documents_transmitted"]) for row in values) / len(values),
                "complete_transmitted_at_15": sum(float(row["complete_transmitted_at_15"]) for row in values) / len(values),
                "complete_merged_at_10": sum(float(row["complete_merged_at_10"]) for row in values) / len(values),
                "support_recall_merged_at_10": sum(float(row["support_recall_merged_at_10"]) for row in values) / len(values),
            })
    write_csv(args.output_dir / "allocation_merge_matrix.csv", summary)

    baseline = next(row for row in summary if row["allocation"] == "A0_equal_5_5_5" and row["merge"] == "M0_raw")
    candidates = [row for row in summary if not int(row["is_oracle_allocation"])]
    best = max(candidates, key=lambda row: (float(row["complete_merged_at_10"]), float(row["support_recall_merged_at_10"])))
    best_details = [row for row in detail_rows if row["allocation"] == best["allocation"]]
    best_raw_details = [row for row in detail_rows if row["allocation"] == best["allocation"]]
    if best["merge"] == "M1_rank_percentile":
        rescues = sum(int(row["support_rescued_by_calibration"]) for row in best_raw_details)
        harms = sum(int(row["support_harmed_by_calibration"]) for row in best_raw_details)
    else:
        rescues, harms = 0, 0
    merge_best = max([row for row in summary if row["allocation"] == "A0_equal_5_5_5" and not int(row["is_oracle_allocation"])], key=lambda row: float(row["complete_merged_at_10"]))
    decision = {
        "status": "m0_m1_matrix_complete",
        "dataset": args.dataset,
        "queries": len(ids),
        "formal_client_set": "frozen Bc=3 inherited selected_clients",
        "query_all_used_only_for": "candidate materialization",
        "reader_started": False,
        "baseline_complete_merged_at_10": float(baseline["complete_merged_at_10"]),
        "best_non_oracle": {"allocation": best["allocation"], "merge": best["merge"], "complete_merged_at_10": float(best["complete_merged_at_10"])},
        "best_a0_label_free_merge": {"merge": merge_best["merge"], "complete_merged_at_10": float(merge_best["complete_merged_at_10"])},
        "calibration_rescue_count_for_best": rescues,
        "calibration_harm_count_for_best": harms,
        "next_gate": "use only a separately frozen, cross-dataset replay before any reader evaluation",
    }
    (args.output_dir / "m0_m1_go_no_go.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
