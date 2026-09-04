#!/usr/bin/env python3
"""Offline H0 candidate evaluation; this is the first code allowed to read gold support."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


def doc_id(dataset: str, title: str, text: str = "") -> str:
    key = normalized(title) if dataset != "musique" else normalized(title) + "\n" + normalized(text)
    return f"{dataset}:{hashlib.sha1(key.encode()).hexdigest()[:20]}"


def support_docs(row: dict[str, Any], dataset: str) -> set[str]:
    if dataset == "musique":
        return {doc_id(dataset, item.get("title", ""), item.get("paragraph_text", "")) for item in row.get("paragraphs", []) if item.get("is_supporting", item.get("is_support", False))}
    facts = row.get("supporting_facts", {})
    titles = facts.get("title", []) if isinstance(facts, dict) else [item[0] for item in facts if item]
    return {doc_id(dataset, title) for title in titles}


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args()

    data = list(jsonl(args.split))
    if args.max_queries:
        data = data[: args.max_queries]
    assignment = {str(row["doc_id"]): int(row["client_id"]) for row in jsonl(args.assignment)}
    rankings = {str(row["query_id"]): row for row in jsonl(args.rankings)}
    timing = {str(row["query_id"]): float(row["candidate_inference_elapsed_ms"]) for row in jsonl(args.timing)}
    if set(rankings) != {query_id(row) for row in data}:
        raise AssertionError("ranking/query split mismatch")
    per_query: list[dict[str, Any]] = []
    for row in data:
        qid = query_id(row)
        gold_clients = sorted({assignment[doc] for doc in support_docs(row, args.dataset) if doc in assignment})
        for method, ranking in rankings[qid]["methods"].items():
            for cutoff in (3, 5, 8):
                chosen = [int(client) for client in ranking[:cutoff]]
                per_query.append({
                    "dataset": args.dataset,
                    "query_id": qid,
                    "method": method,
                    "L": cutoff,
                    "candidate_clients": json.dumps(chosen),
                    "gold_clients_offline_only": json.dumps(gold_clients),
                    "candidate_gold_client_recall_at_L": len(set(chosen) & set(gold_clients)) / max(1, len(gold_clients)),
                    "candidate_complete_client_set_recall_at_L": int(set(gold_clients) <= set(chosen)),
                    "candidate_inference_elapsed_ms": timing[qid],
                    "gold_used_only_for_offline_metrics": True,
                    "reader_started": False,
                    "final_test_accessed": False,
                })
    baseline = {(row["query_id"], row["L"]): int(row["candidate_complete_client_set_recall_at_L"]) for row in per_query if row["method"] == "P0_single_centroid"}
    profile_bytes = args.profiles.stat().st_size
    summary: list[dict[str, Any]] = []
    for method in ("P0_single_centroid", "REMP_rrf_p0_dense_lexical"):
        for cutoff in (3, 5, 8):
            subset = [row for row in per_query if row["method"] == method and row["L"] == cutoff]
            complete = mean(float(row["candidate_complete_client_set_recall_at_L"]) for row in subset)
            p0_complete = mean(baseline[(row["query_id"], cutoff)] for row in subset)
            rescue = sum(baseline[(row["query_id"], cutoff)] == 0 and int(row["candidate_complete_client_set_recall_at_L"]) == 1 for row in subset)
            harm = sum(baseline[(row["query_id"], cutoff)] == 1 and int(row["candidate_complete_client_set_recall_at_L"]) == 0 for row in subset)
            elapsed = [float(row["candidate_inference_elapsed_ms"]) for row in subset]
            summary.append({
                "dataset": args.dataset,
                "method": method,
                "L": cutoff,
                "queries": len(subset),
                "candidate_gold_client_recall_at_L": mean(float(row["candidate_gold_client_recall_at_L"]) for row in subset),
                "candidate_complete_client_set_recall_at_L": complete,
                "delta_vs_p0_complete": complete - p0_complete,
                "rescue_vs_p0": rescue,
                "harm_vs_p0": harm,
                "profile_file_bytes": profile_bytes,
                "profile_file_bytes_per_client": profile_bytes / 20,
                "candidate_query_latency_mean_ms": mean(elapsed),
                "candidate_query_latency_p95_ms": sorted(elapsed)[max(0, int(0.95 * len(elapsed)) - 1)],
                "gold_used_only_for_offline_metrics": True,
                "reader_started": False,
                "final_test_accessed": False,
            })
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "candidate_per_query.csv", per_query)
    write_csv(args.output_root / "candidate_summary.csv", summary)
    decision = {"stage": "CTD-CSR-H0", "dataset": args.dataset, "queries": len(data), "gold_used_only_for_offline_evaluation": True, "reader_started": False, "final_test_accessed": False}
    (args.output_root / "h0_evaluation_manifest.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
