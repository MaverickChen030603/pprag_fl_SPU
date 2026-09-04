#!/usr/bin/env python3
"""Freeze LR method rankings, top-5 transmission, and raw merge before evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def ids(docs: list[dict[str, Any]]) -> list[str]:
    return [str(doc["doc_id"]) for doc in docs]


def titles(docs: list[dict[str, Any]]) -> list[str]:
    return [str(doc["title"]) for doc in docs]


def rrf(*rankings: list[dict[str, Any]], k: int = 60) -> list[dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = str(doc["doc_id"])
            docs.setdefault(key, dict(doc))
            scores[key] += 1.0 / (k + rank)
    selected = sorted(docs, key=lambda key: (-scores[key], -float(docs[key]["dense_score"]), key))
    return [docs[key] | {"local_rank": rank, "local_score": float(scores[key])} for rank, key in enumerate(selected)]


def dense(ranking: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(doc) | {"local_rank": rank, "local_score": float(doc["dense_score"])} for rank, doc in enumerate(ranking)]


def raw_merge(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(docs, key=lambda doc: (-float(doc["local_score"]), -float(doc["dense_score"]), str(doc["doc_id"])))


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def method_ranking(local: dict[str, Any], method: str) -> list[dict[str, Any]]:
    q0_dense = list(local["q0_dense_top50"])
    if method in ("L0", "L1"):
        return dense(q0_dense)
    if method == "L2":
        return rrf(q0_dense, list(local["q0_bm25_top50"]))
    if method == "L3":
        return rrf(q0_dense[:25], list(local["q1_dense_top25"]))
    if method == "L4":
        return rrf(q0_dense[:25], list(local["q1_dense_top25"]), list(local["q0_bm25_top50"])[:25], list(local["q1_bm25_top25"]))
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--methods", default="L0")
    args = parser.parse_args()

    selections = list(jsonl(args.selections))
    rankings = {(str(row["query_id"]), int(row["client_id"])): row for row in jsonl(args.rankings)}
    methods = args.methods.split(",")
    frozen: list[dict[str, Any]] = []
    for selection in selections:
        query_id = str(selection["query_id"])
        selected = [int(client) for client in selection["selected_clients"]]
        local = {client: rankings[(query_id, client)] for client in selected}
        for method in methods:
            ranked = {client: method_ranking(local[client], method) for client in selected}
            local5 = [doc for client in selected for doc in ranked[client][:5]]
            local10 = [doc for client in selected for doc in ranked[client][:10]]
            local20 = [doc for client in selected for doc in ranked[client][:20]]
            local50 = [doc for client in selected for doc in ranked[client][:50]]
            merged = raw_merge(local5)
            frozen.append({
                "dataset": selection["dataset"], "query_id": query_id, "track": selection["track"], "method": method,
                "selected_clients": json.dumps(selected), "candidate_top5": json.dumps(selection["candidate_top5"]),
                "selection_gold_only_diagnostic": bool(selection.get("gold_used_for_selection", False)),
                "local_top5_doc_ids": json.dumps(ids(local5)), "local_top5_titles": json.dumps(titles(local5)),
                "local_top10_doc_ids": json.dumps(ids(local10)), "local_top10_titles": json.dumps(titles(local10)),
                "local_top20_doc_ids": json.dumps(ids(local20)), "local_top20_titles": json.dumps(titles(local20)),
                "local_top50_doc_ids": json.dumps(ids(local50)), "local_top50_titles": json.dumps(titles(local50)),
                "transmitted_doc_ids": json.dumps(ids(local5)), "transmitted_titles": json.dumps(titles(local5)),
                "raw_merged_top10_doc_ids": json.dumps(ids(merged[:10])), "raw_merged_top10_titles": json.dumps(titles(merged[:10])),
                "gold_or_answer_used": False, "reader_started": False,
            })
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "frozen_lr_outputs.csv", frozen)
    manifest = {"stage": "LR", "queries": len({row["query_id"] for row in frozen}), "tracks": sorted({row["track"] for row in frozen}),
                "methods": methods, "documents_per_client": 5, "total_documents": 15, "merged_depth": 10,
                "gold_or_answer_used": False, "reader_started": False, "final_test_accessed": False}
    (args.output_root / "frozen_output_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
