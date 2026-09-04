#!/usr/bin/env python3
"""Enumerate all Bc=3 subsets and freeze retrieval artifacts without labels."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Iterable


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield dict(row)


def pool_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def ids(docs: list[dict[str, Any]]) -> list[str]:
    return [str(doc["doc_id"]) for doc in docs]


def titles(docs: list[dict[str, Any]]) -> list[str]:
    return [str(doc.get("title", "")) for doc in docs]


def merge_raw(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(docs, key=lambda doc: (-float(doc["local_score"]), -float(doc["dense_score"]), str(doc["doc_id"])))


def merge_percentile(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(docs, key=lambda doc: (int(doc["local_rank"]), -float(doc["dense_score"]), str(doc["doc_id"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--local-pool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    pools = {str(row["query_id"]): row for row in pool_rows(args.local_pool)}
    frozen: list[dict[str, Any]] = []
    for candidate in rows(args.candidates):
        query_id = str(candidate["query_id"])
        top5 = [int(client) for client in json.loads(candidate["candidate_clients_top5"])]
        if len(top5) != 5 or len(set(top5)) != 5:
            raise ValueError(f"invalid Top-5 candidate list for {query_id}")
        if query_id not in pools:
            raise KeyError(f"missing local pool for {query_id}")
        rankers = pools[query_id]["rankers"]
        for subset in itertools.combinations(top5, 3):
            local_by_client = {str(client): list(rankers[str(client)]["L0_dense"]) for client in subset}
            local5 = [doc for docs in local_by_client.values() for doc in docs[:5]]
            local10 = [doc for docs in local_by_client.values() for doc in docs[:10]]
            raw = merge_raw(local5)
            percentile = merge_percentile(local5)
            frozen.append(
                {
                    "dataset": candidate["dataset"],
                    "query_id": query_id,
                    "candidate_method": candidate["candidate_method"],
                    "subset_clients": json.dumps(list(subset)),
                    "is_naive_top3": int(list(subset) == top5[:3]),
                    "local_top5_doc_ids": json.dumps(ids(local5)),
                    "local_top5_titles": json.dumps(titles(local5)),
                    "local_top10_doc_ids": json.dumps(ids(local10)),
                    "local_top10_titles": json.dumps(titles(local10)),
                    "transmitted_doc_ids": json.dumps(ids(local5)),
                    "transmitted_titles": json.dumps(titles(local5)),
                    "raw_merged_top5_doc_ids": json.dumps(ids(raw[:5])),
                    "raw_merged_top5_titles": json.dumps(titles(raw[:5])),
                    "raw_merged_top10_doc_ids": json.dumps(ids(raw[:10])),
                    "raw_merged_top10_titles": json.dumps(titles(raw[:10])),
                    "percentile_merged_top5_doc_ids": json.dumps(ids(percentile[:5])),
                    "percentile_merged_top5_titles": json.dumps(titles(percentile[:5])),
                    "percentile_merged_top10_doc_ids": json.dumps(ids(percentile[:10])),
                    "percentile_merged_top10_titles": json.dumps(titles(percentile[:10])),
                    "retrieval_or_answer_or_gold_used": False,
                    "reader_started": False,
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "frozen_subset_retrieval.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frozen[0]))
        writer.writeheader()
        writer.writerows(frozen)
    manifest = {
        "stage": "R2-B0",
        "queries": len({row["query_id"] for row in frozen}),
        "subsets_per_query_per_method": 10,
        "local_ranker": "L0_dense",
        "local_depth": 10,
        "documents_per_client_transmitted": 5,
        "merged_depth": 10,
        "gold_or_answer_used": False,
        "reader_started": False,
        "final_test_accessed": False,
    }
    (args.output_root / "frozen_retrieval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
