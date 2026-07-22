#!/usr/bin/env python3
"""Generate exact fixed-budget action contexts with client metadata."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("v16_oracle_generator", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module_root = str(path.resolve().parents[1])
    if module_root not in sys.path:
        sys.path.insert(0, module_root)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stable_id(query_id: str, context: list[str], family: str) -> str:
    return hashlib.sha1((query_id + "|" + family + "|" + "|".join(context)).encode("utf-8")).hexdigest()[:16]


def enrich(candidate: dict[str, Any], pool_row: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    context = list(map(str, candidate["context_doc_ids"]))
    clients = [int(lookup[doc_id]["client_id"]) for doc_id in context]
    origin = int(pool_row["origin_client"])
    candidate.update({
        "dataset": pool_row["dataset"],
        "partition": pool_row["partition"],
        "origin_client": origin,
        "selected_clients": pool_row["selected_clients"],
        "client_budget": int(pool_row["client_budget"]),
        "local_k": int(pool_row["local_k"]),
        "context_client_ids": clients,
        "distinct_clients": len(set(clients)),
        "cross_client_docs": sum(client != origin for client in clients),
        "is_single_cross_action": int(candidate["depth"]) == 1 and any(client != origin for client in clients),
        "is_cross_composition": 2 <= int(candidate["depth"]) <= 3 and len(set(clients)) >= 2,
        "is_within_client_composition": 2 <= int(candidate["depth"]) <= 3 and len(set(clients)) == 1,
    })
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--v16-generator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-queries", type=int)
    args = parser.parse_args()
    impl = load_module(args.v16_generator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    context_count, query_count = 0, 0
    with args.output.open("w", encoding="utf-8") as handle:
        for index, pool_row in enumerate(rows(args.pool)):
            if args.max_queries is not None and index >= args.max_queries:
                break
            query_id = str(pool_row["query_id"])
            action_docs = impl.docs_from_row(pool_row, 10)
            baseline = list(map(str, pool_row["baseline_doc_ids"]))
            lookup = {str(doc["doc_id"]): doc for doc in pool_row["pool"]}
            seen: set[tuple[str, ...]] = set()
            candidates = list(impl.all_single_edits(query_id, baseline, action_docs))
            candidates.extend(impl.top10_subsets(query_id, baseline, action_docs))
            for candidate in candidates:
                key = tuple(map(str, candidate["context_doc_ids"]))
                if key in seen:
                    continue
                seen.add(key)
                handle.write(json.dumps(enrich(candidate, pool_row, lookup), ensure_ascii=False) + "\n")
                context_count += 1
            for client, context in pool_row.get("single_client_contexts", {}).items():
                context = list(map(str, context))
                key = tuple(context)
                if len(context) != 5 or key in seen:
                    continue
                seen.add(key)
                candidate = {
                    "query_id": query_id,
                    "trajectory_id": stable_id(query_id, context, f"single_client_{client}"),
                    "candidate_type": "single_client_context",
                    "single_client_id": int(client),
                    "depth": 5,
                    "actions": [],
                    "context_doc_ids": context,
                    "cheap_score": 0.0,
                    "is_baseline": key == tuple(baseline),
                }
                handle.write(json.dumps(enrich(candidate, pool_row, lookup), ensure_ascii=False) + "\n")
                context_count += 1
            query_count += 1
    print(json.dumps({"status": "complete", "queries": query_count, "contexts": context_count, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
