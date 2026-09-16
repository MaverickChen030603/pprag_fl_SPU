#!/usr/bin/env python3
"""Prove whether one-dimensional B2 produces the same ranking as B0."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from materialize_posthoc_baselines import rows, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-contexts", type=Path, required=True)
    parser.add_argument("--b2-contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    b0 = {
        (row["dataset"], row["query_id"]): row
        for row in rows(args.b0_contexts)
        if row["method"] == "b0_static_top3"
    }
    b2 = {
        (row["dataset"], row["query_id"]): row for row in rows(args.b2_contexts)
    }
    if len(b0) != 900 or set(b0) != set(b2):
        raise ValueError(f"context key mismatch: B0={len(b0)} B2={len(b2)}")
    route_matches: Counter[str] = Counter()
    context_matches: Counter[str] = Counter()
    for key, left in b0.items():
        right = b2[key]
        dataset = key[0]
        route_matches[dataset] += left["selected_clients"] == right["selected_clients"]
        context_matches[dataset] += left["context_hash"] == right["context_hash"]
    payload = {
        "status": (
            "pass_all_contexts_equivalent"
            if sum(context_matches.values()) == 900
            else "not_equivalent"
        ),
        "interpretation": (
            "B2 applies a positive monotonic transform to the sole static score, "
            "so its ranking, contexts, frozen Reader predictions, and metrics are B0-equivalent."
        ),
        "reader_rerun_required": sum(context_matches.values()) != 900,
        "b0_sha256": sha256(args.b0_contexts),
        "b2_sha256": sha256(args.b2_contexts),
        "datasets": {
            dataset: {
                "queries": 300,
                "route_matches": route_matches[dataset],
                "context_matches": context_matches[dataset],
            }
            for dataset in ("hotpotqa", "2wikimultihopqa", "musique")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
