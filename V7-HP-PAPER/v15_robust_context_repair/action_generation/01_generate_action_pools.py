#!/usr/bin/env python3
"""Generate V15 enumerated or beam complete-context actions from frozen pools."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from action_generation import Document, beam_sequence_repairs, enumerate_set_repairs


def tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9]+", str(value).lower()) if len(token) > 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--beam-depth", type=int, default=2)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats = []
    with args.pool.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            raw_docs = row["documents"]
            title_tokens = [tokens(doc["title"]) for doc in raw_docs]
            docs = []
            for index, doc in enumerate(raw_docs):
                bridge = max((len(title_tokens[index] & other) / len(title_tokens[index] | other) if title_tokens[index] and other else 0.0 for j, other in enumerate(title_tokens) if j != index), default=0.0)
                docs.append(Document(doc_id=str(doc["doc_id"]), title=str(doc["title"]), text=str(doc["text"]), retrieval_score=float(doc.get("hybrid_score", 0.0)), cross_score=float(doc.get("cross_score", 0.0)), bridge_score=bridge, anchor_score=float(doc.get("hybrid_score", 0.0)) if index < 2 else 0.0, source_rank=index))
            baseline = docs[:5]
            started = time.perf_counter()
            if len(docs) <= 12:
                actions = enumerate_set_repairs(docs, baseline, args.top_k)
                generator = "enumerated"
            else:
                actions = beam_sequence_repairs(docs, baseline, args.beam_width, args.beam_depth, args.top_k)
                generator = "beam"
            elapsed = 1000.0 * (time.perf_counter() - started)
            for action in actions:
                target.write(json.dumps({"query_id": row["query_id"], "generator": generator, **action.to_dict()}, ensure_ascii=False) + "\n")
            stats.append({"query_id": row["query_id"], "pool_size": len(docs), "generator": generator, "actions": len(actions), "non_null_actions": sum(not action.is_baseline for action in actions), "generation_latency_ms": elapsed})
    stats_path = args.output.with_name(f"{args.output.stem}_statistics.csv")
    with stats_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]) if stats else ["query_id"])
        writer.writeheader()
        writer.writerows(stats)
    print(json.dumps({"status": "complete", "queries": len(stats), "output": str(args.output), "statistics": str(stats_path)}, indent=2))


if __name__ == "__main__":
    main()
