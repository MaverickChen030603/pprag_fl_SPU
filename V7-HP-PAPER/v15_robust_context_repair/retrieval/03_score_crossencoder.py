#!/usr/bin/env python3
"""Attach matched CrossEncoder scores to a frozen candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval_common import iter_rows, query_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    from sentence_transformers import CrossEncoder

    questions = {query_id(row): str(row["question"]) for row in iter_rows(args.split)}
    model = CrossEncoder(args.checkpoint, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.pool.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            question = questions[str(row["query_id"])]
            docs = row["documents"]
            scores = model.predict([(question, f"{doc['title']}. {doc['text']}") for doc in docs], batch_size=args.batch_size, show_progress_bar=False)
            for doc, score in zip(docs, scores):
                doc["cross_score"] = float(score)
            row["crossencoder_order"] = [doc["doc_id"] for doc in sorted(docs, key=lambda doc: (-doc["cross_score"], doc["doc_id"]))]
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "complete", "pool": str(args.pool), "output": str(args.output), "checkpoint": args.checkpoint}, indent=2))


if __name__ == "__main__":
    main()

