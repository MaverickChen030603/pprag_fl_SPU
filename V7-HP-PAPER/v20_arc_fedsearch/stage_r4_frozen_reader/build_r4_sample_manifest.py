#!/usr/bin/env python3
"""Create the reader-facing manifest from immutable materialized contexts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.contexts.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {}
    output = []
    for row in rows:
        expected[row["dataset"]] = expected.get(row["dataset"], set()) | {row["query_id"]}
        for reader in ("flan", "unifiedqa"):
            output.append({"dataset": row["dataset"], "query_id": row["query_id"], "method": row["method"], "reader": reader,
                           "retrieval_artifact_hash": hashlib.sha256((row["source"] + "|" + "|".join(row["retrieved_doc_ids"])).encode()).hexdigest(),
                           "context_hash": row["context_hash"], "expected_n": 300})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"expected_query_counts": {key: len(value) for key, value in expected.items()}, "rows": output}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"datasets": {key: len(value) for key, value in expected.items()}, "reader_method_rows": len(output)}, indent=2))


if __name__ == "__main__":
    main()
