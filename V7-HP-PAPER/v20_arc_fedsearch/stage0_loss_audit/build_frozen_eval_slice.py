#!/usr/bin/env python3
"""Seal a deterministic development-only evaluation slice for V20 replays.

The script copies records verbatim and never inspects answer or support fields.
Those fields remain unavailable to routing and candidate construction; separate
offline evaluation code is the only consumer of them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=100)
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen slice: {args.output}")
    selected = []
    for index, row in enumerate(rows(args.source)):
        if args.start_index <= index < args.start_index + args.count:
            selected.append(row)
    if len(selected) != args.count:
        raise ValueError(f"expected {args.count} rows, found {len(selected)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "frozen",
        "dataset": args.dataset,
        "source_split": "development",
        "source": str(args.source.resolve()),
        "start_index_zero_based": args.start_index,
        "count": len(selected),
        "query_ids": [query_id(row) for row in selected],
        "query_id_sha256": hashlib.sha256("\n".join(query_id(row) for row in selected).encode("utf-8")).hexdigest(),
        "output_sha256": sha256(args.output),
        "routing_or_candidate_code_reads_gold": False,
        "final_test_accessed": False,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "dataset", "count", "query_id_sha256", "output_sha256")}, indent=2))


if __name__ == "__main__":
    main()
