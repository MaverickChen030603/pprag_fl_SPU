#!/usr/bin/env python3
"""Prepare REM-P router-dev files from existing non-final V17 data splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as h:
        for line in h:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def write_jsonl(path: Path, data: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as h:
        for row in data:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.data_root / args.dataset / "calibration.jsonl"
    data = list(rows(source))
    if len(data) < 300:
        raise ValueError(f"{source} has {len(data)} rows; need at least 300")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev300 = data[:300]
    smoke100 = data[:100]
    write_jsonl(args.output_dir / "router_dev.jsonl", dev300)
    write_jsonl(args.output_dir / "router_dev_smoke100.jsonl", smoke100)

    manifest = {
        "stage": "REM-P",
        "dataset": args.dataset,
        "source_split": str(source.resolve()),
        "final_test_accessed": False,
        "splits": {
            "router_dev": {
                "source_split": "calibration",
                "start_index_zero_based": 0,
                "count": len(dev300),
                "query_id_sha256": digest([qid(x) for x in dev300]),
                "path": str((args.output_dir / "router_dev.jsonl").resolve()),
            },
            "router_dev_smoke100": {
                "source_split": "calibration",
                "start_index_zero_based": 0,
                "count": len(smoke100),
                "query_id_sha256": digest([qid(x) for x in smoke100]),
                "path": str((args.output_dir / "router_dev_smoke100.jsonl").resolve()),
            },
        },
        "reader_started": False,
    }
    (args.output_dir / "remp_split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"dataset": args.dataset, "router_dev": 300, "router_dev_smoke100": 100}, indent=2))


if __name__ == "__main__":
    main()

