#!/usr/bin/env python3
"""Freeze a public training prefix after checking it is disjoint from R5.

This importer is deliberately separate from the R5 artifacts.  It preserves
only public source rows for router training and never reads R5 label files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def read_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        yield from payload
        return
    for key in ("data", "examples", "train"):
        if isinstance(payload.get(key), list):
            yield from payload[key]
            return
    raise ValueError(f"unsupported public source container: {path}")


def query_id(row: dict[str, Any]) -> str:
    for key in ("query_id", "_id", "id", "qid", "question_id"):
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError("source row has no query identifier")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--public-source", type=Path, required=True)
    parser.add_argument("--r5-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    final_path = args.r5_inputs / f"{args.dataset}_final_test_inputs_n300.jsonl"
    final_ids = {query_id(row) for row in read_records(final_path)}
    selected: list[dict[str, Any]] = []
    skipped_final = 0
    seen: set[str] = set()
    for row in read_records(args.public_source):
        qid = query_id(row)
        if qid in final_ids:
            skipped_final += 1
            continue
        if qid in seen:
            raise ValueError(f"duplicate public query ID: {qid}")
        seen.add(qid)
        selected.append(row)
        if len(selected) == args.limit:
            break
    if len(selected) != args.limit:
        raise ValueError(f"public source has only {len(selected)} usable rows")

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "router_train_public.jsonl"
    with output.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "frozen_public_router_train",
        "dataset": args.dataset,
        "selection": "source_order_prefix_after_r5_query_id_exclusion",
        "public_source": str(args.public_source.resolve()),
        "public_source_sha256": sha256(args.public_source),
        "r5_inputs": str(final_path.resolve()),
        "r5_inputs_sha256": sha256(final_path),
        "queries": len(selected),
        "query_id_sha256": hashlib.sha256("\n".join(query_id(row) for row in selected).encode()).hexdigest(),
        "output_sha256": sha256(output),
        "r5_final_overlap": len(set(query_id(row) for row in selected) & final_ids),
        "source_rows_skipped_due_to_r5_overlap": skipped_final,
        "r5_labels_opened": False,
        "reader_started": False,
    }
    if manifest["r5_final_overlap"] != 0:
        raise AssertionError("public router train overlaps R5")
    atomic_json(args.output_dir / "training_split_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
