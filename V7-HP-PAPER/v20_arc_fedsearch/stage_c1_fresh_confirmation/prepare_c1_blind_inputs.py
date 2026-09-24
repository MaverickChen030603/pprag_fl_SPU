#!/usr/bin/env python3
"""Materialize C1 query/source-document inputs while stripping every gold field."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCES = {
    "hotpotqa": "hotpot_train_v1.1.json",
    "2wikimultihopqa": "2wikimultihop_train.json",
    "musique": "musique_ans_v1.0_train.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def blind_row(dataset: str, row: dict) -> dict:
    identifier = str(row.get("_id", row.get("id")))
    if dataset == "musique":
        paragraphs = [
            {key: value for key, value in paragraph.items() if key in {"idx", "title", "paragraph_text"}}
            for paragraph in row.get("paragraphs", [])
        ]
        return {"id": identifier, "query_id": identifier, "question": str(row["question"]), "paragraphs": paragraphs}
    return {"_id": identifier, "query_id": identifier, "question": str(row["question"]), "context": row.get("context", [])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output_root.mkdir(parents=True, exist_ok=False)
    result = {"status": "complete_blind_input_materialization", "labels_loaded": False, "datasets": {}}
    banned = {"answer", "answer_aliases", "supporting_facts", "evidences", "is_supporting", "question_decomposition"}
    for dataset, selected in manifest["datasets"].items():
        source = load_source(args.source_root / SOURCES[dataset])
        by_id = {str(row.get("_id", row.get("id"))): row for row in source}
        out_path = args.output_root / f"{dataset}_c1_blind_n500.jsonl"
        rows = []
        for entry in selected:
            query_id = str(entry["query_id"])
            row = by_id.get(query_id)
            if row is None:
                raise KeyError(f"{dataset}: missing frozen query {query_id}")
            blinded = blind_row(dataset, row)
            if banned & set(blinded) or any(banned & set(value) for value in blinded.get("paragraphs", []) if isinstance(value, dict)):
                raise ValueError(f"{dataset}/{query_id}: blind-field firewall violation")
            rows.append(blinded)
        if len(rows) != 500 or len({row["query_id"] for row in rows}) != 500:
            raise ValueError(f"{dataset}: expected 500 unique rows")
        with out_path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        result["datasets"][dataset] = {"rows": len(rows), "path": str(out_path), "sha256": sha256(out_path)}
    (args.output_root / "blind_input_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
