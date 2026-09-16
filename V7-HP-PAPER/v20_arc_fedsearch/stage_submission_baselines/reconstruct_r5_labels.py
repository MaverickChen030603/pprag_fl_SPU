#!/usr/bin/env python3
"""Reconstruct R5 labels from canonical sources after the historical unseal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
TOP_LEVEL_LABELS = {
    "answer",
    "answers",
    "answer_aliases",
    "gold_answer",
    "supporting_facts",
    "supporting_titles",
    "evidence",
    "evidences",
    "supporting_paragraphs",
    "reasoning_path",
    "question_decomposition",
    "answerable",
}


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
    raise ValueError(f"unsupported source container: {path}")


def query_id(row: dict[str, Any]) -> str:
    for key in ("query_id", "_id", "id", "qid", "question_id"):
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError("row has no query ID")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_labels(row: dict[str, Any], dataset: str) -> dict[str, Any]:
    labels = {
        "query_id": query_id(row),
        **{key: value for key, value in row.items() if key in TOP_LEVEL_LABELS},
    }
    if dataset == "musique" and isinstance(row.get("paragraphs"), list):
        labels["supporting_paragraphs"] = [
            {"idx": paragraph.get("idx"), "title": paragraph.get("title")}
            for paragraph in row["paragraphs"]
            if paragraph.get("is_supporting", paragraph.get("is_support", False))
        ]
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--hotpot-source", type=Path, required=True)
    parser.add_argument("--two-wiki-source", type=Path, required=True)
    parser.add_argument("--musique-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "hotpotqa": args.hotpot_source,
        "2wikimultihopqa": args.two_wiki_source,
        "musique": args.musique_source,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "reconstructed_after_historical_r5_unseal",
        "evidence_role": "retrospective_post_hoc_r5_revealed",
        "datasets": {},
    }
    for dataset in DATASETS:
        sample_path = args.sample_root / f"{dataset}_final_test_inputs_n300.jsonl"
        sample_rows = list(read_records(sample_path))
        target_order = [query_id(row) for row in sample_rows]
        if len(target_order) != 300 or len(set(target_order)) != 300:
            raise ValueError(f"invalid R5 sample for {dataset}")
        targets = set(target_order)
        matched = {}
        scanned = 0
        for source_row in read_records(sources[dataset]):
            scanned += 1
            qid = query_id(source_row)
            if qid in targets:
                matched[qid] = extract_labels(source_row, dataset)
        if set(matched) != targets:
            missing = sorted(targets - set(matched))
            raise ValueError(f"{dataset}: missing {len(missing)} labels: {missing[:3]}")
        output = args.output_root / f"{dataset}_final_test_labels.jsonl"
        if output.exists():
            raise FileExistsError(output)
        with output.open("x", encoding="utf-8") as handle:
            for qid in target_order:
                handle.write(json.dumps(matched[qid], ensure_ascii=False) + "\n")
        manifest["datasets"][dataset] = {
            "queries": len(target_order),
            "source": str(sources[dataset].resolve()),
            "source_rows_scanned": scanned,
            "source_sha256": sha256(sources[dataset]),
            "sample_sha256": sha256(sample_path),
            "label_sha256": sha256(output),
        }
    manifest_path = args.output_root / "reconstruction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
