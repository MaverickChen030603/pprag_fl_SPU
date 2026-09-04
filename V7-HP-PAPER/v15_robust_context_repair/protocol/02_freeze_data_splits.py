#!/usr/bin/env python3
"""Freeze fresh, stratified, non-overlapping V15 data splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SIZES = {"train": 5000, "development": 1000, "calibration": 1000, "final_test": 2000}
LABEL_KEYS = {
    "answer", "answers", "gold_answer", "supporting_facts", "supporting_titles",
    "evidence", "evidences", "supporting_paragraphs", "reasoning_path",
}


def read_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "examples", "train"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported dataset container in {path}")


def query_id(row: dict[str, Any]) -> str:
    for key in ("query_id", "_id", "id", "qid", "question_id"):
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError("row has no query ID")


def stratum(row: dict[str, Any], dataset: str) -> str:
    if dataset == "hotpotqa":
        return f"{row.get('type', 'unknown')}::{row.get('level', 'unknown')}"
    for key in ("type", "question_type", "reasoning_type"):
        if row.get(key):
            return str(row[key])
    evidence = row.get("evidences", row.get("evidence", row.get("supporting_facts", [])))
    return f"evidence_count::{min(4, len(evidence) if isinstance(evidence, list) else 0)}"


def fingerprint_ids(ids: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(map(str, ids))) + "\n").encode("utf-8")).hexdigest()


def normalize_question(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def allocate_stratified(
    rows: list[dict[str, Any]], dataset: str, sizes: dict[str, int], seed: int
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[stratum(row, dataset)].append(row)
    rng = random.Random(seed)
    for key in sorted(grouped):
        rng.shuffle(grouped[key])

    ordered: list[dict[str, Any]] = []
    while any(grouped.values()):
        for key in sorted(grouped, key=lambda value: (-len(grouped[value]), value)):
            if grouped[key]:
                ordered.append(grouped[key].pop())

    required = sum(sizes.values())
    if len(ordered) < required:
        raise ValueError(f"{dataset}: need {required} unused examples, found {len(ordered)}")
    splits: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for name, size in sizes.items():
        splits[name] = ordered[offset:offset + size]
        offset += size
    return splits


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_labels(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    qid = query_id(row)
    inputs = {key: value for key, value in row.items() if key not in LABEL_KEYS}
    inputs["query_id"] = qid
    labels = {"query_id": qid}
    labels.update({key: value for key, value in row.items() if key in LABEL_KEYS})
    return inputs, labels


def load_used(path: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = {
        dataset: set(map(str, values))
        for dataset, values in payload.get("used_query_ids_by_dataset", {}).items()
    }
    questions = {
        dataset: set(map(str, values))
        for dataset, values in payload.get("used_normalized_questions_by_dataset", {}).items()
    }
    return ids, questions


def normalize_rows(rows: list[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    output = []
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        qid = query_id(row)
        if qid in seen:
            continue
        seen.add(qid)
        row["query_id"] = qid
        row["dataset"] = dataset
        row["source_split"] = "train"
        output.append(row)
    return output


def freeze_dataset(
    dataset: str,
    source: Path,
    used: set[str],
    used_questions: set[str],
    output_root: Path,
    sizes: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    rows = normalize_rows(read_json_records(source), dataset)
    available = [
        row for row in rows
        if query_id(row) not in used and normalize_question(row.get("question")) not in used_questions
    ]
    splits = allocate_stratified(available, dataset, sizes, seed)
    files: dict[str, dict[str, Any]] = {}
    dataset_root = output_root / dataset
    for split, split_rows in splits.items():
        ids = [query_id(row) for row in split_rows]
        if split != "final_test":
            path = dataset_root / f"{split}.jsonl"
            write_jsonl(path, split_rows)
            files[split] = {"path": str(path), "sha256": sha256(path), "query_fingerprint": fingerprint_ids(ids)}
            continue
        inputs, labels = zip(*(strip_labels(row) for row in split_rows))
        input_path = dataset_root / "final_test_inputs.jsonl"
        label_path = output_root / "sealed" / f"{dataset}_final_test_labels.jsonl"
        write_jsonl(input_path, inputs)
        write_jsonl(label_path, labels)
        os.chmod(label_path, 0o400)
        files[split] = {
            "input_path": str(input_path),
            "input_sha256": sha256(input_path),
            "sealed_label_path": str(label_path),
            "sealed_label_sha256": sha256(label_path),
            "query_fingerprint": fingerprint_ids(ids),
        }
    all_split_ids = {name: set(query_id(row) for row in values) for name, values in splits.items()}
    names = list(all_split_ids)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if all_split_ids[left] & all_split_ids[right]:
                raise AssertionError(f"{dataset}: split overlap {left}/{right}")
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "source_rows": len(rows),
        "previously_used_id_excluded": len({query_id(row) for row in rows} & used),
        "previously_used_question_excluded": sum(normalize_question(row.get("question")) in used_questions for row in rows),
        "available_after_exclusion": len(available),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "strata": {name: dict(Counter(stratum(row, dataset) for row in values)) for name, values in splits.items()},
        "files": files,
    }


def write_hash_manifest(root: Path, manifest: dict[str, Any]) -> None:
    paths: list[Path] = []
    for dataset in manifest["datasets"].values():
        for record in dataset["files"].values():
            for key in ("path", "input_path", "sealed_label_path"):
                if record.get(key):
                    paths.append(Path(record[key]))
    lines = [f"{sha256(path)}  {path}" for path in sorted(paths)]
    (root.parent / "artifact_sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpot-source", type=Path, required=True)
    parser.add_argument("--two-wiki-source", type=Path, required=True)
    parser.add_argument("--used-inventory", type=Path, default=Path(__file__).resolve().parent / "used_query_inventory.json")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--seed", type=int, default=20260721)
    for name, value in DEFAULT_SIZES.items():
        parser.add_argument(f"--{name.replace('_', '-')}-size", type=int, default=value)
    args = parser.parse_args()
    sizes = {
        "train": args.train_size,
        "development": args.development_size,
        "calibration": args.calibration_size,
        "final_test": args.final_test_size,
    }
    used, used_questions = load_used(args.used_inventory)
    unknown_questions = used_questions.get("unknown", set())
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "split_sizes": sizes,
        "provenance": "official_train_derived_frozen_evaluation",
        "confirmatory_warning": "not_official_dev_or_test; all prior used IDs excluded",
        "datasets": {
            "hotpotqa": freeze_dataset("hotpotqa", args.hotpot_source, used.get("hotpotqa", set()), used_questions.get("hotpotqa", set()) | unknown_questions, output, sizes, args.seed),
            "2wikimultihopqa": freeze_dataset("2wikimultihopqa", args.two_wiki_source, used.get("2wikimultihopqa", set()), used_questions.get("2wikimultihopqa", set()) | unknown_questions, output, sizes, args.seed + 1),
        },
    }
    manifest_path = Path(__file__).resolve().parent / "data_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_hash_manifest(output, manifest)
    print(json.dumps({"status": "complete", "manifest": str(manifest_path), "datasets": {k: v["split_counts"] for k, v in manifest["datasets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
