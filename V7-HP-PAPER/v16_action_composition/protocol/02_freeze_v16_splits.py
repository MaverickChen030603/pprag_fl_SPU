#!/usr/bin/env python3
"""Freeze non-overlapping V16 train/dev/calibration/final splits for three datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SIZES = {"train": 5000, "development": 1000, "calibration": 1000, "final_test": 2000}
TOP_LEVEL_LABELS = {
    "answer", "answers", "answer_aliases", "gold_answer", "supporting_facts",
    "supporting_titles", "evidence", "evidences", "supporting_paragraphs",
    "reasoning_path", "question_decomposition", "answerable",
}


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("data", "examples", "train"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"unsupported source container: {path}")


def qid(row: dict[str, Any]) -> str:
    for key in ("query_id", "_id", "id", "qid", "question_id"):
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError("row has no query ID")


def norm_question(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("question", "")).strip().lower().split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(values: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(set(map(str, values)))) + "\n").encode("utf-8")).hexdigest()


def hop_count(row: dict[str, Any]) -> int:
    for key in ("hop_count", "num_hops", "hops"):
        if row.get(key) is not None:
            try:
                return int(row[key])
            except (TypeError, ValueError):
                pass
    decomposition = row.get("question_decomposition")
    if isinstance(decomposition, list) and decomposition:
        return min(4, len(decomposition))
    supports = row.get("supporting_facts", row.get("evidences", []))
    return min(4, len(supports)) if isinstance(supports, list) else 0


def stratum(row: dict[str, Any], dataset: str) -> str:
    if dataset == "hotpotqa":
        return f"{row.get('type', 'unknown')}::{row.get('level', 'unknown')}"
    kind = row.get("type", row.get("question_type", row.get("reasoning_type", "unknown")))
    return f"{kind}::hop{hop_count(row)}"


def split_rows(rows: list[dict[str, Any]], dataset: str, sizes: dict[str, int], seed: int) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum(row, dataset)].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    if sum(len(values) for values in groups.values()) < sum(sizes.values()):
        raise ValueError(f"{dataset}: need {sum(sizes.values())} unused examples; found {sum(len(values) for values in groups.values())}")
    output: dict[str, list[dict[str, Any]]] = {}
    for name, size in sizes.items():
        available = sum(len(values) for values in groups.values())
        exact = {key: size * len(values) / available for key, values in groups.items()}
        quota = {key: min(len(groups[key]), int(exact[key])) for key in groups}
        remaining = size - sum(quota.values())
        priority = sorted(groups, key=lambda key: (-(exact[key] - int(exact[key])), -len(groups[key]), key))
        while remaining:
            progressed = False
            for key in priority:
                if quota[key] < len(groups[key]):
                    quota[key] += 1
                    remaining -= 1
                    progressed = True
                    if not remaining:
                        break
            if not progressed:
                raise ValueError(f"{dataset}: unable to fill stratified split {name}")
        selected: list[dict[str, Any]] = []
        for key in sorted(groups):
            take = quota[key]
            selected.extend(groups[key][-take:] if take else [])
            if take:
                del groups[key][-take:]
        rng.shuffle(selected)
        output[name] = selected
    return output


def strip_final_labels(row: dict[str, Any], dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    query_id = qid(row)
    inputs = {key: value for key, value in row.items() if key not in TOP_LEVEL_LABELS}
    labels = {"query_id": query_id, **{key: value for key, value in row.items() if key in TOP_LEVEL_LABELS}}
    inputs["query_id"] = query_id
    if dataset == "musique" and isinstance(row.get("paragraphs"), list):
        clean_paragraphs, support_labels = [], []
        for paragraph in row["paragraphs"]:
            clean = {key: value for key, value in paragraph.items() if key not in {"is_supporting", "is_support"}}
            clean_paragraphs.append(clean)
            if paragraph.get("is_supporting", paragraph.get("is_support", False)):
                support_labels.append({"idx": paragraph.get("idx"), "title": paragraph.get("title")})
        inputs["paragraphs"] = clean_paragraphs
        labels["supporting_paragraphs"] = support_labels
    return inputs, labels


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def freeze_one(dataset: str, source: Path, used_ids: set[str], used_questions: set[str], output: Path, sizes: dict[str, int], seed: int) -> dict[str, Any]:
    source_rows = read_records(source)
    unique, seen, seen_questions = [], set(), set()
    duplicate_question_rows = 0
    for source_row in source_rows:
        row = dict(source_row)
        query_id = qid(row)
        if query_id in seen:
            continue
        question = norm_question(row)
        if question and question in seen_questions:
            duplicate_question_rows += 1
            continue
        seen.add(query_id)
        if question:
            seen_questions.add(question)
        row.update({"query_id": query_id, "dataset": dataset, "source_split": "train"})
        unique.append(row)
    available = [row for row in unique if qid(row) not in used_ids and norm_question(row) not in used_questions]
    splits = split_rows(available, dataset, sizes, seed)
    files: dict[str, Any] = {}
    split_ids: dict[str, set[str]] = {}
    for name, values in splits.items():
        split_ids[name] = {qid(row) for row in values}
        if name != "final_test":
            path = output / dataset / f"{name}.jsonl"
            write_jsonl(path, values)
            files[name] = {"path": str(path), "sha256": sha256(path), "query_fingerprint": fingerprint(split_ids[name])}
        else:
            pairs = [strip_final_labels(row, dataset) for row in values]
            input_path = output / dataset / "final_test_inputs.jsonl"
            label_path = output / "sealed" / f"{dataset}_final_test_labels.jsonl"
            write_jsonl(input_path, (pair[0] for pair in pairs))
            write_jsonl(label_path, (pair[1] for pair in pairs))
            os.chmod(label_path, 0o400)
            files[name] = {"input_path": str(input_path), "input_sha256": sha256(input_path), "sealed_label_path": str(label_path), "sealed_label_sha256": sha256(label_path), "query_fingerprint": fingerprint(split_ids[name])}
    for left, right in __import__("itertools").combinations(split_ids, 2):
        if split_ids[left] & split_ids[right]:
            raise AssertionError(f"{dataset}: {left}/{right} overlap")
    return {
        "source": str(source.resolve()), "source_sha256": sha256(source), "source_rows": len(source_rows),
        "unique_rows_after_question_deduplication": len(unique), "duplicate_question_rows_excluded": duplicate_question_rows,
        "excluded_used_ids": sum(qid(row) in used_ids for row in unique),
        "excluded_used_questions": sum(norm_question(row) in used_questions for row in unique),
        "available_after_exclusion": len(available), "split_counts": {key: len(value) for key, value in splits.items()},
        "strata": {key: dict(Counter(stratum(row, dataset) for row in value)) for key, value in splits.items()}, "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpot-source", type=Path, required=True)
    parser.add_argument("--two-wiki-source", type=Path, required=True)
    parser.add_argument("--musique-source", type=Path, required=True)
    parser.add_argument("--used-inventory", type=Path, default=Path(__file__).with_name("used_query_inventory.json"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--seed", type=int, default=20260722)
    for name, value in SIZES.items():
        parser.add_argument(f"--{name.replace('_', '-')}-size", type=int, default=value)
    args = parser.parse_args()
    sizes = {name: getattr(args, f"{name}_size") for name in SIZES}
    inventory = json.loads(args.used_inventory.read_text(encoding="utf-8"))
    used_ids = {key: set(map(str, values)) for key, values in inventory.get("used_query_ids_by_dataset", {}).items()}
    used_questions = {key: set(values) for key, values in inventory.get("used_normalized_questions_by_dataset", {}).items()}
    unknown = used_questions.get("unknown", set())
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {"hotpotqa": args.hotpot_source, "2wikimultihopqa": args.two_wiki_source, "musique": args.musique_source}
    manifest = {"schema_version": 1, "experiment": "V7-HP-PAPER-v16-synergy-aware-action-composition", "seed": args.seed, "split_sizes": sizes, "final_test_status": "sealed_not_for_development", "datasets": {}}
    for offset, (dataset, source) in enumerate(sources.items()):
        manifest["datasets"][dataset] = freeze_one(dataset, source, used_ids.get(dataset, set()), used_questions.get(dataset, set()) | unknown, output, sizes, args.seed + offset)
    manifest_path = Path(__file__).with_name("dataset_split_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hashes = {str(path): sha256(path) for path in output.rglob("*.jsonl")}
    Path(__file__).with_name("artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "splits": {key: value["split_counts"] for key, value in manifest["datasets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
