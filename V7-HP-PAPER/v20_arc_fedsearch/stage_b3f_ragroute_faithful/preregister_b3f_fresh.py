#!/usr/bin/env python3
"""Pre-register the untouched, label-free B3F-Fresh evaluation split."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
SOURCES = {
    "hotpotqa": "hf_hotpot/hotpot_train_v1.json",
    "2wikimultihopqa": "2wikimultihop_train.json",
    "musique": "musique_ans_v1.0_train.jsonl",
}
SALT = "v20-b3f-ragroute-fresh-20260928-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        yield from json.loads(path.read_text(encoding="utf-8"))


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def question_hash(question: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(question)).lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return hashlib.sha256(" ".join(normalized.split()).encode()).hexdigest()


def manifest_entries(path: Path) -> dict[str, list[dict[str, Any]]]:
    return json.loads(path.read_text(encoding="utf-8"))["datasets"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--c1-stage", type=Path, required=True)
    parser.add_argument("--c2-stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    historical = json.loads((args.c1_stage / "pre_registration_recovery/historical_manifests/historical_exclusion_union.json").read_text(encoding="utf-8"))["datasets"]
    c1 = manifest_entries(args.c1_stage / "pre_registration_recovery/protocol/c1_fresh_split_manifest.json")
    c2 = manifest_entries(args.c2_stage / "pre_registration/protocol/c2_fresh_split_manifest.json")

    selected: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    for dataset in DATASETS:
        historical_ids = set(map(str, historical[dataset]))
        c1_ids = {str(row["query_id"]) for row in c1[dataset]}
        c2_ids = {str(row["query_id"]) for row in c2[dataset]}
        c1_questions = {str(row["normalized_question_sha256"]) for row in c1[dataset]}
        c2_questions = {str(row["normalized_question_sha256"]) for row in c2[dataset]}
        excluded_ids = historical_ids | c1_ids | c2_ids
        excluded_questions = c1_questions | c2_questions
        eligible: list[tuple[str, int, str, str]] = []
        for index, row in enumerate(rows(args.base / "public_training_sources" / SOURCES[dataset])):
            identifier, digest = query_id(row), question_hash(str(row["question"]))
            if identifier in excluded_ids or digest in excluded_questions:
                continue
            rank = hashlib.sha256(f"{SALT}|{dataset}|{identifier}".encode()).hexdigest()
            eligible.append((rank, index, identifier, digest))
        sample = sorted(eligible)[:500]
        if len(sample) != 500:
            raise ValueError(f"{dataset}: insufficient unused public training queries")
        selected[dataset] = [
            {
                "dataset": dataset,
                "query_id": identifier,
                "source_row_zero_based": index,
                "normalized_question_sha256": digest,
                "source_split": "public_train_unused_after_historical_c1_c2_exclusion",
            }
            for _, index, identifier, digest in sample
        ]
        ids = {row["query_id"] for row in selected[dataset]}
        hashes = {row["normalized_question_sha256"] for row in selected[dataset]}
        audit.append({
            "dataset": dataset,
            "selected_n": len(ids),
            "historical_id_overlap_n": len(ids & historical_ids),
            "c1_id_overlap_n": len(ids & c1_ids),
            "c2_id_overlap_n": len(ids & c2_ids),
            "c1_question_overlap_n": len(hashes & c1_questions),
            "c2_question_overlap_n": len(hashes & c2_questions),
            "pass": len(ids) == 500 and len(hashes) == 500 and not (ids & excluded_ids) and not (hashes & excluded_questions),
        })

    args.output.mkdir(parents=True)
    protocol = args.output / "protocol"
    protocol.mkdir()
    manifest = {
        "stage": "V20-B3F-Fresh",
        "status": "frozen_before_retrieval",
        "purpose": "independent_protocol_faithful_RAGRoute_baseline_evaluation",
        "sample_size_per_dataset": 500,
        "selection_salt": SALT,
        "selection_rule": "smallest SHA256 salt rank after historical, C1 and C2 ID plus normalized-question exclusions",
        "datasets": selected,
        "source_sha256": {dataset: sha256(args.base / "public_training_sources" / SOURCES[dataset]) for dataset in DATASETS},
        "retrieval_started": False,
        "reader_started": False,
        "labels_scored": False,
    }
    (protocol / "b3f_fresh_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (protocol / "b3f_fresh_zero_overlap_audit.json").write_text(json.dumps({"status": "pass" if all(row["pass"] for row in audit) else "fail", "datasets": audit}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ready_for_blind_b3f_fresh" if all(row["pass"] for row in audit) else "integrity_failure", "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
