#!/usr/bin/env python3
"""Validate V17 split exclusivity, exclusion, hashes, labels, and permissions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROTOCOL = Path(__file__).resolve().parent
FORBIDDEN_TOP_LEVEL = {
    "answer", "answers", "answer_aliases", "gold_answer", "supporting_facts",
    "supporting_titles", "evidence", "evidences", "supporting_paragraphs",
    "reasoning_path", "question_decomposition", "answerable",
}
FORBIDDEN_NESTED = {"is_supporting", "is_support"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def norm(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("question", "")).strip().lower().split())


def nested_forbidden(value: Any, path: str = "") -> list[str]:
    findings = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_NESTED:
                findings.append(f"{path}/{key}")
            findings.extend(nested_forbidden(child, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(nested_forbidden(child, f"{path}/{index}"))
    return findings


def main() -> None:
    manifest = json.loads((PROTOCOL / "dataset_split_manifest.json").read_text(encoding="utf-8"))
    inventory = json.loads((PROTOCOL / "used_query_inventory.json").read_text(encoding="utf-8"))
    failures, datasets = [], {}
    for dataset, record in manifest["datasets"].items():
        used_ids = set(map(str, inventory.get("used_query_ids_by_dataset", {}).get(dataset, [])))
        used_questions = (
            set(inventory.get("used_normalized_questions_by_dataset", {}).get(dataset, []))
            | set(inventory.get("used_normalized_questions_by_dataset", {}).get("unknown", []))
        )
        split_ids, split_questions = {}, {}
        for split, file_record in record["files"].items():
            path = Path(file_record.get("path", file_record.get("input_path")))
            rows = read(path)
            split_ids[split] = {qid(row) for row in rows}
            split_questions[split] = {norm(row) for row in rows}
            expected_hash = file_record.get("sha256", file_record.get("input_sha256"))
            if sha256(path) != expected_hash:
                failures.append(f"{dataset}/{split}: input hash mismatch")
            overlap_ids = split_ids[split] & used_ids
            overlap_questions = split_questions[split] & used_questions
            if overlap_ids:
                failures.append(f"{dataset}/{split}: {len(overlap_ids)} previously used IDs")
            if overlap_questions:
                failures.append(f"{dataset}/{split}: {len(overlap_questions)} previously used questions")
            if split == "final_test":
                for index, row in enumerate(rows):
                    top = FORBIDDEN_TOP_LEVEL & set(row)
                    nested = nested_forbidden(row)
                    if top or nested:
                        failures.append(
                            f"{dataset}/final_test row {index}: label leakage "
                            f"top={sorted(top)} nested={nested[:3]}"
                        )
                        break
                label_path = Path(file_record["sealed_label_path"])
                if sha256(label_path) != file_record["sealed_label_sha256"]:
                    failures.append(f"{dataset}: sealed label hash mismatch")
                if os.stat(label_path).st_mode & 0o222:
                    failures.append(f"{dataset}: sealed labels are writable")
                if {qid(row) for row in read(label_path)} != split_ids[split]:
                    failures.append(f"{dataset}: final input/label ID mismatch")
        names = list(split_ids)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                if split_ids[left] & split_ids[right]:
                    failures.append(f"{dataset}: split ID overlap {left}/{right}")
                if split_questions[left] & split_questions[right]:
                    failures.append(f"{dataset}: split question overlap {left}/{right}")
        datasets[dataset] = {
            "counts": {name: len(values) for name, values in split_ids.items()},
            "strata": record["strata"],
            "used_id_overlap": 0,
            "used_question_overlap": 0,
        }
    payload = {"status": "pass" if not failures else "fail", "datasets": datasets, "failures": failures}
    (PROTOCOL / "frozen_split_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
