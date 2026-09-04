#!/usr/bin/env python3
"""Inventory query IDs already exposed during V1--V14 development.

The audit is deliberately conservative: a query appearing in an outcome,
reporting, action, calibration, or exported data artifact is considered used.
Large raw corpora, model environments, archives, and the V15 directory itself
are excluded. The resulting inventory is an exclusion list, not proof that an
unlisted query was never observed by a human.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ID_KEYS = ("query_id", "qid", "question_id", "_id", "id")
SUPPORTED = {".json", ".jsonl", ".csv", ".tsv"}
DEFAULT_MAX_BYTES = 160 * 1024 * 1024
SKIP_PARTS = {
    ".git", "__pycache__", "node_modules", "site-packages", "tmp", "raw",
    "checkpoints", "models", "wandb", "sealed", "v15_robust_context_repair",
}
USAGE_RULES = {
    "architecture_selection": ("train", "generator", "selector", "candidate", "model", "architecture"),
    "threshold_tuning": ("threshold", "calibrat", "gate", "budget", "risk_coverage"),
    "result_inspection": ("result", "metric", "report", "prediction", "output", "reader", "evaluation"),
    "ablation": ("ablation", "sweep", "variant", "tau", "hardgate"),
    "subgroup_analysis": ("subgroup", "case", "failure", "diagnos", "taxonomy", "oracle"),
    "transfer_calibration": ("2wiki", "musique", "cross_dataset", "transfer", "unifiedqa", "multi_reader"),
}


def sha256_bytes(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(map(str, values))):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_question(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    text = " ".join(str(value).strip().lower().split())
    return text if len(text) >= 4 else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_from_path(path: Path) -> str:
    lower = str(path).lower()
    if "2wiki" in lower:
        return "2wikimultihopqa"
    if "musique" in lower:
        return "musique"
    if "hotpot" in lower or "hp4" in lower or "v7-hp" in lower:
        return "hotpotqa"
    return "unknown"


def usage_from_path(path: Path) -> list[str]:
    lower = str(path).lower()
    matches = [name for name, needles in USAGE_RULES.items() if any(value in lower for value in needles)]
    return matches or ["result_inspection"]


def plausible_id(value: Any) -> str | None:
    if isinstance(value, (dict, list, tuple, set)) or value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > 256 or text.lower() in {"none", "null", "nan"}:
        return None
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return None
    return text


def id_from_row(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    for key in ID_KEYS:
        if key in row:
            value = plausible_id(row[key])
            if value:
                return value
    return None


def iter_json_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        yield from (row for row in payload if isinstance(row, dict))
        return
    if not isinstance(payload, dict):
        return
    if id_from_row(payload):
        yield payload
    for key in ("data", "examples", "rows", "items", "predictions", "records"):
        values = payload.get(key)
        if isinstance(values, list):
            yield from (row for row in values if isinstance(row, dict))


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
    elif path.suffix == ".json":
        yield from iter_json_rows(path)
    else:
        delimiter = "\t" if path.suffix == ".tsv" else ","
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle, delimiter=delimiter)


def candidate_files(root: Path, max_bytes: int) -> tuple[list[Path], list[dict[str, Any]]]:
    found: list[Path] = []
    skipped: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        size = path.stat().st_size
        if size > max_bytes:
            skipped.append({"path": str(relative), "reason": "over_size_limit", "bytes": size})
            continue
        found.append(path)
    return sorted(found), skipped


def audit(root: Path, max_bytes: int) -> dict[str, Any]:
    files, skipped = candidate_files(root, max_bytes)
    ids_by_dataset: dict[str, set[str]] = defaultdict(set)
    questions_by_dataset: dict[str, set[str]] = defaultdict(set)
    ids_by_usage: dict[str, set[str]] = defaultdict(set)
    sources: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in files:
        try:
            rows = list(iter_rows(path))
            ids = {value for row in rows if (value := id_from_row(row))}
            questions = {
                value for row in rows
                if isinstance(row, dict) and (value := normalize_question(row.get("question", row.get("query"))))
            }
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as exc:
            errors.append({"path": str(path.relative_to(root)), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not ids and not questions:
            continue
        dataset = dataset_from_path(path)
        usages = usage_from_path(path)
        ids_by_dataset[dataset].update(ids)
        questions_by_dataset[dataset].update(questions)
        for usage in usages:
            ids_by_usage[usage].update(ids)
        sources.append({
            "path": str(path.relative_to(root)),
            "dataset": dataset,
            "usage_categories": usages,
            "query_count": len(ids),
            "question_fingerprint_count": len(questions),
            "query_fingerprint": sha256_bytes(ids),
            "question_fingerprint": sha256_bytes(questions),
            "file_sha256": file_sha256(path),
        })

    return {
        "schema_version": 1,
        "project_root": str(root),
        "policy": "conservative_union_of_top_level_query_ids",
        "id_keys": list(ID_KEYS),
        "max_file_bytes": max_bytes,
        "counts_by_dataset": {key: len(value) for key, value in sorted(ids_by_dataset.items())},
        "question_counts_by_dataset": {key: len(value) for key, value in sorted(questions_by_dataset.items())},
        "counts_by_usage": {key: len(value) for key, value in sorted(ids_by_usage.items())},
        "query_fingerprints_by_dataset": {key: sha256_bytes(value) for key, value in sorted(ids_by_dataset.items())},
        "used_query_ids_by_dataset": {key: sorted(value) for key, value in sorted(ids_by_dataset.items())},
        "used_normalized_questions_by_dataset": {key: sorted(value) for key, value in sorted(questions_by_dataset.items())},
        "sources": sources,
        "skipped_files": skipped,
        "parse_errors": errors,
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V15 Used-Query Inventory",
        "",
        "This inventory is a conservative exclusion list. A query found in any",
        "architecture, calibration, evaluation, ablation, diagnostic, or transfer",
        "artifact is treated as previously used. Absence from this list is not by",
        "itself proof of non-exposure.",
        "",
        "## Dataset counts",
        "",
        "| Dataset | Unique query IDs | Unique normalized questions | ID fingerprint |",
        "|---|---:|---:|---|",
    ]
    for dataset, count in payload["counts_by_dataset"].items():
        question_count = payload["question_counts_by_dataset"].get(dataset, 0)
        lines.append(f"| {dataset} | {count:,} | {question_count:,} | `{payload['query_fingerprints_by_dataset'][dataset]}` |")
    lines += ["", "## Usage counts", "", "| Usage | Unique query IDs |", "|---|---:|"]
    for usage, count in payload["counts_by_usage"].items():
        lines.append(f"| {usage} | {count:,} |")
    lines += [
        "", "## Audit coverage", "",
        f"- Files containing query IDs: {len(payload['sources']):,}",
        f"- Files skipped by size policy: {len(payload['skipped_files']):,}",
        f"- Parse errors: {len(payload['parse_errors']):,}",
        "- Raw corpora, environments, archives, and V15 itself are intentionally excluded.",
        "- The complete per-source inventory and exclusion IDs are in `used_query_inventory.json`.",
        "",
        "## Confirmatory boundary",
        "",
        "The HotpotQA validation set previously used throughout HP1--HP4 and V4--V14",
        "must not be described as a new confirmatory test. V15 confirmation uses",
        "train-derived IDs absent from this inventory and reports that provenance",
        "explicitly.",
    ]
    if payload["parse_errors"]:
        lines += ["", "## Parse errors requiring review", ""]
        lines += [f"- `{row['path']}`: {row['error']}" for row in payload["parse_errors"][:50]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--max-file-mb", type=int, default=DEFAULT_MAX_BYTES // (1024 * 1024))
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = audit(root, args.max_file_mb * 1024 * 1024)
    (output / "used_query_inventory.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "used_query_inventory.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "counts_by_dataset": payload["counts_by_dataset"],
        "sources": len(payload["sources"]),
        "parse_errors": len(payload["parse_errors"]),
    }, indent=2))


if __name__ == "__main__":
    main()
