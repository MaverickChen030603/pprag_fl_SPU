#!/usr/bin/env python3
"""Conservative V1--V15 query exposure audit for the V16 freeze."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path


def load_v15_auditor(project_root: Path):
    path = project_root / "v15_robust_context_repair" / "protocol" / "01_protocol_and_data_audit.py"
    spec = importlib.util.spec_from_file_location("v15_inventory_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # V15 excluded itself. V16 must include V15 and exclude only its own outputs.
    module.SKIP_PARTS.discard("v15_robust_context_repair")
    module.SKIP_PARTS.add("v16_action_composition")
    return module


def audit_streaming(root: Path, max_json_bytes: int, auditor) -> dict:
    """Stream large tabular artifacts while excluding raw source corpora."""
    ids_by_dataset = defaultdict(set)
    questions_by_dataset = defaultdict(set)
    ids_by_usage = defaultdict(set)
    sources, skipped, errors = [], [], []

    def rows(path: Path):
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        errors.append({"path": str(path.relative_to(root)), "line": line_number, "error": f"JSONDecodeError: {exc}"})
                        continue
                    if isinstance(value, dict):
                        yield value
            return
        if path.suffix in {".csv", ".tsv"}:
            with path.open(encoding="utf-8", newline="") as handle:
                yield from csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ",")
            return
        yield from auditor.iter_json_rows(path)

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in auditor.SUPPORTED:
            continue
        relative = path.relative_to(root)
        parts = set(relative.parts)
        if any(part in auditor.SKIP_PARTS for part in relative.parts):
            continue
        # Raw dataset sources are provenance inputs, not evidence of human or
        # model development exposure. Derived split/action/result files remain.
        lower = str(relative).lower()
        if "/data/sources/" in f"/{lower}" or "/sources/musique_official/" in f"/{lower}":
            continue
        if path.suffix == ".json" and path.stat().st_size > max_json_bytes:
            skipped.append({"path": str(relative), "reason": "oversized_nonstreamable_json", "bytes": path.stat().st_size})
            continue
        file_ids, file_questions = set(), set()
        try:
            for row in rows(path):
                value = auditor.id_from_row(row)
                if value:
                    file_ids.add(value)
                question = auditor.normalize_question(row.get("question", row.get("query"))) if isinstance(row, dict) else None
                if question:
                    file_questions.add(question)
        except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
            errors.append({"path": str(relative), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not file_ids and not file_questions:
            continue
        dataset = auditor.dataset_from_path(path)
        usages = auditor.usage_from_path(path)
        ids_by_dataset[dataset].update(file_ids)
        questions_by_dataset[dataset].update(file_questions)
        for usage in usages:
            ids_by_usage[usage].update(file_ids)
        sources.append({
            "path": str(relative), "dataset": dataset, "usage_categories": usages,
            "query_count": len(file_ids), "question_fingerprint_count": len(file_questions),
            "query_fingerprint": auditor.sha256_bytes(file_ids), "question_fingerprint": auditor.sha256_bytes(file_questions),
            "file_sha256": auditor.file_sha256(path),
        })
    return {
        "schema_version": 2, "project_root": str(root),
        "policy": "streaming_conservative_union_of_top_level_query_ids",
        "id_keys": list(auditor.ID_KEYS), "max_nonstreamable_json_bytes": max_json_bytes,
        "counts_by_dataset": {key: len(value) for key, value in sorted(ids_by_dataset.items())},
        "question_counts_by_dataset": {key: len(value) for key, value in sorted(questions_by_dataset.items())},
        "counts_by_usage": {key: len(value) for key, value in sorted(ids_by_usage.items())},
        "query_fingerprints_by_dataset": {key: auditor.sha256_bytes(value) for key, value in sorted(ids_by_dataset.items())},
        "used_query_ids_by_dataset": {key: sorted(value) for key, value in sorted(ids_by_dataset.items())},
        "used_normalized_questions_by_dataset": {key: sorted(value) for key, value in sorted(questions_by_dataset.items())},
        "sources": sources, "skipped_files": skipped, "parse_errors": errors,
    }


def markdown(payload: dict) -> str:
    lines = [
        "# V16 Used-Query Inventory",
        "",
        "This is a conservative exclusion inventory over V1--V15 artifacts. IDs or normalized questions found in architecture, feature, threshold, utility, ablation, transfer, or reporting artifacts are treated as previously exposed.",
        "",
        "| Dataset | IDs | Normalized questions | Fingerprint |",
        "|---|---:|---:|---|",
    ]
    for dataset, count in payload["counts_by_dataset"].items():
        lines.append(f"| {dataset} | {count:,} | {payload['question_counts_by_dataset'].get(dataset, 0):,} | `{payload['query_fingerprints_by_dataset'][dataset]}` |")
    lines += [
        "",
        "## Boundary",
        "",
        "- V15 artifacts are included; V16 outputs are excluded to prevent self-contamination of the inventory.",
        "- The repeatedly inspected 7,405 HotpotQA validation queries remain ineligible for V16 confirmatory evaluation.",
        "- Absence from this inventory is not proof of non-exposure; source provenance and question-fingerprint exclusion are both enforced during split freezing.",
        f"- Parsed sources: {len(payload['sources']):,}; parse errors: {len(payload['parse_errors']):,}; oversized files skipped: {len(payload['skipped_files']):,}.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True, help="V7-HP-PAPER root")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--max-file-mb", type=int, default=160)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    auditor = load_v15_auditor(root)
    payload = audit_streaming(root, args.max_file_mb * 1024 * 1024, auditor)
    payload["schema_version"] = 2
    payload["experiment"] = "V7-HP-PAPER-v16-synergy-aware-action-composition"
    payload["audit_scope"] = "V1--V15"
    (output / "used_query_inventory.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "used_query_inventory.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"status": "complete", "counts": payload["counts_by_dataset"], "sources": len(payload["sources"]), "errors": len(payload["parse_errors"])}, indent=2))


if __name__ == "__main__":
    main()
