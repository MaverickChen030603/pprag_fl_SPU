#!/usr/bin/env python3
"""Freeze R5 inputs and audit final-test isolation without opening label records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
FROZEN_COMMIT = "13091c6deb6b3868705a49041e89f578f14b4e0e"
SAMPLE_SIZE = 300
SAMPLE_SALT = "v20-r5-sealed-final-20260812"


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id", ""))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def ids_from_structured(path: Path) -> set[str]:
    output: set[str] = set()
    try:
        if path.suffix == ".jsonl":
            source = rows(path)
        else:
            source = csv.DictReader(path.open(encoding="utf-8", errors="ignore"))
        for row in source:
            value = qid(row)
            if value:
                output.add(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error):
        pass
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output_root.resolve()
    if out.exists() and any(out.rglob("*")):
        raise FileExistsError(f"R5 output already exists; one-shot audit refuses overwrite: {out}")
    for name in ("protocol", "retrieval", "reader_predictions", "statistics", "mechanism", "gap_recovery", "cost", "checksums", "reports", "logs"):
        (out / name).mkdir(parents=True, exist_ok=True)

    v17 = root / "V7-HP-PAPER/v17_fedaction_rag"
    current = git(root, "rev-parse", "HEAD")
    if current != FROZEN_COMMIT:
        raise RuntimeError(f"R5 must start from frozen R4 commit {FROZEN_COMMIT}; found {current}")
    status = git(root, "status", "--porcelain=v1").splitlines()

    final_ids: dict[str, set[str]] = {}
    sample_rows: dict[str, list[dict[str, Any]]] = {}
    input_records: dict[str, dict[str, Any]] = {}
    label_records: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        input_path = v17 / f"data/{dataset}/final_test_inputs.jsonl"
        label_path = v17 / f"data/sealed/{dataset}_final_test_labels.jsonl"
        values = list(rows(input_path))
        if len(values) != 2000:
            raise ValueError(f"{dataset}: expected frozen final 2000, found {len(values)}")
        if any(set(row) & {"answer", "answer_aliases", "supporting_facts", "supporting_paragraphs", "is_supporting"} for row in values):
            raise ValueError(f"{dataset}: label-bearing final input")
        ids = [qid(row) for row in values]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{dataset}: duplicate final query ID")
        chosen = set(sorted(ids, key=lambda value: hashlib.sha256(f"{SAMPLE_SALT}|{dataset}|{value}".encode()).hexdigest())[:SAMPLE_SIZE])
        selected = [row for row in values if qid(row) in chosen]
        sample_path = out / f"protocol/{dataset}_final_test_inputs_n{SAMPLE_SIZE}.jsonl"
        with sample_path.open("x", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        final_ids[dataset] = set(ids)
        sample_rows[dataset] = selected
        input_records[dataset] = {"path": str(input_path), "rows": len(values), "sha256": sha256(input_path), "sample_path": str(sample_path), "sample_sha256": sha256(sample_path)}
        # Labels remain semantically unopened. Only filesystem metadata and byte hash are inspected.
        label_records[dataset] = {"path": str(label_path), "exists": label_path.is_file(), "mode": oct(label_path.stat().st_mode & 0o777), "sha256": sha256(label_path)}

    overlap_roots = [
        root / "V7-HP-PAPER/v20_arc_fedsearch/stage_r2_mars_route",
        root / "V7-HP-PAPER/v20_arc_fedsearch/stage_r2a6_resource_memory",
        root / "V7-HP-PAPER/v20_arc_fedsearch/stage_r3_probe_route",
        root / "V7-HP-PAPER/v20_arc_fedsearch/stage_r4_frozen_reader",
    ]
    overlap: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in DATASETS}
    scanned = 0
    for scan_root in overlap_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*"):
            if not path.is_file() or path.suffix not in {".jsonl", ".csv"}:
                continue
            scanned += 1
            present = ids_from_structured(path)
            for dataset in DATASETS:
                shared = present & final_ids[dataset]
                if shared:
                    overlap[dataset].append({"path": str(path), "count": len(shared)})
    contaminated = {dataset: values for dataset, values in overlap.items() if values}
    if contaminated:
        (out / "protocol/r5_contamination_findings.json").write_text(json.dumps(contaminated, indent=2) + "\n")
        raise RuntimeError(f"final_test_contaminated: {contaminated}")

    sample_manifest = {
        "status": "frozen_before_label_unseal", "sample_size_per_dataset": SAMPLE_SIZE,
        "selection_rule": f"smallest SHA256({SAMPLE_SALT}|dataset|query_id), then retain source order",
        "label_features_used": False,
        "datasets": {dataset: {"query_ids": [qid(row) for row in sample_rows[dataset]], "query_id_order_hash": canonical_hash([qid(row) for row in sample_rows[dataset]]), **input_records[dataset]} for dataset in DATASETS},
    }
    (out / "protocol/final_test_sample_manifest.json").write_text(json.dumps(sample_manifest, indent=2) + "\n")

    audit = {
        "status": "pass", "decision": "authorized_for_unlabeled_materialization_only",
        "final_test_kind": "V17 train-derived untouched held-out split; not official hidden test",
        "git_commit": current, "required_git_commit": FROZEN_COMMIT,
        "working_tree_dirty": bool(status), "working_tree_status_entries": len(status),
        "working_tree_status_sha256": canonical_hash(status), "structured_files_scanned": scanned,
        "r2_r3_r4_query_id_overlap": {dataset: 0 for dataset in DATASETS},
        "labels_semantically_opened": False, "label_files": label_records,
        "sample_manifest_sha256": sha256(out / "protocol/final_test_sample_manifest.json"),
        "prohibitions": ["no label read before all reader predictions are frozen", "no intermediate metrics", "no method change", "no rerun after unseal"],
    }
    (out / "protocol/no_leak_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (out / "protocol/r5_final_test_unseal_audit.md").write_text(
        "# R5-P0 Final-Test Unseal Audit\n\n"
        "**Decision:** `pass_for_unlabeled_phases`; labels remain sealed.\n\n"
        f"- Frozen scientific commit: `{current}`.\n"
        f"- Audited {scanned} R2--R4 structured artifacts; final-query overlap is 0 for all datasets.\n"
        "- V17 final inputs contain query/corpus fields only; sealed labels were checked by path, mode and SHA-256 only.\n"
        f"- Frozen sample: N={SAMPLE_SIZE}/dataset, label-free hash selection retained in source order.\n"
        "- This is a train-derived untouched held-out split, not an official hidden test.\n"
        "- Authorization covers Phase 1 retrieval and Phase 2 unscored reader inference only.\n",
        encoding="utf-8",
    )
    prereg = """# V20 Stage R5 Preregistration

- Nature: one-shot confirmatory evaluation; method development is closed.
- Data: V17 untouched train-derived final split, deterministic hash sample N=300/dataset.
- Methods: federated baseline, label-free ProbeRoute, logistic ProbeRoute, centralized retrieval reference.
- Primary: Logistic ProbeRoute minus federated baseline Joint F1.
- Readers: frozen FLAN-T5-Large and UnifiedQA-T5-Large.
- Statistics: 5,000 paired bootstrap resamples; primary Joint uncorrected; Answer/SP secondary BH-FDR.
- SP is context-level and reader-independent because both readers share the frozen V16 support predictor.
- Labels may be opened only after all 7,200 unscored predictions pass checksum validation.
- No observed result may trigger tuning, replacement, or rerun.
"""
    (out / "protocol/r5_preregistration.md").write_text(prereg, encoding="utf-8")
    print(json.dumps({"status": "pass", "sample": SAMPLE_SIZE, "datasets": list(DATASETS), "labels_opened": False}, indent=2))


if __name__ == "__main__":
    main()
