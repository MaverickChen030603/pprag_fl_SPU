#!/usr/bin/env python3
"""Verify the frozen V17 Checkpoint-A artifact contract after completion.

This audit intentionally refuses to inspect a partial run. It only validates
provenance and deterministic artifact contracts; it does not make a Go/No-Go
decision and does not read final-test labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
READERS = ("flan", "unifiedqa")
CONDITIONS = {
    "centralized": {"partition": "centralized", "budget": 1},
    "topic_silo_bc2": {"partition": "topic_silo", "budget": 2},
    "topic_silo": {"partition": "topic_silo", "budget": 3},
    "entity_community": {"partition": "entity_community", "budget": 3},
    "random_control": {"partition": "random_control", "budget": 3},
}
EXPECTED_FIELDS = {
    "dataset", "reader", "partition", "query_id", "client_budget",
    "local_k", "origin_client", "cross_client_strict_syn_joint_f1",
    "cross_client_composition_only_joint_f1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def validate_pool(path: Path, dataset: str, condition: str) -> tuple[bool, str, set[str]]:
    rows = list(jsonl(path))
    query_ids = {str(row.get("query_id", "")) for row in rows}
    if len(rows) != len(query_ids):
        return False, "duplicate or missing pool query IDs", query_ids
    expected = CONDITIONS[condition]
    violations: list[str] = []
    for row in rows:
        if str(row.get("dataset")) != dataset:
            violations.append("dataset")
        if str(row.get("partition")) != expected["partition"]:
            violations.append("partition")
        if int(row.get("client_budget", -1)) != expected["budget"]:
            violations.append("budget")
        if int(row.get("pool_size", -1)) != 10:
            violations.append("pool_size")
        if expected["partition"] != "centralized":
            selected = [int(value) for value in row.get("selected_clients", [])]
            if len(selected) != expected["budget"] or len(set(selected)) != len(selected):
                violations.append("selected_clients")
            if any(value < 0 or value >= 20 for value in selected):
                violations.append("client_id")
        docs = row.get("pool", [])
        doc_ids = [str(doc.get("doc_id", "")) for doc in docs]
        # ``pool`` retains all client-local candidates for auditability; the
        # separately recorded ``pool_size=10`` caps the action-search pool.
        if len(doc_ids) != len(set(doc_ids)):
            violations.append("pool_docs")
        if any(int(doc.get("client_id", 0)) < 0 or int(doc.get("client_id", 0)) >= 20 for doc in docs):
            violations.append("document_client_id")
    return not violations, "ok" if not violations else ",".join(sorted(set(violations))), query_ids


def validate_reader_cell(path: Path, dataset: str, condition: str, reader: str, expected_n: int) -> tuple[bool, str, set[str]]:
    rows = csv_rows(path)
    if not rows:
        return False, "empty reader-backed cell", set()
    fields = set(rows[0])
    missing = EXPECTED_FIELDS - fields
    query_ids = {str(row.get("query_id", "")) for row in rows}
    spec = CONDITIONS[condition]
    violations: list[str] = []
    if missing:
        violations.append("missing_fields=" + "|".join(sorted(missing)))
    if len(rows) != expected_n or len(query_ids) != expected_n:
        violations.append(f"expected_n={expected_n},rows={len(rows)},unique={len(query_ids)}")
    for row in rows:
        if row["dataset"] != dataset or row["reader"] != reader:
            violations.append("dataset_or_reader")
        if row["partition"] != spec["partition"]:
            violations.append("partition")
        if int(row["client_budget"]) != spec["budget"]:
            violations.append("budget")
        expected_local_k = 10 if spec["partition"] == "centralized" else 5
        if int(row["local_k"]) != expected_local_k:
            violations.append("local_k")
        origin = int(row["origin_client"])
        if origin < 0 or origin >= 20:
            violations.append("origin_client")
    return not violations, "ok" if not violations else ";".join(sorted(set(violations))), query_ids


def validate_contexts(path: Path, pool_path: Path, dataset: str, condition: str, expected_n: int) -> tuple[bool, str]:
    pool_docs = {
        str(row["query_id"]): {str(doc["doc_id"]) for doc in row["pool"]}
        for row in jsonl(pool_path)
    }
    query_ids: set[str] = set()
    violations: list[str] = []
    spec = CONDITIONS[condition]
    for row in jsonl(path):
        query_id = str(row.get("query_id", ""))
        query_ids.add(query_id)
        docs = [str(value) for value in row.get("context_doc_ids", [])]
        if len(docs) != 5 or len(set(docs)) != 5:
            violations.append("context_k_or_duplicate")
        if not set(docs).issubset(pool_docs.get(query_id, set())):
            violations.append("context_outside_pool")
        if str(row.get("dataset")) != dataset or str(row.get("partition")) != spec["partition"]:
            violations.append("context_metadata")
        expected_local_k = 10 if spec["partition"] == "centralized" else 5
        if int(row.get("client_budget", -1)) != spec["budget"] or int(row.get("local_k", -1)) != expected_local_k:
            violations.append("context_budget")
        client_ids = [int(value) for value in row.get("context_client_ids", [])]
        if len(client_ids) != 5 or any(value < 0 or value >= 20 for value in client_ids):
            violations.append("context_client_ids")
    if len(query_ids) != expected_n:
        violations.append(f"expected_n={expected_n},contexts={len(query_ids)}")
    return not violations, "ok" if not violations else ";".join(sorted(set(violations)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v17-root", type=Path, required=True)
    parser.add_argument("--phase-tag", default="phase_a_checkpoint100")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", default="fac9f62")
    parser.add_argument("--expected-n", type=int, default=100)
    args = parser.parse_args()

    phase = args.v17_root / "oracle" / args.phase_tag
    results = phase / "results"
    completion = results / "federated_go_no_go_phase_a.json"
    checks: list[dict[str, Any]] = []
    add(checks, "phase_completion_marker", completion.is_file(), str(completion))
    if not completion.is_file():
        report = {
            "status": "refused_partial_run", "phase": str(phase),
            "checks": checks,
            "reason": "Run the audit only after V17 has emitted its formal decision artifact.",
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "checkpoint_a_integrity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "checkpoint_a_integrity.md").write_text("# Checkpoint-A Integrity\n\nStatus: **refused_partial_run**\n", encoding="utf-8")
        return 2

    provenance = phase / "run_provenance.json"
    provenance_value = "missing"
    provenance_ok = False
    if provenance.is_file():
        provenance_value = str(json.loads(provenance.read_text(encoding="utf-8")).get("formal_start_commit", ""))
        provenance_ok = provenance_value.startswith(args.expected_commit)
    add(checks, "frozen_aggregator_commit", provenance_ok, f"run_provenance={provenance_value}; expected_prefix={args.expected_commit}")
    run_config = phase / "frozen_run_config.txt"
    add(checks, "frozen_run_config", run_config.is_file(), str(run_config))
    if run_config.is_file():
        text = run_config.read_text(encoding="utf-8")
        required = ("MAX_QUERIES=100", "BATCH_SIZE=32", "LOCAL_SPARSE_CANDIDATES=20", "readers=flan,unifiedqa")
        add(checks, "frozen_values", all(value in text for value in required), text.strip())

    pool_query_sets: dict[tuple[str, str], set[str]] = {}
    artifact_hashes: dict[str, str] = {}
    for dataset in DATASETS:
        for condition in CONDITIONS:
            pool = phase / "pools" / f"{dataset}_{condition}.jsonl"
            ok, detail, query_ids = validate_pool(pool, dataset, condition) if pool.is_file() else (False, "missing", set())
            add(checks, f"pool:{dataset}:{condition}", ok and len(query_ids) == args.expected_n, detail)
            pool_query_sets[(dataset, condition)] = query_ids
            if pool.is_file():
                artifact_hashes[str(pool.relative_to(args.v17_root))] = sha256(pool)
            context = phase / "contexts" / f"{dataset}_{condition}.jsonl"
            context_ok, context_detail = validate_contexts(context, pool, dataset, condition, args.expected_n) if context.is_file() and pool.is_file() else (False, "missing")
            add(checks, f"contexts:{dataset}:{condition}", context_ok, context_detail)
            if context.is_file():
                artifact_hashes[str(context.relative_to(args.v17_root))] = sha256(context)
        canonical = pool_query_sets[(dataset, "centralized")]
        aligned = all(pool_query_sets[(dataset, condition)] == canonical for condition in CONDITIONS)
        add(checks, f"pool_query_alignment:{dataset}", aligned, f"canonical_n={len(canonical)}")

    reader_query_sets: dict[tuple[str, str, str], set[str]] = {}
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for reader in READERS:
                cell = results / f"{dataset}_{condition}_{reader}_per_query.csv"
                ok, detail, query_ids = validate_reader_cell(cell, dataset, condition, reader, args.expected_n) if cell.is_file() else (False, "missing", set())
                add(checks, f"reader_cell:{dataset}:{condition}:{reader}", ok, detail)
                reader_query_sets[(dataset, condition, reader)] = query_ids
                if cell.is_file():
                    artifact_hashes[str(cell.relative_to(args.v17_root))] = sha256(cell)
                add(checks, f"pool_reader_alignment:{dataset}:{condition}:{reader}", query_ids == pool_query_sets[(dataset, condition)], f"queries={len(query_ids)}")
        for reader in READERS:
            canonical = reader_query_sets[(dataset, "centralized", reader)]
            aligned = all(reader_query_sets[(dataset, condition, reader)] == canonical for condition in CONDITIONS)
            add(checks, f"reader_query_alignment:{dataset}:{reader}", aligned, f"canonical_n={len(canonical)}")

    no_leak = args.v17_root / "protocol" / "no_leak_audit.json"
    no_leak_ok = False
    if no_leak.is_file():
        no_leak_ok = json.loads(no_leak.read_text(encoding="utf-8")).get("status") == "pass"
    add(checks, "no_final_label_access_static_audit", no_leak_ok, str(no_leak))

    all_passed = all(check["passed"] for check in checks)
    report = {
        "status": "pass" if all_passed else "fail",
        "phase": str(phase),
        "expected_n": args.expected_n,
        "expected_commit": args.expected_commit,
        "artifact_hashes": artifact_hashes,
        "checks": checks,
        "note": "No recovery manifest was found or required by this first execution; a recovered run must add deterministic rerun hashes before this audit can pass recovery equivalence.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoint_a_integrity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Checkpoint-A Integrity", "", f"Status: **{report['status']}**", ""]
    lines.extend(f"- {'PASS' if check['passed'] else 'FAIL'} `{check['name']}`: {check['detail']}" for check in checks)
    (args.output_dir / "checkpoint_a_integrity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
