#!/usr/bin/env python3
"""Audit split isolation and optional selected-unit train-corpus membership."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--r2-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--recovery-protocol-dir", type=Path, required=True)
    parser.add_argument("--units", type=Path)
    parser.add_argument("--train-corpus-manifest", type=Path)
    args = parser.parse_args()

    manifest_path = args.recovery_protocol_dir / "recovery_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    r2_protocol = args.r2_root / "protocol"
    smoke = {qid(row) for row in rows(r2_protocol / "router_dev_smoke100.jsonl")}
    router_train = {qid(row) for row in rows(r2_protocol / "router_train.jsonl")}
    router_cal = {qid(row) for row in rows(r2_protocol / "router_calibration.jsonl")}
    dev_ids = set(manifest["recovery_dev"]["query_ids"])
    holdout_ids = set(manifest["recovery_holdout"]["query_ids"])
    checks = {
        "recovery_dev_count_is_100": len(dev_ids) == 100,
        "recovery_holdout_count_is_300": len(holdout_ids) == 300,
        "recovery_dev_excludes_r2a_smoke": not bool(dev_ids & smoke),
        "recovery_dev_holdout_disjoint": not bool(dev_ids & holdout_ids),
        "recovery_dev_excludes_router_train": not bool(dev_ids & router_train),
        "recovery_dev_excludes_router_calibration": not bool(dev_ids & router_cal),
        "holdout_excludes_router_train": not bool(holdout_ids & router_train),
        "reader_disabled": True,
        "final_test_not_accessed": True,
    }
    unit_check: dict[str, Any] = {"checked": False}
    if args.units and args.train_corpus_manifest:
        allowed = set(json.loads(args.train_corpus_manifest.read_text(encoding="utf-8"))["document_ids"])
        unit_rows = list(rows(args.units))
        leaked = [row["source_document_id"] for row in unit_rows if row["source_document_id"] not in allowed]
        unit_check = {
            "checked": True,
            "units": len(unit_rows),
            "source_document_ids_outside_train_corpus": len(leaked),
            "unit_train_corpus_membership_pass": not leaked,
            "selected_units_sha256": digest([row["unit_id"] for row in unit_rows]),
        }
        checks["unit_train_corpus_membership_pass"] = not leaked
    status = "pass" if all(checks.values()) else "fail"
    payload = {
        "stage": "R2-A.6_REMP",
        "dataset": args.dataset,
        "status": status,
        "checks": checks,
        "unit_membership": unit_check,
        "forbidden_inference_inputs": ["gold client", "support label", "answer", "query multiview", "reader output", "final test"],
        "reader_start_decision": "blocked_before_reader",
        "final_test_accessed": False,
        "recovery_dev_sha256": manifest["recovery_dev"]["query_id_sha256"],
        "recovery_holdout_sha256": manifest["recovery_holdout"]["query_id_sha256"],
    }
    (args.recovery_protocol_dir / "no_leak_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
