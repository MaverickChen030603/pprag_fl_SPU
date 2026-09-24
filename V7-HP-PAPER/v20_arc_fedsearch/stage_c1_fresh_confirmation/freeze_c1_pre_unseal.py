#!/usr/bin/env python3
"""Freeze and audit C1 blind artifacts.  This script never opens labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
METHODS = {"b0_static_top3", "b3_ragroute_style_mlp", "b6_dense_centroid_top3", "m2_logistic_proberoute", "b4a_all_candidate_top8_high_cost_reference", *{f"b1_random_top3_seed_{seed:02d}" for seed in range(20)}}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    retrieval = args.stage / "retrieval/blind_retrieval_outputs.jsonl"
    predictions = {reader: args.stage / f"readers/{reader}_unscored_predictions.jsonl" for reader in ("flan", "unifiedqa")}
    contexts = rows(retrieval)
    context_keys = {(row["dataset"], row["query_id"], row["method"]) for row in contexts}
    expected_keys = {(dataset, str(row["query_id"]), method) for dataset in DATASETS for row in rows(args.stage / f"inputs/{dataset}_c1_blind_n500.jsonl") for method in METHODS}
    reader_checks = {}
    for reader, path in predictions.items():
        values = rows(path)
        keys = {(row["dataset"], row["query_id"], row["method"]) for row in values}
        reader_checks[reader] = {"rows": len(values), "unique_keys": len(keys), "missing_keys": len(expected_keys - keys), "extra_keys": len(keys - expected_keys), "labels_loaded": any(row.get("labels_loaded") is not False for row in values), "metrics_computed": any(row.get("metrics_computed") is not False for row in values), "complete_marker": (args.stage / f"readers/{reader}_unscored_predictions.completed.json").is_file()}
    all_complete = len(contexts) == len(context_keys) == len(expected_keys) == 37500 and context_keys == expected_keys and not any(row.get("gold_or_answer_used") for row in contexts) and all(value["rows"] == 37500 and value["unique_keys"] == 37500 and value["missing_keys"] == 0 and value["extra_keys"] == 0 and not value["labels_loaded"] and not value["metrics_computed"] and value["complete_marker"] for value in reader_checks.values())
    files = [retrieval, *predictions.values(), *(args.stage / f"readers/{reader}_unscored_predictions.completed.json" for reader in predictions), args.stage / "protocol/c1_execution_preflight.json", args.stage / "pre_registration_recovery/protocol/c1_fresh_split_manifest.json"]
    code = [args.stage / "materialize_c1_blind_retrieval.py", args.repo / "V7-HP-PAPER/v20_arc_fedsearch/stage_submission_baselines/run_reader_unscored.py", args.stage / "freeze_c1_pre_unseal.py"]
    manifest = {"stage": "V20-C1", "status": "ready_for_c1_label_unseal" if all_complete else "integrity_failure", "all_complete": all_complete, "labels_unsealed": False, "labels_scored": False, "statistics_computed": False, "retrieval": {"rows": len(contexts), "unique_keys": len(context_keys), "expected_keys": len(expected_keys), "gold_or_answer_used": any(row.get("gold_or_answer_used") for row in contexts)}, "readers": reader_checks, "artifacts": [record(path) for path in files], "code": [record(path) for path in code], "git_head": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()}
    target = args.stage / "checksums/pre_unseal_artifact_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    report = args.stage / "reports/c1_pre_unseal_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# C1 Pre-Unseal Report\n\n"
                      f"- Status: `{manifest['status']}`\n"
                      f"- Blind retrieval: `{len(contexts)}/37500` unique method-query cells\n"
                      f"- FLAN predictions: `{reader_checks['flan']['rows']}/37500`\n"
                      f"- UnifiedQA predictions: `{reader_checks['unifiedqa']['rows']}/37500`\n"
                      f"- Missing predictions: FLAN `{reader_checks['flan']['missing_keys']}`, UnifiedQA `{reader_checks['unifiedqa']['missing_keys']}`\n"
                      "- Artifact hashes: frozen in `checksums/pre_unseal_artifact_manifest.json`.\n"
                      "- Labels were not loaded; no metrics or statistics were computed.\n"
                      "- Scientific-contract deviation: none recorded after the invalid pre-repair Hotpot partial output was quarantined and the full index preflight passed.\n\n"
                      "STOP: wait for explicit human instruction `UNSEAL C1 LABELS`.\n")
    print(json.dumps({"status": manifest["status"], "all_complete": all_complete, "manifest": str(target)}, indent=2))
    if not all_complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
