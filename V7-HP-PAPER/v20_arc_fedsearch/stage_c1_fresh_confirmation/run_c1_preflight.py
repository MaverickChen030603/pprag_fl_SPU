#!/usr/bin/env python3
"""Fail-closed integrity preflight for the V20-C1 blind confirmation.

This program deliberately does not materialize retrieval packets, open labels, or
invoke a Reader.  It records the exact executable assets immediately before the
one-shot blind stage and exits non-zero on any contract mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
EXPECTED = {
    "global_indexes": {
        "hotpotqa": "815147ae102f7822f1e934290426a074c2b9f2f9f2b763483038c9558e1e58d1",
        "2wikimultihopqa": "d4a5b856ee2b6fbae5a4d030f198f4706eafd12081246177c0f5c96a3ddc2bf7",
        "musique": "755af6e0630797b4e8d9a492bad3c6e17e5fc849abafd28767ad70357a214344",
    },
    "assignments": {
        "hotpotqa": "474cbe75e632582ab277a69947b8e8eac72840a151358494a323a0150962225e",
        "2wikimultihopqa": "75b4e9bc06854488a26403fb98b29c21d258db1c73fcf87d78afa7eb09a39245",
        "musique": "a2f0e17029f27da9340010ed05501fdf2baf621b5ef83c8ceb7a33ea8219b723",
    },
    "m2": {
        "hotpotqa": "0d07b5cd47c3b6c666c6a1fd70c1ca9e4185307bece207d43b8b6b21ecdcc166",
        "2wikimultihopqa": "ae97397f4f9eb35ae59f7b530ec5edd32285c77f06218289e5e40ac4754081aa",
        "musique": "6955cc51c22370747d2d9b80c7741fb8ce695de022da555296dc632e7a4bd497",
    },
    "local_manifests": {
        "hotpotqa": "ef9cc56177bab922da219fad7d0dc97d0b889ee5cdf00f0b5da5bc9e4b8aa435",
        "2wikimultihopqa": "92a602032afbed61a6c98fb6adb18ddf87944aa8992de6b39646c85e37bc6e08",
        "musique": "6d73f8b6c00bfd6af192761ba8889ad11953771d372163aeb47d83e9e51614b0",
    },
    "p0_centroids": {
        "hotpotqa": "aa0a175a1afe94adeb110e36b088cb2ad0204b5fce074218a9e3bd3d866dcfa0",
        "2wikimultihopqa": "4c337ced6ab7f5dc44eb3bd5ab62649d2d26a0b112c984a663dcdd9acc7498be",
        "musique": "28d720fda907e2837c09b684e5804fd4a690d49761a7bf70380883be406024b4",
    },
    "profiles": {
        "hotpotqa": "e3b4cafd26bbbdef77993cfd90d5065c88ef40c259a27409c3eb69a5e5937b36",
        "2wikimultihopqa": "e2c601c8fe90878bc318d9f7e5761009cbf7f14b8675e70837e64306dafa41cb",
        "musique": "1ed9982cef993429d1c0c83984312d8df4be285f6c7eec1a627966c726d71930",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(path: Path, expected: str | None) -> dict[str, Any]:
    exists = path.is_file()
    actual = sha256(path) if exists else None
    return {"path": str(path), "exists": exists, "actual_sha256": actual,
            "expected_sha256": expected, "match": exists and (expected is None or actual == expected)}


def profile_path(asset_root: Path, dataset: str) -> Path:
    if dataset == "hotpotqa":
        return asset_root / "V7-HP-PAPER/v20_arc_fedsearch/stage_r3_probe_route/hotpot_transfer/resource_profiles/client_profiles.json"
    return asset_root / f"V7-HP-PAPER/v20_arc_fedsearch/stage_r2_mars_route/{dataset}/resource_profiles/client_profiles.json"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    stage = args.repo / "V7-HP-PAPER/v20_arc_fedsearch/stage_c1_fresh_confirmation"
    recovery = stage / "pre_registration_recovery"
    split = json.loads((recovery / "protocol/c1_fresh_split_manifest.json").read_text())
    overlap = json.loads((recovery / "overlap_audit/c1_zero_overlap_audit.json").read_text())
    asset_root = args.base / "inputs/c1_frozen_assets"
    entries: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        entries[dataset] = {
            "manifest_rows": len(split["datasets"].get(dataset, [])),
            "global_index": check(args.base / f"inputs/indexes/{dataset}.sqlite", EXPECTED["global_indexes"][dataset]),
            "assignment": check(args.base / f"inputs/v17/partitions/assignments/{dataset}/topic_silo_m20.jsonl", EXPECTED["assignments"][dataset]),
            "m2_model": check(args.base / f"inputs/models/{dataset}/logistic_seed_20260807.pkl", EXPECTED["m2"][dataset]),
            "p0_centroid": check(asset_root / f"V7-HP-PAPER/v17_fedaction_rag/partitions/centroids/{dataset}/topic_silo_m20.npy", EXPECTED["p0_centroids"][dataset]),
            "p0_profile": check(profile_path(asset_root, dataset), EXPECTED["profiles"][dataset]),
            "local_index_manifest": check(asset_root / f"V7-HP-PAPER/v17_fedaction_rag/retrieval/local_indexes/{dataset}/topic_silo/manifest.json", EXPECTED["local_manifests"][dataset]),
            "b3_model": check(args.base / f"runs/ragroute_b3_r5_posthoc_20260916/routes/{dataset}/ragroute_mlp.pkl", None),
            "b3_centroid": check(args.base / f"runs/ragroute_b3_r5_posthoc_20260916/centroids/{dataset}/source_centroids.npy", None),
        }
    checks = []
    for dataset, value in entries.items():
        checks.append(value["manifest_rows"] == 500)
        checks.extend(item["match"] for key, item in value.items() if isinstance(item, dict))
    freshness_rows = overlap.get("datasets", [])
    zero_overlap = (
        overlap.get("status") == "pass"
        and len(freshness_rows) == len(DATASETS)
        and all(
            row.get("dataset") in DATASETS
            and row.get("candidate_n") == 500
            and row.get("freshness_pass") is True
            and all(value == 0 for key, value in row.items() if key.endswith("_overlap_n"))
            for row in freshness_rows
        )
    )
    checks.append(zero_overlap)
    output = {
        "stage": "V20-C1-A",
        "status": "pass" if all(checks) else "integrity_failure",
        "retrieval_started": False,
        "reader_started": False,
        "labels_scored": False,
        "git_head": git(args.repo, "rev-parse", "HEAD"),
        "git_status_porcelain": git(args.repo, "status", "--porcelain"),
        "split_sha256": sha256(recovery / "protocol/c1_fresh_split_manifest.json"),
        "zero_overlap_audit_sha256": sha256(recovery / "overlap_audit/c1_zero_overlap_audit.json"),
        "zero_overlap_pass": zero_overlap,
        "datasets": entries,
        "contract": {"P": 8, "B": 3, "docs_per_client": 5, "fixed_budget_docs": 15,
                     "b4a_clients": 8, "b4a_docs": 40, "reader_top_k": 5,
                     "b1_seeds": list(range(20))},
    }
    target = stage / "protocol/c1_execution_preflight.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n")
    audit = stage / "protocol/c1_baseline_contract_audit.md"
    audit.write_text("# C1 Baseline Contract Audit\n\n"
                     + ("PASS" if output["status"] == "pass" else "INTEGRITY FAILURE")
                     + "\n\nAll fixed-budget methods use P=8, B=3, depth=10, 5 documents/client, raw dense Top-10 and Reader Top-5. "
                     "B4a alone returns all 8 frozen candidates (40 documents). B1 uses seeds 0--19.\n")
    print(json.dumps({"status": output["status"], "preflight": str(target)}, indent=2))
    if output["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
