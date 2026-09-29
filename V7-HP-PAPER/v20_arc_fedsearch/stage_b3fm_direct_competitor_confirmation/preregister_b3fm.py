#!/usr/bin/env python3
"""Fail-closed B3F-M freshness audit and protocol freeze before retrieval."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
SOURCES = {
    "hotpotqa": "hf_hotpot/hotpot_train_v1.json",
    "2wikimultihopqa": "2wikimultihop_train.json",
    "musique": "musique_ans_v1.0_train.jsonl",
}
SALT = "v20-b3fm-unified-fresh-direct-competitor-20260929-v1"


def rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        yield from json.loads(path.read_text(encoding="utf-8"))


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def question_hash(question: str) -> str:
    value = unicodedata.normalize("NFKC", str(question)).lower()
    value = re.sub(r"[^\w\s]", " ", value)
    return hashlib.sha256(" ".join(value.split()).encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_from(path: Path, dataset: str) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if "query_ids" in value:
        return set(map(str, value["query_ids"]))
    return set(map(str, value["datasets"][dataset]))


def manifest_ids_hashes(path: Path, dataset: str) -> tuple[set[str], set[str]]:
    values = json.loads(path.read_text(encoding="utf-8"))["datasets"][dataset]
    return (
        {str(row["query_id"]) for row in values},
        {str(row["normalized_question_sha256"]) for row in values},
    )


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--b3f-stage", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage
    split_dir, protocol = stage / "splits", stage / "protocol"
    targets = [
        split_dir / "b3fm_fresh_split_manifest.json",
        split_dir / "b3fm_zero_overlap_audit.json",
        protocol / "b3fm_frozen_method_contract.json",
        protocol / "b3fm_preregistration.md",
        protocol / "training_fairness_audit.md",
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError("B3F-M pre-registration already exists; refusing to revise it")

    c1_root = args.repo / "V7-HP-PAPER/v20_arc_fedsearch/stage_c1_fresh_confirmation/pre_registration_recovery"
    c2_manifest = args.repo / "V7-HP-PAPER/v20_arc_fedsearch/stage_c2_replication_confirmation/pre_registration/protocol/c2_fresh_split_manifest.json"
    historical = json.loads((c1_root / "historical_manifests/historical_exclusion_union.json").read_text())["datasets"]
    historical_training = json.loads((c1_root / "historical_manifests/historical_training_ids.json").read_text())["datasets"]
    historical_method = json.loads((c1_root / "historical_manifests/historical_method_selection_ids.json").read_text())["datasets"]
    historical_revealed = json.loads((c1_root / "historical_manifests/historical_revealed_ids.json").read_text())["datasets"]
    c1_manifest = c1_root / "protocol/c1_fresh_split_manifest.json"
    b3f_manifest = args.b3f_stage / "protocol/b3f_fresh_split_manifest.json"

    split_dir.mkdir(parents=True, exist_ok=True)
    protocol.mkdir(parents=True, exist_ok=True)
    selected: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    fairness: list[dict[str, Any]] = []
    for dataset in DATASETS:
        m2_path = c1_root / f"historical_manifests/m2_probe_train_{dataset}.json"
        m2_train = ids_from(m2_path, dataset)
        ragroute_npz = args.b3f_stage / f"features/{dataset}/candidate_rows.npz"
        ragroute_train = set(map(str, np.load(ragroute_npz, allow_pickle=False)["query_ids"]))
        c1_ids, c1_hashes = manifest_ids_hashes(c1_manifest, dataset)
        c2_ids, c2_hashes = manifest_ids_hashes(c2_manifest, dataset)
        b3f_ids, b3f_hashes = manifest_ids_hashes(b3f_manifest, dataset)
        groups = {
            "m2_train": m2_train,
            "ragroute_train": ragroute_train,
            "historical_union": set(map(str, historical[dataset])),
            "historical_training": set(map(str, historical_training[dataset])),
            "historical_method_selection": set(map(str, historical_method[dataset])),
            "historical_revealed": set(map(str, historical_revealed[dataset])),
            "c1": c1_ids,
            "c2": c2_ids,
            "b3f": b3f_ids,
        }
        excluded_ids = set().union(*groups.values())
        # Reconstruct hashes for every historical ID from the public source so
        # hash-level exclusion is as strict as ID-level exclusion.
        excluded_hashes = set(c1_hashes) | set(c2_hashes) | set(b3f_hashes)
        eligible: list[tuple[str, int, str, str]] = []
        for index, source in enumerate(rows(args.experiment_root / "public_training_sources" / SOURCES[dataset])):
            identifier = qid(source)
            digest = question_hash(str(source["question"]))
            if identifier in excluded_ids:
                excluded_hashes.add(digest)
                continue
            if digest in excluded_hashes:
                continue
            key = hashlib.sha256(f"{SALT}|{dataset}|{identifier}".encode()).hexdigest()
            eligible.append((key, index, identifier, digest))
        sample: list[tuple[str, int, str, str]] = []
        used_hashes: set[str] = set()
        for candidate in sorted(eligible):
            if candidate[3] in used_hashes:
                continue
            sample.append(candidate)
            used_hashes.add(candidate[3])
            if len(sample) == 500:
                break
        if len(sample) < 300:
            raise ValueError(f"{dataset}: only {len(sample)} untouched text-unique queries remain")
        selected[dataset] = [
            {
                "dataset": dataset,
                "query_id": identifier,
                "source_row_zero_based": index,
                "normalized_question_sha256": digest,
                "selection_rule": "smallest salted SHA-256 rank after all exclusions, retaining first text-unique 500",
            }
            for _, index, identifier, digest in sample
        ]
        chosen_ids = {row["query_id"] for row in selected[dataset]}
        chosen_hashes = {row["normalized_question_sha256"] for row in selected[dataset]}
        overlap = {
            name: {
                "query_id_overlap_n": len(chosen_ids & values),
                "normalized_question_hash_overlap_n": len(chosen_hashes & excluded_hashes)
                if name in {"historical_union", "historical_training", "historical_method_selection", "historical_revealed", "m2_train", "ragroute_train"}
                else len(chosen_hashes & {"c1": c1_hashes, "c2": c2_hashes, "b3f": b3f_hashes}[name]),
            }
            for name, values in groups.items()
        }
        audit.append(
            {
                "dataset": dataset,
                "selected_n": len(selected[dataset]),
                "unique_query_ids": len(chosen_ids),
                "unique_normalized_question_hashes": len(chosen_hashes),
                "overlap": overlap,
                "pass": len(selected[dataset]) >= 300
                and len(chosen_ids) == len(selected[dataset])
                and len(chosen_hashes) == len(selected[dataset])
                and all(entry["query_id_overlap_n"] == 0 and entry["normalized_question_hash_overlap_n"] == 0 for entry in overlap.values()),
            }
        )
        fairness.append(
            {
                "dataset": dataset,
                "m2_train_queries": len(m2_train),
                "ragroute_train_queries": len(ragroute_train),
                "m2_ragroute_train_intersection": len(m2_train & ragroute_train),
                "fresh_vs_m2_train_intersection": len(chosen_ids & m2_train),
                "fresh_vs_ragroute_train_intersection": len(chosen_ids & ragroute_train),
                "candidate_rows_per_training_query": 8,
                "supervision": "candidate client contains at least one canonical supporting document",
            }
        )

    passed = all(row["pass"] for row in audit)
    manifest = {
        "stage": "V20-B3F-M-Fresh",
        "status": "frozen_before_retrieval" if passed else "integrity_failure",
        "purpose": "one-shot direct competitor confirmation: M2 ProbeRoute vs RAGRoute-Fixed3",
        "selection_salt": SALT,
        "selection_rule": "freshness first; 500 when available, otherwise at least 300",
        "datasets": selected,
        "source_sha256": {dataset: sha256(args.experiment_root / "public_training_sources" / SOURCES[dataset]) for dataset in DATASETS},
        "retrieval_started": False,
        "reader_started": False,
        "labels_scored": False,
    }
    (split_dir / "b3fm_fresh_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (split_dir / "b3fm_zero_overlap_audit.json").write_text(json.dumps({"status": "pass" if passed else "integrity_failure", "datasets": audit}, indent=2) + "\n")

    frozen_files = {
        "m2_logistic": {dataset: file_record(args.experiment_root / f"inputs/models/{dataset}/logistic_seed_20260807.pkl") for dataset in DATASETS},
        "ragroute_ensemble": {dataset: [file_record(args.b3f_stage / f"models/{dataset}/seed_{seed}.pt") for seed in (0, 1, 2)] for dataset in DATASETS},
        "ragroute_scaler": {dataset: file_record(args.b3f_stage / f"models/{dataset}/scaler.npz") for dataset in DATASETS},
        "support_predictor": {dataset: file_record(args.experiment_root / f"inputs/v16_evaluation/checkpoints/{dataset}_support.joblib") for dataset in DATASETS},
        "reader_runner": file_record(args.repo / "V7-HP-PAPER/v20_arc_fedsearch/stage_submission_baselines/run_reader_unscored.py"),
    }
    contract = {
        "status": "frozen_before_retrieval",
        "methods": {
            "b0_static_top3": "static coordinator score, frozen Top-8, select three",
            "ragroute_fixed3": "frozen B3F protocol-adapted three-seed mean sigmoid, select Top-3",
            "m2_logistic_proberoute": "frozen 18 probe features plus static score, standardized logistic Top-3",
            "b4a_all_candidate_top8_high_cost_reference": "frozen Top-8, select all eight",
            "ragroute_original_threshold": "same RAGRoute ensemble, mean sigmoid > 0.5, no fallback",
        },
        "matched_budget": {"candidate_pool": 8, "selected_clients": 3, "local_dense_depth": 10, "documents_per_client": 5, "transmitted_documents": 15, "merge": "raw dense score Top-10", "reader_context": "Top-5"},
        "readers": {"flan": "google/flan-t5-large@0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a", "unifiedqa": "allenai/unifiedqa-v2-t5-large-1363200@1d3b8e13b29dbd161494b0b15428378f4713c418", "max_chars": 4000, "max_tokens": 1024, "max_new_tokens": 32, "decoding": "greedy, num_beams=1, do_sample=False"},
        "support_prediction": "reuse frozen C1/C2 support predictor and output schema before label unseal",
        "primary_metrics": ["reader_context_complete_at_5", "joint_f1"],
        "primary_family": "3 datasets x [context complete plus FLAN joint plus UnifiedQA joint], Holm corrected",
        "statistics": "paired query-level bootstrap=5000 and randomization=5000",
        "success_criteria": "as supplied in V20-B3F-M task specification",
        "frozen_files": frozen_files,
        "git_head": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
    }
    (protocol / "b3fm_frozen_method_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (protocol / "training_fairness_audit.md").write_text(
        "# B3F-M Training Information Fairness Audit\n\n"
        "Both routers are frozen. Neither will be retrained or calibrated on B3F-M. "
        "The candidate universe is the same P0 Top-8 and the supervision definition is "
        "a candidate client containing at least one canonical supporting document.\n\n"
        "| Dataset | M2 train queries | RAGRoute train queries | Intersection | Fresh/M2 overlap | Fresh/RAGRoute overlap | Candidate rows/query |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + "".join(f"| {row['dataset']} | {row['m2_train_queries']} | {row['ragroute_train_queries']} | {row['m2_ragroute_train_intersection']} | {row['fresh_vs_m2_train_intersection']} | {row['fresh_vs_ragroute_train_intersection']} | 8 |\n" for row in fairness)
    )
    (protocol / "b3fm_preregistration.md").write_text(
        "# V20-B3F-M Pre-Registration\n\n"
        "This is a one-shot independent direct-competitor confirmation. It freezes B0, "
        "RAGRoute-Fixed3 (protocol-adapted), M2 ProbeRoute Logistic, B4a, and RAGRoute-Threshold; "
        "no baseline tuning, architecture search, threshold search, retriever change, reader change, or post-hoc rescue is allowed. "
        "Stages are freshness audit, contract freeze, blind routing/context, blind FLAN, blind UnifiedQA, blind support prediction, checksum freeze, STOP, and only then explicit `UNSEAL B3FM LABELS`. "
        "M0/M1/M2 share P=8, B=3, depth=10, five documents/client, raw Top-10 merge, and Top-5 Reader context.\n"
    )
    print(json.dumps({"status": "ready_for_b3fm_preflight" if passed else "integrity_failure", "datasets": {row["dataset"]: row["selected_n"] for row in audit}}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
