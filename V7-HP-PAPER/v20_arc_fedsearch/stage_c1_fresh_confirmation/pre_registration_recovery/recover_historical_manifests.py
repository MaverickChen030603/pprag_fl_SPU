#!/usr/bin/env python3
"""Recover auditable historical query-ID manifests before any C1 execution.

This utility is intentionally limited to data provenance.  It never invokes a
retriever, reader, model, or metric evaluator, and it only reads identifier and
question fields from the prospective C1 source files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
SAMPLE_SIZE = 500
SELECTION_SALT = "v20-c1-fresh-confirmation-20260924-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list):
        raise ValueError(f"expected JSON list: {path}")
    yield from values


def query_id(row: dict[str, Any]) -> str:
    for key in ("query_id", "_id", "id", "qid", "question_id"):
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError(f"record has no recognized query ID: {list(row)[:8]}")


def normalized_question(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).lower()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def ordered_digest(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "query_id", "stage", "role", "used_for_training",
        "used_for_feature_selection", "used_for_model_selection",
        "used_for_threshold_selection", "used_for_error_analysis",
        "used_for_confirmation", "labels_revealed", "source_type",
        "source_path", "source_commit", "confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def git_commit(root: Path, path: Path) -> str:
    relative = path.resolve().relative_to(root.resolve())
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", str(relative)],
            cwd=root, text=True,
        ).strip() or "unavailable"
    except subprocess.CalledProcessError:
        return "unavailable"


def distribution_ids(path: Path) -> dict[str, dict[str, set[str]]]:
    output: dict[str, dict[str, set[str]]] = {
        dataset: defaultdict(set) for dataset in DATASETS
    }
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = row.get("dataset")
            if dataset in output and row.get("partition") == "topic_silo":
                output[dataset][str(row["split"])].add(str(row["query_id"]))
    return output


def model_stats(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError(f"empty model result: {path}")
    first = records[0]
    expected = {
        "train_queries": int(first["train_queries"]),
        "candidate_rows": int(first["candidate_rows"]),
        "positive_rows": int(first["positive_rows"]),
        "hard_negative_rows": int(first["hard_negative_rows"]),
    }
    if any({key: int(row[key]) for key in expected} != expected for row in records):
        raise ValueError(f"seed rows disagree: {path}")
    return expected


def c1_source_rows(path: Path) -> Iterable[tuple[int, str, str]]:
    """Yield only source row number, ID, and question; never inspect label fields."""
    for index, row in enumerate(json_rows(path)):
        yield index, query_id(row), str(row["question"])


def registry_row(
    dataset: str, identifier: str, stage: str, role: str, source_type: str,
    source_path: Path, source_commit: str, confidence: str, **flags: Any,
) -> dict[str, Any]:
    return {
        "dataset": dataset, "query_id": identifier, "stage": stage,
        "role": role, "used_for_training": bool(flags.get("training", False)),
        "used_for_feature_selection": bool(flags.get("feature", False)),
        "used_for_model_selection": bool(flags.get("model", False)),
        "used_for_threshold_selection": bool(flags.get("threshold", False)),
        "used_for_error_analysis": bool(flags.get("error", False)),
        "used_for_confirmation": bool(flags.get("confirmation", False)),
        "labels_revealed": flags.get("revealed", "not_asserted"),
        "source_type": source_type, "source_path": str(source_path),
        "source_commit": source_commit, "confidence": confidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root, base, out = (value.resolve() for value in (args.repo_root, args.base, args.output_root))
    allowed_existing = {Path(__file__).name, "__pycache__"}
    if out.exists() and any(path.name not in allowed_existing for path in out.iterdir()):
        raise FileExistsError(f"refusing to overwrite recovery output: {out}")
    for name in ("historical_manifests", "provenance", "overlap_audit", "protocol", "reports"):
        (out / name).mkdir(parents=True, exist_ok=True)

    v17 = root / "V7-HP-PAPER/v17_fedaction_rag"
    split_manifest_path = v17 / "protocol/dataset_split_manifest.json"
    inventory_path = v17 / "protocol/used_query_inventory.json"
    distribution_path = v17 / "partitions/client_query_distribution.csv"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    distribution = distribution_ids(distribution_path)
    distribution_commit = git_commit(root, distribution_path)
    inventory_commit = git_commit(root, inventory_path)

    registry: list[dict[str, Any]] = []
    train_ids: dict[str, set[str]] = {}
    m2_manifests: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        ids = distribution[dataset].get("train", set())
        if len(ids) != 5000:
            raise ValueError(f"{dataset}: expected 5000 direct V17 train IDs, found {len(ids)}")
        stats_path = base / f"inputs/models/{dataset}/model_results.csv"
        stats = model_stats(stats_path)
        if stats["train_queries"] != len(ids) or stats["candidate_rows"] != len(ids) * 8:
            raise ValueError(f"{dataset}: M2 training statistics do not match direct train artifact")
        train_ids[dataset] = ids
        details = {
            "dataset": dataset,
            "query_ids": sorted(ids),
            "count": len(ids),
            "query_id_sha256": ordered_digest(sorted(ids)),
            "source_split": "V17 train",
            "selection_rule": "V17 frozen train split; R2 router_train=train[0:5000]; R3 Probe-Train copied router_train (Hotpot transfer uses V17 train)",
            "training_artifact": str(distribution_path),
            "training_artifact_sha256": sha256(distribution_path),
            "training_artifact_commit": distribution_commit,
            "dataset_source": split_manifest["datasets"][dataset]["source"],
            "dataset_source_sha256": split_manifest["datasets"][dataset]["source_sha256"],
            "historical_training_statistics": stats,
            "source_type": "training_artifact",
            "confidence": "A",
            "candidate_packets_recovered": False,
            "b2p_status": "B2p_unavailable_due_to_missing_training_packets",
        }
        atomic_json(out / f"historical_manifests/m2_probe_train_{dataset}.json", details)
        m2_manifests[dataset] = details
        for identifier in sorted(ids):
            registry.append(registry_row(
                dataset, identifier, "R3_train", "M2_ProbeTrain",
                "training_artifact", distribution_path, distribution_commit, "A", training=True,
            ))

    # V16 inventory is the authoritative conservative union inherited by V17.
    prior_ids = inventory.get("used_query_ids_by_dataset", {})
    prior_questions = inventory.get("used_normalized_questions_by_dataset", {})
    method_ids: dict[str, set[str]] = {dataset: set(map(str, prior_ids.get(dataset, []))) for dataset in DATASETS}
    question_hashes: dict[str, set[str]] = {
        dataset: {hashlib.sha256(normalized_question(question).encode("utf-8")).hexdigest()
                  for question in prior_questions.get(dataset, [])}
        for dataset in DATASETS
    }
    for dataset in DATASETS:
        for identifier in sorted(method_ids[dataset]):
            registry.append(registry_row(
                dataset, identifier, "V16_prior", "historical_method_selection_union",
                "direct_manifest", inventory_path, inventory_commit, "A", feature=True, model=True,
                threshold=True, error=True, revealed="conservative_union",
            ))

    # R2/R3/R4 used subsets of V17 calibration/development.  Retain the complete
    # source splits as a conservative superset where per-stage files were pruned.
    for dataset in DATASETS:
        for split, stage, flags in (
            ("development", "R2_R4_development_superset", {"feature": True, "model": True, "error": True}),
            ("calibration", "R2_R3_calibration_superset", {"feature": True, "model": True, "threshold": True}),
        ):
            values = distribution[dataset].get(split, set())
            if len(values) != 1000:
                raise ValueError(f"{dataset}: expected 1000 {split} IDs, found {len(values)}")
            method_ids[dataset].update(values)
            for identifier in sorted(values):
                registry.append(registry_row(
                    dataset, identifier, stage, "conservative_split_superset",
                    "training_artifact", distribution_path, distribution_commit, "A", **flags,
                ))

    r5_manifest_path = base / "inputs/r5_run/protocol/final_test_sample_manifest.json"
    r5_manifest = json.loads(r5_manifest_path.read_text(encoding="utf-8"))
    r5_ids: dict[str, set[str]] = {}
    r5_commit = "unversioned_server_recovery_asset"
    for dataset in DATASETS:
        values = set(map(str, r5_manifest["datasets"][dataset]["query_ids"]))
        if len(values) != 300:
            raise ValueError(f"{dataset}: expected 300 revealed R5 IDs, found {len(values)}")
        r5_ids[dataset] = values
        for identifier in sorted(values):
            registry.append(registry_row(
                dataset, identifier, "R5_revealed", "revealed_diagnostic_query",
                "direct_manifest", r5_manifest_path, r5_commit, "A", confirmation=True, revealed=True,
            ))

    training_payload = {dataset: sorted(train_ids[dataset]) for dataset in DATASETS}
    selection_payload = {dataset: sorted(method_ids[dataset]) for dataset in DATASETS}
    revealed_payload = {dataset: sorted(r5_ids[dataset]) for dataset in DATASETS}
    union_payload = {
        dataset: sorted(train_ids[dataset] | method_ids[dataset] | r5_ids[dataset])
        for dataset in DATASETS
    }
    for name, payload in (
        ("historical_training_ids.json", training_payload),
        ("historical_method_selection_ids.json", selection_payload),
        ("historical_revealed_ids.json", revealed_payload),
        ("historical_exclusion_union.json", union_payload),
    ):
        atomic_json(out / f"historical_manifests/{name}", {
            "schema_version": 1, "datasets": payload,
            "per_dataset_counts": {dataset: len(values) for dataset, values in payload.items()},
        })
    write_csv(out / "historical_manifests/historical_query_registry.csv", registry)

    for dataset in DATASETS:
        provenance = m2_manifests[dataset]
        text = [
            f"# M2 Probe-Train Provenance: {dataset}", "",
            "**Recovery status:** `recovered_confidence_A`", "",
            f"- Direct historical query-ID artifact: `{provenance['training_artifact']}`.",
            f"- Artifact commit: `{provenance['training_artifact_commit']}`.",
            f"- Source split: `{provenance['source_split']}`; recovered query count: `{provenance['count']}`.",
            f"- Query-ID SHA256: `{provenance['query_id_sha256']}`.",
            f"- Historical dataset source SHA256: `{provenance['dataset_source_sha256']}`.",
            f"- Historical training rows: `{provenance['historical_training_statistics']}`.",
            "- Candidate packets were not recovered; their absence leaves B2p unavailable but does not invalidate the direct M2 train-ID artifact.",
            "",
            "The manifest is recovered from a version-controlled historical training artifact, not inferred from a current sampling rule.",
        ]
        (out / f"provenance/probe_train_{dataset}_provenance.md").write_text("\n".join(text) + "\n", encoding="utf-8")

    source_paths = {
        "hotpotqa": base / "public_training_sources/hf_hotpot/hotpot_train_v1.json",
        "2wikimultihopqa": base / "public_training_sources/2wikimultihop_train.json",
        "musique": base / "public_training_sources/musique_ans_v1.0_train.jsonl",
    }
    candidates: dict[str, list[dict[str, Any]]] = {}
    audit_rows: list[dict[str, Any]] = []
    for dataset, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        eligible = []
        source_ids: set[str] = set()
        for index, identifier, question in c1_source_rows(path):
            if identifier in source_ids:
                raise ValueError(f"{dataset}: duplicate source ID {identifier}")
            source_ids.add(identifier)
            question_hash = hashlib.sha256(normalized_question(question).encode("utf-8")).hexdigest()
            if identifier in union_payload[dataset] or question_hash in question_hashes[dataset]:
                continue
            key = hashlib.sha256(f"{SELECTION_SALT}|{dataset}|{identifier}".encode("utf-8")).hexdigest()
            eligible.append((key, index, identifier, question_hash))
        selected = sorted(eligible)[:SAMPLE_SIZE]
        if len(selected) != SAMPLE_SIZE:
            raise ValueError(f"{dataset}: only {len(selected)} eligible fresh candidates")
        candidates[dataset] = [
            {"dataset": dataset, "query_id": identifier, "source_split": "public_train_unused_after_historical_exclusion",
             "source_row_zero_based": index, "normalized_question_sha256": question_hash}
            for _, index, identifier, question_hash in selected
        ]
        ids = {row["query_id"] for row in candidates[dataset]}
        probe_overlap = len(ids & train_ids[dataset])
        selection_overlap = len(ids & method_ids[dataset])
        r5_overlap = len(ids & r5_ids[dataset])
        union_overlap = len(ids & set(union_payload[dataset]))
        question_overlap = sum(row["normalized_question_sha256"] in question_hashes[dataset] for row in candidates[dataset])
        audit_rows.append({
            "dataset": dataset, "candidate_n": len(ids), "probe_train_overlap_n": probe_overlap,
            "method_selection_overlap_n": selection_overlap, "r3_overlap_n": probe_overlap,
            "r4_overlap_n": selection_overlap, "r5_overlap_n": r5_overlap,
            "historical_union_overlap_n": union_overlap, "normalized_question_overlap_n": question_overlap,
            "freshness_pass": all(value == 0 for value in (probe_overlap, selection_overlap, r5_overlap, union_overlap, question_overlap)),
        })
    with (out / "overlap_audit/candidate_overlap_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0])); writer.writeheader(); writer.writerows(audit_rows)
    audit = {"status": "pass" if all(row["freshness_pass"] for row in audit_rows) else "fail", "selection_salt": SELECTION_SALT, "datasets": audit_rows}
    atomic_json(out / "overlap_audit/c1_zero_overlap_audit.json", audit)

    split_manifest = {
        "schema_version": 1, "stage": "V20-C1", "status": "frozen_before_execution",
        "sample_size_per_dataset": SAMPLE_SIZE,
        "source_kind": "Level_B_historically_unused_large_public_train_pool",
        "selection_rule": f"retain IDs not in historical_exclusion_union and no normalized-question collision, then take smallest SHA256({SELECTION_SALT}|dataset|query_id)",
        "datasets": {dataset: candidates[dataset] for dataset in DATASETS},
        "source_sha256": {dataset: sha256(path) for dataset, path in source_paths.items()},
        "freshness_audit_sha256": sha256(out / "overlap_audit/c1_zero_overlap_audit.json"),
        "retrieval_started": False, "reader_started": False, "labels_scored": False,
    }
    atomic_json(out / "protocol/c1_fresh_split_manifest.json", split_manifest)
    method_manifest = {
        "M2": "standardized class-balanced Logistic Regression; static_score plus 18 frozen query-time local probe features",
        "routing_contract": {"P": 8, "B": 3, "docs_per_client": 5, "transmitted_docs": 15, "merge": "raw_dense_top10", "reader_context_k": 5},
        "baselines": ["B0 Static Top-3", "B1 Random Top-3", "B3 RAGRoute-style baseline", "B6 Dense Centroid Top-3", "B4a All-candidate Top-8 high-cost reference"],
        "B2p": "B2p_unavailable_due_to_missing_training_packets",
        "methods_mutable": False,
    }
    atomic_json(out / "protocol/c1_frozen_method_manifest.json", method_manifest)
    prereg = """# V20-C1 Fresh Confirmation Preregistration

This C1 split was frozen before any C1 retrieval, Reader inference, label scoring, bootstrap, or significance test.  It contains 500 query IDs per dataset selected from a historically unused public-train pool by a fixed SHA256 rule after ID-level and normalized-question exclusions.

The primary method, baselines, routing contract, frozen Readers, metrics, and statistical protocol remain exactly those defined in the C1 submission-confirmation protocol.  M2/B0, M2/B3, and M2/B6 are the primary paired comparisons; bootstrap has 5,000 query-level resamples, paired randomization is also run, and Holm correction applies to the primary comparison family.  B4a is named only the All-candidate Top-8 high-cost reference.

No C1 method output exists at preregistration time.  B2p remains unavailable because original Probe-Train packets were not recovered.
"""
    (out / "protocol/c1_preregistration.md").write_text(prereg, encoding="utf-8")
    report = [
        "# V20-C1 Historical Manifest Recovery Report", "",
        "**Decision:** `ready_for_one_shot_fresh_confirmation`", "",
        "## Recovered M2 Probe-Train", "",
        "All three M2 Probe-Train query sets were recovered from the version-controlled V17 `topic_silo` client-query distribution artifact. R2 freezes `router_train` as V17 `train[0:5000]`; R3 copies that router train set for 2Wiki/MuSiQue, while the Hotpot transfer uses the V17 train split directly.", "",
        "| Dataset | Queries | Candidate rows | Positive rows | Negative rows | Confidence |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for dataset in DATASETS:
        stats = m2_manifests[dataset]["historical_training_statistics"]
        report.append(f"| {dataset} | {stats['train_queries']} | {stats['candidate_rows']} | {stats['positive_rows']} | {stats['hard_negative_rows']} | A |")
    report += [
        "", "The original Probe-Train packet files were not recovered. This is sufficient to keep B2p unavailable, but does not weaken recovery of the M2 training query IDs because the IDs and their 5,000-query cardinality are direct historical artifacts and the row statistics agree with the frozen M2 result files.",
        "", "## Historical Exclusion", "",
        "The exclusion union combines: M2 Probe-Train IDs, the V16 direct used-query inventory, conservative V17 development/calibration supersets covering R2--R4 selection activity, and all 300 revealed R5 IDs per dataset. Conservative supersets avoid an unsupported claim about exact pruned per-stage membership.",
        "", "## Freshness Audit", "",
        "Each dataset has 500 C1 IDs. The ID-level Probe-Train, method-selection, R5, and full-union intersections are zero. A normalized-question SHA256 secondary audit is also zero. Candidate source rows are not ranked, retrieved, read by a Reader, or scored here.",
        "", "## Remaining Boundary", "",
        "This recovery authorizes only the one-shot C1 execution under the separately frozen protocol. It does not authorize B2p, B5, dynamic Top-k, new features, new baselines, or any result-triggered method change.",
    ]
    (out / "reports/historical_manifest_recovery_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "reports/c1_readiness_report.md").write_text(
        "# V20-C1 Readiness\n\n**Status:** `ready_for_one_shot_fresh_confirmation`\n\n"
        "Historical M2 train IDs are confidence-A direct artifacts, historical development exclusions are conservatively registered, all revealed R5 IDs are registered, and the frozen 500-query-per-dataset C1 manifest has zero ID and normalized-question overlap. C1 retrieval, Reader inference, and label scoring remain not started.\n",
        encoding="utf-8",
    )
    decision = {
        "status": "ready_for_one_shot_fresh_confirmation" if audit["status"] == "pass" else "candidate_pool_not_fresh",
        "m2_probe_train_confidence": {dataset: "A" for dataset in DATASETS},
        "b2p": "B2p_unavailable_due_to_missing_training_packets",
        "retrieval_started": False, "reader_started": False, "labels_scored": False,
        "freshness_audit_pass": audit["status"] == "pass",
        "manifest_sha256": sha256(out / "protocol/c1_fresh_split_manifest.json"),
    }
    atomic_json(out / "reports/c1_final_pre_registration_decision.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
