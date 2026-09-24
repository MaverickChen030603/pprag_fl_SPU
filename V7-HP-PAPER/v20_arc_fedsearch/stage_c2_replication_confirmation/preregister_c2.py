#!/usr/bin/env python3
"""Create a fresh, label-free C2 replication split after C1 was revealed."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
SOURCES = {
    "hotpotqa": "hf_hotpot/hotpot_train_v1.json",
    "2wikimultihopqa": "2wikimultihop_train.json",
    "musique": "musique_ans_v1.0_train.jsonl",
}
SALT = "v20-c2-independent-replication-20260924-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip(): yield json.loads(line)
    else:
        yield from json.loads(path.read_text(encoding="utf-8"))


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def question_hash(question: str) -> str:
    value = unicodedata.normalize("NFKC", str(question)).lower()
    value = re.sub(r"[^\w\s]", " ", value)
    return hashlib.sha256(" ".join(value.split()).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--c1-stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    historical = json.loads((args.c1_stage / "pre_registration_recovery/historical_manifests/historical_exclusion_union.json").read_text())["datasets"]
    c1 = json.loads((args.c1_stage / "pre_registration_recovery/protocol/c1_fresh_split_manifest.json").read_text())["datasets"]
    args.output.mkdir(parents=True)
    selected, audit = {}, []
    for dataset in DATASETS:
        excluded_ids = set(map(str, historical[dataset])) | {str(row["query_id"]) for row in c1[dataset]}
        excluded_questions = {str(row["normalized_question_sha256"]) for row in c1[dataset]}
        eligible = []
        for index, row in enumerate(rows(args.base / "public_training_sources" / SOURCES[dataset])):
            identifier, digest = qid(row), question_hash(str(row["question"]))
            if identifier in excluded_ids or digest in excluded_questions: continue
            key = hashlib.sha256(f"{SALT}|{dataset}|{identifier}".encode()).hexdigest()
            eligible.append((key, index, identifier, digest))
        sample = sorted(eligible)[:500]
        if len(sample) != 500: raise ValueError(f"{dataset}: insufficient fresh queries")
        selected[dataset] = [{"dataset":dataset,"query_id":identifier,"source_row_zero_based":index,"normalized_question_sha256":digest,"source_split":"public_train_unused_after_historical_and_c1_exclusion"} for _,index,identifier,digest in sample]
        ids={row["query_id"] for row in selected[dataset]}; hashes={row["normalized_question_sha256"] for row in selected[dataset]}
        audit.append({"dataset":dataset,"selected_n":len(ids),"historical_id_overlap_n":len(ids & set(historical[dataset])),"c1_id_overlap_n":len(ids & {str(row["query_id"]) for row in c1[dataset]}),"c1_question_overlap_n":len(hashes & {str(row["normalized_question_sha256"]) for row in c1[dataset]}),"pass":len(ids)==500 and len(hashes)==500 and not (ids & excluded_ids) and not (hashes & excluded_questions)})
    manifest={"stage":"V20-C2","status":"frozen_before_execution","purpose":"independent_confirmatory_replication_of_C1","sample_size_per_dataset":500,"selection_salt":SALT,"selection_rule":"smallest SHA256 salt rank after historical plus C1 ID and normalized-question exclusions","datasets":selected,"source_sha256":{d:sha256(args.base/"public_training_sources"/SOURCES[d]) for d in DATASETS},"retrieval_started":False,"reader_started":False,"labels_scored":False}
    protocol=args.output/"protocol"; protocol.mkdir(); (protocol/"c2_fresh_split_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (protocol/"c2_zero_overlap_audit.json").write_text(json.dumps({"status":"pass" if all(row["pass"] for row in audit) else "fail","datasets":audit},indent=2)+"\n")
    frozen={"methods":["B0 Static Top-3","B1 Random Top-3 fixed seeds 0..19","B3 RAGRoute-style MLP","B6 Dense Centroid Top-3","M2 Logistic ProbeRoute","B4a All-candidate Top-8 high-cost reference"],"routing_contract":{"P":8,"B":3,"depth":10,"docs_per_client":5,"fixed_budget_docs":15,"merge":"raw dense Top-10","reader_top_k":5},"readers":{"flan":"google/flan-t5-large@0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a","unifiedqa":"allenai/unifiedqa-v2-t5-large-1363200@1d3b8e13b29dbd161494b0b15428378f4713c418"},"methods_mutable":False,"no_tuning_after_c1":True}
    (protocol/"c2_frozen_contract.json").write_text(json.dumps(frozen,indent=2)+"\n")
    (protocol/"c2_cost_contract.json").write_text(json.dumps({"probe":{"candidate_clients":8,"feature_dimension":18,"float32_payload_bytes":576,"protocol_overhead_bytes":"report separately"},"fixed_budget":{"clients":3,"documents":15},"b4a":{"clients":8,"documents":40},"byte_audit":"record per-query document UTF-8 payload and total payload without calling it a pure document-count reduction"},indent=2)+"\n")
    (args.output/"c2_preregistration.md").write_text("# V20-C2 Independent Replication Preregistration\n\nC2 excludes all historical development/training/revealed IDs and all C1 IDs plus normalized-question hashes. Methods, readers, budgets, merge, and statistics are frozen from C1. No C2 result may alter the contract. C2 is a replication, not a retuning round.\n")
    print(json.dumps({"status":"ready_for_c2_preflight" if all(row["pass"] for row in audit) else "integrity_failure","audit":audit},indent=2))


if __name__ == "__main__": main()
