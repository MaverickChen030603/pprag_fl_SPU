#!/usr/bin/env python3
"""Materialize B2 contexts with frozen static-only logistic models."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

from materialize_posthoc_baselines import (
    DATASETS,
    atomic_json,
    context_hash,
    lookup_documents,
    query_id,
    raw_merge,
    rows,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "contexts_unlabeled.jsonl"
    dataset_manifests = {}
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output_dir, delete=False
    ) as handle:
        for dataset in DATASETS:
            split_path = (
                args.input_root
                / "protocol"
                / f"{dataset}_final_test_inputs_n300.jsonl"
            )
            packet_path = (
                args.input_root / "retrieval" / f"{dataset}_probe_packets.jsonl"
            )
            index_path = args.index_root / f"{dataset}.sqlite"
            model_path = args.model_root / dataset / "static_only_logistic.pkl"
            with model_path.open("rb") as model_handle:
                model_payload = pickle.load(model_handle)
            split_rows = list(rows(split_path))
            packets = {str(row["query_id"]): row for row in rows(packet_path)}
            connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
            try:
                for source_row in split_rows:
                    qid = query_id(source_row)
                    packet = packets[qid]
                    records = packet["p0_candidate_records"]
                    matrix = np.asarray(
                        [[float(record["static_score"])] for record in records],
                        dtype=np.float64,
                    )
                    probability = model_payload["model"].predict_proba(
                        model_payload["scaler"].transform(matrix)
                    )[:, 1]
                    order = np.argsort(-probability, kind="stable")[:3]
                    clients = [int(records[index]["client_id"]) for index in order]
                    transmitted, merged = raw_merge(packet, clients)
                    merged_ids = [str(document["doc_id"]) for document in merged]
                    documents = lookup_documents(connection, merged_ids[:5])
                    question = str(source_row["question"])
                    row = {
                        "dataset": dataset,
                        "query_id": qid,
                        "question": question,
                        "method": "b2_static_only_logistic",
                        "selected_clients": clients,
                        "transmitted_doc_ids": [
                            str(document["doc_id"]) for document in transmitted
                        ],
                        "retrieved_doc_ids": merged_ids,
                        "reader_context_doc_ids": merged_ids[:5],
                        "reader_context_docs": documents,
                        "context_hash": context_hash(question, documents),
                        "candidate_clients": 8,
                        "client_budget": 3,
                        "local_depth": 10,
                        "documents_per_client": 5,
                        "transmission_budget": 15,
                        "global_pool_size": 10,
                        "reader_context_k": 5,
                        "probe_bytes": 0,
                        "gold_or_answer_used": False,
                        "reader_started": False,
                        "evidence_role": "retrospective_post_hoc_r5_revealed",
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            finally:
                connection.close()
            dataset_manifests[dataset] = {
                "queries": len(split_rows),
                "model_sha256": sha256(model_path),
                "split_sha256": sha256(split_path),
                "packets_sha256": sha256(packet_path),
                "index_sha256": sha256(index_path),
            }
        temporary = Path(handle.name)
    os.replace(temporary, output)
    row_count = sum(1 for _ in rows(output))
    if row_count != 900:
        raise ValueError(f"expected 900 rows, found {row_count}")
    manifest = {
        "status": "complete_unlabeled_contexts",
        "method": "b2_static_only_logistic",
        "rows": row_count,
        "probe_bytes": 0,
        "labels_used_at_inference": False,
        "test_or_confirmation_labels_used_for_training": False,
        "evidence_role": "retrospective_post_hoc_r5_revealed",
        "output_sha256": sha256(output),
        "datasets": dataset_manifests,
    }
    atomic_json(args.output_dir / "materialization_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
