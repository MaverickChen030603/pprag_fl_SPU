#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import pickle
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

FEATURE_SCHEMA = (
    "dense_top1_score", "dense_top3_mean", "dense_top1_top2_margin",
    "dense_score_std", "dense_score_entropy", "dense_local_rank_percentile",
    "bm25_top1_score", "bm25_top3_mean", "bm25_top1_top2_margin",
    "dense_bm25_top1_same", "dense_bm25_top3_overlap",
    "dense_sparse_rank_correlation", "matched_query_entity_count",
    "matched_query_token_count", "matched_title_token_count",
    "query_title_embedding_similarity", "top3_title_diversity", "top3_entity_diversity",
)
METHODS = {
    "inherited": "B0_inherited_route",
    "label_free": "B3_label_free_probe",
    "logistic": "B4_logistic_seed_20260807",
}
FORBIDDEN = {
    "answer", "answers", "aliases", "supporting_facts", "supporting_titles",
    "gold_answer", "gold_answers", "gold_support", "gold_document_ids",
    "answer_em", "answer_f1", "support_em", "support_f1", "reader_outcome",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    output = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                output.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid JSON at {path}:{number}: {exc}") from exc
    return output


def nested_keys(value) -> set[str]:
    found = set()
    if isinstance(value, dict):
        found.update(map(str, value))
        for item in value.values():
            found.update(nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(nested_keys(item))
    return found


def stable_rank(values: list[float]) -> list[int]:
    return [int(i) for i in np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")]


def minmax(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    span = float(array.max() - array.min())
    return np.zeros_like(array) if span <= 1e-12 else (array - array.min()) / span


def label_free_select(records: list[dict]) -> list[int]:
    score = 0.25 * minmax([float(row["static_score"]) for row in records])
    score += 0.75 * minmax([float(row["dense_top3_mean"]) for row in records])
    return [int(records[i]["client_id"]) for i in stable_rank(score.tolist())[:3]]


def feature_matrix(records: list[dict]) -> np.ndarray:
    return np.asarray(
        [[float(row["static_score"]), *[float(row[key]) for key in FEATURE_SCHEMA]] for row in records],
        dtype=np.float64,
    )


def logistic_select(records: list[dict], payload: dict) -> list[int]:
    probabilities = payload["model"].predict_proba(payload["scaler"].transform(feature_matrix(records)))[:, 1]
    return [int(records[i]["client_id"]) for i in stable_rank(probabilities.tolist())[:3]]


def raw_merge(packet: dict, selected: list[int]) -> tuple[list[dict], list[dict]]:
    available = packet["local_dense_docs_top10"]
    transmitted = [doc for client in selected for doc in available[str(client)][:5]]
    merged = sorted(transmitted, key=lambda doc: (-float(doc["dense_score"]), str(doc["doc_id"])))[:10]
    return transmitted, merged


def load_published(path: Path) -> dict[tuple[str, str], list[int]]:
    output = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] in METHODS.values():
                output[(row["query_id"], row["method"])] = [int(v) for v in ast.literal_eval(row["selected_clients"])]
    return output


def write_frozen_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    path.chmod(0o444)


def write_manifest(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/iiserver31/projects/FedE4RAG-main"))
    parser.add_argument("--u1-root", type=Path, required=True)
    args = parser.parse_args()
    v20 = args.root / "V7-HP-PAPER/v20_arc_fedsearch"
    r3 = v20 / "stage_r3_probe_route/hotpot_transfer"
    old_r4 = v20 / "stage_r4_frozen_reader"
    packet_path = r3 / "packets/probe_holdout.jsonl"
    model_path = r3 / "models/logistic_seed_20260807.pkl"
    published_path = r3 / "holdout/main_results/per_query_results.csv"
    index_path = args.root / "V7-HP-PAPER/v15_robust_context_repair/retrieval/indexes/hotpotqa.sqlite"
    packets = read_jsonl(packet_path)
    if len(packets) != 300 or len({str(row["query_id"]) for row in packets}) != 300:
        raise ValueError("sealed Hotpot packet must contain 300 unique query IDs")
    if any(row.get("gold_or_answer_used") is not False for row in packets):
        raise ValueError("sealed packet does not assert gold_or_answer_used=false")
    if nested_keys(packets) & FORBIDDEN:
        raise ValueError(f"forbidden fields in sealed packet: {sorted(nested_keys(packets) & FORBIDDEN)}")
    query_ids = [str(row["query_id"]) for row in packets]
    published = load_published(published_path)
    with model_path.open("rb") as f:
        logistic_payload = pickle.load(f)
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    outputs = {method: [] for method in METHODS}
    selection_mismatches = []
    try:
        for packet in packets:
            query_id = str(packet["query_id"])
            records = packet["p0_candidate_records"]
            selected_by_method = {
                "inherited": [int(v) for v in packet["inherited_selected_clients"]],
                "label_free": label_free_select(records),
                "logistic": logistic_select(records, logistic_payload),
            }
            for method, selected in selected_by_method.items():
                expected = published.get((query_id, METHODS[method]))
                if expected != selected:
                    selection_mismatches.append({"query_id": query_id, "method": method, "expected": expected, "actual": selected})
                    continue
                transmitted, merged = raw_merge(packet, selected)
                retrieved_ids = [str(doc["doc_id"]) for doc in merged]
                reader_ids = retrieved_ids[:5]
                docs = []
                for doc_id in reader_ids:
                    row = connection.execute("SELECT doc_id,title,text FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
                    if row is None:
                        raise KeyError(f"document missing from frozen index: {doc_id}")
                    docs.append({"doc_id": str(row[0]), "title": str(row[1]), "text": str(row[2])})
                context_hash = canonical_hash({"reader_context_docs": docs})
                outputs[method].append({
                    "query_id": query_id,
                    "method": method,
                    "retrieval_method": METHODS[method],
                    "selected_clients": selected,
                    "transmitted_doc_ids": [str(doc["doc_id"]) for doc in transmitted],
                    "retrieved_doc_ids": retrieved_ids,
                    "reader_context_doc_ids": reader_ids,
                    "reader_context_docs": docs,
                    "document_order_sha256": digest_lines(reader_ids),
                    "context_sha256": context_hash,
                    "global_pool_size": 10,
                    "reader_context_k": 5,
                    "source": "sealed_r3_packet_raw_dense_merge",
                    "gold_or_answer_used": False,
                })
    finally:
        connection.close()
    if selection_mismatches:
        raise ValueError(f"published R3 selected-client mismatch: {selection_mismatches[:3]}")

    manifests = {}
    for method, rows in outputs.items():
        if len(rows) != 300 or [row["query_id"] for row in rows] != query_ids:
            raise ValueError(f"{method}: context order/count mismatch")
        if any(len(row["retrieved_doc_ids"]) != 10 or len(row["reader_context_docs"]) != 5 for row in rows):
            raise ValueError(f"{method}: document budget mismatch")
        context_path = args.u1_root / "contexts" / f"{method}.jsonl"
        write_frozen_jsonl(context_path, rows)
        manifest = {
            "status": "frozen",
            "dataset": "hotpotqa",
            "method": method,
            "retrieval_method": METHODS[method],
            "n": 300,
            "query_id_order_sha256": digest_lines(query_ids),
            "query_id_set_sha256": digest_lines(sorted(query_ids)),
            "context_file": str(context_path),
            "context_file_sha256": sha256(context_path),
            "per_query_context_hashes_sha256": digest_lines([row["context_sha256"] for row in rows]),
            "document_count_distribution": dict(sorted(Counter(len(row["reader_context_docs"]) for row in rows).items())),
            "reader_context_k": 5,
            "global_pool_size": 10,
            "packet_sha256": sha256(packet_path),
            "logistic_model_sha256": sha256(model_path) if method == "logistic" else None,
            "published_r3_results_sha256": sha256(published_path),
            "frozen_index_sha256": sha256(index_path),
            "published_selected_clients_match": True,
            "gold_or_answer_used": False,
            "schema_forbidden_fields_found": [],
        }
        manifest_path = args.u1_root / "input_manifests" / f"{method}_manifest.json"
        write_manifest(manifest_path, manifest)
        manifests[method] = manifest

    partial_path = old_r4 / "centralized_reference/hotpotqa_centralized_pool.jsonl"
    partial_rows = read_jsonl(partial_path)
    partial_ids = [str(row["query_id"]) for row in partial_rows]
    forbidden_found = sorted(nested_keys(partial_rows) & FORBIDDEN)
    contract_ok = all(
        row.get("dataset") == "hotpotqa"
        and row.get("partition") == "centralized"
        and row.get("selected_clients") == [0]
        and int(row.get("client_budget", -1)) == 1
        and int(row.get("local_k", -1)) == 10
        and int(row.get("pool_size", -1)) == 10
        and row.get("router") == "centralized_global_hybrid"
        and len(row.get("pool", [])) >= 10
        for row in partial_rows
    )
    partial_audit = {
        "path": str(partial_path),
        "sha256": sha256(partial_path),
        "rows": len(partial_rows),
        "last_line_newline_complete": partial_path.read_bytes().endswith(b"\n"),
        "all_lines_valid_json": True,
        "unique_query_ids": len(set(partial_ids)) == len(partial_ids),
        "canonical_prefix_match": partial_ids == query_ids[:len(partial_ids)],
        "forbidden_fields_found": forbidden_found,
        "centralized_contract_match": contract_ok,
        "safe_to_reuse": bool(partial_rows) and len(partial_rows) < 300 and len(set(partial_ids)) == len(partial_ids) and partial_ids == query_ids[:len(partial_ids)] and not forbidden_found and contract_ok and partial_path.read_bytes().endswith(b"\n"),
    }
    if partial_audit["safe_to_reuse"]:
        cache_path = args.u1_root / "centralized_cache" / f"partial_{len(partial_rows)}.jsonl"
        if cache_path.exists():
            raise FileExistsError(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(partial_path, cache_path)
        cache_path.chmod(0o444)
        partial_audit["copied_cache_path"] = str(cache_path)
        partial_audit["copied_cache_sha256"] = sha256(cache_path)
        central_contexts = []
        for row in partial_rows:
            docs = [
                {"doc_id": str(doc["doc_id"]), "title": str(doc["title"]), "text": str(doc["text"])}
                for doc in row["pool"][:5]
            ]
            ids = [doc["doc_id"] for doc in docs]
            central_contexts.append({
                "query_id": str(row["query_id"]),
                "method": "centralized",
                "retrieval_method": "v17_frozen_centralized_global_hybrid_replay",
                "selected_clients": [0],
                "retrieved_doc_ids": [str(doc["doc_id"]) for doc in row["pool"][:10]],
                "reader_context_doc_ids": ids,
                "reader_context_docs": docs,
                "document_order_sha256": digest_lines(ids),
                "context_sha256": canonical_hash({"reader_context_docs": docs}),
                "global_pool_size": 10,
                "reader_context_k": 5,
                "source": str(cache_path),
                "gold_or_answer_used": False,
            })
        central_partial_path = args.u1_root / "contexts" / "centralized.partial.jsonl"
        write_frozen_jsonl(central_partial_path, central_contexts)
        partial_audit["partial_context_path"] = str(central_partial_path)
        partial_audit["partial_context_sha256"] = sha256(central_partial_path)
    write_manifest(args.u1_root / "input_manifests/centralized_partial_audit.json", partial_audit)

    summary = {
        "status": "federated_contexts_ready_centralized_partial",
        "federated": manifests,
        "centralized_partial": partial_audit,
        "query_id_order_sha256": digest_lines(query_ids),
        "query_id_set_sha256": digest_lines(sorted(query_ids)),
        "labels_read": False,
        "final_test_touched": False,
    }
    write_manifest(args.u1_root / "r4_context_materialization_audit.json", summary)
    print(json.dumps({"status": summary["status"], "centralized_rows": len(partial_rows)}, indent=2))


if __name__ == "__main__":
    main()
