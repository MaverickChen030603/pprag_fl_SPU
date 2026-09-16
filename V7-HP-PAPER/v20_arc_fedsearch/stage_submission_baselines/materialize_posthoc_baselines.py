#!/usr/bin/env python3
"""Materialize frozen B0/B1 contexts on the revealed V20 R5 query set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
DEFAULT_SEEDS = tuple(range(20))


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def query_id(row: Mapping[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def static_top3(records: Sequence[Mapping[str, Any]]) -> list[int]:
    if len(records) != 8:
        raise ValueError(f"expected 8 candidate clients, found {len(records)}")
    ranked = sorted(
        records,
        key=lambda row: (
            int(row["static_candidate_rank"]),
            -float(row["static_score"]),
            int(row["client_id"]),
        ),
    )
    return [int(row["client_id"]) for row in ranked[:3]]


def random_top3(
    records: Sequence[Mapping[str, Any]], qid: str, seed: int
) -> list[int]:
    if len(records) != 8:
        raise ValueError(f"expected 8 candidate clients, found {len(records)}")

    def rank_key(row: Mapping[str, Any]) -> tuple[bytes, int]:
        client = int(row["client_id"])
        payload = f"{seed}|{qid}|{client}".encode("utf-8")
        return hashlib.sha256(payload).digest(), client

    return [int(row["client_id"]) for row in sorted(records, key=rank_key)[:3]]


def raw_merge(
    packet: Mapping[str, Any], clients: Sequence[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transmitted = [
        dict(document)
        for client in clients
        for document in packet["local_dense_docs_top10"][str(client)][:5]
    ]
    if len(transmitted) != 15:
        raise ValueError(f"expected 15 transmitted documents, found {len(transmitted)}")
    merged = sorted(
        transmitted,
        key=lambda document: (
            -float(document["dense_score"]),
            str(document["doc_id"]),
        ),
    )[:10]
    return transmitted, merged


def lookup_documents(
    connection: sqlite3.Connection, document_ids: Sequence[str]
) -> list[dict[str, str]]:
    output = []
    for document_id in document_ids:
        value = connection.execute(
            "SELECT doc_id,title,text FROM docs WHERE doc_id=?", (document_id,)
        ).fetchone()
        if value is None:
            raise KeyError(f"missing canonical document {document_id}")
        output.append(
            {"doc_id": str(value[0]), "title": str(value[1]), "text": str(value[2])}
        )
    return output


def context_hash(question: str, documents: Sequence[Mapping[str, str]]) -> str:
    payload = json.dumps(
        {"question": question, "docs": list(documents)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def materialize_dataset(
    dataset: str,
    input_root: Path,
    index_root: Path,
    output_handle: Any,
    seeds: Sequence[int],
) -> dict[str, Any]:
    split_path = input_root / "protocol" / f"{dataset}_final_test_inputs_n300.jsonl"
    packet_path = input_root / "retrieval" / f"{dataset}_probe_packets.jsonl"
    index_path = index_root / f"{dataset}.sqlite"
    split_rows = list(rows(split_path))
    packets = {str(row["query_id"]): row for row in rows(packet_path)}
    split_ids = [query_id(row) for row in split_rows]
    if len(split_ids) != 300 or set(split_ids) != set(packets):
        raise ValueError(
            f"query mismatch for {dataset}: split={len(split_ids)} packets={len(packets)}"
        )

    methods = [("b0_static_top3", None)] + [
        (f"b1_random_top3_seed_{seed:02d}", seed) for seed in seeds
    ]
    route_signatures: Counter[str] = Counter()
    selected_frequency: dict[str, Counter[int]] = {
        method: Counter() for method, _ in methods
    }
    static_overlap: dict[str, list[float]] = {
        method: [] for method, seed in methods if seed is not None
    }
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        for source_row in split_rows:
            qid = query_id(source_row)
            packet = packets[qid]
            records = packet["p0_candidate_records"]
            static_clients = static_top3(records)
            question = str(source_row["question"])
            for method, seed in methods:
                clients = (
                    static_clients if seed is None else random_top3(records, qid, seed)
                )
                transmitted, merged = raw_merge(packet, clients)
                merged_ids = [str(document["doc_id"]) for document in merged]
                documents = lookup_documents(connection, merged_ids[:5])
                payload = {
                    "dataset": dataset,
                    "query_id": qid,
                    "question": question,
                    "method": method,
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
                output_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                signature = f"{dataset}|{method}|{','.join(map(str, clients))}"
                route_signatures[signature] += 1
                selected_frequency[method].update(clients)
                if seed is not None:
                    static_overlap[method].append(
                        len(set(clients) & set(static_clients)) / 3.0
                    )
    finally:
        connection.close()

    return {
        "dataset": dataset,
        "queries": len(split_ids),
        "methods": len(methods),
        "rows": len(split_ids) * len(methods),
        "input_hashes": {
            "split": sha256(split_path),
            "packets": sha256(packet_path),
            "index": sha256(index_path),
        },
        "unique_route_signatures": len(route_signatures),
        "selected_client_frequency": {
            method: {str(client): count for client, count in sorted(counts.items())}
            for method, counts in selected_frequency.items()
        },
        "mean_static_route_overlap": {
            method: sum(values) / len(values)
            for method, values in static_overlap.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if seeds != DEFAULT_SEEDS:
        raise ValueError(f"frozen post-hoc seeds must be {list(DEFAULT_SEEDS)}")

    output_path = args.output_dir / "contexts_unlabeled.jsonl"
    manifest_path = args.output_dir / "materialization_manifest.json"
    audit_path = args.output_dir / "selection_audit.json"
    for path in (output_path, manifest_path, audit_path):
        if path.exists():
            raise FileExistsError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output_dir, delete=False
    ) as handle:
        summaries = [
            materialize_dataset(
                dataset, args.input_root, args.index_root, handle, seeds
            )
            for dataset in DATASETS
        ]
        temporary = Path(handle.name)
    os.replace(temporary, output_path)

    expected_rows = len(DATASETS) * 300 * (1 + len(seeds))
    actual_rows = sum(1 for _ in rows(output_path))
    if actual_rows != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {actual_rows}")
    code_path = Path(__file__).resolve()
    manifest = {
        "status": "complete_unlabeled_contexts",
        "evidence_role": "retrospective_post_hoc_r5_revealed",
        "claim_restriction": "diagnostic_only_not_fresh_confirmation",
        "datasets": list(DATASETS),
        "random_seeds": list(seeds),
        "rows": actual_rows,
        "methods_per_dataset": 1 + len(seeds),
        "labels_used": False,
        "reader_started": False,
        "code_sha256": sha256(code_path),
        "output_sha256": sha256(output_path),
    }
    atomic_json(manifest_path, manifest)
    atomic_json(audit_path, {"status": "pass", "datasets": summaries})
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
