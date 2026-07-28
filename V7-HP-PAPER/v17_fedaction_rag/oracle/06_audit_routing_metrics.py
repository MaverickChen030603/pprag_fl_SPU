#!/usr/bin/env python3
"""Offline gold audit of routing, retrieval, support access, and communication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def norm(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def doc_id(dataset: str, title: str, text: str = "") -> str:
    identity = norm(title) if dataset != "musique" else norm(title) + "\n" + norm(text)
    return f"{dataset}:{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:20]}"


def gold_support_docs(row: dict[str, Any], dataset: str) -> set[str]:
    if dataset == "musique":
        return {
            doc_id(dataset, paragraph.get("title", ""), paragraph.get("paragraph_text", ""))
            for paragraph in row.get("paragraphs", [])
            if paragraph.get("is_supporting", paragraph.get("is_support", False))
        }
    facts = row.get("supporting_facts", [])
    titles = facts.get("title", []) if isinstance(facts, dict) else [value[0] for value in facts if value]
    return {doc_id(dataset, str(title)) for title in titles}


def redundancy(docs: list[dict[str, Any]]) -> float:
    token_sets = [set(TOKEN_RE.findall((doc["title"] + " " + doc["text"]).lower())) for doc in docs]
    values = [len(left & right) / max(1, len(left | right)) for left, right in combinations(token_sets, 2)]
    return float(np.mean(values)) if values else 0.0


def write_csv(path: Path, output: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dispersion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    source = {
        dataset: {qid(row): row for row in rows(args.data_root / dataset / "development.jsonl")}
        for dataset in ("hotpotqa", "2wikimultihopqa", "musique")
    }
    dispersion = {}
    with args.dispersion.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dispersion[(row["dataset"], row["partition"], row["query_id"])] = row

    output = []
    for path in sorted(args.pool_dir.glob("*.jsonl")):
        for pool in rows(path):
            dataset, partition, query_id = pool["dataset"], pool["partition"], str(pool["query_id"])
            source_row = source[dataset][query_id]
            support_docs = gold_support_docs(source_row, dataset)
            selected_clients = set(map(int, pool["selected_clients"]))
            action_docs = pool["pool"][: int(pool["pool_size"])]
            returned_docs = [doc for doc in pool["pool"] if int(doc["client_id"]) in selected_clients]
            action_ids = {str(doc["doc_id"]) for doc in action_docs}
            returned_ids = {str(doc["doc_id"]) for doc in returned_docs}
            audit = dispersion.get((dataset, partition, query_id))
            support_clients = set()
            if audit:
                support_clients = {int(value) for value in audit["support_clients"].split("|") if value}
            answer = norm(source_row.get("answer", ""))
            answer_access = int(
                bool(answer)
                and any(answer in norm(doc["title"] + " " + doc["text"]) for doc in action_docs)
            )
            transmitted = len(str(source_row["question"]).encode("utf-8")) * len(selected_clients)
            transmitted += sum(len((doc["title"] + doc["text"]).encode("utf-8")) for doc in returned_docs)
            output.append({
                "dataset": dataset,
                "partition": partition,
                "query_id": query_id,
                "client_budget": int(pool["client_budget"]),
                "local_k": int(pool["local_k"]),
                "clients_contacted": len(selected_clients),
                "gold_client_recall": len(support_clients & selected_clients) / max(1, len(support_clients)) if audit else "",
                "complete_evidence_client_recall": int(support_clients.issubset(selected_clients)) if audit else "",
                "local_return_support_recall": len(support_docs & returned_ids) / max(1, len(support_docs)),
                "action_pool_support_recall": len(support_docs & action_ids) / max(1, len(support_docs)),
                "complete_support_in_action_pool": int(support_docs.issubset(action_ids)),
                "answer_access": answer_access,
                "candidate_redundancy": redundancy(action_docs),
                "action_client_diversity": len({int(doc["client_id"]) for doc in action_docs}),
                "documents_transmitted": len(returned_docs),
                "estimated_online_bytes": transmitted,
                "retrieval_latency_ms": float(pool["retrieval_latency_ms"]),
            })
    write_csv(args.output, output)
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        grouped[(row["dataset"], row["partition"], row["client_budget"], row["local_k"])].append(row)
    summary = []
    metrics = [key for key in output[0] if key not in {"dataset", "partition", "query_id", "client_budget", "local_k"}]
    for key, values in sorted(grouped.items()):
        result = {"dataset": key[0], "partition": key[1], "client_budget": key[2], "local_k": key[3], "queries": len(values)}
        for metric in metrics:
            available = [float(row[metric]) for row in values if row[metric] != ""]
            result[f"mean_{metric}"] = float(np.mean(available)) if available else ""
        summary.append(result)
    write_csv(args.summary, summary)
    print(json.dumps({"status": "complete", "queries": len(output), "cells": len(summary), "output": str(args.output), "summary": str(args.summary)}, indent=2))


if __name__ == "__main__":
    main()
