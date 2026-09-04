#!/usr/bin/env python3
"""Create a label-blind 2Wiki sample and the unchanged V4 hybrid Top-5 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from completion_common import EXTERNAL, PROJECT_ROOT, ensure_layout, query_fingerprint, sha256, write_json, write_jsonl


DEFAULT_SOURCE = Path(
    "/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/"
    "cross_dataset_validation/outputs/2wiki_adapter/2wiki_dev_converted.json"
)
SEED = 20260714
ALPHA = 0.55
TOP_K = 5


def text_doc(query_id: str, index: int, raw: list[Any]) -> dict[str, Any]:
    title = str(raw[0])
    sentences = [str(value) for value in raw[1]]
    return {
        "doc_id": f"{query_id}::doc_{index}",
        "title": title,
        "text": " ".join(sentences),
        "sentences": sentences,
        "source_rank": index,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(os.environ.get("V4_2WIKI_DEV", DEFAULT_SOURCE)))
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()
    ensure_layout()
    if args.sample_size < 1000:
        raise AssertionError("Submission external validation requires at least 1,000 examples")

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.v7_hp4.hybrid_retriever import HybridDocument, HybridSoftRetriever

    raw = json.loads(args.source.read_text(encoding="utf-8"))
    ranked = sorted(
        raw,
        key=lambda row: hashlib.sha256(f"{SEED}:{row['query_id']}".encode("utf-8")).hexdigest(),
    )
    sampled = ranked[: args.sample_size]
    normalized_rows = []
    context_rows = []
    for source_index, item in enumerate(sampled):
        query_id = str(item["query_id"])
        docs = [text_doc(query_id, index, value) for index, value in enumerate(item["context"])]
        retriever_docs = [
            HybridDocument(doc_id=doc["doc_id"], title=doc["title"], text=doc["text"], soft_weight=1.0)
            for doc in docs
        ]
        retriever = HybridSoftRetriever(retriever_docs, alpha=ALPHA)
        ranked_docs = [doc for doc, _ in retriever.rank(str(item["question"]), top_k=min(TOP_K, len(docs)))]
        by_id = {doc["doc_id"]: doc for doc in docs}
        baseline_docs = [by_id[doc.doc_id] for doc in ranked_docs]
        normalized_rows.append({
            "query_id": query_id,
            "question": str(item["question"]),
            "answer": str(item["answer"]),
            "supporting_facts": list(item.get("supporting_facts", [])),
            "supporting_titles": [str(value) for value in item.get("supporting_titles", [])],
            "context": list(item["context"]),
            "type": str(item.get("type", "")),
            "source_dataset": "2WikiMultiHopQA",
            "split": "dev",
            "sample_index": source_index,
        })
        context_rows.append({
            "query_id": query_id,
            "question": str(item["question"]),
            "baseline_doc_ids": [doc["doc_id"] for doc in baseline_docs],
            "baseline_titles": [doc["title"] for doc in baseline_docs],
            "baseline_context": baseline_docs,
            "all_docs": docs,
            "retriever": {
                "implementation": "src.v7_hp4.hybrid_retriever.HybridSoftRetriever",
                "alpha": ALPHA,
                "top_k": TOP_K,
                "weights": "uniform_1.0",
                "bm25_only": False,
            },
        })

    source_path = EXTERNAL / "2wiki_frozen_1000.json"
    contexts_path = EXTERNAL / "frozen_baseline_contexts_1000.jsonl"
    write_json(source_path, normalized_rows)
    write_jsonl(contexts_path, context_rows)
    source_ids = {str(row["query_id"]) for row in raw}
    sample_ids = {str(row["query_id"]) for row in normalized_rows}
    audit = {
        "status": "pass",
        "dataset": "2WikiMultiHopQA",
        "source_split": "dev",
        "source_size": len(raw),
        "sample_size": len(normalized_rows),
        "sampling": f"sha256({SEED}:query_id), first {args.sample_size}",
        "sample_query_fingerprint": query_fingerprint(sample_ids),
        "source_query_fingerprint": query_fingerprint(source_ids),
        "labels_used_for_sampling": False,
        "labels_used_for_retrieval": False,
        "baseline_retriever": "HybridSoftRetriever(alpha=0.55, uniform weights, top_k=5)",
        "bm25_only_substitution_used": False,
        "hotpot_generator_frozen": True,
        "hotpot_selector_frozen": True,
        "target_threshold_tuning": False,
        "source_path": str(args.source),
        "source_sha256": sha256(args.source),
        "sample_path": str(source_path),
        "sample_sha256": sha256(source_path),
        "contexts_path": str(contexts_path),
        "contexts_sha256": sha256(contexts_path),
    }
    write_json(EXTERNAL / "data_and_baseline_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
