#!/usr/bin/env python3
"""Freeze a disjoint 3,000-query HotpotQA context set from the exact v1 baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from v4_common import (
    HERE,
    OUTPUTS,
    PROJECT_ROOT,
    context_snapshot_path,
    ensure_layout,
    query_fingerprint,
    read_json,
    read_jsonl,
    sha256,
    source_1000_path,
    write_json,
    write_jsonl,
)


SOURCE_NAME = "huggingface:hotpot_qa/distractor/validation"
SOURCE_SEED = 44
DEVELOPMENT_SIZE = 1000
SCALEUP_SIZE = 3000
ALPHA = 0.55
TOP_K = 5


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruct_source() -> list[dict[str, Any]]:
    from datasets import load_dataset

    fullval = load_module(PROJECT_ROOT / "V7-HP4/run_hp4_full_validation_eval.py", "v4_scale_fullval")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    candidates = []
    for index, item in enumerate(dataset):
        normalized = fullval._normalize_hotpot_item(dict(item), index)
        if normalized is not None:
            candidates.append(normalized)
    random.Random(SOURCE_SEED).shuffle(candidates)
    return candidates


def baseline_row(item: dict[str, Any], source_index: int, selector_v1: Any, reader: Any) -> dict[str, Any]:
    built = reader.build_dev300_case(item, source_index)
    if built is None:
        raise AssertionError(f"Failed to materialize {item.get('_id', source_index)}")
    case, raw_docs = built
    docs = selector_v1.sanitize_docs(raw_docs)
    weights = {doc.doc_id: 1.0 for doc in docs}
    candidates = selector_v1.generate_candidates(
        str(case.get("question", "")), docs, weights, alpha=ALPHA, top_k=TOP_K
    )
    baseline = next(row for row in candidates if row["mode"] == "baseline")
    top_docs = baseline["top_docs"]
    return {
        "query_id": str(case["id"]),
        "question": str(case["question"]),
        "baseline_doc_ids": [doc.doc_id for doc in top_docs],
        "baseline_titles": [doc.title for doc in top_docs],
        "baseline_context": [
            {"doc_id": doc.doc_id, "title": doc.title, "text": doc.text, "source_rank": int(doc.doc_id.rsplit("_", 1)[-1])}
            for doc in top_docs
        ],
        "all_docs": [
            {"doc_id": doc.doc_id, "title": doc.title, "text": doc.text, "source_rank": int(doc.doc_id.rsplit("_", 1)[-1])}
            for doc in docs
        ],
        "retriever": {
            "implementation": "src.v7_hp4.hybrid_retriever.HybridSoftRetriever",
            "weights": "uniform_1.0",
            "alpha": ALPHA,
            "top_k": TOP_K,
            "dense_component": "deterministic lexical/entity dense proxy",
            "sparse_component": "BM25Scorer",
            "bm25_only": False,
        },
    }


def canonical_rows(rows: list[dict[str, Any]]) -> list[str]:
    return [json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaleup-size", type=int, default=SCALEUP_SIZE)
    parser.add_argument("--development-size", type=int, default=DEVELOPMENT_SIZE)
    args = parser.parse_args()
    if args.scaleup_size < 3000:
        raise AssertionError("Paper scale-up requires at least 3,000 untouched queries")
    ensure_layout()
    scale_dir = OUTPUTS / "scaleup"
    source_path = scale_dir / "same_source_hotpot_validation_3000.json"
    contexts_path = scale_dir / "frozen_baseline_contexts_3000.jsonl"

    reconstructed = reconstruct_source()
    needed = args.development_size + args.scaleup_size
    if len(reconstructed) < needed:
        raise AssertionError(f"Need {needed} examples, found {len(reconstructed)}")

    existing_1000 = read_json(source_1000_path())
    prefix = reconstructed[: args.development_size]
    source_exact = canonical_rows(prefix) == canonical_rows(existing_1000)
    if not source_exact:
        mismatches = [
            index for index, (left, right) in enumerate(zip(canonical_rows(prefix), canonical_rows(existing_1000))) if left != right
        ]
        raise AssertionError(f"Seed-44 source reconstruction differs from frozen 1,000 at {mismatches[:10]}")

    selector_v1 = load_module(PROJECT_ROOT / "V7-HP-PAPER/run_support_insertion_selector_v1.py", "v4_scale_selector_v1")
    reader = selector_v1.READER
    snapshots = {str(row["query_id"]): row for row in read_jsonl(context_snapshot_path())}
    reproduction_rows = []
    title_mismatches = []
    for index, item in enumerate(prefix):
        row = baseline_row(item, index, selector_v1, reader)
        frozen_titles = list(snapshots[row["query_id"]]["baseline_titles"])
        matches = row["baseline_titles"] == frozen_titles
        reproduction_rows.append({
            "query_id": row["query_id"],
            "generated_titles": row["baseline_titles"],
            "frozen_titles": frozen_titles,
            "exact_title_order_match": matches,
        })
        if not matches:
            title_mismatches.append(reproduction_rows[-1])
    if title_mismatches:
        write_jsonl(scale_dir / "baseline_reproduction_mismatches.jsonl", title_mismatches)
        raise AssertionError(f"Baseline reproduction failed for {len(title_mismatches)}/1000 queries")

    scale_source = reconstructed[args.development_size:needed]
    development_ids = {str(row["_id"]) for row in prefix}
    scale_ids = {str(row["_id"]) for row in scale_source}
    overlap = development_ids & scale_ids
    if overlap:
        raise AssertionError(f"Development/scale-up overlap: {len(overlap)}")
    write_json(source_path, scale_source)

    context_rows = [
        baseline_row(item, args.development_size + index, selector_v1, reader)
        for index, item in enumerate(scale_source)
    ]
    write_jsonl(contexts_path, context_rows)
    invalid_context_rows = [
        row for row in context_rows
        if not row["all_docs"] or len(row["baseline_doc_ids"]) != min(TOP_K, len(row["all_docs"]))
    ]
    if len(context_rows) != args.scaleup_size or invalid_context_rows:
        raise AssertionError("Scale-up context cardinality audit failed")
    available_doc_distribution = Counter(len(row["all_docs"]) for row in context_rows)
    frozen_context_distribution = Counter(len(row["baseline_doc_ids"]) for row in context_rows)

    audit = {
        "status": "pass",
        "source": SOURCE_NAME,
        "source_seed": SOURCE_SEED,
        "full_validation_size": len(reconstructed),
        "development_size": args.development_size,
        "scaleup_size": args.scaleup_size,
        "scaleup_slice": [args.development_size, needed],
        "development_scaleup_overlap": len(overlap),
        "available_doc_count_distribution": {str(key): value for key, value in sorted(available_doc_distribution.items())},
        "frozen_context_count_distribution": {str(key): value for key, value in sorted(frozen_context_distribution.items())},
        "top_k_semantics": "up to 5 documents; preserve official examples with fewer available contexts",
        "source_1000_exact_reconstruction": source_exact,
        "baseline_1000_exact_title_order_matches": len(reproduction_rows) - len(title_mismatches),
        "baseline_1000_reproduction_rate": 1.0,
        "baseline_retriever": "HybridSoftRetriever(alpha=0.55, uniform weights, top_k=5)",
        "bm25_only_substitution_used": False,
        "thresholds_retuned": False,
        "development_query_fingerprint": query_fingerprint(development_ids),
        "scaleup_query_fingerprint": query_fingerprint(scale_ids),
        "source_1000_path": str(source_1000_path()),
        "source_1000_sha256": sha256(source_1000_path()),
        "scaleup_source_path": str(source_path),
        "scaleup_source_sha256": sha256(source_path),
        "frozen_contexts_path": str(contexts_path),
        "frozen_contexts_sha256": sha256(contexts_path),
        "selector_v1_sha256": sha256(PROJECT_ROOT / "V7-HP-PAPER/run_support_insertion_selector_v1.py"),
        "hybrid_retriever_sha256": sha256(PROJECT_ROOT / "src/v7_hp4/hybrid_retriever.py"),
    }
    write_json(scale_dir / "same_source_context_audit.json", audit)
    write_jsonl(scale_dir / "baseline_reproduction_1000.jsonl", reproduction_rows)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
