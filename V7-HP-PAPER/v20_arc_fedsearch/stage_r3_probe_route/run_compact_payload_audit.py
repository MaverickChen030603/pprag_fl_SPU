#!/usr/bin/env python3
"""Audit a compact, no-text wire format against frozen R3 probe decisions.

This program never calls a retriever or modifies routing.  It consumes the
already materialized Probe-Dev transcripts, round-trips their scalar features
through a fixed float32 payload, and checks the original P0--P5 client choices
and downstream retrieval metrics byte-for-byte at the semantic level.
"""
from __future__ import annotations

import argparse
import csv
import json
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FEATURE_SCHEMA = (
    "dense_top1_score",
    "dense_top3_mean",
    "dense_top1_top2_margin",
    "dense_score_std",
    "dense_score_entropy",
    "dense_local_rank_percentile",
    "bm25_top1_score",
    "bm25_top3_mean",
    "bm25_top1_top2_margin",
    "dense_bm25_top1_same",
    "dense_bm25_top3_overlap",
    "dense_sparse_rank_correlation",
    "matched_query_entity_count",
    "matched_query_token_count",
    "matched_title_token_count",
    "query_title_embedding_similarity",
    "top3_title_diversity",
    "top3_entity_diversity",
)
HEADER_BYTES = 16  # protocol version, schema ID, count, integrity checksum
BYTES_PER_CLIENT = 4 * len(FEATURE_SCHEMA)
RRF_K = 60


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_ranking(values: list[float]) -> list[int]:
    return [int(index) for index in np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")]


def minmax(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    span = float(array.max() - array.min())
    return np.zeros_like(array) if span <= 1e-12 else (array - array.min()) / span


def unpack_feature(record: dict[str, Any]) -> dict[str, float]:
    """Fixed order enables schema-once binary transport without text fields."""
    packed = struct.pack("<" + "f" * len(FEATURE_SCHEMA), *(float(record[name]) for name in FEATURE_SCHEMA))
    restored = struct.unpack("<" + "f" * len(FEATURE_SCHEMA), packed)
    return dict(zip(FEATURE_SCHEMA, restored))


def selections(records: list[dict[str, Any]], limit: int) -> dict[str, list[int]]:
    by_client = {int(record["client_id"]): record for record in records}
    ordered = sorted(records, key=lambda record: int(record["static_candidate_rank"]))
    candidates = [int(record["client_id"]) for record in ordered[:limit]]
    restored = {client: unpack_feature(by_client[client]) for client in candidates}
    dense_top1 = [restored[client]["dense_top1_score"] for client in candidates]
    dense_top3 = [restored[client]["dense_top3_mean"] for client in candidates]
    percentile = [restored[client]["dense_local_rank_percentile"] for client in candidates]
    bm25_top1 = [restored[client]["bm25_top1_score"] for client in candidates]
    dense_rank = stable_ranking(dense_top1)
    bm25_rank = stable_ranking(bm25_top1)
    rrf = [1.0 / (RRF_K + dense_rank[index] + 1) + 1.0 / (RRF_K + bm25_rank[index] + 1) for index in range(limit)]
    static = [float(by_client[client]["static_score"]) for client in candidates]
    result = {
        "P0_static_single_centroid": [int(record["client_id"]) for record in ordered[:3]],
        "P1_probe_dense_top1": [candidates[index] for index in stable_ranking(dense_top1)[:3]],
        "P2_probe_dense_top3_mean": [candidates[index] for index in stable_ranking(dense_top3)[:3]],
        "P3_probe_dense_rank_percentile": [candidates[index] for index in stable_ranking(percentile)[:3]],
        "P4_probe_dense_bm25_rrf": [candidates[index] for index in stable_ranking(rrf)[:3]],
    }
    static_norm, probe_norm = minmax(static), minmax(dense_top3)
    for alpha in (0.25, 0.50, 0.75):
        values = alpha * static_norm + (1.0 - alpha) * probe_norm
        result[f"P5_static_plus_probe_alpha_{alpha:.2f}"] = [candidates[index] for index in stable_ranking(values.tolist())[:3]]
    return result


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.stage_root
    transcript = list(jsonl(root / "probe_features" / args.dataset / "per_query_client_probe.jsonl"))
    frozen = csv_rows(root / "label_free_baselines" / args.dataset / "per_query_results.csv")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in transcript:
        grouped[str(record["query_id"])].append(record)
    frozen_by_key = {(row["query_id"], int(row["candidate_L"]), row["method"]): row for row in frozen}

    audit_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for query_id, records in sorted(grouped.items()):
        for candidate_limit in (5, 8):
            expected = selections(records, candidate_limit)
            for method, selected in expected.items():
                original = frozen_by_key[(query_id, candidate_limit, method)]
                original_selected = json.loads(original["selected_clients"])
                exact = selected == original_selected
                if not exact:
                    mismatches.append({
                        "query_id": query_id,
                        "candidate_L": candidate_limit,
                        "method": method,
                        "original_selected_clients": original["selected_clients"],
                        "float32_roundtrip_selected_clients": json.dumps(selected),
                    })
                uses_probe = method != "P0_static_single_centroid"
                compact_bytes = HEADER_BYTES + candidate_limit * BYTES_PER_CLIENT if uses_probe else 0
                audit_rows.append({
                    "dataset": args.dataset,
                    "query_id": query_id,
                    "candidate_L": candidate_limit,
                    "method": method,
                    "selection_exact_after_float32_roundtrip": exact,
                    "probe_clients_contacted": candidate_limit if uses_probe else 0,
                    "wire_schema_header_bytes": HEADER_BYTES if uses_probe else 0,
                    "wire_feature_bytes_per_client": BYTES_PER_CLIENT if uses_probe else 0,
                    "compact_probe_wire_bytes": compact_bytes,
                    "verbose_probe_debug_bytes": int(original["probe_bytes"]),
                    "document_bytes": int(original["document_bytes"]),
                    "local_complete_at_10_frozen": original["local_complete_at_10"],
                    "transmitted_complete_at_15_frozen": original["transmitted_complete_at_15"],
                    "raw_merged_complete_at_10_frozen": original["raw_merged_complete_at_10"],
                    "percentile_merged_complete_at_10_frozen": original["percentile_merged_complete_at_10"],
                    "retrieval_recomputed": False,
                    "reader_started": False,
                    "final_test_accessed": False,
                })
    if mismatches:
        raise AssertionError(f"float32 payload altered frozen choices: {mismatches[:3]}")

    summary: list[dict[str, Any]] = []
    for candidate_limit in (5, 8):
        for method in sorted({row["method"] for row in audit_rows if row["candidate_L"] == candidate_limit}):
            rows = [row for row in audit_rows if row["candidate_L"] == candidate_limit and row["method"] == method]
            compact = float(np.mean([float(row["compact_probe_wire_bytes"]) for row in rows]))
            document = float(np.mean([float(row["document_bytes"]) for row in rows]))
            summary.append({
                "dataset": args.dataset,
                "candidate_L": candidate_limit,
                "method": method,
                "queries": len(rows),
                "mean_compact_probe_wire_bytes": compact,
                "mean_verbose_probe_debug_bytes": float(np.mean([float(row["verbose_probe_debug_bytes"]) for row in rows])),
                "mean_document_bytes": document,
                "probe_to_document_ratio": compact / document if document else 0.0,
                "selection_roundtrip_exact": all(str(row["selection_exact_after_float32_roundtrip"]) == "True" for row in rows),
                "local_complete_at_10_frozen": float(np.mean([float(row["local_complete_at_10_frozen"]) for row in rows])),
                "transmitted_complete_at_15_frozen": float(np.mean([float(row["transmitted_complete_at_15_frozen"]) for row in rows])),
                "raw_merged_complete_at_10_frozen": float(np.mean([float(row["raw_merged_complete_at_10_frozen"]) for row in rows])),
                "percentile_merged_complete_at_10_frozen": float(np.mean([float(row["percentile_merged_complete_at_10_frozen"]) for row in rows])),
                "reader_started": False,
            })

    communication = root / "communication" / args.dataset
    write_csv(communication / "per_query_compact_payload_audit.csv", audit_rows)
    write_csv(communication / "compact_probe_wire_payload_audit.csv", summary)
    decision_rows = [row for row in summary if row["candidate_L"] == 8 and row["method"] != "P0_static_single_centroid"]
    cost_pass = all(float(row["probe_to_document_ratio"]) <= 0.10 for row in decision_rows)
    decision = {
        "dataset": args.dataset,
        "feature_schema": list(FEATURE_SCHEMA),
        "schema_version": "r3-fixed-f32-v1",
        "wire_header_bytes": HEADER_BYTES,
        "wire_feature_bytes_per_probed_client": BYTES_PER_CLIENT,
        "all_frozen_p0_p5_selections_exact_after_float32_roundtrip": True,
        "all_retrieval_metrics_frozen_not_recomputed": True,
        "candidate_l8_probe_to_document_ratio_leq_0_10_for_all_probe_methods": cost_pass,
        "reader_started": False,
        "final_test_accessed": False,
    }
    (communication / "compact_payload_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
