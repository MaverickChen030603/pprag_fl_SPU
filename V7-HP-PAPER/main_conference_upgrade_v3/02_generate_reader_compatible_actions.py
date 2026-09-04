#!/usr/bin/env python3
"""Generate bounded reader-compatible actions using inference-safe signals only."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from typing import Any

from v3_common import OUTPUTS, ensure_layout, doc_feature_rows, load_context_snapshots, load_source_examples, markdown_table, normalize_title, write_json, write_jsonl


FORBIDDEN_KEYS = {
    "answer", "gold_answer", "supporting_titles", "gold_support", "reader_outcome",
    "oracle_action", "answer_f1", "joint_f1", "sp_f1", "support_recall",
}
ALLOWED_SIGNALS = {
    "query_text", "candidate_text", "baseline_context_text", "bm25_score",
    "query_overlap", "entity_overlap", "title_overlap", "anchor_proxy_score",
    "baseline_rank", "displaced_score_margin", "bridge_connection",
    "novel_entity_ratio", "redundancy", "routing_metadata",
}


def context_from_snapshot(snapshot: dict[str, Any], source_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_title = {normalize_title(doc["title"]): doc for doc in source_docs}
    baseline: list[dict[str, Any]] = []
    context_rows = snapshot.get("baseline_context", [])
    for index, title in enumerate(snapshot.get("baseline_titles", [])):
        source_doc = by_title.get(normalize_title(title))
        context_row = context_rows[index] if index < len(context_rows) else {}
        text = " ".join(str(value) for value in context_row.get("sentences", []))
        baseline.append({
            "doc_id": source_doc["doc_id"] if source_doc else f"{snapshot['query_id']}::frozen_baseline_{index}",
            "title": str(title),
            "text": text or (source_doc["text"] if source_doc else ""),
            "source_rank": source_doc.get("source_rank", -1) if source_doc else -1,
        })
    if not baseline:
        raise AssertionError(f"No frozen baseline context for {snapshot.get('query_id')}")
    return baseline


def aggregate_features(added: list[dict[str, Any]], removed: list[dict[str, Any]], preserve: int, ordering: str) -> dict[str, Any]:
    def mean(key: str, rows: list[dict[str, Any]]) -> float:
        return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0

    return {
        "added_bm25_mean": mean("bm25_score", added),
        "added_query_overlap_mean": mean("query_overlap", added),
        "added_entity_overlap_mean": mean("entity_overlap", added),
        "added_title_overlap_mean": mean("title_overlap", added),
        "added_bridge_connection_mean": mean("bridge_connection", added),
        "added_novel_entity_ratio_mean": mean("novel_entity_ratio", added),
        "added_redundancy_mean": mean("redundancy", added),
        "added_anchor_proxy_mean": mean("anchor_proxy_score", added),
        "removed_anchor_proxy_mean": mean("anchor_proxy_score", removed),
        "displaced_score_margin": mean("anchor_proxy_score", added) - mean("anchor_proxy_score", removed),
        "preserved_prefix_length": preserve,
        "num_added_docs": len(added),
        "num_removed_docs": len(removed),
        "ordering_anchor_first": float(ordering == "anchor_first"),
        "ordering_bridge_first": float(ordering == "bridge_first"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=16)
    args = parser.parse_args()
    ensure_layout()
    source, snapshots = load_source_examples(), load_context_snapshots()
    rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()

    for query_index, query_id in enumerate(sorted(source)):
        if args.max_queries and query_index >= args.max_queries:
            break
        item, snapshot = source[query_id], snapshots[query_id]
        source_docs = item["docs"]
        baseline = context_from_snapshot(snapshot, source_docs)
        frozen_by_id = {doc["doc_id"]: doc for doc in baseline}
        docs = [frozen_by_id.get(doc["doc_id"], doc) for doc in source_docs]
        docs.extend(doc for doc in baseline if doc["doc_id"] not in {value["doc_id"] for value in source_docs})
        baseline_ids = [doc["doc_id"] for doc in baseline]
        context_size = len(baseline_ids)
        by_id = {doc["doc_id"]: doc for doc in docs}
        features = doc_feature_rows(str(item["question"]), docs, baseline_ids)
        feature_by_id = {row["doc_id"]: row for row in features}
        candidates = [feature_by_id[doc["doc_id"]] for doc in docs if doc["doc_id"] not in baseline_ids]
        candidates.sort(key=lambda row: (row["anchor_proxy_score"] + 0.25 * row["bridge_connection"] + 0.10 * row["novel_entity_ratio"] - 0.15 * row["redundancy"]), reverse=True)
        baseline_features = [feature_by_id[doc_id] for doc_id in baseline_ids]
        actions: list[dict[str, Any]] = []
        seen = {tuple(baseline_ids)}

        def add_action(family: str, name: str, ordered_ids: list[str], added_ids: list[str], removed_ids: list[str], preserve: int, ordering: str, safety_prior: str) -> None:
            if len(actions) >= args.max_actions - 1 or len(ordered_ids) != context_size or len(set(ordered_ids)) != context_size:
                return
            key = tuple(ordered_ids)
            if key in seen:
                return
            seen.add(key)
            added = [feature_by_id[doc_id] for doc_id in added_ids]
            removed = [feature_by_id[doc_id] for doc_id in removed_ids]
            action = {
                "query_id": query_id,
                "question": item["question"],
                "action_id": f"{query_id}::{len(actions):02d}",
                "action_family": family,
                "action_name": name,
                "context_doc_ids": ordered_ids,
                "context_titles": [by_id[doc_id]["title"] for doc_id in ordered_ids],
                "context_docs": [by_id[doc_id] for doc_id in ordered_ids],
                "added_doc_ids": added_ids,
                "removed_doc_ids": removed_ids,
                "safety_prior": safety_prior,
                "inference_safe_features": aggregate_features(added, removed, preserve, ordering),
            }
            actions.append(action)
            family_counts[family] += 1

        # 1. Anchor-preserving single-document tail replacement.
        for preserve in tuple(value for value in (3, 2, 1) if value < context_size):
            removable = list(range(preserve, context_size))
            tail_index = min(removable, key=lambda idx: baseline_features[idx]["anchor_proxy_score"])
            for rank, candidate in enumerate(candidates[:2]):
                ordered = list(baseline_ids)
                removed_id = ordered[tail_index]
                ordered[tail_index] = candidate["doc_id"]
                add_action("anchor_preserving_tail_replacement", f"keep_top{preserve}_candidate{rank + 1}", ordered, [candidate["doc_id"]], [removed_id], preserve, "stable", "high" if preserve >= 3 else "medium")

        # 2. Bridge-aware complementary insertion.
        bridge_ranked = sorted(candidates, key=lambda row: (row["bridge_connection"] + 0.35 * row["novel_entity_ratio"] + 0.25 * row["bm25_score"] - 0.30 * row["redundancy"]), reverse=True)
        if context_size >= 5:
            for rank, candidate in enumerate(bridge_ranked[:2]):
                ordered = baseline_ids[:3] + [candidate["doc_id"], baseline_ids[3]]
                add_action("bridge_aware_complementary_insertion", f"keep_top3_bridge{rank + 1}", ordered, [candidate["doc_id"]], [baseline_ids[4]], 3, "anchor_first", "high")

        # 3. Newly bounded two-document chain with top-3 anchors preserved.
        pair_rows: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for left, right in combinations(candidates[:6], 2):
            complement = abs(left["novel_entity_ratio"] - right["novel_entity_ratio"]) + 0.5 * (left["bridge_connection"] + right["bridge_connection"])
            score = left["anchor_proxy_score"] + right["anchor_proxy_score"] + 0.45 * complement - 0.35 * (left["redundancy"] + right["redundancy"])
            pair_rows.append((score, left, right))
        pair_rows.sort(key=lambda value: value[0], reverse=True)
        if context_size >= 5:
            for pair_rank, (_, left, right) in enumerate(pair_rows[:2]):
                if min(left["anchor_proxy_score"], right["anchor_proxy_score"]) < 0.12:
                    continue
                ordered = baseline_ids[:3] + [left["doc_id"], right["doc_id"]]
                add_action("bounded_two_document_chain", f"keep_top3_chain{pair_rank + 1}", ordered, [left["doc_id"], right["doc_id"]], baseline_ids[3:5], 3, "anchor_first", "strict")

        # 4. Replace the most redundant tail only when redundancy is visible.
        redundant_tail = [(idx, baseline_features[idx]) for idx in range(2, context_size) if baseline_features[idx]["redundancy"] >= 0.35]
        if redundant_tail and candidates:
            tail_index, _ = max(redundant_tail, key=lambda pair: pair[1]["redundancy"])
            candidate = min(candidates[:4], key=lambda row: row["redundancy"] - row["anchor_proxy_score"] - row["bridge_connection"])
            ordered = list(baseline_ids)
            removed_id = ordered[tail_index]
            ordered[tail_index] = candidate["doc_id"]
            add_action("redundancy_aware_replacement", "replace_redundant_tail", ordered, [candidate["doc_id"]], [removed_id], min(2, tail_index), "stable", "high")

        # 5. One insertion followed by one deterministic ordering template.
        if bridge_ranked and context_size >= 5:
            candidate = bridge_ranked[0]
            add_action("joint_reorder_and_insert", "anchor_first_insert", baseline_ids[:3] + [candidate["doc_id"], baseline_ids[3]], [candidate["doc_id"]], [baseline_ids[4]], 3, "anchor_first", "high")
            add_action("joint_reorder_and_insert", "bridge_first_insert", [baseline_ids[0], candidate["doc_id"], baseline_ids[1], baseline_ids[2], baseline_ids[3]], [candidate["doc_id"]], [baseline_ids[4]], 1, "bridge_first", "medium")

        rows.extend(actions)
        rows.append({
            "query_id": query_id,
            "question": item["question"],
            "action_id": f"{query_id}::fallback",
            "action_family": "fallback",
            "action_name": "baseline_fallback",
            "context_doc_ids": baseline_ids,
            "context_titles": [doc["title"] for doc in baseline],
            "context_docs": baseline,
            "added_doc_ids": [],
            "removed_doc_ids": [],
            "safety_prior": "absolute",
            "inference_safe_features": aggregate_features([], [], 5, "stable"),
        })
        family_counts["fallback"] += 1

    output_path = OUTPUTS / "candidate_generation/v3_candidate_actions.jsonl"
    write_jsonl(output_path, rows)
    effective = [row for row in rows if row["action_family"] != "fallback"]
    all_query_ids = sorted({row["query_id"] for row in rows})
    counts_per_query = Counter(row["query_id"] for row in effective)
    per_query_counts = [counts_per_query[query_id] for query_id in all_query_ids]
    summary = {
        "num_queries": len(all_query_ids),
        "num_effective_actions": len(effective),
        "num_fallback_actions": len(rows) - len(effective),
        "effective_actions_per_query_min": min(per_query_counts),
        "effective_actions_per_query_mean": sum(per_query_counts) / len(per_query_counts),
        "effective_actions_per_query_max": max(per_query_counts),
        "family_counts": dict(family_counts),
        "bounded_search": True,
        "maximum_effective_actions_config": args.max_actions - 1,
    }
    write_json(OUTPUTS / "candidate_generation/v3_candidate_generation_summary.json", summary)

    serialized = output_path.read_text(encoding="utf-8").lower()
    key_hits = sorted(key for key in FORBIDDEN_KEYS if f'"{key}"' in serialized)
    audit = {
        "status": "pass" if not key_hits else "fail",
        "generator_type": "deterministic_non_learned",
        "fully_nested_training_required": False,
        "target_gold_answer_used": False,
        "target_gold_support_used": False,
        "target_reader_outcome_used": False,
        "oracle_action_used": False,
        "test_fold_calibration_used": False,
        "allowed_signal_inventory": sorted(ALLOWED_SIGNALS),
        "forbidden_output_key_hits": key_hits,
        "diagnostic_taxonomy_imported": False,
        "source_code_contract": "Generator accepts question, source documents, baseline context, and inference-safe lexical/routing metadata only.",
    }
    write_json(OUTPUTS / "audits/candidate_generation_no_leak_audit.json", audit)
    if key_hits:
        raise AssertionError(f"No-leak audit failed: {key_hits}")

    v2_rows = [["v2 main eligible", "4", "fixed single edit/reorder", "yes"], ["v3", f"{summary['effective_actions_per_query_mean']:.1f}", "bounded anchor/bridge/chain/redundancy actions", "yes"]]
    (OUTPUTS / "tables/v2_vs_v3_action_space.md").write_text("# v2 vs v3 Action Space\n\n" + markdown_table(["Action space", "Mean effective/query", "Families", "Fallback"], v2_rows) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
