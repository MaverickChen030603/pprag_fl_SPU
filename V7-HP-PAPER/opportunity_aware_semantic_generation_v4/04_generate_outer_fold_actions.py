#!/usr/bin/env python3
"""Generate bounded semantic actions with frozen outer-fold models."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from typing import Any

import joblib
import numpy as np

from semantic_features import pair_key
from v4_common import OUTPUTS, REPORTS, ensure_layout, query_fingerprint, read_json, read_jsonl, sha256, v3_actions_path, write_json, write_jsonl


FORBIDDEN_OUTPUT_KEYS = {
    "answer", "gold_answer", "supporting_titles", "gold_support", "reader_outcome",
    "oracle_action", "answer_f1", "joint_f1", "sp_f1", "support_recall",
}


def positive_probability(model: Any, features: list[float]) -> float:
    values = model.predict_proba(np.asarray([features], dtype=np.float32))[0]
    classes = list(model.classes_)
    return float(values[classes.index(1)]) if 1 in classes else 0.0


def missing_probabilities(model: Any, features: list[float]) -> dict[str, float]:
    values = model.predict_proba(np.asarray([features], dtype=np.float32))[0]
    return {str(label): float(value) for label, value in zip(model.classes_, values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args()
    ensure_layout()
    cache_path = OUTPUTS / "semantic_generator/semantic_feature_cache.joblib"
    cache = joblib.load(cache_path)
    manifest = read_json(OUTPUTS / "semantic_generator/foldwise_generator_models.json")
    v3_contexts: dict[str, set[tuple[str, ...]]] = {}
    for row in read_jsonl(v3_actions_path()):
        if row["action_family"] == "fallback":
            continue
        v3_contexts.setdefault(str(row["query_id"]), set()).add(tuple(row["context_doc_ids"]))

    rows: list[dict[str, Any]] = []
    audit_folds = []
    family_counts: Counter[str] = Counter()
    budget_counts: Counter[int] = Counter()
    selected_query_ids: list[str] = []
    for fold_record in manifest["folds"]:
        fold_id = int(fold_record["fold_id"])
        bundle = joblib.load(fold_record["model_path"])
        test_ids = list(fold_record["test_query_ids"])
        if args.max_queries:
            remaining = max(0, args.max_queries - len(selected_query_ids))
            test_ids = test_ids[:remaining]
        if not test_ids:
            continue
        if bundle["metadata"]["train_query_fingerprint"] == query_fingerprint(test_ids):
            raise AssertionError(f"Fold {fold_id} train/test fingerprint collision")
        for query_id in test_ids:
            query = cache["queries"][query_id]
            docs_by_id = {str(doc["doc_id"]): doc for doc in query["docs"]}
            baseline_ids = list(query["baseline_ids"])
            candidate_ids = list(query["candidate_ids"])
            missing = missing_probabilities(bundle["missing_model"], query["query_features"])
            doc_scores = {
                doc_id: positive_probability(bundle["doc_model"], query["doc_features"][doc_id])
                for doc_id in candidate_ids
            }
            semantic_scores = {
                doc_id: 0.45 * query["doc_feature_details"][doc_id]["query_doc_cosine"]
                + 0.25 * query["doc_feature_details"][doc_id]["cross_encoder_relevance"]
                + 0.20 * query["doc_feature_details"][doc_id]["bridge_entity_match"]
                + 0.10 * query["doc_feature_details"][doc_id]["novel_information"]
                - 0.15 * query["doc_feature_details"][doc_id]["redundancy"]
                for doc_id in candidate_ids
            }
            ranked_candidates = sorted(candidate_ids, key=lambda doc_id: (0.65 * doc_scores[doc_id] + 0.35 * semantic_scores[doc_id]), reverse=True)
            pair_scores: dict[str, float] = {}
            for left_id, right_id in combinations(candidate_ids, 2):
                key = pair_key(left_id, right_id)
                learned = positive_probability(bundle["pair_model"], query["pair_features"][key]) if bundle["pair_model"] is not None else 0.0
                pair_scores[key] = 0.65 * learned + 0.175 * doc_scores[left_id] + 0.175 * doc_scores[right_id]
            ranked_pairs = sorted(pair_scores, key=pair_scores.get, reverse=True)
            baseline_anchor = {
                doc_id: 0.55 * query["doc_feature_details"][doc_id]["anchor_proxy"]
                + 0.45 * query["doc_feature_details"][doc_id]["query_doc_cosine"]
                for doc_id in baseline_ids
            }
            removable_start = min(2, max(0, len(baseline_ids) - 1))
            removable_positions = sorted(range(removable_start, len(baseline_ids)), key=lambda index: baseline_anchor[baseline_ids[index]])
            if not removable_positions:
                removable_positions = [len(baseline_ids) - 1]
            weakest_tail = removable_positions[0]
            most_redundant = max(removable_positions, key=lambda index: query["doc_feature_details"][baseline_ids[index]]["redundancy"])
            pool: list[dict[str, Any]] = []
            seen = {tuple(baseline_ids)}

            def add(family: str, name: str, context_ids: list[str], added: list[str], removed: list[str], ordering: str, opportunity: float) -> None:
                key = tuple(context_ids)
                if key in seen or len(context_ids) != len(baseline_ids) or len(set(context_ids)) != len(context_ids):
                    return
                seen.add(key)
                removal_risk = max((baseline_anchor[doc_id] for doc_id in removed), default=0.0)
                family_probability = {
                    "single_complementary_insertion": missing.get("missing_bridge", 0.0),
                    "anchor_preserving_replacement": missing.get("missing_answer_resolution", 0.0) + 0.5 * missing.get("missing_bridge", 0.0),
                    "semantic_two_document_chain": missing.get("missing_bridge", 0.0),
                    "redundancy_replacement": missing.get("redundant_context", 0.0),
                    "bridge_first_reorder": missing.get("ordering_problem", 0.0) + 0.5 * missing.get("missing_bridge", 0.0),
                    "answer_anchor_first_reorder": missing.get("ordering_problem", 0.0) + 0.5 * missing.get("missing_answer_resolution", 0.0),
                }[family]
                score = 0.55 * opportunity + 0.30 * family_probability - 0.15 * removal_risk
                pool.append({
                    "family": family,
                    "name": name,
                    "context_ids": context_ids,
                    "added": added,
                    "removed": removed,
                    "ordering": ordering,
                    "generator_score": float(score),
                    "removal_risk": float(removal_risk),
                })

            for rank, doc_id in enumerate(ranked_candidates[:3]):
                context = list(baseline_ids)
                removed = context[weakest_tail]
                context[weakest_tail] = doc_id
                add("single_complementary_insertion", f"semantic_single_{rank + 1}", context, [doc_id], [removed], "stable", doc_scores[doc_id])

            for rank, doc_id in enumerate(ranked_candidates[:3]):
                position = removable_positions[min(rank, len(removable_positions) - 1)]
                context = list(baseline_ids)
                removed = context[position]
                context[position] = doc_id
                add("anchor_preserving_replacement", f"anchor_preserve_{rank + 1}", context, [doc_id], [removed], "anchor_first", 0.7 * doc_scores[doc_id] + 0.3 * semantic_scores[doc_id])

            for rank, key in enumerate(ranked_pairs[:3]):
                left_id, right_id = key.split("|||")
                preserve = max(0, len(baseline_ids) - 2)
                context = baseline_ids[:preserve] + [left_id, right_id]
                add("semantic_two_document_chain", f"semantic_pair_{rank + 1}", context, [left_id, right_id], baseline_ids[preserve:], "anchor_first", pair_scores[key])
                if rank == 0 and len(baseline_ids) >= 5:
                    alternate = baseline_ids[:2] + [left_id, right_id, baseline_ids[2]]
                    add("semantic_two_document_chain", "semantic_pair_bridge_middle", alternate, [left_id, right_id], baseline_ids[3:], "bridge_middle", pair_scores[key] - 0.02)

            if ranked_candidates:
                doc_id = ranked_candidates[0]
                context = list(baseline_ids)
                removed = context[most_redundant]
                context[most_redundant] = doc_id
                add("redundancy_replacement", "replace_semantic_redundancy", context, [doc_id], [removed], "stable", doc_scores[doc_id] + 0.15 * query["doc_feature_details"][removed]["redundancy"])

                if len(baseline_ids) >= 4:
                    bridge_ranked = sorted(ranked_candidates, key=lambda value: query["doc_feature_details"][value]["bridge_entity_match"], reverse=True)
                    bridge_id = bridge_ranked[0]
                    bridge_context = [baseline_ids[0], bridge_id] + baseline_ids[1:-1]
                    add("bridge_first_reorder", "learned_bridge_first", bridge_context, [bridge_id], [baseline_ids[-1]], "bridge_first", doc_scores[bridge_id])

                    anchor_prefix = baseline_ids[: min(3, len(baseline_ids) - 1)]
                    anchor_order = sorted(anchor_prefix, key=baseline_anchor.get, reverse=True)
                    anchor_context = anchor_order + [doc_id] + baseline_ids[len(anchor_prefix):-1]
                    add("answer_anchor_first_reorder", "learned_anchor_order", anchor_context, [doc_id], [baseline_ids[-1]], "answer_anchor_first", doc_scores[doc_id])

            no_intervention = missing.get("no_intervention_needed", 0.0)
            action_budget = min(args.max_actions, 4 if no_intervention >= 0.65 else 6 if no_intervention >= 0.45 else args.max_actions)
            budget_counts[action_budget] += 1
            selected = sorted(pool, key=lambda row: row["generator_score"], reverse=True)[:action_budget]
            for action_index, candidate in enumerate(selected):
                context_ids = candidate["context_ids"]
                is_new = tuple(context_ids) not in v3_contexts.get(query_id, set())
                action = {
                    "query_id": query_id,
                    "question": cache["queries"][query_id].get("question", "") or "",
                    "action_id": f"{query_id}::v4::{action_index:02d}",
                    "outer_fold": fold_id,
                    "action_family": candidate["family"],
                    "action_name": candidate["name"],
                    "context_doc_ids": context_ids,
                    "context_titles": [docs_by_id[doc_id]["title"] for doc_id in context_ids],
                    "context_docs": [docs_by_id[doc_id] for doc_id in context_ids],
                    "added_doc_ids": candidate["added"],
                    "removed_doc_ids": candidate["removed"],
                    "ordering": candidate["ordering"],
                    "generator_score": candidate["generator_score"],
                    "is_new_vs_v3_action_table": is_new,
                    "inference_safe_features": {
                        "missing_hop_probabilities": missing,
                        "added_doc_opportunity_mean": float(np.mean([doc_scores[doc_id] for doc_id in candidate["added"]])) if candidate["added"] else 0.0,
                        "added_doc_semantic_mean": float(np.mean([semantic_scores[doc_id] for doc_id in candidate["added"]])) if candidate["added"] else 0.0,
                        "removal_risk": candidate["removal_risk"],
                    },
                }
                rows.append(action)
                family_counts[action["action_family"]] += 1
            rows.append({
                "query_id": query_id,
                "question": cache["queries"][query_id].get("question", "") or "",
                "action_id": f"{query_id}::v4::fallback",
                "outer_fold": fold_id,
                "action_family": "fallback",
                "action_name": "baseline_fallback",
                "context_doc_ids": baseline_ids,
                "context_titles": [docs_by_id[doc_id]["title"] for doc_id in baseline_ids],
                "context_docs": [docs_by_id[doc_id] for doc_id in baseline_ids],
                "added_doc_ids": [],
                "removed_doc_ids": [],
                "ordering": "stable",
                "generator_score": 0.0,
                "is_new_vs_v3_action_table": False,
                "inference_safe_features": {"missing_hop_probabilities": missing, "removal_risk": 0.0},
            })
            selected_query_ids.append(query_id)
        audit_folds.append({
            "fold_id": fold_id,
            "train_query_fingerprint": bundle["metadata"]["train_query_fingerprint"],
            "test_query_fingerprint": query_fingerprint(test_ids),
            "n_test_generated": len(test_ids),
            "model_sha256": fold_record["model_sha256"],
        })

    output_path = OUTPUTS / "generated_actions/v4_outer_test_actions.jsonl"
    write_jsonl(output_path, rows)
    effective = [row for row in rows if row["action_family"] != "fallback"]
    generated_text = output_path.read_text(encoding="utf-8")
    found_forbidden = []
    for line in generated_text.splitlines():
        keys = set(json.loads(line))
        found_forbidden.extend(sorted(keys & FORBIDDEN_OUTPUT_KEYS))
    audit = {
        "status": "pass" if not found_forbidden else "fail",
        "protocol": "outer-fold frozen semantic action generation",
        "num_queries": len(selected_query_ids),
        "num_effective_actions": len(effective),
        "num_new_actions_vs_v3_table": sum(int(row["is_new_vs_v3_action_table"]) for row in effective),
        "max_actions_per_query": args.max_actions,
        "family_counts": dict(family_counts),
        "action_budget_counts": {str(key): value for key, value in sorted(budget_counts.items())},
        "target_query_gold_support_used": False,
        "target_query_answer_used": False,
        "target_query_reader_outcome_used": False,
        "target_query_oracle_action_used": False,
        "post_hoc_test_coverage_used": False,
        "forbidden_output_key_hits": sorted(set(found_forbidden)),
        "folds": audit_folds,
        "output_sha256": sha256(output_path),
    }
    write_json(OUTPUTS / "audits/generator_nested_no_leak_audit.json", audit)
    write_json(OUTPUTS / "generated_actions/v4_generation_summary.json", audit)
    if found_forbidden:
        raise AssertionError(f"Generator no-leak audit failed: {sorted(set(found_forbidden))}")

    report_path = REPORTS / "semantic_generator_report.md"
    prior = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Semantic Generator Report\n"
    prior += f"""

## Frozen Outer-Test Generation

- Queries: **{len(selected_query_ids)}**.
- Effective actions: **{len(effective)}**.
- Actions not present in the v3 table: **{audit['num_new_actions_vs_v3_table']}**.
- Per-query action budget: learned no-intervention gate with a hard maximum of **{args.max_actions}**.
- No-leak audit: **{audit['status']}**.

The generator ranks full per-query distractor documents semantically, estimates pair complementarity, and then materializes a bounded set of six reader-compatible action types. V3 action membership is recorded only after generation for marginal-efficiency accounting and is not used in action scoring.
"""
    report_path.write_text(prior, encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ["status", "num_queries", "num_effective_actions", "num_new_actions_vs_v3_table", "family_counts"]}, indent=2))


if __name__ == "__main__":
    main()
