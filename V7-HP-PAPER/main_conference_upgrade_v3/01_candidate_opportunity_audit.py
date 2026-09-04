#!/usr/bin/env python3
"""Diagnostic-only taxonomy of v2 queries with no main-eligible positive action."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from v3_common import OUTPUTS, REPORTS, ensure_layout, jaccard, load_source_examples, markdown_table, normalize_title, read_jsonl, tokens, v2_action_labels_path, write_json


def fnum(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def is_positive(row: dict[str, Any]) -> bool:
    return fnum(row, "answer_f1_delta") >= 0 and fnum(row, "joint_f1_delta") > 0 and (fnum(row, "support_recall_delta") > 0 or fnum(row, "sp_f1_delta") >= 0)


def main() -> None:
    ensure_layout()
    source = load_source_examples()
    rows = read_jsonl(v2_action_labels_path())
    main_rows = [row for row in rows if str(row.get("candidate_family", "")) != "insert2"]
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in main_rows:
        by_query[str(row["query_id"])].append(row)
    no_positive = {qid: values for qid, values in by_query.items() if not any(is_positive(row) for row in values)}
    if len(no_positive) != 797:
        raise AssertionError(f"Frozen v2 expected 797 no-positive queries, found {len(no_positive)}")

    taxonomy_rows: list[dict[str, Any]] = []
    primary_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for query_id, actions in sorted(no_positive.items()):
        item = source[query_id]
        source_titles = {normalize_title(doc["title"]) for doc in item["docs"]}
        gold_titles = {normalize_title(title) for title in item.get("supporting_titles", [])}
        baseline_titles = [str(value) for value in actions[0].get("baseline_titles", [])]
        baseline_norm = {normalize_title(value) for value in baseline_titles}
        exposed_titles = set(baseline_norm)
        for action in actions:
            exposed_titles.update(normalize_title(value) for value in action.get("candidate_titles", []))
        missing_source = sorted(gold_titles - source_titles)
        missing_exposed = sorted(gold_titles - exposed_titles)
        missing_baseline = sorted(gold_titles - baseline_norm)
        support_gain_rows = [row for row in actions if fnum(row, "support_recall_delta") > 0 or fnum(row, "sp_f1_delta") > 0]
        harmful_gain_rows = [row for row in support_gain_rows if fnum(row, "answer_f1_delta") < 0]
        answer_safe_neutral = [row for row in actions if fnum(row, "answer_f1_delta") >= 0 and fnum(row, "joint_f1_delta") <= 0 and fnum(row, "support_recall_delta") <= 0]
        order_rows = [
            row for row in actions
            if {normalize_title(value) for value in row.get("candidate_titles", [])} == baseline_norm
            and [normalize_title(value) for value in row.get("candidate_titles", [])] != [normalize_title(value) for value in baseline_titles]
            and (abs(fnum(row, "answer_f1_delta")) > 1e-12 or abs(fnum(row, "joint_f1_delta")) > 1e-12)
        ]
        baseline_docs = [doc for doc in item["docs"] if normalize_title(doc["title"]) in baseline_norm]
        redundancy = any(
            jaccard(tokens(f"{left['title']} {left['text']}"), tokens(f"{right['title']} {right['text']}")) >= 0.65
            for index, left in enumerate(baseline_docs)
            for right in baseline_docs[index + 1:]
        ) or len(baseline_norm) < len(baseline_titles)
        strong_baseline = fnum(actions[0], "baseline_answer_f1") >= 1.0 - 1e-12 and fnum(actions[0], "baseline_support_recall") >= 1.0 - 1e-12

        labels: list[str] = []
        if missing_source or missing_exposed:
            labels.append("candidate_pool_lacks_gold_support_title")
        if len(missing_baseline) >= 2:
            labels.append("single_document_action_insufficient")
        if missing_baseline and not missing_source and not harmful_gain_rows and not strong_baseline:
            labels.append("candidate_exists_but_wrong_insertion_slot")
        if harmful_gain_rows:
            labels.extend(["candidate_exists_but_displaces_answer_anchor", "evidence_positive_but_answer_harmful"])
        if order_rows:
            labels.append("document_order_failure")
        if redundancy:
            labels.append("redundancy_or_duplicate_occupation")
        if strong_baseline:
            labels.append("strong_baseline_with_no_improvable_context")
        if answer_safe_neutral:
            labels.append("answer_safe_but_evidence_neutral")
        labels.append("reader_prediction_variance_unmeasured")

        hierarchy = [
            "candidate_pool_lacks_gold_support_title",
            "strong_baseline_with_no_improvable_context",
            "single_document_action_insufficient",
            "candidate_exists_but_displaces_answer_anchor",
            "document_order_failure",
            "redundancy_or_duplicate_occupation",
            "candidate_exists_but_wrong_insertion_slot",
            "answer_safe_but_evidence_neutral",
            "reader_prediction_variance_unmeasured",
        ]
        primary = next(label for label in hierarchy if label in labels)
        labels = list(dict.fromkeys(labels))
        primary_counts[primary] += 1
        label_counts.update(labels)
        taxonomy_rows.append({
            "query_id": query_id,
            "diagnostic_only": True,
            "primary_category": primary,
            "categories": labels,
            "baseline_titles": baseline_titles,
            "gold_support_titles": sorted(gold_titles),
            "missing_from_source_pool": missing_source,
            "missing_from_v2_exposed_pool": missing_exposed,
            "missing_from_baseline": missing_baseline,
            "support_gain_action_count": len(support_gain_rows),
            "evidence_positive_answer_harmful_count": len(harmful_gain_rows),
            "answer_safe_evidence_neutral_count": len(answer_safe_neutral),
            "order_sensitive_action_count": len(order_rows),
            "baseline_answer_f1": fnum(actions[0], "baseline_answer_f1"),
            "baseline_title_recall": fnum(actions[0], "baseline_support_recall"),
        })

    summary = {
        "diagnostic_only": True,
        "target_query_outcomes_used": True,
        "features_reusable_at_inference": False,
        "n_no_positive_queries": len(taxonomy_rows),
        "primary_category_counts": dict(primary_counts),
        "overlapping_category_counts": dict(label_counts),
        "answers": {
            "source_candidate_pool_lacks_gold_support": sum(bool(row["missing_from_source_pool"]) for row in taxonomy_rows),
            "v2_exposed_pool_lacks_gold_support": sum(bool(row["missing_from_v2_exposed_pool"]) for row in taxonomy_rows),
            "suitable_document_but_current_template_not_safe": sum("candidate_exists_but_wrong_insertion_slot" in row["categories"] or "candidate_exists_but_displaces_answer_anchor" in row["categories"] for row in taxonomy_rows),
            "needs_two_document_action": sum("single_document_action_insufficient" in row["categories"] for row in taxonomy_rows),
            "answer_anchor_displacement": sum("candidate_exists_but_displaces_answer_anchor" in row["categories"] for row in taxonomy_rows),
            "strong_baseline_no_improvement": sum("strong_baseline_with_no_improvable_context" in row["categories"] for row in taxonomy_rows),
        },
        "recommended_action_families": [
            "anchor_preserving_tail_replacement",
            "bridge_aware_complementary_insertion",
            "bounded_two_document_chain",
        ],
    }
    write_json(OUTPUTS / "audits/no_positive_query_taxonomy.json", {"summary": summary, "queries": taxonomy_rows})
    table_rows = [[name, count, f"{count / len(taxonomy_rows):.1%}"] for name, count in primary_counts.most_common()]
    table = markdown_table(["Primary diagnostic category", "Queries", "Share of 797"], table_rows)
    (OUTPUTS / "tables/no_positive_query_taxonomy.md").write_text("# No-Positive Query Taxonomy\n\n" + table + "\n", encoding="utf-8")
    answers = summary["answers"]
    report = f"""# Candidate Opportunity Report

## Scope and leakage boundary

This is a **diagnostic-only** analysis of the frozen 797 no-positive queries. It uses gold support and reader outcomes to explain failures. None of its labels or target-derived values is exported to the v3 candidate generator or held-out selector features.

## Primary taxonomy

{table}

## Required answers

1. Full source pool lacks a gold-support title: **{answers['source_candidate_pool_lacks_gold_support']} / 797**.
2. The older exposed action pool lacks a gold-support title: **{answers['v2_exposed_pool_lacks_gold_support']} / 797**.
3. A suitable source document exists but the current templates do not place it safely: **{answers['suitable_document_but_current_template_not_safe']} / 797**.
4. Baseline misses both support titles and therefore needs a bounded two-document action: **{answers['needs_two_document_action']} / 797**.
5. Evidence-positive edits displace an answer anchor: **{answers['answer_anchor_displacement']} / 797**.
6. Baseline already has perfect answer F1 and title recall: **{answers['strong_baseline_no_improvement']} / 797**.

## Generator priorities

The three most defensible families are anchor-preserving tail replacement, bridge-aware complementary insertion, and a newly bounded two-document chain. They directly target safe placement, cross-document complementarity, and the single-action expressivity limit while keeping the search space finite.
"""
    (REPORTS / "candidate_opportunity_report.md").write_text(report, encoding="utf-8")
    print(OUTPUTS / "audits/no_positive_query_taxonomy.json")


if __name__ == "__main__":
    main()

