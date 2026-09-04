#!/usr/bin/env python3
"""Reinterpret frozen v3 outcomes with ceiling-aware and marginal coverage metrics."""

from __future__ import annotations

from collections import defaultdict

from v4_common import OUTPUTS, REPORTS, ensure_layout, grouped_outcomes, load_v3_merged_rows, markdown_table, v2_main_positive_query_ids, write_json


def main() -> None:
    ensure_layout()
    grouped = grouped_outcomes(load_v3_merged_rows())
    all_query_ids = set(grouped)
    v2_positive = v2_main_positive_query_ids()
    if len(v2_positive) != 203:
        raise AssertionError(f"Frozen main-eligible v2 positive-query count changed: {len(v2_positive)}")

    baseline_by_query = {}
    family_positive: dict[str, set[str]] = defaultdict(set)
    for query_id, rows in grouped.items():
        baseline = next(row for row in rows if row["action_family"] == "fallback")
        baseline_by_query[query_id] = baseline
        for row in rows:
            if row["action_family"] != "fallback" and bool(row.get("positive_action")):
                family_positive[str(row["action_family"])].add(query_id)

    v3_positive = set().union(*family_positive.values())
    ceiling = {
        query_id for query_id, row in baseline_by_query.items()
        if float(row["answer_f1"]) >= 1.0 - 1e-12 and float(row["title_recall"]) >= 1.0 - 1e-12
    }
    non_ceiling = all_query_ids - ceiling
    v3_non_ceiling_positive = v3_positive & non_ceiling
    v2_non_ceiling_positive = v2_positive & non_ceiling

    ceiling_payload = {
        "status": "complete_from_frozen_v3_outputs",
        "n_queries": len(all_query_ids),
        "ceiling_definition": "baseline_answer_f1 == 1 and baseline_title_recall == 1",
        "n_ceiling_queries": len(ceiling),
        "n_non_ceiling_queries": len(non_ceiling),
        "v2": {
            "queries_with_positive_action": len(v2_positive),
            "overall_positive_query_coverage": len(v2_positive) / len(all_query_ids),
            "non_ceiling_positive_queries": len(v2_non_ceiling_positive),
            "non_ceiling_positive_query_coverage": len(v2_non_ceiling_positive) / len(non_ceiling),
        },
        "v3": {
            "queries_with_positive_action": len(v3_positive),
            "overall_positive_query_coverage": len(v3_positive) / len(all_query_ids),
            "non_ceiling_positive_queries": len(v3_non_ceiling_positive),
            "non_ceiling_positive_query_coverage": len(v3_non_ceiling_positive) / len(non_ceiling),
            "positive_ceiling_queries": len(v3_positive & ceiling),
        },
        "interpretation_guard": "Ceiling-aware coverage is retrospective analysis and does not overturn the pre-registered v3 stop decision.",
        "ceiling_query_ids": sorted(ceiling),
        "non_ceiling_query_ids": sorted(non_ceiling),
    }
    write_json(OUTPUTS / "audits/v3_ceiling_aware_opportunity.json", ceiling_payload)

    all_v3_positive = set(v3_positive)
    marginal = {}
    for family, positive_queries in sorted(family_positive.items()):
        without_family = set().union(*(values for name, values in family_positive.items() if name != family))
        new_vs_v2 = positive_queries - v2_positive
        marginal[family] = {
            "positive_actions": sum(
                1 for rows in grouped.values() for row in rows
                if row["action_family"] == family and bool(row.get("positive_action"))
            ),
            "unique_positive_queries": len(positive_queries),
            "positive_queries_already_covered_by_v2": len(positive_queries & v2_positive),
            "new_positive_queries_not_covered_by_v2": len(new_vs_v2),
            "incremental_query_coverage_after_adding_family_to_v2": len((v2_positive | positive_queries) - v2_positive) / len(all_query_ids),
            "coverage_after_adding_family_to_v2": len(v2_positive | positive_queries) / len(all_query_ids),
            "leave_one_family_out_unique_query_loss": len(all_v3_positive - without_family),
            "positive_query_ids": sorted(positive_queries),
            "new_query_ids_vs_v2": sorted(new_vs_v2),
        }

    families = sorted(family_positive)
    overlap_counts = {
        left: {right: len(family_positive[left] & family_positive[right]) for right in families}
        for left in families
    }
    marginal_payload = {
        "status": "complete_from_frozen_v3_outputs",
        "v2_positive_queries": len(v2_positive),
        "v3_positive_queries": len(v3_positive),
        "v3_new_positive_queries_vs_v2": len(v3_positive - v2_positive),
        "families": marginal,
        "pairwise_positive_query_overlap_counts": overlap_counts,
    }
    write_json(OUTPUTS / "audits/v3_family_marginal_coverage.json", marginal_payload)

    table_rows = [[family] + [overlap_counts[family][other] for other in families] for family in families]
    (OUTPUTS / "tables/v3_family_overlap_matrix.md").write_text(
        "# V3 Family Positive-Query Overlap\n\n" + markdown_table(["Family"] + families, table_rows) + "\n",
        encoding="utf-8",
    )

    family_rows = []
    for family in families:
        value = marginal[family]
        family_rows.append([
            family,
            value["positive_actions"],
            value["unique_positive_queries"],
            value["positive_queries_already_covered_by_v2"],
            value["new_positive_queries_not_covered_by_v2"],
            value["leave_one_family_out_unique_query_loss"],
        ])
    report = f"""# V3 Opportunity Reanalysis

## Frozen Result

- Overall positive-query coverage: **{len(v3_positive)}/{len(all_query_ids)} = {len(v3_positive) / len(all_query_ids):.1%}**.
- V2 main-eligible coverage: **{len(v2_positive)}/{len(all_query_ids)} = {len(v2_positive) / len(all_query_ids):.1%}**.
- Baseline ceiling queries: **{len(ceiling)}**; non-ceiling queries: **{len(non_ceiling)}**.
- Conditional v3 opportunity among non-ceiling queries: **{len(v3_non_ceiling_positive)}/{len(non_ceiling)} = {len(v3_non_ceiling_positive) / len(non_ceiling):.1%}**.
- This conditional value is retrospective and does **not** reverse the frozen v3 stop decision.

## Marginal Family Coverage

{markdown_table(["Family", "Positive actions", "Unique queries", "Already v2", "New vs v2", "Leave-one-out loss"], family_rows)}

Raw positive-action count is not a sufficient family contribution measure. The table distinguishes overlap from genuinely new query opportunity; the pairwise matrix is in `outputs/tables/v3_family_overlap_matrix.md`.

## Interpretation

V3 nearly doubled the action table. Its **net** coverage gain was only **{len(v3_positive) - len(v2_positive)} queries** (+3.1 points). Set-level comparison finds **{len(v3_positive - v2_positive)} newly covered** v2-negative queries and **{len(v2_positive - v3_positive)} v2-covered queries not recovered by v3**, so raw coverage delta and marginal new-query coverage must be reported separately. Positive actions remain concentrated on overlapping, already-improvable queries. The next bottleneck is semantic candidate construction and query-specific opportunity creation, not selector tuning or another fixed-template expansion.
"""
    (REPORTS / "v3_reanalysis_report.md").write_text(report, encoding="utf-8")
    print(REPORTS / "v3_reanalysis_report.md")


if __name__ == "__main__":
    main()
