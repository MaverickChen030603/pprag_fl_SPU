#!/usr/bin/env python3
"""Freeze the v4 opportunity metric definitions before outcome inspection."""

from __future__ import annotations

from v4_common import OUTPUTS, REPORTS, ensure_layout, read_json, write_json


def main() -> None:
    ensure_layout()
    v3 = read_json(OUTPUTS / "audits/v3_ceiling_aware_opportunity.json")
    v3_new_queries_vs_v2 = read_json(OUTPUTS / "audits/v3_family_marginal_coverage.json")["v3_new_positive_queries_vs_v2"]
    definitions = {
        "status": "pre_registered",
        "metric_scope": "outer-test actions only",
        "definitions": {
            "overall_positive_query_coverage": "queries with >=1 positive action / all queries",
            "non_ceiling_positive_query_coverage": "non-ceiling queries with >=1 positive action / all non-ceiling queries",
            "positive_action_density": "positive actions / effective generated actions",
            "marginal_new_query_coverage": "queries newly covered beyond frozen comparator / all queries",
            "answer_safe_action_rate": "actions with answer_f1_delta >= 0 / effective generated actions",
            "positive_actions_per_covered_query": "positive actions / queries with >=1 positive action",
            "family_diversity_per_query": "mean number of positive action families over covered queries",
            "new_query_efficiency": "previously uncovered queries gaining a positive action / effective newly generated actions",
        },
        "frozen_v3_reference": {
            "overall_positive_query_coverage": v3["v3"]["overall_positive_query_coverage"],
            "non_ceiling_positive_query_coverage": v3["v3"]["non_ceiling_positive_query_coverage"],
            "net_coverage_gain_queries_vs_v2": 31,
            "new_queries_vs_v2": v3_new_queries_vs_v2,
            "new_actions_vs_v2": 3882,
            "new_query_efficiency": v3_new_queries_vs_v2 / 3882,
        },
        "v4_gates": {
            "overall_positive_query_coverage_min": 0.30,
            "non_ceiling_positive_query_coverage_min": 0.45,
            "additional_queries_vs_v3_min": 70,
            "absolute_coverage_gain_vs_v3_alternative": 0.07,
            "positive_action_density_min": 0.12,
            "new_query_efficiency_multiplier_vs_v3_min": 1.25,
            "strong_pass_gate_count": 4,
            "mandatory_stop_failed_gate_count": 4,
            "total_gate_count": 5,
        },
    }
    write_json(OUTPUTS / "opportunity/metric_preregistration.json", definitions)
    report = f"""# Ceiling-Aware Opportunity Metrics

V4 freezes seven opportunity metrics before outer-test reader outcomes are examined. The headline quantities are overall coverage, conditional coverage among non-ceiling queries, positive-action density, and new-query efficiency.

The frozen v3 reference is **{v3['v3']['overall_positive_query_coverage']:.1%}** overall and **{v3['v3']['non_ceiling_positive_query_coverage']:.1%}** among {v3['n_non_ceiling_queries']} non-ceiling queries. V3 has a net gain of 31 covered queries, but set-level comparison finds {v3_new_queries_vs_v2} newly covered v2-negative queries. Its exact new-query efficiency is therefore **{v3_new_queries_vs_v2 / 3882:.4f}** queries per added action. V4 must improve breadth, not merely increase row count.

The pre-registered v4 gates remain: 30% overall coverage, 45% conditional coverage, at least 70 new queries or +7 points over v3, 12% positive-action density, and at least 1.25x v3 new-query efficiency. Four passes constitute a strong opportunity result. Following the supplied stop rule literally, selector training is mandatory-stopped only when at least four of five gates fail; a weaker continuation is explicitly labeled borderline.
"""
    (REPORTS / "ceiling_aware_opportunity_report.md").write_text(report, encoding="utf-8")
    print(OUTPUTS / "opportunity/metric_preregistration.json")


if __name__ == "__main__":
    main()
