#!/usr/bin/env python3
"""Apply the frozen v4 opportunity gates without post-hoc threshold changes."""

from __future__ import annotations

import json

from v4_common import OUTPUTS, REPORTS, ensure_layout, markdown_table, read_json, write_json


def main() -> None:
    ensure_layout()
    metrics = read_json(OUTPUTS / "action_outcomes/v4_action_summary.json")
    registration = read_json(OUTPUTS / "opportunity/metric_preregistration.json")
    thresholds = registration["v4_gates"]
    v3_efficiency = registration["frozen_v3_reference"]["new_query_efficiency"]
    gates = {
        "A_overall_opportunity": metrics["overall_positive_query_coverage"] >= thresholds["overall_positive_query_coverage_min"],
        "B_conditional_opportunity": metrics["non_ceiling_positive_query_coverage"] >= thresholds["non_ceiling_positive_query_coverage_min"],
        "C_marginal_breadth": (
            metrics["new_queries_covered_beyond_v3"] >= thresholds["additional_queries_vs_v3_min"]
            or metrics["overall_positive_query_coverage"] - registration["frozen_v3_reference"]["overall_positive_query_coverage"] >= thresholds["absolute_coverage_gain_vs_v3_alternative"]
        ),
        "D_action_quality_density": metrics["positive_action_density"] >= thresholds["positive_action_density_min"],
        "E_new_query_efficiency": metrics["new_query_efficiency"] >= thresholds["new_query_efficiency_multiplier_vs_v3_min"] * v3_efficiency,
    }
    passed = sum(gates.values())
    failed = len(gates) - passed
    strong_pass = passed >= int(thresholds["strong_pass_gate_count"])
    mandatory_stop = failed >= int(thresholds["mandatory_stop_failed_gate_count"])
    proceed = not mandatory_stop
    decision = {
        "status": "pass" if strong_pass else "borderline_continue" if proceed else "stop",
        "gates": gates,
        "passed_gate_count": passed,
        "failed_gate_count": failed,
        "strong_pass_gate_count": thresholds["strong_pass_gate_count"],
        "opportunity_strong_pass": strong_pass,
        "mandatory_stop_triggered": mandatory_stop,
        "proceed_to_nested_selector": proceed,
        "v4_main_conference_status": "opportunity_pass" if strong_pass else "opportunity_borderline_continue" if proceed else "not_ready",
        "thresholds": thresholds,
        "observed": {
            "overall_positive_query_coverage": metrics["overall_positive_query_coverage"],
            "non_ceiling_positive_query_coverage": metrics["non_ceiling_positive_query_coverage"],
            "new_queries_covered_beyond_v3": metrics["new_queries_covered_beyond_v3"],
            "coverage_delta_vs_v3": metrics["overall_positive_query_coverage"] - registration["frozen_v3_reference"]["overall_positive_query_coverage"],
            "positive_action_density": metrics["positive_action_density"],
            "new_query_efficiency": metrics["new_query_efficiency"],
            "v3_new_query_efficiency": v3_efficiency,
        },
        "stop_rule": "A strong pass requires four of five gates; the supplied mandatory stop triggers when at least four of five gates fail.",
    }
    write_json(OUTPUTS / "opportunity/v4_opportunity_gate.json", decision)
    rows = [[name, "PASS" if value else "FAIL"] for name, value in gates.items()]
    report = f"""# V4 Opportunity Gate Report

{markdown_table(["Gate", "Result"], rows)}

- Passed: **{passed}/5**; strong-pass requirement: **{thresholds['strong_pass_gate_count']}/5**.
- Overall coverage: **{metrics['overall_positive_query_coverage']:.1%}**.
- Conditional non-ceiling coverage: **{metrics['non_ceiling_positive_query_coverage']:.1%}**.
- Newly covered queries beyond v3: **{metrics['new_queries_covered_beyond_v3']}**.
- Positive-action density: **{metrics['positive_action_density']:.2%}**.
- New-query efficiency: **{metrics['new_query_efficiency']:.4f}**, versus frozen v3 **{v3_efficiency:.4f}**.

Decision: **{'strong pass; continue to fully nested selector' if strong_pass else 'borderline continuation under the supplied mandatory-stop rule' if proceed else 'stop before selector training'}**. No individual threshold was changed after outcome inspection.
"""
    (REPORTS / "opportunity_gate_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
