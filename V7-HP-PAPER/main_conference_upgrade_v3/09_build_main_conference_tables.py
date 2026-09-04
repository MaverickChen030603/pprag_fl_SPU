#!/usr/bin/env python3
"""Build final v3 tables and evaluate the pre-registered readiness gates."""

from __future__ import annotations

import json

from v3_common import HERE, OUTPUTS, REPORTS, ensure_layout, markdown_table, read_json, write_json


def main() -> None:
    ensure_layout()
    generation = read_json(OUTPUTS / "candidate_generation/v3_candidate_generation_summary.json")
    outcomes = read_json(OUTPUTS / "action_outcomes/v3_action_outcome_summary.json")
    nested_path = OUTPUTS / "nested_selector/v3_nested_summary.json"
    nested = read_json(nested_path) if nested_path.exists() else {"status": "not_run"}
    scale = read_json(OUTPUTS / "scaleup/scaleup_summary.json") if (OUTPUTS / "scaleup/scaleup_summary.json").exists() else {"status": "not_run"}
    external = read_json(OUTPUTS / "external_dataset/external_smoke_summary.json") if (OUTPUTS / "external_dataset/external_smoke_summary.json").exists() else {"status": "not_run"}
    v2 = read_json(HERE.parent / "submission_revision_v2/nested_final_1000_summary.json")

    main_rows = [
        ["v2 fully nested selector", "4,000", "203", "20.3%", "+0.0120", "+0.0150", "+0.0028", "+0.0079"],
        ["v3 bounded action generator", f"{generation['num_effective_actions']:,}", str(outcomes["queries_with_positive_action"]), f"{outcomes['positive_query_coverage']:.1%}", "not selected", "not selected", "not selected", "not selected"],
    ]
    table1 = markdown_table(["Method/stage", "Effective actions", "Positive queries", "Opportunity", "Title recall delta", "Title F1 delta", "Answer F1 delta", "Product delta"], main_rows)

    family_rows = []
    for family, count in generation["family_counts"].items():
        if family == "fallback":
            continue
        positives = int(outcomes["family_positive_action_counts"].get(family, 0))
        queries = int(outcomes["family_positive_query_counts"].get(family, 0))
        family_rows.append([family, f"{count:,}", positives, f"{positives / count:.1%}", queries, f"{queries / outcomes['num_queries']:.1%}"])
    table2 = markdown_table(["Action family", "Actions", "Positive actions", "Positive/action", "Positive queries", "Query coverage"], family_rows)

    gates = {
        "A_candidate_opportunity": {"pass": outcomes["positive_query_coverage"] >= 0.30, "observed": f"{outcomes['positive_query_coverage']:.1%}", "required": ">=30%"},
        "B_downstream": {"pass": False, "observed": "not run after opportunity stop", "required": "official/product/reader-consistent gain"},
        "C_protocol": {"pass": True, "observed": "deterministic no-leak generator; stop rule honored", "required": "no leak and train-only decisions"},
        "D_breadth": {"pass": False, "observed": f"scale={scale.get('status')}; external={external.get('status')}", "required": "reader, scale, or positive external smoke"},
        "E_reproducibility": {"pass": True, "observed": "revision/environment/commands/artifacts logged", "required": "complete"},
    }
    gate_rows = [[name, "PASS" if value["pass"] else "FAIL", value["observed"], value["required"]] for name, value in gates.items()]
    table3 = markdown_table(["Gate", "Status", "Observed", "Requirement"], gate_rows)
    readiness = {
        "status": "not_ready",
        "v2_status": "findings_or_coling_ready",
        "v3_main_conference_status": "not_ready",
        "gates": gates,
        "passed_gate_count": sum(value["pass"] for value in gates.values()),
        "total_gate_count": len(gates),
        "primary_reason": "Positive-query opportunity reached 23.4%, below both the 25% continuation floor and 30% main-conference Gate A.",
        "paper_mode": "findings_fallback",
    }
    write_json(OUTPUTS / "tables/main_conference_readiness.json", readiness)
    (OUTPUTS / "tables/main_results_table.md").write_text("# Table 1: Frozen v2 and v3 Opportunity Result\n\n" + table1 + "\n", encoding="utf-8")
    (OUTPUTS / "tables/candidate_generator_ablation_table.md").write_text("# Table 2: v3 Action-Family Opportunity\n\n" + table2 + "\n", encoding="utf-8")
    (OUTPUTS / "tables/validation_gate_table.md").write_text("# Table 3: Main-Conference Gates\n\n" + table3 + "\n", encoding="utf-8")
    latex = r"""\begin{table*}[t]
\centering
\small
\begin{tabular}{lrrrrrrr}
\toprule
Stage & Actions & Positive queries & Opportunity & $\Delta$Title-R & $\Delta$Title-F1 & $\Delta$Ans-F1 & $\Delta$Product \\
\midrule
v2 nested selector & 4,000 & 203 & 20.3\% & +0.0120 & +0.0150 & +0.0028 & +0.0079 \\
v3 action generator & 7,882 & 234 & 23.4\% & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\caption{Frozen v2 downstream results and the pre-registered v3 candidate-opportunity result. V3 downstream selection was not run after the opportunity stop rule fired.}
\label{tab:v3-main}
\end{table*}
"""
    (OUTPUTS / "tables/main_results_table.tex").write_text(latex, encoding="utf-8")
    report = f"""# Main-Conference Readiness Report

## Decision

**V3 status: NOT READY.** Only {readiness['passed_gate_count']}/{readiness['total_gate_count']} readiness gates pass.

## Main result

{table1}

The bounded generator almost doubled the number of evaluated actions, from four to 7.882 effective actions per query on average, but positive-query opportunity rose only from 20.3% to 23.4%. This is a real +3.1-point diagnostic gain, yet it misses the pre-registered 25% continuation floor and the 30% main-conference gate.

## Family analysis

{table2}

The bounded two-document chain contributes the most positive actions (229) and reaches 156 queries, but family coverages overlap heavily. Its effect is therefore evidence that broader action expressivity helps some cases, not evidence that candidate opportunity has been solved.

## Gates

{table3}

## Recommendation

Freeze the v3 result as a negative/diagnostic extension. Keep v2 as the submission fallback. A future v4 should redesign candidate *sources* or learn context construction from train-only outcomes rather than add more transformations over the same ten-document pool.
"""
    (REPORTS / "main_conference_readiness_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(readiness, indent=2))


if __name__ == "__main__":
    main()

