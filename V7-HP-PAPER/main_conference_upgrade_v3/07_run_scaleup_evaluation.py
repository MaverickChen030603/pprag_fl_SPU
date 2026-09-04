#!/usr/bin/env python3
"""Apply the frozen scale-up gate before any 3k/full HotpotQA reader run."""

from __future__ import annotations

import json

from v3_common import OUTPUTS, REPORTS, ensure_layout, read_json, write_json


def main() -> None:
    ensure_layout()
    opportunity = read_json(OUTPUTS / "action_outcomes/v3_action_outcome_summary.json")
    coverage = float(opportunity["positive_query_coverage"])
    if coverage < 0.25:
        summary = {
            "status": "skipped_by_pre_registered_opportunity_gate",
            "positive_query_coverage": coverage,
            "required_to_continue": 0.25,
            "planned_scale": "full HotpotQA distractor validation (7405) or at least 3000",
            "protocol_frozen_before_scaleup": False,
            "reason": "The candidate generator did not expose enough positive-query opportunity to justify expensive scale-up.",
        }
        significance = {"status": "not_run", "reason": summary["reason"]}
    else:
        nested = read_json(OUTPUTS / "nested_selector/v3_nested_summary.json")
        summary = {
            "status": "ready_for_frozen_scaleup_execution",
            "positive_query_coverage": coverage,
            "planned_scale": "full HotpotQA distractor validation (7405)",
            "frozen_protocol": {
                "candidate_generator": "02_generate_reader_compatible_actions.py",
                "selector_architecture": "04_train_nested_selector_v3.py",
                "coverage": nested.get("coverage"),
                "reader": "google/flan-t5-large@0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
                "metrics": ["answer_f1", "official_sp_f1", "official_joint_f1"],
            },
            "note": "Execution is permitted only with the frozen manifest above; no scaleup-set tuning is allowed.",
        }
        significance = {"status": "pending_scaleup_execution"}
    write_json(OUTPUTS / "scaleup/scaleup_summary.json", summary)
    write_json(OUTPUTS / "scaleup/scaleup_significance.json", significance)
    report = f"""# Scale-Up Report

Status: **{summary['status']}**

- Positive-query opportunity: {coverage:.1%}
- Pre-registered continuation floor: 25.0%
- Intended scale: full HotpotQA distractor validation or at least 3,000 queries

{summary.get('reason', summary.get('note', ''))}

No larger-set result is claimed unless the complete frozen protocol is executed.
"""
    (REPORTS / "scaleup_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

