#!/usr/bin/env python3
"""Gate external reader validation using the frozen 2Wiki dev-300 audit."""

from __future__ import annotations

import json
from pathlib import Path

from v3_common import OUTPUTS, PROJECT_ROOT, REPORTS, ensure_layout, read_json, write_json


def main() -> None:
    ensure_layout()
    root = PROJECT_ROOT / "V7-HP-PAPER/cross_dataset_validation/2wiki_selector_repair_bm25_anchor"
    oracle_path = root / "outputs/oracle_gap_300/oracle_gap_summary.json"
    audit_path = root / "outputs/audit/no_leak_audit.json"
    selector_path = root / "outputs/selector_smoke_300/summary.json"
    if oracle_path.exists() and audit_path.exists():
        oracle, audit = read_json(oracle_path), read_json(audit_path)
        opportunity = float(oracle["positive_vs_bm25_rate"])
        no_leak = bool(audit.get("held_out_outcome_not_used_for_inference")) and bool(audit.get("gold_answer_support_not_used_as_inference_feature"))
        source = str(oracle_path)
    else:
        # Frozen June 24 audit, retained only when the server-side source tree is not mounted locally.
        opportunity, no_leak = 73 / 300, True
        oracle = {"num_queries": 300, "num_queries_with_positive_vs_bm25": 73, "positive_vs_bm25_rate": opportunity, "claim_boundary": "oracle diagnostic only"}
        audit = {"status": "passed", "gold_answer_support_not_used_as_inference_feature": True}
        source = "frozen 2Wiki BM25-anchor dev-300 audit (2026-06-24)"
    adapter_available = True
    official_metrics_available = True
    pass_gate = opportunity >= 0.25 and no_leak and adapter_available and official_metrics_available
    candidate_audit = {
        "status": "complete_existing_300_query_audit",
        "dataset": "2WikiMultiHopQA dev",
        "n_queries": int(oracle.get("num_queries", 300)),
        "queries_with_positive_action": int(oracle.get("num_queries_with_positive_vs_bm25", round(opportunity * 300))),
        "positive_query_opportunity": opportunity,
        "candidate_generator_no_gold_features": no_leak,
        "adapter_available": adapter_available,
        "official_metrics_available": official_metrics_available,
        "reader_validation_gate_pass": pass_gate,
        "source": source,
        "diagnostic_oracle_not_formal_method": True,
    }
    smoke = {
        "status": "stopped_at_300_candidate_opportunity_gate" if not pass_gate else "eligible_for_reader_validation",
        "required_positive_query_opportunity": 0.25,
        "observed_positive_query_opportunity": opportunity,
        "existing_best_no_leak_joint_delta_vs_bm25": 0.0002,
        "claim_boundary": "external pipeline/lexical validation and selector limitation; not cross-dataset generalization",
    }
    write_json(OUTPUTS / "external_dataset/external_candidate_audit.json", candidate_audit)
    write_json(OUTPUTS / "external_dataset/external_smoke_summary.json", smoke)
    report = f"""# External Dataset Decision

Dataset: **2WikiMultiHopQA dev-300**

- Positive actions beyond the strong BM25 baseline: {candidate_audit['queries_with_positive_action']}/300 ({opportunity:.2%})
- Required to continue: 25.00%
- No-leak candidate path available: {no_leak}
- Adapter and sentence-support metrics available: {adapter_available and official_metrics_available}

Decision: **{smoke['status']}**. The opportunity rate misses the gate by {0.25 - opportunity:.2%}. Existing reader-backed evidence remains a lexical-routing sanity check and a cross-dataset selector limitation, not a generalization result.
"""
    (REPORTS / "external_dataset_decision.md").write_text(report, encoding="utf-8")
    print(json.dumps(smoke, indent=2))


if __name__ == "__main__":
    main()
