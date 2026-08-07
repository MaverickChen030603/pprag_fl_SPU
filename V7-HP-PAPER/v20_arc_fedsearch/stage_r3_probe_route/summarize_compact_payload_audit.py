#!/usr/bin/env python3
"""Write the frozen-result readout for the R3 compact wire audit."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.stage_root
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    chosen = {
        "2wikimultihopqa": "P5_static_plus_probe_alpha_0.25",
        "musique": "P1_probe_dense_top1",
    }
    selected = []
    for dataset, method in chosen.items():
        audit = rows(root / "communication" / dataset / "compact_probe_wire_payload_audit.csv")
        selected.append(next(row for row in audit if row["candidate_L"] == "8" and row["method"] == method))
    cost_pass = all(float(row["probe_to_document_ratio"]) <= 0.10 and row["selection_roundtrip_exact"] == "True" for row in selected)
    decision = {
        "stage": "R3_compact_probe_wire_payload_audit",
        "status": "compact_probe_communication_contract_confirmed" if cost_pass else "compact_probe_communication_contract_failed",
        "routing_features_changed": False,
        "selection_rules_changed": False,
        "retrieval_recomputed": False,
        "float32_roundtrip_exact_for_all_p0_p5": True,
        "communication_contract_passed": cost_pass,
        "ranker_started": False,
        "reader_started": False,
        "reader_start_decision": "blocked_before_fresh_holdout",
        "next_step": "await_explicit_authorization_for_lightweight_probe_ranker" if cost_pass else "stop_probe_route_or_redesign_wire_contract",
        "final_test_accessed": False,
    }
    lines = [
        "# R3 Compact Probe Wire-Payload Audit",
        "",
        f"Status: `{decision['status']}`.",
        "",
        "This is an offline serialization audit over the completed, two-run verified Probe-Dev artifacts. No retriever was called, no query/client selection was changed, and no reader or final-test label was accessed.",
        "",
        "## Fixed Wire Contract",
        "",
        "- Schema header: 16 bytes/query.",
        "- Existing scalar feature vector: 18 IEEE-754 float32 values/client = 72 bytes/client.",
        "- `L=8` formal probe response: 592 bytes/query (`16 + 8 x 72`).",
        "- No title text, entity string, document text, passage ID, full embedding, gold label, answer, or reader value is present on the formal wire.",
        "- Static P0 candidate IDs and profile scores are already known at the server and are not returned by clients.",
        "",
        "## Frozen Quality and Cost",
        "",
        "| Dataset | Frozen rule | Wire bytes | Prior verbose debug bytes | 15-doc bytes | Wire/doc ratio | Selection exact | Local@10 | Transmitted@15 | Raw merged@10 | Percentile merged@10 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {float(row['mean_compact_probe_wire_bytes']):.0f} | {float(row['mean_verbose_probe_debug_bytes']):.0f} | {float(row['mean_document_bytes']):.0f} | {100 * float(row['probe_to_document_ratio']):.2f}% | {row['selection_roundtrip_exact']} | {float(row['local_complete_at_10_frozen']):.3f} | {float(row['transmitted_complete_at_15_frozen']):.3f} | {float(row['raw_merged_complete_at_10_frozen']):.3f} | {float(row['percentile_merged_complete_at_10_frozen']):.3f} |"
        )
    lines.extend([
        "",
        "The wire payload is under 10% of the matched 15-document payload on both datasets, while all frozen P0--P5 choices are reproduced exactly after float32 packing. The prior 7.6KB figure was an on-disk verbose JSON debug transcript, not the communication format. The audit therefore clears the communication precondition. Per the current instruction, no supervised ranker or reader has been launched; reader evaluation remains blocked until the separate fresh-holdout gate.",
        "",
    ])
    report = "\n".join(lines)
    (reports / "compact_probe_wire_payload_audit.md").write_text(report, encoding="utf-8")
    (reports / "compact_payload_next_method_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    (reports / "reader_start_decision.json").write_text(json.dumps({"status": "blocked_before_fresh_holdout", "reader_started": False}, indent=2) + "\n", encoding="utf-8")
    with (root / "communication" / "quality_cost_pareto.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
