#!/usr/bin/env python3
"""Freeze v2 evidence and state the pre-registered v3 upgrade gates."""

from __future__ import annotations

from pathlib import Path

from v3_common import HERE, OUTPUTS, REPORTS, context_snapshot_path, ensure_layout, markdown_table, sha256, source_1000_path, v2_action_labels_path, write_json


FROZEN = {
    "n_queries": 1000,
    "outer_folds": 5,
    "selected_queries": 500,
    "fallback_queries": 500,
    "title_recall_delta": 0.0120,
    "title_recall_p": 0.007,
    "title_f1_delta": 0.0150,
    "title_f1_p": 0.018,
    "answer_f1_delta": 0.0028,
    "answer_f1_p": 0.344,
    "answer_title_product_delta": 0.0079,
    "answer_title_product_p": 0.1245,
    "main_eligible_positive_actions": 379,
    "main_eligible_actions": 4000,
    "queries_with_positive_action": 203,
    "queries_without_positive_action": 797,
    "selected_answer_drop_rate": 0.058,
}


def main() -> None:
    ensure_layout()
    v2_dir = HERE.parent / "submission_revision_v2"
    artifacts = {
        "paper_full_clean_v2": v2_dir / "paper_full_clean_v2.md",
        "nested_summary": v2_dir / "nested_final_1000_summary.json",
        "significance": v2_dir / "nested_significance_report.json",
        "action_scope": v2_dir / "action_scope_statistics.json",
        "source_1000": source_1000_path(),
        "context_snapshot": context_snapshot_path(),
        "v2_action_labels": v2_action_labels_path(),
    }
    manifest = {name: {"path": str(path), "sha256": sha256(path)} for name, path in artifacts.items()}
    requirements = [
        ["fully nested evaluation", "complete", "complete", "freeze"],
        ["significant title metrics", "complete", "insufficient alone", "preserve"],
        ["significant answer/product gain", "absent", "strongly preferred", "candidate redesign"],
        ["official support/joint metrics", "absent", "strongly preferred", "official pipeline"],
        ["positive-query coverage", "20.3%", "too low", "new action generator"],
        ["multi-reader", "absent", "desirable", "frozen-context replay"],
        ["scale", "1000", "limited", "full/large validation"],
        ["second dataset", "failed diagnostic", "desirable", "gated retry"],
        ["exact strong baselines", "partial proxies", "desirable", "controlled faithful baselines"],
        ["reproducibility", "mostly complete", "complete", "archive manifest"],
    ]
    payload = {
        "task": "V7-HP-PAPER-main_conference_upgrade_v3",
        "v2_frozen": True,
        "v2_directory_modified": False,
        "frozen_metrics": FROZEN,
        "artifact_manifest": manifest,
        "scientific_question": "Can reader-compatible candidate action generation increase positive-action opportunity and enable answer-safe selection to produce significant downstream multi-hop QA gains?",
        "primary_bottleneck": "candidate opportunity, not selector capacity",
        "pre_registered_gates": {
            "candidate_opportunity": {"meaningful": 0.30, "strong": 0.40, "unlikely": 0.25},
            "downstream": "official joint/support significance, product significance, or non-negative answer trend with stronger evidence across readers",
            "protocol": ["fully_nested", "no_target_outcome_leakage", "train_only_coverage", "frozen_scaleup"],
            "breadth": ["second_reader", "large_hotpot", "positive_external_smoke"],
            "reproducibility": ["model_revision", "environment", "commands", "archive_manifest"],
        },
    }
    write_json(OUTPUTS / "audits/main_conference_gap_audit.json", payload)
    report = f"""# Main-Conference Gap Audit

## Frozen v2 result

The existing `submission_revision_v2/` is frozen as the Findings/COLING fallback. Its files were read only and fingerprinted; v3 is a separate experiment directory.

{markdown_table(["Requirement", "Current v2", "Main-conference need", "Planned action"], requirements)}

## Primary diagnosis

**The primary main-conference bottleneck is candidate opportunity, not selector capacity.** In the frozen main-eligible action table, only 203/1,000 queries expose at least one answer-safe positive action; 797 expose none. A stronger selector cannot select an action that does not exist.

## Frozen claim boundary

Fully nested reader-safe action selection improves title-level evidence coverage without a demonstrated answer-quality or product gain. The title remains **Reader-Safe Context Action Selection for Multi-Hop Question Answering**. Federated/distributed routing is motivation and diagnostic history, not the paper title or an evaluated systems claim.

## v3 decision rule

The new generator must raise positive-query coverage materially above 20.3% using inference-safe signals only. Coverage below 25% will trigger a `not_ready` main-conference decision; at least 30% is meaningful and at least 40% is strong. Downstream, protocol, breadth, and reproducibility gates remain independent and cannot be waived after seeing test results.
"""
    (REPORTS / "main_conference_gap_audit.md").write_text(report, encoding="utf-8")
    print(OUTPUTS / "audits/main_conference_gap_audit.json")


if __name__ == "__main__":
    main()

