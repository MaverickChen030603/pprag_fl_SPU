#!/usr/bin/env python3
"""Inventory V4 evidence and write the mandatory review weakness audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from v5_common import HERE, V4, V4_COMPLETION, V4_FINAL, artifact, config, read_json, write_json, write_text


def locate_paper() -> dict[str, Path]:
    candidates = {
        "overleaf_pdf": [
            V4_FINAL / "overleaf_v4_final.pdf",
            V4_FINAL / "overleaf_v4_final" / "overleaf_v4_final.pdf",
            V4_FINAL / "overleaf_v4_final" / "main.pdf",
        ],
        "paper_full": [
            V4_COMPLETION / "paper_full_clean_v4_submission.md",
            V4 / "paper" / "paper_full_clean_v4.md",
        ],
        "paper_main": [
            V4_COMPLETION / "paper_main_conference_v4_submission.md",
            V4 / "paper" / "paper_main_conference_v4.md",
        ],
        "paper_appendix": [
            V4_COMPLETION / "paper_appendix_v4_submission.md",
            V4 / "paper" / "paper_appendix_v4.md",
        ],
    }
    return {name: next((path for path in paths if path.exists()), paths[0]) for name, paths in candidates.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when a P0 source is missing.")
    args = parser.parse_args()
    cfg = config()
    papers = locate_paper()
    sources = {
        **{name: artifact(path) for name, path in papers.items()},
        "v4_actions": artifact(V4 / "outputs/generated_actions/v4_outer_test_actions.jsonl", minimum_rows=1000),
        "v4_action_outcomes": artifact(V4 / "outputs/action_outcomes/v4_action_outputs.jsonl", minimum_rows=1000),
        "v4_nested_selector": artifact(V4 / "outputs/nested_selector/v4_nested_per_query.jsonl", minimum_rows=1000),
        "v4_official_per_query": artifact(V4 / "outputs/official_metrics/official_hotpotqa_per_query.jsonl", minimum_rows=2000),
        "holdout_3000_contexts": artifact(V4 / "outputs/scaleup/frozen_baseline_contexts_3000.jsonl", minimum_rows=3000),
        "holdout_3000_reader": artifact(V4 / "outputs/scaleup/official_metrics/flan_per_query.jsonl", minimum_rows=6000),
        "holdout_source_audit": artifact(V4 / "outputs/scaleup/same_source_context_audit.json"),
        "two_wiki_summary": artifact(V4_COMPLETION / "outputs/external_2wiki_frozen/external_validation_results.json"),
        "recomp_top1_summary": artifact(V4_COMPLETION / "outputs/faithful_baseline/faithful_baseline_results.json"),
        "generator_ablation_summary": artifact(V4_COMPLETION / "outputs/generator_ablation/generator_ablation_results.json"),
        "reader_manifest": artifact(V4 / "outputs/action_outcomes/reader_environment_manifest.json"),
        "fold_manifest": artifact(V4 / "outputs/semantic_generator/foldwise_generator_models.json"),
        "support_outputs": artifact(V4 / "outputs/official_metrics/official_hotpotqa_summary.json"),
    }
    source_audit = read_json(V4 / "outputs/scaleup/same_source_context_audit.json", {}) or {}
    source_section = source_audit.get("source", {})
    full_size = source_section.get("full_validation_size") if isinstance(source_section, dict) else None
    if full_size is None:
        full_size = source_audit.get("full_validation_size")
    untouched_expected = cfg["hotpot"]["revision_holdout_expected_size"]
    untouched = {
        "expected_slice": cfg["hotpot"]["revision_holdout_slice"],
        "expected_size": untouched_expected,
        "full_validation_size_from_audit": full_size if full_size is not None else "[NEEDS SOURCE FILE]",
        "eligible": bool(full_size == 7405),
        "status": "identity audit required before outcomes are opened",
    }
    weaknesses = [
        {
            "name": "Marginal absolute gains",
            "priority": "P1",
            "is_valid": "true",
            "evidence": "The frozen 3,000-query result reports modest population-level gains (Answer +0.0088, SP +0.0056, Joint +0.0064).",
            "risk": "A reviewer may consider the average effect too small without conditional intervention effects and deployment cost.",
            "experiment": "Compute exact selected-query and fallback-query effects from paired per-query outputs; report gain per 100 interventions and online overhead.",
            "writing": "Lead with selective intervention scope, report both population and conditional effects, and avoid practical-impact inflation.",
            "blocks_submission": "partially",
        },
        {
            "name": "Limited domain transfer",
            "priority": "P1",
            "is_valid": "true",
            "evidence": "Frozen 2Wiki transfer is non-significant, support is flat, and selected answer-drop rises from 2.0% in-domain to 6.92%.",
            "risk": "The current draft cannot claim robust cross-dataset generalization.",
            "experiment": "Run nested K-shot safety calibration on 2Wiki train only and preserve the existing 1,000-query dev evaluation set.",
            "writing": "Separate zero-shot failure from few-shot calibration and retain distribution shift as a limitation.",
            "blocks_submission": "no, if claims are narrowed",
        },
        {
            "name": "Mixed semantic component ablations",
            "priority": "P0/P1",
            "is_valid": "true",
            "evidence": "Generator ablations show the most stable mechanism-level signal for pair complementarity and bounded two-document construction; other semantic features are mixed.",
            "risk": "Presenting every semantic feature as an independent contribution overstates the evidence and leaves an unnecessarily complex method.",
            "experiment": "Evaluate fully nested Lite-Lexical-Pair, Lite-Semantic-Pair, and PairChain-Ablation; freeze a 0.002 Joint-F1 non-inferiority margin before holdout access.",
            "writing": "Center pair complementarity, bounded chains, anchor preservation, and reader-safe selection; move full semantic implementation details to the appendix.",
            "blocks_submission": "yes unless method claims are simplified",
        },
        {
            "name": "Unfair RECOMP comparison",
            "priority": "P0",
            "is_valid": "true",
            "evidence": "The original RECOMP context averages 47.13 tokens versus 660.57 for V4 and 668.18 for the baseline.",
            "risk": "Top-1 sentence compression cannot support a general superiority claim against near-full Top-5 contexts.",
            "experiment": "Run official RECOMP sentence scoring at 64/128/256/384/512/660 tokens, Baseline-Truncated controls, the same reader, support predictor, and paired protocol; freeze 660 before the 3,000-query holdout.",
            "writing": "Remove the old superiority statement. Retain a main-table comparison only if the matched-budget experiment completes.",
            "blocks_submission": "yes",
        },
        {
            "name": "High operational complexity",
            "priority": "P1",
            "is_valid": "partially_true",
            "evidence": "The current method includes MPNet, a cross-encoder, multiple generator heads, nested selection, and expensive offline action labeling; online cost has not been isolated from offline development cost.",
            "risk": "Readers may infer that every candidate action requires an online reader call or that the small gain cannot justify deployment overhead.",
            "experiment": "Benchmark baseline, Full, Lite, RECOMP Top-1, and budget-matched RECOMP; separately measure offline labels and online single-reader inference.",
            "writing": "State prominently that deployment runs the reader once on the selected final context, while candidate reader outcomes are offline supervision only.",
            "blocks_submission": "partially",
        },
    ]
    audit = {
        "task": cfg["task_name"],
        "protocol_frozen_at": cfg["protocol_frozen_at"],
        "frozen_directories_modified": False,
        "source_inventory": sources,
        "untouched_revision_holdout": untouched,
        "weaknesses": weaknesses,
        "p0_ready": all(sources[key]["exists"] for key in ("paper_full", "v4_actions", "v4_action_outcomes", "holdout_3000_contexts")),
    }
    write_json(HERE / "outputs/audits/review_weakness_audit.json", audit)
    lines = [
        "# Review Weakness Audit",
        "",
        f"- Task: `{cfg['task_name']}`",
        f"- Protocol frozen: `{cfg['protocol_frozen_at']}`",
        "- Frozen V4 directories modified: `false`",
        f"- Untouched revision holdout eligible: `{str(untouched['eligible']).lower()}` (expected slice 4000:7405; identity audit still required)",
        "",
        "## Source Inventory",
        "",
        "| Artifact | Exists | Rows/bytes | Path |",
        "|---|---:|---:|---|",
    ]
    for name, item in sources.items():
        size = item.get("rows", item.get("bytes", 0))
        lines.append(f"| {name} | {str(item['exists']).lower()} | {size} | `{item['path']}` |")
    for index, item in enumerate(weaknesses, 1):
        lines.extend(
            [
                "",
                f"## {index}. {item['name']} ({item['priority']})",
                "",
                f"- **Reviewer concern:** {item['name']}",
                f"- **is_valid:** `{item['is_valid']}`",
                f"- **Current paper evidence:** {item['evidence']}",
                f"- **Scientific risk:** {item['risk']}",
                f"- **Required experiment:** {item['experiment']}",
                f"- **Required writing change:** {item['writing']}",
                f"- **Blocks submission:** `{item['blocks_submission']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Revision Order",
            "",
            "1. Budget-match RECOMP or remove its numeric superiority comparison.",
            "2. Test the pre-registered Lite variants under fully nested evaluation.",
            "3. Measure online and offline cost separately and compute exact selected-query effects.",
            "4. Calibrate 2Wiki safety with target-train examples only.",
            "5. Rewrite the paper around complementary pairs, bounded chains, anchor preservation, and reader-safe selective intervention.",
        ]
    )
    write_text(HERE / "reports/review_weakness_audit.md", "\n".join(lines))
    write_text(HERE / "review_weakness_audit.md", "\n".join(lines))
    print(HERE / "reports/review_weakness_audit.md")
    if args.strict and not audit["p0_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
