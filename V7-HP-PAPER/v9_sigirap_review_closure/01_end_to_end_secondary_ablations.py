#!/usr/bin/env python3
"""Audit whether requested secondary holdout ablations are legally runnable.

This script deliberately does not train removal variants. It inventories the
frozen artifacts, emits the Full reference rows, and reports unavailable
removals as unavailable when no pre-holdout checkpoint/action set exists.
Development opportunity ablations are retained as diagnostic evidence only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
DEV_RESULTS = (
    PAPER_ROOT
    / "v4_submission_completion/outputs/generator_ablation/generator_ablation_results.json"
)
DEV_AUDIT = (
    PAPER_ROOT
    / "v4_submission_completion/outputs/generator_ablation/generator_ablation_preparation_audit.json"
)
ORIGINAL_RESULTS = (
    PAPER_ROOT
    / "opportunity_aware_semantic_generation_v4/outputs/scaleup/official_metrics/scaleup_official_summary.json"
)
REVISION_RESULTS = (
    PAPER_ROOT / "review_driven_revision_v5/outputs/lite_model/lite_holdout_metrics.json"
)

REMOVALS = {
    "Full without pair complementarity": "without_pair_complementarity",
    "Full without two-document-chain actions": "without_two_document_actions",
    "Full without CrossEncoder document feature": "without_cross_encoder_features",
}

FIELDS = [
    "scope",
    "split",
    "n_queries",
    "variant",
    "status",
    "answer_f1",
    "sp_f1",
    "joint_f1",
    "answer_delta_vs_baseline",
    "sp_delta_vs_baseline",
    "joint_delta_vs_baseline",
    "joint_delta_vs_full",
    "coverage",
    "selected_answer_drop",
    "selected_joint_drop",
    "latency_ms",
    "joint_ci95_low",
    "joint_ci95_high",
    "joint_p_value",
    "positive_action_density",
    "positive_query_coverage",
    "answer_safe_action_rate",
    "evidence_boundary",
]


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def full_rows() -> list[dict[str, Any]]:
    original = read_json(ORIGINAL_RESULTS)
    revision = read_json(REVISION_RESULTS)

    original_base = original["readers"]["flan"]["metrics"]["baseline"]
    original_full = original["readers"]["flan"]["metrics"]["v4_selected"]
    original_sig = original["readers"]["flan"]["significance"]["joint_f1"]
    revision_base = revision["metrics"]["frozen_top5_baseline"]
    revision_full = revision["metrics"]["full_v4"]
    revision_sig = revision["comparisons"]["full_v4_vs_frozen_top5_baseline"]["joint_f1"]

    return [
        {
            "scope": "frozen_holdout_reference",
            "split": "Original holdout",
            "n_queries": 3000,
            "variant": "Full",
            "status": "available_frozen_reference",
            "answer_f1": original_full["answer_f1"],
            "sp_f1": original_full["sp_f1"],
            "joint_f1": original_full["joint_f1"],
            "answer_delta_vs_baseline": original_full["answer_f1"] - original_base["answer_f1"],
            "sp_delta_vs_baseline": original_full["sp_f1"] - original_base["sp_f1"],
            "joint_delta_vs_baseline": original_full["joint_f1"] - original_base["joint_f1"],
            "joint_delta_vs_full": 0.0,
            "coverage": 0.258,
            "selected_answer_drop": 0.0775,
            "selected_joint_drop": 0.1486,
            "latency_ms": 213.48,
            "joint_ci95_low": original_sig["ci95_low"],
            "joint_ci95_high": original_sig["ci95_high"],
            "joint_p_value": original_sig["p_value"],
            "evidence_boundary": "Primary frozen Full result; not a new ablation.",
        },
        {
            "scope": "frozen_holdout_reference",
            "split": "Revision holdout",
            "n_queries": 3405,
            "variant": "Full",
            "status": "available_frozen_reference",
            "answer_f1": revision_full["answer_f1"],
            "sp_f1": revision_full["sp_f1"],
            "joint_f1": revision_full["joint_f1"],
            "answer_delta_vs_baseline": revision_full["answer_f1"] - revision_base["answer_f1"],
            "sp_delta_vs_baseline": revision_full["sp_f1"] - revision_base["sp_f1"],
            "joint_delta_vs_baseline": revision_full["joint_f1"] - revision_base["joint_f1"],
            "joint_delta_vs_full": 0.0,
            "coverage": 0.2587371512481645,
            "selected_answer_drop": 0.07832009080590238,
            "selected_joint_drop": 0.1419,
            "latency_ms": 213.48,
            "joint_ci95_low": revision_sig["ci_low"],
            "joint_ci95_high": revision_sig["ci_high"],
            "joint_p_value": revision_sig["p_value"],
            "evidence_boundary": "Primary frozen Full result; not a new ablation.",
        },
    ]


def unavailable_holdout_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, n_queries in (("Original holdout", 3000), ("Revision holdout", 3405)):
        for display_name in REMOVALS:
            rows.append(
                {
                    "scope": "post_hoc_secondary_end_to_end_ablation",
                    "split": split,
                    "n_queries": n_queries,
                    "variant": display_name,
                    "status": "deferred_no_legal_frozen_checkpoint",
                    "evidence_boundary": (
                        "No corresponding checkpoint/action set was frozen before holdout inspection; "
                        "post-hoc training is prohibited."
                    ),
                }
            )
    return rows


def development_rows() -> list[dict[str, Any]]:
    results = read_json(DEV_RESULTS)
    audit = read_json(DEV_AUDIT)
    if audit.get("holdout_used") is not False:
        raise AssertionError("Development generator audit does not establish holdout isolation")
    rows: list[dict[str, Any]] = []
    variants = {"Full": "full", **REMOVALS}
    for display_name, key in variants.items():
        values = results["variants"][key]
        rows.append(
            {
                "scope": "development_opportunity_diagnostic",
                "split": "Nested development",
                "n_queries": values["queries"],
                "variant": display_name,
                "status": "available_diagnostic_only",
                "positive_action_density": values["positive_action_density"],
                "positive_query_coverage": values["positive_query_coverage"],
                "answer_safe_action_rate": values["answer_safe_action_rate"],
                "evidence_boundary": (
                    "Generator opportunity diagnostic; not a frozen end-to-end holdout removal and "
                    "not evidence that the removed component is necessary."
                ),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    with (HERE / "secondary_ablation_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_reports(rows: list[dict[str, Any]]) -> None:
    dev = {row["variant"]: row for row in rows if row["scope"] == "development_opportunity_diagnostic"}
    table = [
        "# Secondary Ablation Evidence",
        "",
        "## Frozen end-to-end status",
        "",
        "| Variant | Original 3,000 | Revision 3,405 | Evidence status |",
        "| --- | --- | --- | --- |",
        "| Full | A/SP/J = .6271/.4987/.3356 | A/SP/J = .6244/.4923/.3280 | Frozen reference |",
    ]
    for display_name in REMOVALS:
        table.append(
            f"| {display_name} | Not legally available | Not legally available | "
            "No pre-inspection frozen checkpoint/action set |"
        )
    table.extend(
        [
            "",
            "A clean frozen end-to-end ablation is unavailable because the corresponding model was not "
            "frozen before holdout inspection. We therefore retain development opportunity ablations "
            "and identify this as a limitation.",
            "",
            "## Nested-development opportunity diagnostics",
            "",
            "| Variant | Positive-action density | Positive-query coverage | Answer-safe action rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for display_name in ("Full", *REMOVALS.keys()):
        row = dev[display_name]
        table.append(
            f"| {display_name} | {pct(float(row['positive_action_density']))} | "
            f"{pct(float(row['positive_query_coverage']))} | "
            f"{pct(float(row['answer_safe_action_rate']))} |"
        )
    table.extend(
        [
            "",
            "These development values characterize candidate availability inside the bounded generator. "
            "They are not end-to-end holdout scores and do not support causal necessity claims.",
        ]
    )
    (HERE / "secondary_ablation_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    protocol = [
        "# Secondary Ablation Protocol Audit",
        "",
        "## Decision",
        "",
        "**Deferred, with the missing comparison disclosed.** The inventory found no pre-holdout "
        "frozen Full-without-pair, Full-without-chain, or Full-without-CrossEncoder checkpoint/action "
        "set compatible with the frozen selector and coverage contract.",
        "",
        "## Why a new run would be invalid as confirmation",
        "",
        "The holdout outcomes and the selected Full architecture are already known. Training a removal "
        "variant now, choosing its feature schema, or repairing its operating point after observing either "
        "holdout would make the comparison post-inspection and architecture-adaptive. We therefore do not "
        "run or report such a model as a confirmatory ablation.",
        "",
        "## Evidence retained",
        "",
        "The nested-development generator audit was prepared without holdout use and measures action-level "
        "opportunity density, query coverage, and answer-safe action rate. It remains a mechanism diagnostic. "
        "The separately frozen Lite architecture comparison is also excluded because it changes several "
        "modules together and is not a one-component Full removal.",
        "",
        "## Prohibited interpretations",
        "",
        "- The diagnostic does not show that pair complementarity, chain actions, or CrossEncoder features are necessary.",
        "- Missing cells are not zero effects.",
        "- No post-hoc removal is described as pre-specified or independently confirmatory.",
        "- The primary Full operating point and both holdouts remain unchanged.",
    ]
    (HERE / "secondary_ablation_protocol_audit.md").write_text(
        "\n".join(protocol) + "\n", encoding="utf-8"
    )


def main() -> None:
    rows = full_rows() + unavailable_holdout_rows() + development_rows()
    write_csv(rows)
    write_reports(rows)
    print(
        json.dumps(
            {
                "status": "complete",
                "legal_frozen_removal_variants": 0,
                "deferred_holdout_rows": len(unavailable_holdout_rows()),
                "development_diagnostic_rows": len(development_rows()),
                "primary_result_changed": False,
                "holdout_retuning": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
