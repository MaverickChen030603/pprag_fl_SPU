#!/usr/bin/env python3
"""Summarize opportunity for each fully nested generator ablation."""

from __future__ import annotations

import json
from collections import defaultdict

from completion_common import ABLATION, TABLES, V4_ROOT, add_v4_import_path, ensure_layout, read_json, read_jsonl, write_json


def signature(query_id: str, context_ids: list[str]) -> str:
    return f"{query_id}|||{'|||'.join(context_ids)}"


def main() -> None:
    ensure_layout()
    add_v4_import_path()
    from v4_common import load_v3_merged_rows, v2_main_positive_query_ids

    actions = read_jsonl(ABLATION / "generator_ablation_actions.jsonl")
    outcome_rows = read_jsonl(ABLATION / "reused_context_outcomes.jsonl")
    pending_path = ABLATION / "reader/pending_context_outcomes.jsonl"
    if pending_path.exists():
        outcome_rows.extend(read_jsonl(pending_path))
    outcomes = {
        signature(str(row["query_id"]), list(row["context_doc_ids"])): row
        for row in outcome_rows
    }
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in actions:
        grouped[str(row["variant"])][str(row["query_id"])].append(row)
    v3_audit = read_json(V4_ROOT / "outputs/audits/v3_ceiling_aware_opportunity.json")
    ceiling_ids = set(v3_audit.get("ceiling_query_ids", []))
    if not ceiling_ids:
        # Historical audit stores per-query records in some versions.
        ceiling_ids = {str(row["query_id"]) for row in v3_audit.get("per_query", []) if row.get("ceiling")}
    v3_rows = load_v3_merged_rows()
    v3_context_signatures = {
        signature(str(row["query_id"]), list(row["context_doc_ids"]))
        for row in v3_rows
        if row["action_family"] != "fallback"
    }
    v3_positive_ids = {
        str(row["query_id"])
        for row in v3_rows
        if row["action_family"] != "fallback" and bool(row.get("positive_action"))
    }
    v2_positive_ids = v2_main_positive_query_ids()

    summaries = {}
    for variant, query_groups in grouped.items():
        effective = 0
        new_actions_vs_v3 = 0
        positives = 0
        safe = 0
        positive_ids: set[str] = set()
        for query_id, rows in query_groups.items():
            baseline_action = next(row for row in rows if row["action_family"] == "fallback")
            baseline = outcomes[signature(query_id, list(baseline_action["context_doc_ids"]))]
            for action in rows:
                if action["action_family"] == "fallback":
                    continue
                effective += 1
                action_signature = signature(query_id, list(action["context_doc_ids"]))
                new_actions_vs_v3 += int(action_signature not in v3_context_signatures)
                result = outcomes[action_signature]
                answer_delta = float(result["answer_f1"]) - float(baseline["answer_f1"])
                recall_delta = float(result["title_recall"]) - float(baseline["title_recall"])
                f1_delta = float(result["title_f1"]) - float(baseline["title_f1"])
                product_delta = float(result["answer_title_product"]) - float(baseline["answer_title_product"])
                answer_safe = answer_delta >= -1e-12
                positive = answer_safe and product_delta > 1e-12 and (recall_delta > 1e-12 or f1_delta >= -1e-12)
                safe += int(answer_safe)
                positives += int(positive)
                if positive:
                    positive_ids.add(query_id)
        non_ceiling = set(query_groups) - ceiling_ids
        newly_covered = positive_ids - v3_positive_ids
        summaries[variant] = {
            "queries": len(query_groups),
            "effective_actions": effective,
            "new_actions_vs_v3": new_actions_vs_v3,
            "positive_actions": positives,
            "positive_action_density": positives / max(1, effective),
            "positive_query_count": len(positive_ids),
            "positive_query_coverage": len(positive_ids) / len(query_groups),
            "conditional_non_ceiling_coverage": len(positive_ids & non_ceiling) / max(1, len(non_ceiling)),
            "newly_covered_v3_uncovered_queries": len(newly_covered),
            "new_query_efficiency": len(newly_covered) / max(1, new_actions_vs_v3),
            "answer_safe_action_rate": safe / max(1, effective),
            "positive_queries_not_in_v2": len(positive_ids - v2_positive_ids),
        }
    write_json(ABLATION / "generator_ablation_results.json", {"status": "complete", "variants": summaries})
    lines = [
        "# Generator Ablation Table",
        "",
        "| Variant | Effective actions | New actions vs V3 | Positive density | Overall coverage | Non-ceiling coverage | New queries vs V3 | Efficiency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, row in summaries.items():
        lines.append(
            f"| {variant} | {row['effective_actions']} | {row['new_actions_vs_v3']} | {row['positive_action_density']:.3f} | "
            f"{row['positive_query_coverage']:.3f} | {row['conditional_non_ceiling_coverage']:.3f} | "
            f"{row['newly_covered_v3_uncovered_queries']} | {row['new_query_efficiency']:.4f} |"
        )
    lines.extend([
        "",
        "All learned ablations are retrained on each outer-training split and applied to the disjoint outer test fold. Structural action-family removals use the same frozen fold model. No 3,000-query holdout outcome is used.",
    ])
    (TABLES / "generator_ablation_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "variants": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
