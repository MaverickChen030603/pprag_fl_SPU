#!/usr/bin/env python3
"""Post-hoc descriptive disagreement analysis over frozen per-query outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
SOURCE = PAPER_ROOT / "v7_sigirap_targeted_strengthening"
PER_QUERY = SOURCE / "outputs/reranker/ce_reranker_per_query.csv"
SELECTIONS = {
    "holdout3000": PAPER_ROOT / "opportunity_aware_semantic_generation_v4/outputs/scaleup/frozen_selector_selections_3000.jsonl",
    "revision3405": PAPER_ROOT / "review_driven_revision_v5/outputs/lite_model/revision_holdout/full_v4_selections_3405.jsonl",
}
SPLIT_LABELS = {
    "holdout3000": "Original holdout (3,000)",
    "revision3405": "Revision holdout (3,405)",
}
METHODS = ("baseline", "ce_score_order", "full")
METRICS = ("answer_f1", "sp_f1", "joint_f1")
SEED = 20260717
EPS = 1e-12


def load_rows() -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    grouped: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    with PER_QUERY.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            method = row["method"]
            if split not in SPLIT_LABELS or method not in METHODS:
                continue
            grouped.setdefault(split, {}).setdefault(row["query_id"], {})[method] = row
    for split, queries in grouped.items():
        expected = 3000 if split == "holdout3000" else 3405
        if len(queries) != expected:
            raise RuntimeError(f"{split}: expected {expected} queries, found {len(queries)}")
        incomplete = [query_id for query_id, methods in queries.items() if set(methods) != set(METHODS)]
        if incomplete:
            raise RuntimeError(f"{split}: incomplete method rows for {len(incomplete)} queries")
    return grouped


def load_families(path: Path) -> dict[str, str]:
    families: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                families[str(row["query_id"])] = str(row.get("action_family", "unknown"))
    return families


def delta(row: dict[str, str], reference: dict[str, str], metric: str) -> float:
    return float(row[metric]) - float(reference[metric])


def outcome_counts(values: list[float]) -> dict[str, int]:
    return {
        "wins": sum(value > EPS for value in values),
        "losses": sum(value < -EPS for value in values),
        "ties": sum(abs(value) <= EPS for value in values),
    }


def proportion_ci(count: int, n: int, rng: np.random.Generator) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    samples = rng.binomial(n, count / n, size=5000) / n
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def analyze() -> dict[str, object]:
    grouped = load_rows()
    rng = np.random.default_rng(SEED)
    result: dict[str, object] = {
        "status": "complete",
        "analysis_type": "post-hoc descriptive mechanism analysis",
        "uses_frozen_per_query_outputs": True,
        "anchor_analysis": {
            "status": "not_run",
            "reason": "No reliable explicit answer-anchor label is stored in the frozen per-query artifacts; no outcome-derived proxy was created.",
        },
        "splits": {},
    }
    event_rows: list[dict[str, object]] = []
    for split, queries in grouped.items():
        families = load_families(SELECTIONS[split])
        comparisons: dict[str, dict[str, dict[str, int]]] = {
            "crossencoder_vs_baseline": {},
            "full_vs_crossencoder": {},
        }
        for metric in METRICS:
            comparisons["crossencoder_vs_baseline"][metric] = outcome_counts([
                delta(methods["ce_score_order"], methods["baseline"], metric)
                for methods in queries.values()
            ])
            comparisons["full_vs_crossencoder"][metric] = outcome_counts([
                delta(methods["full"], methods["ce_score_order"], metric)
                for methods in queries.values()
            ])

        event_definitions = {
            "ce_sp_up_answer_down": lambda b, c, f: delta(c, b, "sp_f1") > EPS and delta(c, b, "answer_f1") < -EPS,
            "ce_joint_up_answer_down": lambda b, c, f: delta(c, b, "joint_f1") > EPS and delta(c, b, "answer_f1") < -EPS,
            "full_answer_up_ce_answer_down": lambda b, c, f: delta(f, b, "answer_f1") > EPS and delta(c, b, "answer_f1") < -EPS,
            "both_answer_up": lambda b, c, f: delta(c, b, "answer_f1") > EPS and delta(f, b, "answer_f1") > EPS,
            "both_answer_down": lambda b, c, f: delta(c, b, "answer_f1") < -EPS and delta(f, b, "answer_f1") < -EPS,
            "both_joint_up": lambda b, c, f: delta(c, b, "joint_f1") > EPS and delta(f, b, "joint_f1") > EPS,
        }
        events: dict[str, dict[str, object]] = {}
        event_query_ids: dict[str, list[str]] = {}
        n = len(queries)
        for event, predicate in event_definitions.items():
            query_ids = [
                query_id for query_id, methods in queries.items()
                if predicate(methods["baseline"], methods["ce_score_order"], methods["full"])
            ]
            low, high = proportion_ci(len(query_ids), n, rng)
            events[event] = {
                "n": len(query_ids),
                "proportion": len(query_ids) / n,
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
            }
            event_query_ids[event] = query_ids
            event_rows.append({"split": split, "event": event, **events[event]})

        focal_ids = event_query_ids["full_answer_up_ce_answer_down"]
        family_counts = Counter(families.get(query_id, "unknown") for query_id in focal_ids)
        family_profile = [
            {
                "action_family": family,
                "n": count,
                "proportion_within_event": count / len(focal_ids) if focal_ids else 0.0,
            }
            for family, count in family_counts.most_common()
        ]
        result["splits"][split] = {
            "label": SPLIT_LABELS[split],
            "n": n,
            "comparisons": comparisons,
            "cross_events": events,
            "full_answer_up_ce_answer_down_action_families": family_profile,
        }

    (HERE / "outputs/disagreement_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (HERE / "outputs/disagreement_events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(event_rows[0]))
        writer.writeheader()
        writer.writerows(event_rows)
    return result


def comparison_table(result: dict[str, object], comparison: str) -> list[str]:
    lines = [
        "| Split | Metric | Wins | Losses | Ties |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for split in SPLIT_LABELS:
        row = result["splits"][split]
        for metric in METRICS:
            counts = row["comparisons"][comparison][metric]
            lines.append(
                f"| {row['label']} | {metric} | {counts['wins']} | {counts['losses']} | {counts['ties']} |"
            )
    return lines


def write_report(result: dict[str, object]) -> None:
    event_labels = {
        "ce_sp_up_answer_down": "CE SP up, Answer down",
        "ce_joint_up_answer_down": "CE Joint up, Answer down",
        "full_answer_up_ce_answer_down": "Full Answer up, CE Answer down",
        "both_answer_up": "Both Answer up",
        "both_answer_down": "Both Answer down",
        "both_joint_up": "Both Joint up",
    }
    lines = [
        "# CrossEncoder-Full Disagreement Analysis",
        "",
        "This is a post-hoc descriptive mechanism analysis over frozen per-query reader and official-metric outputs. It does not retrain either system, tune a threshold, or establish a causal effect of anchor preservation.",
        "",
        "## CrossEncoder versus Frozen Top-5",
        "",
        *comparison_table(result, "crossencoder_vs_baseline"),
        "",
        "## Full versus CrossEncoder",
        "",
        *comparison_table(result, "full_vs_crossencoder"),
        "",
        "## Cross-events",
        "",
        "| Split | Event | N | Proportion | Bootstrap 95% CI |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for split in SPLIT_LABELS:
        row = result["splits"][split]
        for event, values in row["cross_events"].items():
            lines.append(
                f"| {row['label']} | {event_labels[event]} | {values['n']} | {values['proportion']:.1%} | "
                f"[{values['bootstrap_ci95_low']:.1%}, {values['bootstrap_ci95_high']:.1%}] |"
            )
    lines.extend([
        "",
        "## Full action families when Full improves Answer and CE lowers Answer",
        "",
        "| Split | Action family | N | Share within event |",
        "| --- | --- | ---: | ---: |",
    ])
    for split in SPLIT_LABELS:
        row = result["splits"][split]
        for family in row["full_answer_up_ce_answer_down_action_families"]:
            lines.append(
                f"| {row['label']} | {family['action_family']} | {family['n']} | {family['proportion_within_event']:.1%} |"
            )
    lines.extend([
        "",
        "## Anchor-label boundary",
        "",
        "The frozen artifacts contain inference-time anchor proxies but no reliable explicit answer-anchor label. We therefore do not report anchor retention or create an outcome-derived anchor label. The paired disagreement patterns are associations between system outputs, not causal proof of any particular document mechanism.",
    ])
    (HERE / "crossencoder_full_disagreement_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    compact = [
        "| Event | Original 3,000 | Revision 3,405 |",
        "| --- | ---: | ---: |",
    ]
    for event in ("ce_sp_up_answer_down", "full_answer_up_ce_answer_down", "both_joint_up"):
        values = []
        for split in SPLIT_LABELS:
            row = result["splits"][split]["cross_events"][event]
            values.append(f"{row['n']} ({row['proportion']:.1%})")
        compact.append(f"| {event_labels[event]} | {values[0]} | {values[1]} |")
    (HERE / "outputs/disagreement_compact_table.md").write_text("\n".join(compact) + "\n", encoding="utf-8")


def main() -> None:
    result = analyze()
    write_report(result)
    print(json.dumps({"status": "complete", "splits": {key: value["n"] for key, value in result["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
