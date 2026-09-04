#!/usr/bin/env python3
"""Fixed-grid support-threshold sensitivity over frozen contexts and readers.

The pre-specified primary threshold remains 0.7. This script changes only the
post-hoc metric threshold applied to frozen support probabilities; it never
changes contexts, answer predictions, the Full selector, or model parameters.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
V7 = PAPER_ROOT / "v7_sigirap_targeted_strengthening"
if str(V7) not in sys.path:
    sys.path.insert(0, str(V7))

import sigirap_common as common  # noqa: E402


THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
METHODS = ("baseline", "full", "ce_score_order")
METHOD_LABELS = {
    "baseline": "Frozen Top-5",
    "full": "Full",
    "ce_score_order": "CrossEncoder-Top5",
}


def load_oracle_module() -> Any:
    path = V7 / "01_oracle_action_set_diagnostic.py"
    spec = importlib.util.spec_from_file_location("v9_oracle_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_answer_components() -> dict[tuple[str, str, str], tuple[float, float] | str]:
    path = V7 / "outputs/reranker/ce_reranker_per_query.csv"
    result: dict[tuple[str, str, str], tuple[float, float] | str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split, method = str(row["split"]), str(row["method"])
            if split not in {"holdout3000", "revision3405"} or method not in METHODS:
                continue
            key = (split, str(row["query_id"]), method)
            if row["answer_precision"] != "" and row["answer_recall"] != "":
                result[key] = (float(row["answer_precision"]), float(row["answer_recall"]))
            elif row["prediction"] != "":
                # CrossEncoder rows retain the frozen prediction but omit the
                # redundant answer precision/recall columns. Recompute only
                # those metric components with the same official normalizer.
                result[key] = str(row["prediction"])
            elif float(row["answer_f1"]) == 0.0:
                # Three frozen CE generations are the empty string. Their
                # official answer overlap is exactly zero against non-empty
                # Hotpot answers.
                result[key] = (0.0, 0.0)
            else:
                raise ValueError(f"Missing frozen answer evidence: {split}/{method}/{row['query_id']}")
    return result


def load_actions(split: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    all_actions = {str(row["action_id"]): row for row in common.iter_jsonl(common.SPLITS[split]["actions"])}
    selections = {str(row["query_id"]): row for row in common.iter_jsonl(common.SPLITS[split]["selections"])}
    ce_path = V7 / f"outputs/reranker/ce_actions_{split}.jsonl"
    ce_actions = {
        str(row["query_id"]): row
        for row in common.iter_jsonl(ce_path)
        if str(row["method"]) == "ce_score_order"
    }
    if len(selections) != common.SPLITS[split]["n"] or len(ce_actions) != common.SPLITS[split]["n"]:
        raise AssertionError(f"Incomplete frozen actions for {split}")
    return all_actions, selections, ce_actions


def harmonic(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_split(
    split: str,
    api: Any,
    official: dict[str, dict[str, Any]],
    support_model: Any,
    answer_components: dict[tuple[str, str, str], tuple[float, float] | str],
) -> tuple[list[dict[str, Any]], dict[tuple[float, str], list[dict[str, float]]]]:
    all_actions, selections, ce_actions = load_actions(split)
    values: dict[tuple[float, str], list[dict[str, float]]] = defaultdict(list)

    for index, query_id in enumerate(selections, start=1):
        action_map = {
            "baseline": all_actions[common.baseline_action_id(split, query_id)],
            "full": all_actions[str(selections[query_id]["action_id"])],
            "ce_score_order": ce_actions[query_id],
        }
        gold = official[query_id]
        gold_support = {
            (api.normalize_title(title), int(sentence_id))
            for title, sentence_id in zip(
                gold["supporting_facts"]["title"], gold["supporting_facts"]["sent_id"]
            )
        }
        for method, action in action_map.items():
            instances = api.context_instances(query_id, action, gold)
            probabilities = api.score(support_model, instances)
            answer_value = answer_components[(split, query_id, method)]
            if isinstance(answer_value, str):
                answer_metric = api.answer_metrics(answer_value, gold["answer"])
                answer_precision = float(answer_metric["precision"])
                answer_recall = float(answer_metric["recall"])
            else:
                answer_precision, answer_recall = answer_value
            for threshold in THRESHOLDS:
                support = api.support_set(instances, probabilities, threshold)
                support_metrics = api.support_metrics(support, gold_support)
                joint_precision = answer_precision * float(support_metrics["precision"])
                joint_recall = answer_recall * float(support_metrics["recall"])
                values[(threshold, method)].append(
                    {
                        "sp_f1": float(support_metrics["f1"]),
                        "joint_f1": harmonic(joint_precision, joint_recall),
                    }
                )
        if index % 500 == 0:
            print(f"[{split}] {index}/{len(selections)}", flush=True)

    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        aggregates = {
            method: {
                "sp_f1": mean(row["sp_f1"] for row in values[(threshold, method)]),
                "joint_f1": mean(row["joint_f1"] for row in values[(threshold, method)]),
            }
            for method in METHODS
        }
        baseline = aggregates["baseline"]
        for method in METHODS:
            metric = aggregates[method]
            rows.append(
                {
                    "split": split,
                    "n_queries": len(selections),
                    "threshold": threshold,
                    "primary_threshold": threshold == 0.7,
                    "system": METHOD_LABELS[method],
                    "sp_f1": metric["sp_f1"],
                    "joint_f1": metric["joint_f1"],
                    "sp_delta_vs_baseline": metric["sp_f1"] - baseline["sp_f1"],
                    "joint_delta_vs_baseline": metric["joint_f1"] - baseline["joint_f1"],
                }
            )
    return rows, values


def sign(value: float, eps: float = 1e-12) -> int:
    return int(value > eps) - int(value < -eps)


def write_outputs(rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "n_queries",
        "threshold",
        "primary_threshold",
        "system",
        "sp_f1",
        "joint_f1",
        "sp_delta_vs_baseline",
        "joint_delta_vs_baseline",
    ]
    with (HERE / "support_threshold_sensitivity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Support-Threshold Sensitivity",
        "",
        "This is a post-hoc metric sensitivity analysis over frozen contexts, frozen answer-reader "
        "outputs, and frozen support-predictor probabilities. The pre-specified primary threshold "
        "remains **0.7**; no threshold is re-selected from these results.",
        "",
    ]
    for split in ("holdout3000", "revision3405"):
        label = common.SPLITS[split]["label"]
        report.extend(
            [
                f"## {label}",
                "",
                "| Threshold | System | SP F1 | Joint F1 | SP delta vs baseline | Joint delta vs baseline |",
                "| ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        split_rows = [row for row in rows if row["split"] == split]
        for row in split_rows:
            marker = " **(primary)**" if row["primary_threshold"] else ""
            report.append(
                f"| {float(row['threshold']):.1f}{marker} | {row['system']} | "
                f"{float(row['sp_f1']):.4f} | {float(row['joint_f1']):.4f} | "
                f"{float(row['sp_delta_vs_baseline']):+.4f} | "
                f"{float(row['joint_delta_vs_baseline']):+.4f} |"
            )
        report.append("")
        for system in ("Full", "CrossEncoder-Top5"):
            system_rows = [row for row in split_rows if row["system"] == system]
            sp_signs = {sign(float(row["sp_delta_vs_baseline"])) for row in system_rows}
            joint_signs = {sign(float(row["joint_delta_vs_baseline"])) for row in system_rows}
            report.append(
                f"- **{system}:** SP direction "
                f"{'is stable' if len(sp_signs) == 1 else 'changes sign'}; Joint direction "
                f"{'is stable' if len(joint_signs) == 1 else 'changes sign'} across the fixed grid."
            )
        report.append("")
    report.extend(
        [
            "## Interpretation boundary",
            "",
            "The grid tests whether the reported operating-point comparison is an artifact of the "
            "0.7 support cutoff. It is not a new optimization step and does not alter the frozen Full "
            "selector, its coverage budget, the CrossEncoder ranker, or either answer reader.",
        ]
    )
    (HERE / "support_threshold_sensitivity_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    metadata = {
        "status": "complete",
        "analysis_label": "post-hoc metric sensitivity analysis",
        "threshold_grid": list(THRESHOLDS),
        "primary_threshold": 0.7,
        "threshold_retuned": False,
        "contexts_changed": False,
        "answer_outputs_changed": False,
        "support_model_changed": False,
        "splits": sorted({row["split"] for row in rows}),
    }
    (HERE / "support_threshold_sensitivity_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow", default=str(common.DEFAULT_ARROW))
    parser.add_argument(
        "--splits", nargs="+", choices=("holdout3000", "revision3405"),
        default=["holdout3000", "revision3405"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = common.official_api()
    arrow = Path(args.arrow)
    if not arrow.exists():
        raise FileNotFoundError(arrow)
    official = api.load_official(str(arrow))
    oracle = load_oracle_module()
    _, support_model = oracle.build_support_models(api, official)
    answer_components = read_answer_components()
    all_rows: list[dict[str, Any]] = []
    for split in args.splits:
        split_rows, _ = evaluate_split(
            split, api, official, support_model, answer_components
        )
        all_rows.extend(split_rows)
    write_outputs(all_rows)
    print(
        json.dumps(
            {
                "status": "complete",
                "rows": len(all_rows),
                "primary_threshold": 0.7,
                "threshold_retuned": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
