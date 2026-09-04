#!/usr/bin/env python3
"""Evaluate frozen scale-up contexts with one frozen support predictor and two readers."""

from __future__ import annotations

import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from v4_common import OUTPUTS, ensure_layout, normalize_title, read_json, read_jsonl, write_json, write_jsonl


DEFAULT_ARROW = "/home/iiserver31/.cache/huggingface/datasets/hotpot_qa/distractor/0.0.0/1908d6afbbead072334abe2965f91bd2709910ab/hotpot_qa-validation.arrow"
FROZEN_SUPPORT_THRESHOLD = 0.7
METRICS = ["answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1"]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gold_support(row: dict[str, Any]) -> set[tuple[str, int]]:
    return {
        (normalize_title(title), int(sentence_id))
        for title, sentence_id in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"])
    }


def main() -> None:
    ensure_layout()
    scale_dir = OUTPUTS / "scaleup"
    flan_summary = read_json(scale_dir / "readers/flan/summary.json")
    unified_summary = read_json(scale_dir / "readers/unifiedqa/summary.json")
    if flan_summary.get("status") != "complete" or unified_summary.get("status") != "complete":
        raise AssertionError("Both frozen reader evaluations must complete before official metrics")

    official_module = load_module(Path(__file__).with_name("08_run_official_hotpot_evaluation.py"), "v4_scale_official")
    arrow_path = os.environ.get("V4_HOTPOT_ARROW", DEFAULT_ARROW)
    official = official_module.load_official(arrow_path)

    development_selections = read_jsonl(OUTPUTS / "nested_selector/v4_nested_per_query.jsonl")
    development_actions = {str(row["action_id"]): row for row in read_jsonl(OUTPUTS / "generated_actions/v4_outer_test_actions.jsonl")}
    train_instances = []
    for selection in development_selections:
        query_id = str(selection["query_id"])
        for action_id in (f"{query_id}::v4::fallback", str(selection["action_id"])):
            train_instances.extend(official_module.context_instances(query_id, development_actions[action_id], official[query_id]))
    support_model = official_module.fit(train_instances)
    historical = read_json(OUTPUTS / "official_metrics/official_hotpotqa_summary.json")
    historical_thresholds = {float(value) for value in historical["fold_thresholds"].values()}
    if historical_thresholds != {FROZEN_SUPPORT_THRESHOLD}:
        raise AssertionError(f"Historical support thresholds changed: {historical_thresholds}")

    selections = read_jsonl(scale_dir / "frozen_selector_selections_3000.jsonl")
    actions = {str(row["action_id"]): row for row in read_jsonl(scale_dir / "generated_actions_3000.jsonl")}
    instances: dict[tuple[str, str], list[dict[str, Any]]] = {}
    action_ids: dict[tuple[str, str], str] = {}
    for selection in selections:
        query_id = str(selection["query_id"])
        for method, action_id in (("baseline", f"{query_id}::v4scale::fallback"), ("v4_selected", str(selection["action_id"]))):
            action_ids[(query_id, method)] = action_id
            instances[(query_id, method)] = official_module.context_instances(query_id, actions[action_id], official[query_id])

    support_results: dict[tuple[str, str], tuple[set[tuple[str, int]], dict[str, float]]] = {}
    for key, rows in instances.items():
        query_id, _ = key
        predicted = official_module.support_set(rows, official_module.score(support_model, rows), FROZEN_SUPPORT_THRESHOLD)
        support_results[key] = (predicted, official_module.support_metrics(predicted, gold_support(official[query_id])))

    all_payload = {
        "status": "complete",
        "n_queries": len(selections),
        "support_predictor_training_queries": len(development_selections),
        "support_threshold": FROZEN_SUPPORT_THRESHOLD,
        "support_threshold_retuned": False,
        "readers": {},
    }
    official_dir = scale_dir / "official_metrics"
    official_dir.mkdir(parents=True, exist_ok=True)
    for reader_name in ("flan", "unifiedqa"):
        reader_rows = {
            (str(row["query_id"]), str(row["method"])): row
            for row in read_jsonl(scale_dir / f"readers/{reader_name}/per_query.jsonl")
        }
        metric_rows = []
        for selection in selections:
            query_id = str(selection["query_id"])
            gold = official[query_id]
            for method in ("baseline", "v4_selected"):
                reader_row = reader_rows[(query_id, method)]
                predicted_support, _ = support_results[(query_id, method)]
                metrics = official_module.official_metrics(reader_row["prediction"], gold["answer"], predicted_support, gold_support(gold))
                metric_rows.append({
                    "query_id": query_id,
                    "method": method,
                    "reader": reader_name,
                    "action_id": action_ids[(query_id, method)],
                    **metrics,
                })
        output_path = official_dir / f"{reader_name}_per_query.jsonl"
        write_jsonl(output_path, metric_rows)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in metric_rows:
            grouped[row["method"]].append(row)
            by_query[row["query_id"]][row["method"]] = row
        metrics = {
            method: {metric: mean(float(row[metric]) for row in rows) for metric in METRICS}
            for method, rows in grouped.items()
        }
        significance = {
            metric: official_module.paired_bootstrap([
                rows["v4_selected"][metric] - rows["baseline"][metric] for rows in by_query.values()
            ])
            for metric in METRICS
        }
        answer_drop_rate = mean(
            rows["v4_selected"]["answer_f1"] < rows["baseline"]["answer_f1"] - 1e-12 for rows in by_query.values()
        )
        all_payload["readers"][reader_name] = {
            "metrics": metrics,
            "deltas": {metric: metrics["v4_selected"][metric] - metrics["baseline"][metric] for metric in METRICS},
            "significance": significance,
            "answer_drop_rate": answer_drop_rate,
            "per_query_path": str(output_path),
        }
    flan_delta = all_payload["readers"]["flan"]["deltas"]["answer_f1"]
    unified_delta = all_payload["readers"]["unifiedqa"]["deltas"]["answer_f1"]
    all_payload["dual_reader_direction_consistent"] = (flan_delta >= 0) == (unified_delta >= 0)
    all_payload["systematic_answer_degradation"] = flan_delta < 0 and unified_delta < 0
    write_json(official_dir / "scaleup_official_summary.json", all_payload)
    print(json.dumps(all_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
