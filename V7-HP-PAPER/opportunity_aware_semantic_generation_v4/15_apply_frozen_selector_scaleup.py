#!/usr/bin/env python3
"""Fit only on the frozen development 1,000 and deploy unchanged on scale-up."""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib

from v4_common import OUTPUTS, ensure_layout, query_fingerprint, read_json, read_jsonl, sha256, write_json, write_jsonl


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ensure_layout()
    scale_dir = OUTPUTS / "scaleup"
    generator_audit = read_json(scale_dir / "frozen_generator_audit.json")
    if generator_audit.get("status") != "pass":
        raise AssertionError("Frozen generator audit must pass before selector deployment")

    selector = load_module(Path(__file__).with_name("07_train_nested_selector_v4.py"), "v4_scale_selector")
    _, development_actions = selector.prepare(read_jsonl(OUTPUTS / "action_outcomes/v4_action_outputs.jsonl"))
    nested = read_json(OUTPUTS / "nested_selector/v4_nested_summary.json")
    scale_actions = read_jsonl(scale_dir / "generated_actions_3000.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    baselines: dict[str, dict[str, Any]] = {}
    for row in scale_actions:
        query_id = str(row["query_id"])
        if row["action_family"] == "fallback":
            baselines[query_id] = row
        else:
            grouped[query_id].append(row)

    selected_by_query: dict[str, dict[str, Any]] = {}
    model_records = []
    model_dir = scale_dir / "selector_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold_record in nested["folds"]:
        fold_id = int(fold_record["outer_fold"])
        train_rows = [row for row in development_actions if int(row["outer_fold"]) != fold_id]
        scale_ids = {query_id for query_id, rows in grouped.items() if int(rows[0]["outer_fold"]) == fold_id}
        test_rows = [row for query_id in scale_ids for row in grouped[query_id]]
        safety_model = selector.fit_model(train_rows, "answer_safe")
        opportunity_model = selector.fit_model(train_rows, "positive_action")
        scored = []
        safe_probs = selector.probabilities(safety_model, test_rows)
        positive_probs = selector.probabilities(opportunity_model, test_rows)
        for row, safe, positive in zip(test_rows, safe_probs, positive_probs):
            value = dict(row)
            value["pred_answer_safe_prob"] = safe
            value["pred_positive_prob"] = positive
            scored.append(value)
        config = fold_record["train_selected_config"]
        selected = selector.select(
            scored,
            scale_ids,
            float(config["safe_threshold"]),
            float(config["positive_threshold"]),
            float(config["coverage"]),
        )
        selected_by_query.update(selected)
        model_path = model_dir / f"fold_{fold_id}_selector.joblib"
        joblib.dump({
            "safety_model": safety_model,
            "opportunity_model": opportunity_model,
            "config": config,
            "development_train_query_fingerprint": query_fingerprint({str(row["query_id"]) for row in train_rows}),
        }, model_path, compress=3)
        model_records.append({
            "fold_id": fold_id,
            "n_development_train_actions": len(train_rows),
            "n_scale_queries": len(scale_ids),
            "n_selected": len(selected),
            "frozen_config": config,
            "model_path": str(model_path),
            "model_sha256": sha256(model_path),
            "scaleup_outcomes_used": False,
            "thresholds_retuned": False,
        })

    selections = []
    for query_id in sorted(baselines):
        baseline = baselines[query_id]
        selected = selected_by_query.get(query_id)
        current = selected or baseline
        selections.append({
            "query_id": query_id,
            "outer_fold": int(current["outer_fold"]),
            "selected": selected is not None,
            "fallback": selected is None,
            "action_id": current["action_id"],
            "action_family": current["action_family"],
            "context_doc_ids": current["context_doc_ids"],
            "context_titles": current["context_titles"],
            "pred_answer_safe_prob": float(current.get("pred_answer_safe_prob", 0.0)),
            "pred_positive_prob": float(current.get("pred_positive_prob", 0.0)),
        })
    output_path = scale_dir / "frozen_selector_selections_3000.jsonl"
    write_jsonl(output_path, selections)
    selected_count = sum(bool(row["selected"]) for row in selections)
    manifest = {
        "status": "pass",
        "protocol": "fold-matched frozen development selector deployment",
        "n_queries": len(selections),
        "selected_count": selected_count,
        "coverage": selected_count / len(selections),
        "development_training_queries": 1000,
        "scaleup_labels_or_reader_outcomes_used": False,
        "thresholds_retuned": False,
        "selector_hyperparameters_changed": False,
        "selection_path": str(output_path),
        "selection_sha256": sha256(output_path),
        "folds": model_records,
    }
    write_json(scale_dir / "frozen_selector_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
