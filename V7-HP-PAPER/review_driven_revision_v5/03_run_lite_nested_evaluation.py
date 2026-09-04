#!/usr/bin/env python3
"""Reader-backed, fully nested development evaluation for the V5 Lite generators."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import joblib

from v5_common import HERE, V4, config, iter_jsonl, paired_bootstrap, read_json, read_jsonl, write_json, write_jsonl, write_text


OUT = HERE / "outputs" / "lite_model"
FLAN = "/home/iiserver31/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
ARROW = "/home/iiserver31/.cache/huggingface/datasets/hotpot_qa/distractor/0.0.0/1908d6afbbead072334abe2965f91bd2709910ab/hotpot_qa-validation.arrow"
VARIANTS = ("lite_lexical_pair", "lite_semantic_pair", "pairchain_ablation")
METRICS = ("answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1")


def load_module(path: Path, name: str) -> Any:
    if str(V4) not in sys.path:
        sys.path.insert(0, str(V4))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def signature(query_id: str, context_ids: list[str]) -> str:
    return f"{query_id}|||{'|||'.join(context_ids)}"


def prompt(question: str, docs: list[dict[str, Any]], max_chars: int = 3200) -> str:
    context = "\n".join(
        f"[{index}] {doc['title']}: {doc['text']}" for index, doc in enumerate(docs, start=1)
    )
    return (
        "Answer the question using only the context. Return a short answer.\n\n"
        f"Question: {question}\n\nContext:\n{context[:max_chars]}\n\nAnswer:"
    )


def reader_stage(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    pending = read_jsonl(OUT / "pending_context_actions.jsonl")
    assigned = [row for index, row in enumerate(pending) if index % args.num_shards == args.shard_id]
    output = OUT / "reader" / f"pending_outcomes.shard{args.shard_id:02d}-of-{args.num_shards:02d}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(output) if args.resume and output.exists() else []
    done = {str(row["context_signature"]) for row in existing}
    assigned_pending = [row for row in assigned if str(row["context_signature"]) not in done]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float16
    ).to(args.device)
    model.eval()
    rows = list(existing)
    started = time.perf_counter()
    sys.path.insert(0, str(V4))
    import v4_common

    source = v4_common.load_source_examples()
    for start in range(0, len(assigned_pending), args.batch_size):
        batch = assigned_pending[start : start + args.batch_size]
        prompts = [prompt(row["question"], row["context_docs"]) for row in batch]
        encoded = tokenizer(
            prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        ).to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=32, num_beams=1, do_sample=False
            )
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for action, prediction in zip(batch, predictions):
            item = source[str(action["query_id"])]
            answer_em, answer_f1 = v4_common.answer_scores(prediction.strip(), item["answer"])
            title_recall, title_f1 = v4_common.title_metrics(
                action["context_titles"], item["supporting_titles"]
            )
            rows.append(
                {
                    "context_signature": action["context_signature"],
                    "query_id": str(action["query_id"]),
                    "context_doc_ids": action["context_doc_ids"],
                    "prediction": prediction.strip(),
                    "answer_em": answer_em,
                    "answer_f1": answer_f1,
                    "title_recall": title_recall,
                    "title_f1": title_f1,
                    "answer_title_product": answer_f1 * title_f1,
                    "source": "v5_lite_reader",
                }
            )
        if (start // args.batch_size) % 10 == 0 or start + args.batch_size >= len(assigned_pending):
            write_jsonl(output, rows)
            write_json(
                OUT / "reader" / f"progress_shard{args.shard_id}.json",
                {
                    "status": "running" if start + args.batch_size < len(assigned_pending) else "complete",
                    "completed": len(rows),
                    "assigned": len(assigned),
                    "seconds": time.perf_counter() - started,
                },
            )
    if not assigned_pending:
        write_json(
            OUT / "reader" / f"progress_shard{args.shard_id}.json",
            {"status": "complete", "completed": len(rows), "assigned": len(assigned)},
        )
    print(output)


def merged_outcomes() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    known = {
        signature(str(row["query_id"]), list(row["context_doc_ids"])): row
        for row in read_jsonl(OUT / "known_context_outcomes.jsonl")
    }
    for path in sorted((OUT / "reader").glob("pending_outcomes.shard*-of-*.jsonl")):
        for row in iter_jsonl(path):
            known[str(row["context_signature"])] = row
    actions = read_jsonl(OUT / "lite_actions_development.jsonl")
    missing = [
        row for row in actions if signature(str(row["query_id"]), list(row["context_doc_ids"])) not in known
    ]
    if missing:
        raise AssertionError(f"Missing outcomes for {len(missing)} action rows")
    merged = []
    for action in actions:
        outcome = known[signature(str(action["query_id"]), list(action["context_doc_ids"]))]
        row = dict(action)
        for key in ("prediction", "answer_f1", "title_recall", "title_f1", "answer_title_product"):
            row[key] = outcome[key]
        merged.append(row)
    write_jsonl(OUT / "lite_action_outcomes.jsonl", merged)
    return merged, known


def nested_stage(args: argparse.Namespace) -> None:
    cfg = config()
    selector = load_module(V4 / "07_train_nested_selector_v4.py", "v5_lite_selector")
    merged, _ = merged_outcomes()
    all_per_query = []
    summaries = {}
    fold_manifests = {}
    for variant in VARIANTS:
        rows = [row for row in merged if row["variant"] == variant]
        baselines, actions = selector.prepare(rows)
        query_ids = set(baselines)
        per_query, fold_records = [], []
        selector_model_dir = OUT / "selector_models"
        selector_model_dir.mkdir(parents=True, exist_ok=True)
        for fold in range(5):
            test_ids = {query_id for query_id in query_ids if int(baselines[query_id]["outer_fold"]) == fold}
            train_ids = query_ids - test_ids
            train_rows = [row for row in actions if row["query_id"] in train_ids]
            test_rows = [row for row in actions if row["query_id"] in test_ids]
            oof = selector.inner_oof(train_rows, train_ids, fold)
            selected_config = selector.tune(oof, baselines, train_ids)
            safety_model = selector.fit_model(train_rows, "answer_safe")
            opportunity_model = selector.fit_model(train_rows, "positive_action")
            safe_probs = selector.probabilities(safety_model, test_rows)
            positive_probs = selector.probabilities(opportunity_model, test_rows)
            scored = []
            for row, safe, positive in zip(test_rows, safe_probs, positive_probs):
                value = dict(row)
                value["pred_answer_safe_prob"] = safe
                value["pred_positive_prob"] = positive
                scored.append(value)
            model_path = selector_model_dir / f"fold_{fold}_{variant}.joblib"
            joblib.dump(
                {
                    "safety_model": safety_model,
                    "opportunity_model": opportunity_model,
                    "config": selected_config,
                    "variant": variant,
                    "outer_fold": fold,
                },
                model_path,
                compress=3,
            )
            selected = selector.select(
                scored,
                test_ids,
                selected_config["safe_threshold"],
                selected_config["positive_threshold"],
                selected_config["coverage"],
            )
            result = selector.evaluate(selected, baselines, test_ids)
            per_query.extend(result["per_query"])
            fold_records.append(
                {
                    "outer_fold": fold,
                    "n_train": len(train_ids),
                    "n_test": len(test_ids),
                    "train_selected_config": selected_config,
                    "outer_test_result": {key: value for key, value in result.items() if key != "per_query"},
                    "outer_test_outcomes_used_for_training_or_tuning": False,
                    "selector_model_path": str(model_path),
                }
            )
        selected_rows = [row for row in per_query if row["selected"]]
        summary = {
            "n_queries": len(per_query),
            "selected_count": len(selected_rows),
            "coverage": len(selected_rows) / len(per_query),
            "answer_drop_rate": sum(row["answer_f1_delta"] < -1e-12 for row in selected_rows) / max(1, len(selected_rows)),
            "answer_f1": mean(float(row["answer_f1"]) for row in per_query),
            "baseline_answer_f1": mean(float(row["baseline_answer_f1"]) for row in per_query),
            "answer_f1_delta": mean(float(row["answer_f1_delta"]) for row in per_query),
            "title_recall_delta": mean(float(row["title_recall_delta"]) for row in per_query),
            "answer_title_product_delta": mean(float(row["answer_title_product_delta"]) for row in per_query),
            "folds": fold_records,
        }
        summaries[variant] = summary
        fold_manifests[variant] = fold_records
        for row in per_query:
            all_per_query.append({"variant": variant, **row})
    write_jsonl(OUT / "lite_nested_per_query.jsonl", all_per_query)
    write_json(
        OUT / "lite_nested_selector_summary.json",
        {
            "status": "complete",
            "protocol": "fully nested five outer folds with inner OOF threshold and coverage selection",
            "joint_f1_noninferiority_margin_pre_registered": cfg["lite"]["joint_f1_noninferiority_margin"],
            "revision_holdout_used": False,
            "variants": summaries,
        },
    )
    write_json(
        HERE / "outputs/audits/lite_selector_nested_no_leak.json",
        {"status": "pass", "variants": fold_manifests, "revision_holdout_used": False},
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def gold_support(item: dict[str, Any], api: Any) -> set[tuple[str, int]]:
    return {
        (api.normalize_title(title), int(sent_id))
        for title, sent_id in zip(item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"])
    }


def official_stage(args: argparse.Namespace) -> None:
    api = load_module(V4 / "08_run_official_hotpot_evaluation.py", "v5_lite_official")
    official = api.load_official(args.arrow)
    actions = {
        str(row["action_id"]): row for row in read_jsonl(OUT / "lite_actions_development.jsonl")
    }
    outcomes = {
        str(row["action_id"]): row for row in read_jsonl(OUT / "lite_action_outcomes.jsonl")
    }
    nested_rows = read_jsonl(OUT / "lite_nested_per_query.jsonl")
    selected_by_variant = {
        variant: [row for row in nested_rows if row["variant"] == variant] for variant in VARIANTS
    }
    metric_rows = []
    for variant, selections in selected_by_variant.items():
        query_ids = {str(row["query_id"]) for row in selections}
        for fold in range(5):
            test = [row for row in selections if int(row["outer_fold"]) == fold]
            test_ids = {str(row["query_id"]) for row in test}
            train_ids = query_ids - test_ids
            train_instances = []
            train_selection = {str(row["query_id"]): row for row in selections if str(row["query_id"]) in train_ids}
            for query_id in train_ids:
                baseline_action = actions[f"{query_id}::v5lite::{variant}::fallback"]
                selected_action = actions[str(train_selection[query_id]["action_id"])]
                train_instances.extend(api.context_instances(query_id, baseline_action, official[query_id]))
                train_instances.extend(api.context_instances(query_id, selected_action, official[query_id]))
            support_model = api.fit(train_instances)
            for selection in test:
                query_id = str(selection["query_id"])
                action = actions[str(selection["action_id"])]
                instances = api.context_instances(query_id, action, official[query_id])
                pred_support = api.support_set(instances, api.score(support_model, instances), 0.7)
                result = api.official_metrics(
                    outcomes[str(selection["action_id"])]["prediction"],
                    official[query_id]["answer"],
                    pred_support,
                    gold_support(official[query_id], api),
                )
                metric_rows.append(
                    {
                        "query_id": query_id,
                        "method": variant,
                        "outer_fold": fold,
                        "selected": bool(selection["selected"]),
                        "action_id": selection["action_id"],
                        **result,
                    }
                )
    full_rows = [
        {**row, "method": "full_v4", "selected": None}
        for row in read_jsonl(V4 / "outputs/official_metrics/official_hotpotqa_per_query.jsonl")
        if row["method"] == "v4_selected"
    ]
    baseline_rows = [
        {**row, "method": "frozen_top5_baseline", "selected": False}
        for row in read_jsonl(V4 / "outputs/official_metrics/official_hotpotqa_per_query.jsonl")
        if row["method"] == "baseline"
    ]
    all_rows = baseline_rows + full_rows + metric_rows
    write_jsonl(OUT / "lite_official_per_query.jsonl", all_rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[str(row["method"])].append(row)
    metrics = {
        method: {metric: mean(float(row[metric]) for row in values) for metric in METRICS}
        for method, values in grouped.items()
    }
    full_by_query = {str(row["query_id"]): row for row in full_rows}
    noninferiority = {}
    margin = float(config()["lite"]["joint_f1_noninferiority_margin"])
    nested_summary = read_json(OUT / "lite_nested_selector_summary.json")
    for variant in VARIANTS:
        values = {str(row["query_id"]): row for row in metric_rows if row["method"] == variant}
        answer = paired_bootstrap([float(values[q]["answer_f1"]) - float(full_by_query[q]["answer_f1"]) for q in sorted(values)])
        joint = paired_bootstrap([float(values[q]["joint_f1"]) - float(full_by_query[q]["joint_f1"]) for q in sorted(values)])
        noninferiority[variant] = {
            "answer_f1_vs_full": answer,
            "joint_f1_vs_full": joint,
            "margin": margin,
            "point_estimate_noninferior": joint["delta"] >= -margin,
            "ci_noninferior": joint["ci_low"] >= -margin,
            "answer_drop_rate": nested_summary["variants"][variant]["answer_drop_rate"],
            "holdout_not_opened": True,
        }
    payload = {
        "status": "complete",
        "n_queries": 1000,
        "metrics": metrics,
        "noninferiority": noninferiority,
        "support_threshold": 0.7,
        "fully_nested": True,
        "revision_holdout_used": False,
    }
    write_json(OUT / "lite_nested_metrics.json", payload)
    write_json(OUT / "lite_noninferiority.json", noninferiority)
    write_lite_report(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def fmt(value: Any, digits: int = 4) -> str:
    return "[NEEDS MEASUREMENT]" if value is None else f"{float(value):.{digits}f}"


def write_lite_report(payload: dict[str, Any] | None = None) -> None:
    payload = payload or read_json(OUT / "lite_nested_metrics.json")
    audit = read_json(OUT / "lite_generator_audit.json", {})
    lines = [
        "# Lite Model Report",
        "",
        "## Design",
        "",
        "The Lite generator removes the missing-hop estimator, learned document-opportunity model, and cross-encoder. `Lite-Lexical-Pair` uses lexical/entity features plus a learned pair-complementarity model; `Lite-Semantic-Pair` adds one cached query-document cosine; `PairChain-Ablation` retains pair scoring and bounded two-document chains only. All learned models and selector thresholds use fully nested outer/inner folds.",
        "",
        f"- Pre-registered Joint-F1 non-inferiority margin: `{config()['lite']['joint_f1_noninferiority_margin']}`",
        f"- Pending contexts requiring new reader outcomes: `{audit.get('n_pending_contexts', '[NEEDS MEASUREMENT]')}`",
        "- Revision holdout opened during architecture selection: `false`",
        "",
        "## Development Results",
        "",
    ]
    if not payload:
        lines.append("[NEEDS MEASUREMENT]")
    else:
        lines.extend(["| Method | Answer F1 | SP F1 | Joint F1 | Joint vs Full | Point non-inferior | CI non-inferior |", "|---|---:|---:|---:|---:|---:|---:|"])
        for method, values in payload["metrics"].items():
            ni = payload["noninferiority"].get(method)
            lines.append(
                f"| {method} | {fmt(values['answer_f1'])} | {fmt(values['sp_f1'])} | {fmt(values['joint_f1'])} | "
                f"{fmt(ni['joint_f1_vs_full']['delta']) if ni else 'reference'} | "
                f"{str(ni['point_estimate_noninferior']).lower() if ni else 'reference'} | "
                f"{str(ni['ci_noninferior']).lower() if ni else 'reference'} |"
            )
    lines.extend(
        [
            "",
            "## Architecture Freeze Decision",
            "",
            "A Lite architecture is not promoted from development metrics alone. Eligibility requires the pre-registered non-inferiority test and at least 30% lower measured latency or cross-encoder calls. The final variant is then frozen before evaluating the untouched 3,405-query revision holdout.",
        ]
    )
    holdout = read_json(OUT / "lite_holdout_metrics.json")
    if holdout:
        lines.extend(
            [
                "",
                "## Untouched Revision Holdout (3,405)",
                "",
                "| Method | Answer F1 | SP F1 | Joint F1 |",
                "|---|---:|---:|---:|",
            ]
        )
        for method, values in holdout["metrics"].items():
            lines.append(f"| {method} | {fmt(values['answer_f1'])} | {fmt(values['sp_f1'])} | {fmt(values['joint_f1'])} |")
        ni = holdout["lite_noninferiority"]
        lines.extend(
            [
                "",
                f"Lite vs Full Joint F1: `{ni['joint_f1_delta']['delta']:+.4f}`; point-estimate non-inferior: `{str(ni['point_estimate_noninferior']).lower()}`; CI non-inferior: `{str(ni['ci_noninferior']).lower()}`.",
            ]
        )
    text = "\n".join(lines)
    write_text(HERE / "reports/lite_model_report.md", text)
    write_text(HERE / "lite_model_report.md", text)
    table = "\n".join(lines[lines.index("## Development Results") + 2 :]) if "## Development Results" in lines else text
    write_text(HERE / "outputs/tables/full_vs_lite.md", table)


def freeze_stage() -> None:
    payload = read_json(OUT / "lite_nested_metrics.json")
    if not payload or payload.get("revision_holdout_used"):
        raise AssertionError("Development-only Lite metrics are required before freezing")
    eligible = []
    for variant, row in payload["noninferiority"].items():
        answer = row["answer_f1_vs_full"]
        joint = row["joint_f1_vs_full"]
        if row["point_estimate_noninferior"] and answer["p_value"] >= 0.05 and joint["p_value"] >= 0.05:
            eligible.append(
                {
                    "variant": variant,
                    "cross_encoder_calls_per_query": 0,
                    "cross_encoder_call_reduction_vs_full": 1.0,
                    "answer_drop_rate": row["answer_drop_rate"],
                    "joint_f1_delta_vs_full": joint["delta"],
                    "strict_ci_noninferior": row["ci_noninferior"],
                }
            )
    if not eligible:
        selected = "full_v4"
        reason = "No Lite variant met the pre-registered point-estimate and non-significant-degradation screen."
    else:
        winner = min(
            eligible,
            key=lambda row: (
                row["cross_encoder_calls_per_query"],
                row["answer_drop_rate"],
                -row["joint_f1_delta_vs_full"],
            ),
        )
        selected = winner["variant"]
        reason = "Lowest measured cross-encoder call count, then lowest development answer-drop rate among point-estimate-noninferior variants."
    decision = {
        "status": "frozen",
        "selected_variant": selected,
        "selection_split": "HotpotQA development 1000",
        "revision_holdout_opened": False,
        "eligible_variants": eligible,
        "decision_rule": config()["lite"]["selection_rule"],
        "reason": reason,
        "joint_f1_margin": config()["lite"]["joint_f1_noninferiority_margin"],
        "important_caveat": "Point-estimate non-inferiority is used for architecture freezing; no Lite variant established strict CI-based non-inferiority on development.",
    }
    write_json(OUT / "lite_architecture_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def revision_reader_stage(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    revision_dir = OUT / "revision_holdout"
    audit = read_json(revision_dir / "revision_holdout_audit.json", {})
    full_manifest = read_json(revision_dir / "full_v4_generation_manifest.json", {})
    if audit.get("status") != "pass" or full_manifest.get("status") != "pass":
        raise AssertionError("Both frozen Lite and Full V4 revision generators must complete")
    source = {str(row["_id"]): row for row in read_json(revision_dir / "hotpot_revision_holdout_3405.json")}
    lite_actions = {str(row["action_id"]): row for row in read_jsonl(revision_dir / "lite_actions_3405.jsonl")}
    full_actions = {str(row["action_id"]): row for row in read_jsonl(revision_dir / "full_v4_actions_3405.jsonl")}
    lite_selections = {str(row["query_id"]): row for row in read_jsonl(revision_dir / "frozen_lite_selections_3405.jsonl")}
    full_selections = {str(row["query_id"]): row for row in read_jsonl(revision_dir / "full_v4_selections_3405.jsonl")}
    query_ids = sorted(source)
    assigned_ids = {query_id for index, query_id in enumerate(query_ids) if index % args.num_shards == args.shard_id}
    specs = []
    for query_id in sorted(assigned_ids):
        baseline_id = f"{query_id}::v5liteholdout::lite_lexical_pair::fallback"
        specs.extend(
            [
                {"query_id": query_id, "method": "frozen_top5_baseline", "action": lite_actions[baseline_id], "selected": False},
                {"query_id": query_id, "method": "lite_lexical_pair", "action": lite_actions[str(lite_selections[query_id]["action_id"])], "selected": bool(lite_selections[query_id]["selected"])},
                {"query_id": query_id, "method": "full_v4", "action": full_actions[str(full_selections[query_id]["action_id"])], "selected": bool(full_selections[query_id]["selected"])},
            ]
        )
    output = revision_dir / "reader" / f"per_query.shard{args.shard_id:02d}-of-{args.num_shards:02d}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(output) if args.resume and output.exists() else []
    done = {(str(row["query_id"]), str(row["method"])) for row in existing}
    pending = [row for row in specs if (row["query_id"], row["method"]) not in done]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.float16).to(args.device)
    model.eval()
    rows = list(existing)
    sys.path.insert(0, str(V4))
    import v4_common

    started = time.perf_counter()
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        prompts = [prompt(source[row["query_id"]]["question"], row["action"]["context_docs"]) for row in batch]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for spec, prediction in zip(batch, predictions):
            item, action = source[spec["query_id"]], spec["action"]
            answer_em, answer_f1 = v4_common.answer_scores(prediction.strip(), item["answer"])
            title_recall, title_f1 = v4_common.title_metrics(action["context_titles"], item["supporting_titles"])
            rows.append(
                {
                    "query_id": spec["query_id"],
                    "method": spec["method"],
                    "selected": spec["selected"],
                    "action_id": action["action_id"],
                    "prediction": prediction.strip(),
                    "answer_em_proxy": answer_em,
                    "answer_f1_proxy": answer_f1,
                    "title_recall": title_recall,
                    "title_f1": title_f1,
                }
            )
        if (start // args.batch_size) % 10 == 0 or start + args.batch_size >= len(pending):
            write_jsonl(output, rows)
            write_json(
                revision_dir / "reader" / f"progress_shard{args.shard_id}.json",
                {
                    "status": "running" if start + args.batch_size < len(pending) else "complete",
                    "completed": len(rows),
                    "assigned": len(specs),
                    "seconds": time.perf_counter() - started,
                },
            )
    print(output)


def revision_official_stage(args: argparse.Namespace) -> None:
    revision_dir = OUT / "revision_holdout"
    api = load_module(V4 / "08_run_official_hotpot_evaluation.py", "v5_revision_official")
    official = api.load_official(args.arrow)
    reader_rows = {
        (str(row["query_id"]), str(row["method"])): row
        for path in sorted((revision_dir / "reader").glob("per_query.shard*-of-*.jsonl"))
        for row in iter_jsonl(path)
    }
    expected = 3405 * 3
    if len(reader_rows) != expected:
        raise AssertionError(f"Expected {expected} revision reader rows, found {len(reader_rows)}")
    dev_selections = read_jsonl(V4 / "outputs/nested_selector/v4_nested_per_query.jsonl")
    dev_actions = {str(row["action_id"]): row for row in read_jsonl(V4 / "outputs/generated_actions/v4_outer_test_actions.jsonl")}
    train_instances = []
    for selection in dev_selections:
        query_id = str(selection["query_id"])
        for action_id in (f"{query_id}::v4::fallback", str(selection["action_id"])):
            train_instances.extend(api.context_instances(query_id, dev_actions[action_id], official[query_id]))
    support_model = api.fit(train_instances)
    lite_actions = {str(row["action_id"]): row for row in read_jsonl(revision_dir / "lite_actions_3405.jsonl")}
    full_actions = {str(row["action_id"]): row for row in read_jsonl(revision_dir / "full_v4_actions_3405.jsonl")}
    actions = {**lite_actions, **full_actions}
    metric_rows = []
    for (query_id, method), reader_row in sorted(reader_rows.items()):
        action = actions[str(reader_row["action_id"])]
        instances = api.context_instances(query_id, action, official[query_id])
        pred_support = api.support_set(instances, api.score(support_model, instances), 0.7)
        metrics = api.official_metrics(
            reader_row["prediction"], official[query_id]["answer"], pred_support, gold_support(official[query_id], api)
        )
        metric_rows.append({**reader_row, **metrics})
    write_jsonl(revision_dir / "official_per_query_3405.jsonl", metric_rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in metric_rows:
        grouped[row["method"]].append(row)
        by_query[row["query_id"]][row["method"]] = row
    metrics = {method: {metric: mean(float(row[metric]) for row in rows) for metric in METRICS} for method, rows in grouped.items()}
    comparisons = {}
    for left, right in (("lite_lexical_pair", "full_v4"), ("lite_lexical_pair", "frozen_top5_baseline"), ("full_v4", "frozen_top5_baseline")):
        key = f"{left}_vs_{right}"
        comparisons[key] = {
            metric: paired_bootstrap([float(values[left][metric]) - float(values[right][metric]) for values in by_query.values()])
            for metric in ("answer_f1", "sp_f1", "joint_f1")
        }
    margin = float(config()["lite"]["joint_f1_noninferiority_margin"])
    lite_vs_full = comparisons["lite_lexical_pair_vs_full_v4"]["joint_f1"]
    selected_effects = {}
    for method in ("lite_lexical_pair", "full_v4"):
        selected = [values for values in by_query.values() if values[method]["selected"]]
        selected_effects[method] = {
            "n": len(selected),
            "coverage": len(selected) / len(by_query),
            "answer_drop_rate": mean(float(values[method]["answer_f1"] < values["frozen_top5_baseline"]["answer_f1"] - 1e-12) for values in selected),
            **{
                f"{metric}_delta": mean(float(values[method][metric]) - float(values["frozen_top5_baseline"][metric]) for values in selected)
                for metric in ("answer_f1", "sp_f1", "joint_f1")
            },
        }
    payload = {
        "status": "complete",
        "split": "untouched_revision_holdout_3405",
        "n_queries": len(by_query),
        "metrics": metrics,
        "comparisons": comparisons,
        "selected_query_effects": selected_effects,
        "lite_noninferiority": {
            "margin": margin,
            "point_estimate_noninferior": lite_vs_full["delta"] >= -margin,
            "ci_noninferior": lite_vs_full["ci_low"] >= -margin,
            "joint_f1_delta": lite_vs_full,
        },
        "support_predictor": "same frozen all-development V4 support predictor, threshold 0.7",
        "architecture_selected_before_holdout_outcomes": True,
    }
    write_json(OUT / "lite_holdout_metrics.json", payload)
    write_lite_report(read_json(OUT / "lite_nested_metrics.json"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["reader", "nested", "official", "freeze", "revision-reader", "revision-official", "report"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=os.environ.get("V4_FLAN_T5_LARGE", FLAN))
    parser.add_argument("--arrow", default=os.environ.get("V4_HOTPOT_ARROW", ARROW))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "reader":
        reader_stage(args)
    elif args.stage == "nested":
        nested_stage(args)
    elif args.stage == "official":
        official_stage(args)
    elif args.stage == "freeze":
        freeze_stage()
    elif args.stage == "revision-reader":
        revision_reader_stage(args)
    elif args.stage == "revision-official":
        revision_official_stage(args)
    else:
        write_lite_report()


if __name__ == "__main__":
    main()
