#!/usr/bin/env python3
"""Outcome-aware oracle decomposition over the already frozen bounded actions.

The reader stage only replays existing contexts. The finalize stage computes
official HotpotQA metrics and retrospective oracles; none of its outcomes are
fed back into Full, its selector, or any threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from sigirap_common import (
    DEFAULT_ARROW,
    FIGURES,
    FLAN,
    METRICS,
    OUTPUTS,
    REPORTS,
    SPLITS,
    TABLES,
    baseline_action_id,
    ensure_layout,
    flan_prompt,
    iter_jsonl,
    metric_means,
    official_api,
    paired_bootstrap,
    read_json,
    read_jsonl,
    source_rows,
    stable_shard,
    title_proxy,
    win_loss_tie,
    write_json,
    write_jsonl,
)


ORACLE_DIR = OUTPUTS / "oracle"
EPS = 1e-12


def gold_titles(item: dict[str, Any]) -> list[str]:
    if "supporting_titles" in item:
        return [str(value) for value in item["supporting_titles"]]
    facts = item.get("supporting_facts", {})
    if isinstance(facts, dict):
        return [str(value) for value in facts.get("title", [])]
    return [str(value[0]) for value in facts]


def load_official() -> tuple[Any, dict[str, dict[str, Any]]]:
    api = official_api()
    arrow = Path(DEFAULT_ARROW)
    if not arrow.exists():
        raise FileNotFoundError(arrow)
    return api, api.load_official(str(arrow))


def reader_output_path(split: str, shard_index: int, num_shards: int) -> Path:
    return ORACLE_DIR / f"reader_{split}.shard{shard_index:02d}-of-{num_shards:02d}.jsonl"


def run_reader(args: argparse.Namespace) -> None:
    if args.split == "development1000":
        raise ValueError("Development already has all-action frozen FLAN outcomes")
    config = SPLITS[args.split]
    source = source_rows(args.split)
    output = reader_output_path(args.split, args.shard_index, args.num_shards)
    done = {str(row["action_id"]) for row in read_jsonl(output)} if args.resume else set()
    if output.exists() and not args.resume:
        output.unlink()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_path = Path(args.model_path or FLAN)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float16
    ).to(torch.device(args.device))
    model.eval()
    api = official_api()
    manifest = {
        "status": "running",
        "split": args.split,
        "stage": "all-frozen-action reader replay",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "model_path": str(model_path),
        "device": args.device,
        "batch_size": args.batch_size,
        "same_prompt": True,
        "context_character_cap": 3200,
        "generation": {"max_new_tokens": 32, "num_beams": 1, "do_sample": False},
        "posthoc_diagnostic_only": True,
        "started_at_epoch": time.time(),
    }
    write_json(output.with_suffix(".manifest.json"), manifest)

    pending_batch: list[dict[str, Any]] = []
    completed = len(done)
    total_in_shard = 0

    def flush(batch: list[dict[str, Any]]) -> int:
        if not batch:
            return 0
        prompts = [flan_prompt(source[row["query_id"]]["question"], row["context_docs"]) for row in batch]
        encoded = tokenizer(
            prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        ).to(torch.device(args.device))
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        rows = []
        for action, prediction in zip(batch, predictions):
            item = source[action["query_id"]]
            answer = str(item["answer"])
            answer_metrics = api.answer_metrics(prediction.strip(), answer)
            recall, title_f1 = title_proxy(action["context_titles"], gold_titles(item))
            rows.append({
                "split": args.split,
                "query_id": action["query_id"],
                "action_id": action["action_id"],
                "action_family": action["action_family"],
                "prediction": prediction.strip(),
                "answer_em": float(answer_metrics["em"]),
                "answer_f1": float(answer_metrics["f1"]),
                "title_recall": recall,
                "title_f1": title_f1,
                "answer_title_product": float(answer_metrics["f1"]) * title_f1,
            })
        write_jsonl(output, rows, mode="a")
        return len(rows)

    for action in iter_jsonl(config["actions"]):
        query_id = str(action["query_id"])
        if stable_shard(query_id, args.num_shards) != args.shard_index:
            continue
        total_in_shard += 1
        if str(action["action_id"]) in done:
            continue
        action["query_id"] = query_id
        pending_batch.append(action)
        if len(pending_batch) >= args.batch_size:
            completed += flush(pending_batch)
            pending_batch = []
            if completed % 256 < args.batch_size:
                write_json(output.with_suffix(".progress.json"), {
                    "split": args.split,
                    "shard_index": args.shard_index,
                    "completed": completed,
                    "seen_in_shard": total_in_shard,
                    "epoch": time.time(),
                })
    completed += flush(pending_batch)
    manifest.update({"status": "complete", "completed": completed, "rows_in_shard": total_in_shard})
    write_json(output.with_suffix(".manifest.json"), manifest)
    write_json(output.with_suffix(".progress.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_support_models(api: Any, official: dict[str, dict[str, Any]]) -> tuple[dict[int, Any], Any]:
    dev_actions = {str(row["action_id"]): row for row in iter_jsonl(SPLITS["development1000"]["actions"])}
    dev_selections = read_jsonl(SPLITS["development1000"]["selections"])
    instances: dict[tuple[str, str], list[dict[str, Any]]] = {}
    query_ids = {str(row["query_id"]) for row in dev_selections}
    for selection in dev_selections:
        query_id = str(selection["query_id"])
        for action_id in (baseline_action_id("development1000", query_id), str(selection["action_id"])):
            instances[(query_id, action_id)] = api.context_instances(query_id, dev_actions[action_id], official[query_id])
    fold_models: dict[int, Any] = {}
    for outer_fold in range(5):
        test_ids = {
            str(row["query_id"]) for row in dev_selections if int(row["outer_fold"]) == outer_fold
        }
        train_ids = query_ids - test_ids
        train_rows = [
            instance
            for query_id in train_ids
            for action_id in (
                baseline_action_id("development1000", query_id),
                str(next(row for row in dev_selections if str(row["query_id"]) == query_id)["action_id"]),
            )
            for instance in instances[(query_id, action_id)]
        ]
        fold_models[outer_fold] = api.fit(train_rows)
    # Preserve the original scale-up weighting exactly: when the frozen policy
    # falls back, baseline and selected contribute two identical training
    # instances rather than one deduplicated dictionary entry.
    all_rows = [
        instance
        for selection in dev_selections
        for action_id in (
            baseline_action_id("development1000", str(selection["query_id"])),
            str(selection["action_id"]),
        )
        for instance in instances[(str(selection["query_id"]), action_id)]
    ]
    return fold_models, api.fit(all_rows)


def load_reader_outcomes(split: str) -> dict[str, dict[str, Any]]:
    config = SPLITS[split]
    if split == "development1000":
        return {str(row["action_id"]): row for row in iter_jsonl(config["reader_outcomes"])}
    paths = sorted(ORACLE_DIR.glob(f"reader_{split}.shard*-of-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No reader shards found for {split}")
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in iter_jsonl(path):
            rows[str(row["action_id"])] = row
    expected = sum(1 for _ in iter_jsonl(config["actions"]))
    if len(rows) != expected:
        raise AssertionError(f"Incomplete {split} reader replay: {len(rows)} != {expected}")
    # Reuse the already frozen baseline/Full predictions for their matching
    # actions. New replay predictions are needed only for previously unevaluated
    # actions; this keeps the primary rows byte-for-byte aligned with the paper.
    selections = {str(row["query_id"]): row for row in iter_jsonl(config["selections"])}
    if split == "holdout3000":
        primary_paths = [config["actions"].parent / "readers/flan/per_query.jsonl"]
        baseline_method, full_method = "baseline", "v4_selected"
    else:
        primary_paths = sorted((config["actions"].parent / "reader").glob("per_query.shard*.jsonl"))
        baseline_method, full_method = "frozen_top5_baseline", "full_v4"
    for path in primary_paths:
        for primary in iter_jsonl(path):
            method = str(primary["method"])
            if method not in {baseline_method, full_method}:
                continue
            query_id = str(primary["query_id"])
            action_id = baseline_action_id(split, query_id) if method == baseline_method else str(selections[query_id]["action_id"])
            answer_f1 = float(primary.get("answer_f1", primary.get("answer_f1_proxy", 0.0)))
            title_f1 = float(primary["title_f1"])
            rows[action_id] = {
                **rows[action_id],
                "prediction": str(primary["prediction"]),
                "answer_em": float(primary.get("answer_em", primary.get("answer_em_proxy", 0.0))),
                "answer_f1": answer_f1,
                "title_recall": float(primary["title_recall"]),
                "title_f1": title_f1,
                "answer_title_product": answer_f1 * title_f1,
            }
    return rows


def official_action_path(split: str) -> Path:
    return ORACLE_DIR / f"official_all_actions_{split}.jsonl"


def score_all_actions(
    split: str,
    api: Any,
    official: dict[str, dict[str, Any]],
    fold_models: dict[int, Any],
    global_model: Any,
    reuse: bool,
) -> list[dict[str, Any]]:
    output = official_action_path(split)
    if reuse and output.exists():
        rows = read_jsonl(output)
        expected = sum(1 for _ in iter_jsonl(SPLITS[split]["actions"]))
        if len(rows) == expected:
            return rows
    outcomes = load_reader_outcomes(split)
    actions_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in iter_jsonl(SPLITS[split]["actions"]):
        actions_by_query[str(action["query_id"])].append(action)
    output_rows: list[dict[str, Any]] = []
    for query_index, (query_id, actions) in enumerate(actions_by_query.items(), start=1):
        fallback_id = baseline_action_id(split, query_id)
        baseline_proxy = outcomes[fallback_id]
        gold = official[query_id]
        gold_support = {
            (api.normalize_title(title), int(sentence_id))
            for title, sentence_id in zip(gold["supporting_facts"]["title"], gold["supporting_facts"]["sent_id"])
        }
        for action in actions:
            action_id = str(action["action_id"])
            outcome = outcomes[action_id]
            instances = api.context_instances(query_id, action, gold)
            model = fold_models[int(action["outer_fold"])] if split == "development1000" else global_model
            predicted_support = api.support_set(instances, api.score(model, instances), 0.7)
            metrics = api.official_metrics(outcome["prediction"], gold["answer"], predicted_support, gold_support)
            answer_delta = float(outcome["answer_f1"]) - float(baseline_proxy["answer_f1"])
            title_recall_delta = float(outcome["title_recall"]) - float(baseline_proxy["title_recall"])
            title_f1_delta = float(outcome["title_f1"]) - float(baseline_proxy["title_f1"])
            product_delta = float(outcome["answer_title_product"]) - float(baseline_proxy["answer_title_product"])
            answer_safe = answer_delta >= -EPS
            positive = (
                action_id != fallback_id
                and answer_safe
                and product_delta > EPS
                and (title_recall_delta > EPS or title_f1_delta >= -EPS)
            )
            output_rows.append({
                "split": split,
                "query_id": query_id,
                "action_id": action_id,
                "outer_fold": int(action["outer_fold"]),
                "action_family": action["action_family"],
                "prediction": outcome["prediction"],
                "proxy_answer_f1": float(outcome["answer_f1"]),
                "proxy_title_recall": float(outcome["title_recall"]),
                "proxy_title_f1": float(outcome["title_f1"]),
                "answer_safe": bool(answer_safe),
                "positive_action": bool(positive),
                **{metric: float(metrics[metric]) for metric in METRICS},
            })
        if query_index % 250 == 0:
            print(f"[{split}] official support scoring {query_index}/{len(actions_by_query)}", flush=True)
    write_jsonl(output, output_rows)
    return output_rows


def choose_best(rows: list[dict[str, Any]], baseline_id: str) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row["joint_f1"]),
            float(row["answer_f1"]),
            int(str(row["action_id"]) == baseline_id),
            str(row["action_id"]),
        ),
    )


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def decompose_split(split: str, action_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        grouped[str(row["query_id"])].append(row)
    selections = {str(row["query_id"]): row for row in iter_jsonl(SPLITS[split]["selections"])}
    query_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    effective_actions = 0
    positive_actions = 0
    systems = ("baseline", "policy", "utility_oracle", "answer_preserving_oracle", "available_opportunity_oracle")
    system_values: dict[str, list[dict[str, Any]]] = {name: [] for name in systems}

    for query_id, rows in grouped.items():
        baseline_id = baseline_action_id(split, query_id)
        baseline = next(row for row in rows if str(row["action_id"]) == baseline_id)
        policy_id = str(selections[query_id]["action_id"])
        policy = next(row for row in rows if str(row["action_id"]) == policy_id)
        utility = choose_best(rows, baseline_id)
        answer_preserving_candidates = [
            row for row in rows if float(row["answer_f1"]) >= float(baseline["answer_f1"]) - EPS
        ]
        answer_preserving = choose_best(answer_preserving_candidates, baseline_id)
        available_positive = [row for row in rows if bool(row["positive_action"])]
        available_oracle = choose_best(available_positive, baseline_id) if available_positive else baseline
        effective_actions += sum(str(row["action_id"]) != baseline_id for row in rows)
        positive_actions += len(available_positive)

        selected = policy_id != baseline_id
        policy_positive = bool(policy["positive_action"])
        opportunity = bool(available_positive)
        policy_answer_delta = float(policy["answer_f1"]) - float(baseline["answer_f1"])
        policy_joint_delta = float(policy["joint_f1"]) - float(baseline["joint_f1"])
        regret = float(answer_preserving["joint_f1"]) - float(policy["joint_f1"])
        systems_for_query = {
            "baseline": baseline,
            "policy": policy,
            "utility_oracle": utility,
            "answer_preserving_oracle": answer_preserving,
            "available_opportunity_oracle": available_oracle,
        }
        for name, value in systems_for_query.items():
            system_values[name].append(value)
        flat: dict[str, Any] = {
            "split": split,
            "query_id": query_id,
            "has_available_positive": opportunity,
            "n_effective_actions": sum(str(row["action_id"]) != baseline_id for row in rows),
            "n_positive_actions": len(available_positive),
            "policy_selected": selected,
            "policy_selected_positive": policy_positive,
            "selection_miss": opportunity and not policy_positive,
            "selection_success": opportunity and policy_positive,
            "harmful_selection_answer": selected and policy_answer_delta < -EPS,
            "harmful_selection_joint": selected and policy_joint_delta < -EPS,
            "regret_joint": regret,
        }
        for name, value in systems_for_query.items():
            flat[f"{name}_action_id"] = value["action_id"]
            flat[f"{name}_answer_f1"] = value["answer_f1"]
            flat[f"{name}_sp_f1"] = value["sp_f1"]
            flat[f"{name}_joint_f1"] = value["joint_f1"]
        query_rows.append(flat)
        regret_rows.append({
            "split": split,
            "query_id": query_id,
            "has_available_positive": opportunity,
            "policy_selected": selected,
            "policy_positive": policy_positive,
            "policy_joint_f1": policy["joint_f1"],
            "answer_preserving_oracle_joint_f1": answer_preserving["joint_f1"],
            "regret_joint": regret,
        })

    means = {name: metric_means(values) for name, values in system_values.items()}
    baseline_joint = means["baseline"]["joint_f1"]
    policy_gain = means["policy"]["joint_f1"] - baseline_joint
    ap_gain = means["answer_preserving_oracle"]["joint_f1"] - baseline_joint
    regrets = [float(row["regret_joint"]) for row in regret_rows]
    policy_diffs = [
        float(row["policy_joint_f1"]) - float(row["baseline_joint_f1"]) for row in query_rows
    ]
    ap_diffs = [
        float(row["answer_preserving_oracle_joint_f1"]) - float(row["baseline_joint_f1"])
        for row in query_rows
    ]
    utility_answer_diffs = [
        float(row["utility_oracle_answer_f1"]) - float(row["baseline_answer_f1"]) for row in query_rows
    ]
    summary = {
        "split": split,
        "label": SPLITS[split]["label"],
        "diagnostic_status": SPLITS[split]["diagnostic_status"],
        "n_queries": len(query_rows),
        "metrics": means,
        "deltas_vs_baseline": {
            name: {metric: means[name][metric] - means["baseline"][metric] for metric in METRICS}
            for name in systems if name != "baseline"
        },
        "query_opportunity_coverage": mean(row["has_available_positive"] for row in query_rows),
        "positive_action_density": positive_actions / max(1, effective_actions),
        "oracle_intervention_coverage": mean(
            row["answer_preserving_oracle_action_id"] != row["baseline_action_id"] for row in query_rows
        ),
        "policy_intervention_coverage": mean(row["policy_selected"] for row in query_rows),
        "policy_wins_losses_ties": win_loss_tie(policy_diffs),
        "answer_preserving_oracle_wins_losses_ties": win_loss_tie(ap_diffs),
        "utility_oracle_answer_drop_rate": mean(value < -EPS for value in utility_answer_diffs),
        "answer_preserving_oracle_answer_drop_rate": mean(
            row["answer_preserving_oracle_answer_f1"] < row["baseline_answer_f1"] - EPS for row in query_rows
        ),
        "answer_preserving_oracle_joint_drop_rate": mean(value < -EPS for value in ap_diffs),
        "selector_capture_ratio": policy_gain / ap_gain if ap_gain > EPS else None,
        "decomposition": {
            "no_opportunity": sum(not row["has_available_positive"] for row in query_rows),
            "opportunity_but_missed": sum(row["selection_miss"] for row in query_rows),
            "successful_selection": sum(row["selection_success"] for row in query_rows),
            "harmful_selection_answer": sum(row["harmful_selection_answer"] for row in query_rows),
            "harmful_selection_joint": sum(row["harmful_selection_joint"] for row in query_rows),
        },
        "regret": {
            "mean": mean(regrets),
            "median": median(regrets),
            "p75": quantile(regrets, 0.75),
            "p90": quantile(regrets, 0.90),
            "p95": quantile(regrets, 0.95),
            "zero_regret_proportion": mean(abs(value) <= EPS for value in regrets),
            "positive_regret_proportion": mean(value > EPS for value in regrets),
        },
        "paired_diagnostics": {
            "policy_joint": paired_bootstrap(policy_diffs),
            "answer_preserving_oracle_joint": paired_bootstrap(ap_diffs),
        },
    }
    return summary, query_rows, regret_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(summaries: dict[str, dict[str, Any]], query_rows: list[dict[str, Any]], regret_rows: list[dict[str, Any]]) -> None:
    payload = {
        "status": "complete",
        "protocol": "post-hoc outcome-aware diagnostic restricted to frozen bounded action sets",
        "deployable": False,
        "used_for_model_selection": False,
        "full_or_threshold_changed": False,
        "splits": summaries,
    }
    write_json(ORACLE_DIR / "oracle_metrics.json", payload)
    write_csv(ORACLE_DIR / "oracle_query_rows.csv", query_rows)
    write_csv(ORACLE_DIR / "selector_regret.csv", regret_rows)

    table_a = [
        "# Oracle Opportunity and Selection Decomposition",
        "",
        "Outcome-aware oracles are retrospective diagnostics over the same frozen bounded action sets. They are not deployable competitors.",
        "",
        "## Table A: opportunity and capture",
        "",
        "| Split | Baseline Joint | Policy Joint | Answer-preserving oracle Joint | Opportunity coverage | Policy coverage | Selector capture |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, summary in summaries.items():
        capture = summary["selector_capture_ratio"]
        table_a.append(
            f"| {summary['label']} | {summary['metrics']['baseline']['joint_f1']:.4f} | "
            f"{summary['metrics']['policy']['joint_f1']:.4f} | "
            f"{summary['metrics']['answer_preserving_oracle']['joint_f1']:.4f} | "
            f"{summary['query_opportunity_coverage']:.1%} | "
            f"{summary['policy_intervention_coverage']:.1%} | "
            f"{capture:.1%} |" if capture is not None else
            f"| {summary['label']} | {summary['metrics']['baseline']['joint_f1']:.4f} | "
            f"{summary['metrics']['policy']['joint_f1']:.4f} | "
            f"{summary['metrics']['answer_preserving_oracle']['joint_f1']:.4f} | "
            f"{summary['query_opportunity_coverage']:.1%} | {summary['policy_intervention_coverage']:.1%} | n/a |"
        )
    table_a.extend([
        "",
        "## Table B: query decomposition",
        "",
        "| Split | No opportunity | Opportunity but missed | Successful selection | Harmful selection (Joint) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for summary in summaries.values():
        d = summary["decomposition"]
        table_a.append(
            f"| {summary['label']} | {d['no_opportunity']} | {d['opportunity_but_missed']} | "
            f"{d['successful_selection']} | {d['harmful_selection_joint']} |"
        )
    (TABLES / "oracle_decomposition.md").write_text("\n".join(table_a) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    labels = [summary["label"].replace(" (", "\n(") for summary in summaries.values()]
    no_opportunity = [summary["decomposition"]["no_opportunity"] / summary["n_queries"] for summary in summaries.values()]
    missed = [summary["decomposition"]["opportunity_but_missed"] / summary["n_queries"] for summary in summaries.values()]
    success = [summary["decomposition"]["successful_selection"] / summary["n_queries"] for summary in summaries.values()]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.bar(labels, no_opportunity, label="No available positive action", color="#7a7a7a")
    ax.bar(labels, missed, bottom=no_opportunity, label="Opportunity missed", color="#d95f4b")
    ax.bar(labels, success, bottom=[a + b for a, b in zip(no_opportunity, missed)], label="Positive selected", color="#2a7f62")
    ax.set_ylabel("Proportion of queries")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "opportunity_selection_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)

    report = [
        "# Outcome-Aware Oracle Diagnostic",
        "",
        "## Status and boundary",
        "",
        "This analysis is retrospective. It selects only among contexts that were already present in each frozen bounded action set, but it uses target-query reader outcomes and official metrics. It therefore quantifies mechanism potential and selector regret; it is neither a deployable system nor a confirmatory baseline.",
        "",
        "## Main results",
        "",
    ]
    for summary in summaries.values():
        base = summary["metrics"]["baseline"]["joint_f1"]
        policy = summary["metrics"]["policy"]["joint_f1"]
        oracle = summary["metrics"]["answer_preserving_oracle"]["joint_f1"]
        capture = summary["selector_capture_ratio"]
        report.extend([
            f"### {summary['label']}",
            "",
            f"- Label: **{summary['diagnostic_status']}**.",
            f"- Baseline / policy / answer-preserving oracle Joint F1: {base:.4f} / {policy:.4f} / {oracle:.4f}.",
            f"- Available positive-action coverage: {summary['query_opportunity_coverage']:.1%}; positive-action density: {summary['positive_action_density']:.1%}.",
            f"- Frozen policy coverage: {summary['policy_intervention_coverage']:.1%}; aggregate selector capture ratio: {capture:.1%}." if capture is not None else "- Aggregate selector capture ratio is undefined because the oracle gain is non-positive.",
            f"- Mean / P90 / P95 selector regret: {summary['regret']['mean']:.4f} / {summary['regret']['p90']:.4f} / {summary['regret']['p95']:.4f}.",
            f"- No opportunity / opportunity missed / positive selected: {summary['decomposition']['no_opportunity']} / {summary['decomposition']['opportunity_but_missed']} / {summary['decomposition']['successful_selection']}.",
            "",
        ])
    report.extend([
        "## Interpretation",
        "",
        "The diagnostic separates candidate availability from selection. A large no-opportunity segment points to the bounded generator; a large opportunity-but-missed segment points to selector regret. The answer-preserving oracle includes the baseline and cannot be read as an attainable inference-time score. Holdout oracle values are explicitly post-hoc outcome-aware diagnostics and do not validate significance or generalization.",
    ])
    (REPORTS / "oracle_diagnostic_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def finalize(args: argparse.Namespace) -> None:
    api, official = load_official()
    fold_models, global_model = build_support_models(api, official)
    summaries: dict[str, dict[str, Any]] = {}
    all_queries: list[dict[str, Any]] = []
    all_regrets: list[dict[str, Any]] = []
    for split in SPLITS:
        print(f"Scoring and decomposing {split}", flush=True)
        action_rows = score_all_actions(split, api, official, fold_models, global_model, args.reuse_official)
        summary, query_rows, regret_rows = decompose_split(split, action_rows)
        summaries[split] = summary
        all_queries.extend(query_rows)
        all_regrets.extend(regret_rows)
    write_outputs(summaries, all_queries, all_regrets)
    print(json.dumps({"status": "complete", "splits": summaries}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("reader", "finalize"), required=True)
    parser.add_argument("--split", choices=tuple(SPLITS), default="holdout3000")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-official", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    ensure_layout()
    arguments = parse_args()
    if arguments.stage == "reader":
        run_reader(arguments)
    else:
        finalize(arguments)
