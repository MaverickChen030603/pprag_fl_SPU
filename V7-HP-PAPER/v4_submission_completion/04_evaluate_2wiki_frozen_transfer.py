#!/usr/bin/env python3
"""Evaluate 2Wiki opportunity and official answer/support/joint metrics."""

from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean
from typing import Any

from completion_common import EXTERNAL, REPORTS, TABLES, V4_ROOT, add_v4_import_path, ensure_layout, load_module, read_json, read_jsonl, write_json, write_jsonl


SUPPORT_THRESHOLD = 0.7
METRICS = ["answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1"]


def external_instances(
    query_id: str,
    action: dict[str, Any],
    item: dict[str, Any],
    official_module: Any,
) -> list[dict[str, Any]]:
    by_title = {
        official_module.normalize_title(str(raw[0])): (str(raw[0]), [str(value) for value in raw[1]])
        for raw in item["context"]
    }
    gold = {
        (official_module.normalize_title(str(value[0])), int(value[1]))
        for value in item.get("supporting_facts", [])
    }
    rows = []
    for doc_rank, title in enumerate(action["context_titles"]):
        record = by_title.get(official_module.normalize_title(str(title)))
        if record is None:
            continue
        canonical_title, sentences = record
        for sentence_id, sentence in enumerate(sentences):
            rows.append({
                "query_id": query_id,
                "title": canonical_title,
                "sent_id": sentence_id,
                "features": official_module.sentence_features(
                    str(item["question"]), canonical_title, sentence, doc_rank, sentence_id, len(sentences)
                ),
                "label": int((official_module.normalize_title(canonical_title), sentence_id) in gold),
            })
    return rows


def main() -> None:
    ensure_layout()
    add_v4_import_path()
    official = load_module(V4_ROOT / "08_run_official_hotpot_evaluation.py", "v4_external_official")
    audit = read_json(EXTERNAL / "frozen_generator_selector_audit.json")
    reader_summary = read_json(EXTERNAL / "reader/summary.json")
    if audit.get("status") != "pass" or reader_summary.get("status") != "complete":
        raise AssertionError("Frozen generator/selector and reader stages must complete")

    # Refit the same support model on the original Hotpot development material only.
    hotpot_official = official.load_official(official.DEFAULT_ARROW)
    development_selections = read_jsonl(V4_ROOT / "outputs/nested_selector/v4_nested_per_query.jsonl")
    development_actions = {
        str(row["action_id"]): row
        for row in read_jsonl(V4_ROOT / "outputs/generated_actions/v4_outer_test_actions.jsonl")
    }
    train_instances = []
    for selection in development_selections:
        query_id = str(selection["query_id"])
        for action_id in (f"{query_id}::v4::fallback", str(selection["action_id"])):
            train_instances.extend(official.context_instances(query_id, development_actions[action_id], hotpot_official[query_id]))
    support_model = official.fit(train_instances)

    source = {str(row["query_id"]): row for row in read_json(EXTERNAL / "2wiki_frozen_1000.json")}
    actions = {str(row["action_id"]): row for row in read_jsonl(EXTERNAL / "generated_actions_1000.jsonl")}
    outcomes = {str(row["action_id"]): row for row in read_jsonl(EXTERNAL / "reader/all_action_outcomes.jsonl")}
    selections = read_jsonl(EXTERNAL / "frozen_selector_selections_1000.jsonl")

    grouped_actions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions.values():
        grouped_actions[str(action["query_id"])].append(action)
    positive_actions = 0
    positive_queries: set[str] = set()
    answer_safe_actions = 0
    effective_actions = 0
    for query_id, query_actions in grouped_actions.items():
        fallback = next(row for row in query_actions if row["action_family"] == "fallback")
        baseline = outcomes[str(fallback["action_id"])]
        for action in query_actions:
            if action["action_family"] == "fallback":
                continue
            effective_actions += 1
            result = outcomes[str(action["action_id"])]
            answer_delta = float(result["answer_f1"]) - float(baseline["answer_f1"])
            title_recall_delta = float(result["title_recall"]) - float(baseline["title_recall"])
            title_f1_delta = float(result["title_f1"]) - float(baseline["title_f1"])
            product_delta = float(result["answer_title_product"]) - float(baseline["answer_title_product"])
            safe = answer_delta >= -1e-12
            positive = safe and product_delta > 1e-12 and (title_recall_delta > 1e-12 or title_f1_delta >= -1e-12)
            answer_safe_actions += int(safe)
            positive_actions += int(positive)
            if positive:
                positive_queries.add(query_id)

    metric_rows = []
    selected_answer_drops = 0
    for selection in selections:
        query_id = str(selection["query_id"])
        item = source[query_id]
        baseline_id = f"{query_id}::v4xfer::fallback"
        selected_id = str(selection["action_id"])
        if selection["selected"] and float(outcomes[selected_id]["answer_f1"]) < float(outcomes[baseline_id]["answer_f1"]) - 1e-12:
            selected_answer_drops += 1
        gold_support = {
            (official.normalize_title(str(value[0])), int(value[1]))
            for value in item.get("supporting_facts", [])
        }
        for method, action_id in (("baseline", baseline_id), ("v4_frozen_transfer", selected_id)):
            action = actions[action_id]
            rows = external_instances(query_id, action, item, official)
            pred_support = official.support_set(rows, official.score(support_model, rows), SUPPORT_THRESHOLD)
            metrics = official.official_metrics(outcomes[action_id]["prediction"], item["answer"], pred_support, gold_support)
            metric_rows.append({
                "query_id": query_id,
                "method": method,
                "action_id": action_id,
                "selected": bool(selection["selected"]) if method == "v4_frozen_transfer" else False,
                **metrics,
            })
    write_jsonl(EXTERNAL / "official_per_query.jsonl", metric_rows)

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in metric_rows:
        by_method[row["method"]].append(row)
        by_query[row["query_id"]][row["method"]] = row
    metrics = {
        method: {metric: mean(float(row[metric]) for row in rows) for metric in METRICS}
        for method, rows in by_method.items()
    }
    significance = {
        metric: official.paired_bootstrap([
            rows["v4_frozen_transfer"][metric] - rows["baseline"][metric]
            for rows in by_query.values()
        ])
        for metric in METRICS
    }
    selected_count = sum(bool(row["selected"]) for row in selections)
    payload = {
        "status": "complete",
        "setting": "cross-dataset zero-shot frozen transfer",
        "dataset": "2WikiMultiHopQA dev",
        "n_queries": len(selections),
        "opportunity": {
            "effective_actions": effective_actions,
            "positive_actions": positive_actions,
            "positive_action_density": positive_actions / max(1, effective_actions),
            "positive_query_count": len(positive_queries),
            "positive_query_coverage": len(positive_queries) / len(selections),
            "answer_safe_action_rate": answer_safe_actions / max(1, effective_actions),
        },
        "selector": {
            "selected_count": selected_count,
            "coverage": selected_count / len(selections),
            "selected_answer_drop_count": selected_answer_drops,
            "selected_answer_drop_rate": selected_answer_drops / max(1, selected_count),
        },
        "metrics": metrics,
        "deltas": {
            metric: metrics["v4_frozen_transfer"][metric] - metrics["baseline"][metric]
            for metric in METRICS
        },
        "significance": significance,
        "support_model_training_dataset": "HotpotQA development 1000 only",
        "support_threshold": SUPPORT_THRESHOLD,
        "support_threshold_retuned_on_2wiki": False,
        "target_dataset_training_or_tuning": False,
    }
    answer_non_degrading = payload["deltas"]["answer_f1"] >= -1e-12
    joint_positive = payload["deltas"]["joint_f1"] > 0
    payload["external_success_rule"] = {
        "positive_and_non_degrading_frozen_transfer": bool(answer_non_degrading and joint_positive),
        "answer_non_degrading": bool(answer_non_degrading),
        "joint_positive": bool(joint_positive),
    }
    write_json(EXTERNAL / "external_validation_results.json", payload)

    table = """# External Validation Table\n\n| Setting | N | Coverage | Answer F1 delta | SP F1 delta | Joint F1 delta | Answer-drop rate |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"""
    table += (
        f"| 2Wiki frozen transfer | {len(selections)} | {payload['selector']['coverage']:.3f} | "
        f"{payload['deltas']['answer_f1']:+.4f} | {payload['deltas']['sp_f1']:+.4f} | "
        f"{payload['deltas']['joint_f1']:+.4f} | {payload['selector']['selected_answer_drop_rate']:.3f} |\n"
    )
    (TABLES / "external_validation_table.md").write_text(table, encoding="utf-8")
    report = f"""# 2WikiMultiHopQA Frozen-Transfer Report

- Protocol: fully frozen HotpotQA V4 generator, selector, thresholds, coverage, reader prompt, and support threshold.
- Dataset: deterministic label-blind sample of {len(selections)} 2Wiki dev queries.
- Opportunity: {len(positive_queries)}/{len(selections)} queries ({len(positive_queries)/len(selections):.2%}); positive-action density {positive_actions/max(1,effective_actions):.2%}.
- Selector coverage: {selected_count/len(selections):.2%}; selected-action answer-drop rate {selected_answer_drops/max(1,selected_count):.2%}.
- Answer F1 delta: {payload['deltas']['answer_f1']:+.4f}.
- Supporting-fact F1 delta: {payload['deltas']['sp_f1']:+.4f}.
- Joint F1 delta: {payload['deltas']['joint_f1']:+.4f}.
- Frozen-transfer success rule: {payload['external_success_rule']['positive_and_non_degrading_frozen_transfer']}.

This is a cross-dataset zero-shot transfer result. No 2Wiki label, reader outcome, threshold, or coverage value was used for generation or selection.
"""
    (REPORTS / "external_2wiki_frozen_transfer_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
