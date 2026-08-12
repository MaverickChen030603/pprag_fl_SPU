#!/usr/bin/env python3
"""Validate every unscored prediction, then open labels exactly once and evaluate R5."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
READERS = ("flan", "unifiedqa")
METHODS = ("federated_baseline", "label_free_proberoute", "logistic_proberoute", "centralized_retrieval_reference")
COMPARISONS = (("logistic_proberoute", "federated_baseline"), ("label_free_proberoute", "federated_baseline"), ("logistic_proberoute", "label_free_proberoute"))
METRICS = ("answer_f1", "sp_f1", "joint_f1", "answer_em", "sp_em", "joint_em")


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row):
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in values for key in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(values)


def bootstrap(delta: np.ndarray, seed: int):
    rng = np.random.default_rng(seed); n = len(delta)
    sampled = np.asarray([delta[rng.integers(0, n, n)].mean() for _ in range(5000)])
    low_tail, high_tail = int((sampled <= 0).sum()), int((sampled >= 0).sum())
    p = min(1.0, 2.0 * (min(low_tail, high_tail) + 1) / (len(sampled) + 1))
    return float(delta.mean()), float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975)), p


def apply_bh(values):
    candidates = [row for row in values if row["metric"] != "joint_f1"]
    ordered = sorted(candidates, key=lambda row: row["two_sided_p"])
    running, total = 1.0, len(ordered)
    for rank, row in reversed(list(enumerate(ordered, 1))):
        running = min(running, row["two_sided_p"] * total / rank); row["bh_fdr_q"] = running
    for row in values:
        row.setdefault("bh_fdr_q", "not_applicable_primary")


def transition(base, probe):
    return {(0, 1): "T1_rescue", (1, 1): "T2_preserved", (1, 0): "T3_harm", (0, 0): "T4_persistent_failure"}[base, probe]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--v17", type=Path, required=True)
    parser.add_argument("--v16-eval", type=Path, required=True)
    parser.add_argument("--r4-main", type=Path, required=True)
    parser.add_argument("--r4-bootstrap", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if (run / "reports/r5_final_decision.json").exists():
        raise FileExistsError("R5 already evaluated; one-shot contract forbids rerun")
    contexts_path = run / "retrieval/retrieval_outputs_unlabeled.jsonl"
    contexts = list(rows(contexts_path))
    if len(contexts) != 3600 or any(row.get("gold_or_answer_used") for row in contexts):
        raise ValueError("invalid unlabeled contexts")
    context_map = {(row["dataset"], row["method"], row["query_id"]): row for row in contexts}
    expected = {(d, m, q) for d in DATASETS for m in METHODS for q in [qid(row) for row in rows(run / f"protocol/{d}_final_test_inputs_n300.jsonl")]}
    if set(context_map) != expected:
        raise ValueError("context key mismatch")
    predictions = {}
    pre_unseal = {"validated_at": datetime.now(timezone.utc).isoformat(), "labels_opened": False, "contexts_sha256": sha256(contexts_path), "readers": {}}
    for reader in READERS:
        path = run / f"reader_predictions/{reader}_unscored.jsonl"
        marker = json.loads(path.with_suffix(".completed.json").read_text())
        values = list(rows(path))
        if len(values) != 3600 or marker["prediction_sha256"] != sha256(path) or marker["labels_loaded"] or marker["metrics_computed"]:
            raise ValueError(f"unsealed prediction validation failed: {reader}")
        keys = {(row["dataset"], row["method"], row["query_id"]) for row in values}
        if keys != expected or any(row.get("labels_loaded") or row.get("metrics_computed") for row in values):
            raise ValueError(f"prediction key/firewall mismatch: {reader}")
        for row in values:
            predictions[(reader, row["dataset"], row["method"], row["query_id"])] = row
        pre_unseal["readers"][reader] = {"rows": len(values), "sha256": sha256(path), "marker_sha256": sha256(path.with_suffix(".completed.json"))}
    checks = run / "checksums/pre_unseal_prediction_manifest.json"
    checks.write_text(json.dumps(pre_unseal, indent=2) + "\n")

    # Label files are first opened below, after both readers and all methods validated.
    unseal_time = datetime.now(timezone.utc).isoformat()
    sys.path.insert(0, str(args.v16_eval))
    from eval_common import document_id, normalize_title, official_metrics
    sources, labels = {}, {}
    for dataset in DATASETS:
        sources[dataset] = {qid(row): row for row in rows(run / f"protocol/{dataset}_final_test_inputs_n300.jsonl")}
        selected = set(sources[dataset])
        labels[dataset] = {qid(row): row for row in rows(args.v17 / f"data/sealed/{dataset}_final_test_labels.jsonl") if qid(row) in selected}
        if set(labels[dataset]) != selected:
            raise ValueError(f"label/sample mismatch: {dataset}")

    def gold_row(dataset, query_id):
        row = dict(sources[dataset][query_id]); label = labels[dataset][query_id]; row.update(label)
        if dataset == "musique":
            indices = {int(value["idx"]) for value in label.get("supporting_paragraphs", [])}
            row["paragraphs"] = [{**paragraph, "is_supporting": int(paragraph.get("idx", index)) in indices} for index, paragraph in enumerate(row.get("paragraphs", []))]
        return row

    def gold_docs(dataset, row):
        if dataset == "musique":
            return {document_id(dataset, str(p.get("title", "")), str(p.get("paragraph_text", ""))) for p in row.get("paragraphs", []) if p.get("is_supporting")}
        facts = row.get("supporting_facts", {})
        titles = facts.get("title", []) if isinstance(facts, dict) else [value[0] for value in facts]
        return {document_id(dataset, str(title)) for title in titles}

    assignments = {dataset: {str(row["doc_id"]): int(row["client_id"]) for row in rows(args.v17 / f"partitions/assignments/{dataset}/topic_silo_m20.jsonl")} for dataset in DATASETS}
    packets = {}
    for dataset in DATASETS:
        packets[dataset] = {str(row["query_id"]): row for row in rows(run / f"retrieval/{dataset}_probe_packets.jsonl")}
    per_query = []
    for reader in READERS:
        for dataset in DATASETS:
            for query_id in sources[dataset]:
                gold = gold_row(dataset, query_id); support_ids = gold_docs(dataset, gold)
                gold_clients = {assignments[dataset][doc] for doc in support_ids if doc in assignments[dataset]}
                for method in METHODS:
                    context = context_map[(dataset, method, query_id)]
                    prediction = predictions[(reader, dataset, method, query_id)]
                    support_prediction = {tuple(value) for value in prediction["predicted_support"]}
                    metric = official_metrics(prediction["predicted_answer"], gold, support_prediction, dataset)
                    transmitted = set(context["transmitted_doc_ids"]); merged = set(context["retrieved_doc_ids"]); visible = set(context["reader_context_doc_ids"])
                    selected = set(map(int, context["selected_clients"]))
                    local = set()
                    if method != "centralized_retrieval_reference":
                        packet = packets[dataset][query_id]
                        local = {str(doc["doc_id"]) for client in selected for doc in packet["local_dense_docs_top10"][str(client)][:10]}
                    complete = lambda values: int(bool(support_ids) and support_ids.issubset(values))
                    per_query.append({"dataset": dataset, "reader": reader, "query_id": query_id, "method": method,
                                      "client_complete_at_3": "" if method == "centralized_retrieval_reference" else int(bool(gold_clients) and gold_clients.issubset(selected)),
                                      "local_complete_at_10": "" if method == "centralized_retrieval_reference" else complete(local),
                                      "transmitted_complete_at_15": complete(transmitted), "merged_complete_at_10": complete(merged),
                                      "reader_context_complete_at_5": complete(visible), **metric})

    grouped = defaultdict(list)
    for row in per_query:
        grouped[(row["dataset"], row["reader"], row["method"])].append(row)
    main_rows = []
    for key, values in sorted(grouped.items()):
        dataset, reader, method = key
        metrics = {name: float(np.mean([float(row[name]) for row in values])) for name in METRICS}
        chain = {name: "" if method == "centralized_retrieval_reference" and name in {"client_complete_at_3", "local_complete_at_10"} else float(np.mean([float(row[name]) for row in values])) for name in ("client_complete_at_3", "local_complete_at_10", "transmitted_complete_at_15", "merged_complete_at_10", "reader_context_complete_at_5")}
        main_rows.append({"dataset": dataset, "reader": reader, "method": method, "queries": len(values), **chain, **metrics})
    comparisons = []
    for dataset in DATASETS:
        for reader in READERS:
            by_method = {method: {row["query_id"]: row for row in grouped[(dataset, reader, method)]} for method in METHODS}
            order = list(sources[dataset])
            for method_a, method_b in COMPARISONS:
                for metric in ("answer_f1", "sp_f1", "joint_f1"):
                    delta = np.asarray([float(by_method[method_a][query][metric]) - float(by_method[method_b][query][metric]) for query in order])
                    seed = int(hashlib.sha256(f"r5|{dataset}|{reader}|{method_a}|{method_b}|{metric}".encode()).hexdigest()[:8], 16)
                    mean, low, high, p = bootstrap(delta, seed)
                    comparisons.append({"dataset": dataset, "reader": reader, "comparison": f"{method_a}_vs_{method_b}", "method_a": method_a, "method_b": method_b, "metric": metric, "queries": len(delta), "mean_delta": mean, "ci_low": low, "ci_high": high, "two_sided_p": p, "paired_win": int((delta > 0).sum()), "paired_tie": int((delta == 0).sum()), "paired_loss": int((delta < 0).sum())})
    apply_bh(comparisons)

    transitions = []
    for dataset in DATASETS:
        for reader in READERS:
            lookup = {(row["method"], row["query_id"]): row for method in METHODS for row in grouped[(dataset, reader, method)]}
            for probe in ("label_free_proberoute", "logistic_proberoute"):
                buckets = defaultdict(list)
                for query_id in sources[dataset]:
                    base, candidate = lookup[("federated_baseline", query_id)], lookup[(probe, query_id)]
                    state = transition(int(base["reader_context_complete_at_5"]), int(candidate["reader_context_complete_at_5"]))
                    buckets[state].append((base, candidate))
                for state, pairs in buckets.items():
                    transitions.append({"dataset": dataset, "reader": reader, "comparison": f"{probe}_vs_federated_baseline", "support_transition": state, "n": len(pairs), **{f"delta_{metric}": float(np.mean([float(b[metric]) - float(a[metric]) for a, b in pairs])) for metric in ("answer_f1", "sp_f1", "joint_f1")}})

    gap = []
    for dataset in DATASETS:
        for reader in READERS:
            lookup = {row["method"]: row for row in main_rows if row["dataset"] == dataset and row["reader"] == reader}
            for method in ("label_free_proberoute", "logistic_proberoute"):
                for metric in ("reader_context_complete_at_5", "sp_f1", "joint_f1"):
                    denominator = float(lookup["centralized_retrieval_reference"][metric]) - float(lookup["federated_baseline"][metric])
                    gap.append({"dataset": dataset, "reader": reader, "method": method, "metric": metric, "gap_recovery": "N/A" if denominator <= 0 else (float(lookup[method][metric]) - float(lookup["federated_baseline"][metric])) / denominator})

    r4 = {(row["dataset"], row["reader"]): row for row in csv.DictReader(args.r4_bootstrap.open()) if row["method_a"] == "logistic_proberoute" and row["method_b"] == "federated_baseline" and row["metric"] == "joint_f1"}
    consistency = []
    primary = [row for row in comparisons if row["method_a"] == "logistic_proberoute" and row["method_b"] == "federated_baseline" and row["metric"] == "joint_f1"]
    for row in primary:
        old = r4[(row["dataset"], row["reader"])]
        consistency.append({"dataset": row["dataset"], "reader": row["reader"], "r4_delta_joint": float(old["mean_delta"]), "r5_delta_joint": row["mean_delta"], "same_direction": (float(old["mean_delta"]) >= 0) == (row["mean_delta"] >= 0)})
    by_dataset = defaultdict(list)
    for row in primary: by_dataset[row["dataset"]].append(row)
    positive_datasets = sum(np.mean([value["mean_delta"] for value in values]) > 0 for values in by_dataset.values())
    macro_joint = float(np.mean([row["mean_delta"] for row in primary]))
    clearly_negative = any(all(row["ci_high"] < 0 for row in values) for values in by_dataset.values())
    primary_transitions = [row for row in transitions if row["comparison"].startswith("logistic")]
    rescue = [row for row in primary_transitions if row["support_transition"] == "T1_rescue"]
    harm = [row for row in primary_transitions if row["support_transition"] == "T3_harm"]
    # SP is shared across readers, so count each dataset once via the FLAN rows.
    rescue_n = sum(int(row["n"]) for row in rescue if row["reader"] == "flan")
    harm_n = sum(int(row["n"]) for row in harm if row["reader"] == "flan")
    rescue_joint = float(np.mean([row["delta_joint_f1"] for row in rescue])) if rescue else 0.0
    answer_primary = [row for row in comparisons if row["method_a"] == "logistic_proberoute" and row["method_b"] == "federated_baseline" and row["metric"] == "answer_f1"]
    systematic_answer_harm = np.mean([row["mean_delta"] for row in answer_primary]) < 0 and sum(row["mean_delta"] < 0 for row in answer_primary) >= 4
    if positive_datasets >= 2 and macro_joint > 0 and not clearly_negative and rescue_n > harm_n and rescue_joint > 0 and not systematic_answer_harm:
        decision = "final_test_strongly_confirmed"
    elif positive_datasets >= 2 and macro_joint > 0 and not clearly_negative and not systematic_answer_harm:
        decision = "final_test_partially_confirmed"
    elif macro_joint < 0 and sum(all(row["mean_delta"] < 0 for row in values) for values in by_dataset.values()) >= 2:
        decision = "final_test_contradiction"
    else:
        decision = "final_test_mixed_generalization"

    write_csv(run / "retrieval/final_test_retrieval_results.csv", [{key: value for key, value in row.items() if key not in METRICS} for row in per_query if row["reader"] == "flan"])
    write_csv(run / "statistics/main_final_test_results.csv", main_rows)
    write_csv(run / "statistics/paired_bootstrap.csv", comparisons)
    write_csv(run / "statistics/bh_secondary_tests.csv", comparisons)
    write_csv(run / "statistics/per_query_results.csv", per_query)
    write_csv(run / "statistics/r4_r5_consistency.csv", consistency)
    write_csv(run / "mechanism/support_transition_analysis.csv", transitions)
    write_csv(run / "mechanism/reader_gain_given_support_rescue.csv", rescue)
    write_csv(run / "gap_recovery/final_gap_recovery.csv", gap)
    (run / "protocol/label_unseal_record.json").write_text(json.dumps({"label_unseal_timestamp": unseal_time, "pre_unseal_manifest_sha256": sha256(checks), "all_predictions_complete_before_unseal": True, "rerun_forbidden": True}, indent=2) + "\n")
    final = {"status": decision, "lifecycle_status": "v20_empirical_evaluation_complete", "r5_macro_joint_delta": macro_joint, "r4_macro_joint_delta": float(np.mean([row["r4_delta_joint"] for row in consistency])), "positive_datasets": positive_datasets, "support_rescue_count": rescue_n, "support_harm_count": harm_n, "support_rescue_mean_joint_delta": rescue_joint, "systematic_answer_harm": systematic_answer_harm, "final_test_kind": "V17 train-derived untouched held-out split", "method_development_closed": True}
    (run / "reports/r5_final_decision.json").write_text(json.dumps(final, indent=2) + "\n")
    lines = ["# V20 R5 One-Shot Final-Test Report", "", f"**Decision:** `{decision}`", "", f"Labels were unsealed once at `{unseal_time}` after all 7,200 predictions passed checksum validation.", "", "## Primary Joint F1", ""]
    for row in primary:
        lines.append(f"- {row['dataset']} / {row['reader']}: {row['mean_delta']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}], p={row['two_sided_p']:.4g}, W/T/L={row['paired_win']}/{row['paired_tie']}/{row['paired_loss']}")
    lines += ["", f"R4 macro delta: {final['r4_macro_joint_delta']:+.4f}; R5 macro delta: {macro_joint:+.4f}.", f"Support rescue/harm: {rescue_n}/{harm_n}; rescue mean Joint delta: {rescue_joint:+.4f}.", "", "SP uses the shared frozen V16 support predictor and is context-level, not an independent cross-reader replication."]
    (run / "reports/r5_final_test_report.md").write_text("\n".join(lines) + "\n")
    claims = f"""# V20 Paper Claim Freeze

## Tier 1 - Fully Supported

- ProbeRoute improves federated resource selection under the frozen Bc=3 and 15-document budget.
- ProbeRoute improves complete multi-hop evidence access.
- ProbeRoute improves downstream Joint F1 under frozen readers when R4 and R5 show aligned positive direction.

## Tier 2 - Qualified

- Final confirmation status: `{decision}`; R5 is a train-derived untouched held-out confirmation, not an official hidden-test result.
- Logistic is a lightweight supervised enhancement; its advantage over label-free ProbeRoute must be stated only where directly supported.

## Tier 3 - Unsupported

- ProbeRoute is always better or guarantees no harm.
- Centralized retrieval is an upper bound.
- Logistic is significantly better than label-free everywhere.
- ProbeRoute has zero extra cost or formal privacy/security guarantees.

V20 method development is permanently closed. Only paper writing, visualization, re-statistics of frozen results, and reproducibility packaging remain.
"""
    (run / "reports/paper_claim_freeze.md").write_text(claims)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
