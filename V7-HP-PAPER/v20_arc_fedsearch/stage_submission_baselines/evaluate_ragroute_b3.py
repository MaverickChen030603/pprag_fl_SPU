#!/usr/bin/env python3
"""Score B3 only after both frozen-Reader prediction files validate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluate_posthoc_baselines import CHAIN_METRICS, METRICS, paired_bootstrap, query_id, rows, sha256


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
READERS = ("flan", "unifiedqa")
METHOD = "b3_ragroute_protocol_adapted"


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for value in values for key in value})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--v16-eval", type=Path, required=True)
    parser.add_argument("--historical-per-query", type=Path, required=True)
    parser.add_argument("--b0-per-query", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contexts = list(rows(args.contexts))
    context_map = {(row["dataset"], row["query_id"]): row for row in contexts}
    if len(context_map) != 900 or len(context_map) != len(contexts):
        raise ValueError("B3 needs exactly one context for each frozen R5 query")
    if any(row.get("method") != METHOD or row.get("gold_or_answer_used") for row in contexts):
        raise ValueError("invalid B3 context information boundary")

    predictions = {}
    prediction_hashes = {}
    for reader in READERS:
        path = args.prediction_dir / f"{reader}_unscored.jsonl"
        marker = json.loads(path.with_suffix(".completed.json").read_text())
        values = list(rows(path))
        if len(values) != len(contexts) or marker["contexts_sha256"] != sha256(args.contexts) or marker["prediction_sha256"] != sha256(path) or marker["labels_loaded"] or marker["metrics_computed"]:
            raise ValueError(f"{reader}: Reader firewall validation failed")
        predictions.update({(reader, row["dataset"], row["query_id"]): row for row in values})
        prediction_hashes[reader] = sha256(path)
    if len(predictions) != len(contexts) * len(READERS):
        raise ValueError("incomplete B3 Reader predictions")

    # Labels are intentionally unavailable until routing contexts and both Reader
    # prediction files have passed the validation above.
    sys.path.insert(0, str(args.v16_eval))
    from eval_common import document_id, official_metrics
    sources = {dataset: {query_id(row): row for row in rows(args.sample_root / f"{dataset}_final_test_inputs_n300.jsonl")} for dataset in DATASETS}
    labels = {dataset: {query_id(row): row for row in rows(args.label_root / f"{dataset}_final_test_labels.jsonl")} for dataset in DATASETS}
    assignments = {dataset: {str(row["doc_id"]): int(row["client_id"]) for row in rows(args.assignment_root / dataset / "topic_silo_m20.jsonl")} for dataset in DATASETS}
    packets = {dataset: {str(row["query_id"]): row for row in rows(args.packet_root / f"{dataset}_probe_packets.jsonl")} for dataset in DATASETS}
    if any(set(sources[d]) != set(labels[d]) for d in DATASETS):
        raise ValueError("frozen R5 labels do not match inputs")

    def gold_documents(dataset: str, source: dict[str, Any], label: dict[str, Any]) -> set[str]:
        merged = {**source, **label}
        if dataset == "musique":
            supporting = {int(item["idx"]) for item in label.get("supporting_paragraphs", [])}
            return {document_id(dataset, str(item.get("title", "")), str(item.get("paragraph_text", ""))) for index, item in enumerate(source.get("paragraphs", [])) if int(item.get("idx", index)) in supporting}
        facts = label.get("supporting_facts", {})
        titles = facts.get("title", []) if isinstance(facts, dict) else [item[0] for item in facts if item]
        return {document_id(dataset, str(title)) for title in titles}

    per_query = []
    for dataset in DATASETS:
        for qid, source in sources[dataset].items():
            label, context, packet = labels[dataset][qid], context_map[(dataset, qid)], packets[dataset][qid]
            gold_docs = gold_documents(dataset, source, label)
            selected = {int(value) for value in context["selected_clients"]}
            gold_clients = {assignments[dataset][doc] for doc in gold_docs if doc in assignments[dataset]}
            local = {str(item["doc_id"]) for client in selected for item in packet["local_dense_docs_top10"][str(client)][:10]}
            def complete(values: set[str]) -> int:
                return int(bool(gold_docs) and gold_docs.issubset(values))
            scored = {**source, **label}
            if dataset == "musique":
                supporting = {int(item["idx"]) for item in label.get("supporting_paragraphs", [])}
                scored["paragraphs"] = [{**item, "is_supporting": int(item.get("idx", index)) in supporting} for index, item in enumerate(source.get("paragraphs", []))]
            for reader in READERS:
                prediction = predictions[(reader, dataset, qid)]
                per_query.append({
                    "dataset": dataset, "reader": reader, "query_id": qid, "method": METHOD,
                    "client_complete_at_3": int(bool(gold_clients) and gold_clients.issubset(selected)),
                    "local_complete_at_10": complete(local),
                    "transmitted_complete_at_15": complete(set(context["transmitted_doc_ids"])),
                    "merged_complete_at_10": complete(set(context["retrieved_doc_ids"])),
                    "reader_context_complete_at_5": complete(set(context["reader_context_doc_ids"])),
                    **official_metrics(prediction["predicted_answer"], scored, {tuple(item) for item in prediction["predicted_support"]}, dataset),
                })
    summary = []
    for dataset in DATASETS:
        for reader in READERS:
            values = [row for row in per_query if row["dataset"] == dataset and row["reader"] == reader]
            summary.append({"dataset": dataset, "reader": reader, "method": METHOD, "queries": len(values), **{metric: float(np.mean([float(row[metric]) for row in values])) for metric in (*CHAIN_METRICS, *METRICS)}})

    historical = {(row["reader"], row["dataset"], row["query_id"]): row for row in csv.DictReader(args.historical_per_query.open()) if row["method"] == "logistic_proberoute"}
    b0 = {(row["reader"], row["dataset"], row["query_id"]): row for row in csv.DictReader(args.b0_per_query.open()) if row["method"] == "b0_static_top3"}
    comparisons = []
    for dataset in DATASETS:
        for reader in READERS:
            values = [row for row in per_query if row["dataset"] == dataset and row["reader"] == reader]
            for metric in ("reader_context_complete_at_5", "answer_f1", "sp_f1", "joint_f1"):
                b3_values = np.asarray([float(row[metric]) for row in values])
                for name, reference in (("b3_minus_m2", historical), ("b3_minus_b0", b0)):
                    delta = b3_values - np.asarray([float(reference[(reader, dataset, row["query_id"])][metric]) for row in values])
                    seed = int(hashlib.sha256(f"{name}|{dataset}|{reader}|{metric}".encode()).hexdigest()[:8], 16)
                    comparisons.append({"dataset": dataset, "reader": reader, "comparison": name, "metric": metric, "queries": len(values), **paired_bootstrap(delta, seed)})
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "per_query_results.csv", per_query)
    write_csv(args.output_dir / "method_results.csv", summary)
    write_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    manifest = {"status": "complete_posthoc_b3_evaluation", "method": METHOD, "rows": len(per_query), "contexts_sha256": sha256(args.contexts), "prediction_sha256": prediction_hashes, "evidence_role": "retrospective_post_hoc_r5_revealed", "claim_restriction": "diagnostic_only_not_fresh_confirmation"}
    (args.output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
