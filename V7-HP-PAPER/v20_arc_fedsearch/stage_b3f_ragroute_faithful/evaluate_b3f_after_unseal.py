#!/usr/bin/env python3
"""One-shot post-unseal evaluation for the frozen B3F RAGRoute comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
READERS = ("flan", "unifiedqa")
METHODS = ("ragroute_fixed3", "ragroute_original_threshold")
METRICS = (
    "client_complete_selected",
    "local_complete_at_10",
    "transmitted_complete",
    "merged_complete_at_10",
    "reader_context_complete_at_5",
    "answer_f1",
)


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in values for key in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def source_path(root: Path, dataset: str) -> Path:
    names = {
        "hotpotqa": "hf_hotpot/hotpot_train_v1.json",
        "2wikimultihopqa": "2wikimultihop_train.json",
        "musique": "musique_ans_v1.0_train.jsonl",
    }
    return root / names[dataset]


def load_raw(dataset: str, root: Path, selected: set[str]) -> dict[str, dict[str, Any]]:
    path = source_path(root, dataset)
    if path.suffix == ".jsonl":
        source = (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
    else:
        source = iter(json.loads(path.read_text(encoding="utf-8")))
    found = {qid(row): row for row in source if qid(row) in selected}
    if set(found) != selected:
        raise ValueError(f"{dataset}: unsealed rows do not exactly match frozen B3F IDs")
    return found


def bootstrap(delta: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(5000)])
    return {
        "absolute_delta": float(delta.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()

    context_file = args.stage / "contexts/b3f_ragroute_contexts_unscored.jsonl"
    prediction_files = {
        reader: args.stage / f"predictions/{reader}_b3f_unscored.jsonl" for reader in READERS
    }
    output = args.stage / "statistics"
    if output.exists():
        raise FileExistsError("B3F post-unseal statistics already exist")
    contexts = list(rows(context_file))
    context_map = {(row["dataset"], row["query_id"], row["method"]): row for row in contexts}
    if len(contexts) != 3000 or len(context_map) != 3000:
        raise ValueError("frozen B3F context artifact is incomplete")
    predictions: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for reader, path in prediction_files.items():
        values = list(rows(path))
        if len(values) != 3000:
            raise ValueError(f"{reader}: frozen prediction artifact is incomplete")
        for row in values:
            if row.get("labels_loaded") or row.get("metrics_computed"):
                raise ValueError(f"{reader}: prediction firewall metadata is invalid")
            predictions[(reader, row["dataset"], row["query_id"], row["method"])] = row
    if len(predictions) != 6000:
        raise ValueError("duplicate B3F prediction keys")

    # This file makes the user-authorized transition and preserves pre-score hashes.
    protocol = args.stage / "protocol"
    record = protocol / "b3f_label_unseal_record.json"
    if record.exists():
        raise FileExistsError("B3F label unseal was already recorded")
    record.write_text(
        json.dumps(
            {
                "stage": "V20-B3F-Fresh",
                "authorization": "UNSEAL B3F LABELS",
                "contexts_sha256": sha256(context_file),
                "prediction_sha256": {reader: sha256(path) for reader, path in prediction_files.items()},
                "labels_opened_for_post_unseal_scoring": True,
                "supported_metrics": list(METRICS),
                "unsupported_metrics": ["sp_f1", "joint_f1"],
                "reason": "blind reader runner intentionally emitted answer-only predictions to avoid opening source rows before unseal",
            },
            indent=2,
        )
        + "\n"
    )

    sys.path.insert(0, str(args.experiment_root / "inputs/v16_evaluation"))
    from eval_common import document_id, official_metrics

    selected = {
        dataset: {str(row["query_id"]) for row in rows(args.stage / f"inputs/{dataset}_blind_n500.jsonl")}
        for dataset in DATASETS
    }
    gold = {
        dataset: load_raw(dataset, args.experiment_root / "public_training_sources", selected[dataset])
        for dataset in DATASETS
    }
    assignments = {
        dataset: {
            str(row["doc_id"]): int(row["client_id"])
            for row in rows(
                args.experiment_root
                / f"inputs/v17/partitions/assignments/{dataset}/topic_silo_m20.jsonl"
            )
        }
        for dataset in DATASETS
    }
    packets = {
        dataset: {
            str(row["query_id"]): row
            for row in rows(args.stage / f"retrieval/{dataset}_probe_packets.jsonl")
        }
        for dataset in DATASETS
    }

    per_query: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for query_id, source in gold[dataset].items():
            if dataset == "musique":
                support = {
                    document_id(dataset, str(item.get("title", "")), str(item.get("paragraph_text", "")))
                    for item in source.get("paragraphs", [])
                    if item.get("is_supporting", item.get("is_support", False))
                }
            else:
                facts = source.get("supporting_facts", [])
                titles = facts.get("title", []) if isinstance(facts, dict) else [item[0] for item in facts]
                support = {document_id(dataset, str(title)) for title in titles}
            support_clients = {assignments[dataset][doc] for doc in support if doc in assignments[dataset]}
            for method in METHODS:
                context = context_map[(dataset, query_id, method)]
                selected_clients = set(map(int, context["selected_client_ids"]))
                local = {
                    str(item["doc_id"])
                    for client in selected_clients
                    for item in packets[dataset][query_id]["local_dense_docs_top10"][str(client)][:10]
                }
                complete = lambda docs: int(bool(support) and support.issubset(docs))
                chain = {
                    "client_complete_selected": int(bool(support_clients) and support_clients.issubset(selected_clients)),
                    "local_complete_at_10": complete(local),
                    "transmitted_complete": complete(set(context["transmitted_doc_ids"])),
                    "merged_complete_at_10": complete(set(context["merged_top10_doc_ids"])),
                    "reader_context_complete_at_5": complete(set(context["reader_top5_doc_ids"])),
                    "selected_clients": len(selected_clients),
                    "transmitted_documents": len(context["transmitted_doc_ids"]),
                }
                for reader in READERS:
                    prediction = predictions[(reader, dataset, query_id, method)]
                    answer_f1 = official_metrics(
                        prediction["predicted_answer"], source, set(), dataset
                    )["answer_f1"]
                    per_query.append(
                        {
                            "dataset": dataset,
                            "reader": reader,
                            "query_id": query_id,
                            "method": method,
                            **chain,
                            "answer_f1": answer_f1,
                        }
                    )

    output.mkdir()
    write_csv(output / "per_query_results.csv", per_query)
    summary = []
    for dataset in DATASETS:
        for reader in READERS:
            for method in METHODS:
                values = [
                    row for row in per_query
                    if row["dataset"] == dataset and row["reader"] == reader and row["method"] == method
                ]
                summary.append(
                    {
                        "dataset": dataset,
                        "reader": reader,
                        "method": method,
                        "queries": len(values),
                        **{metric: float(np.mean([row[metric] for row in values])) for metric in METRICS},
                        "mean_selected_clients": float(np.mean([row["selected_clients"] for row in values])),
                        "mean_transmitted_documents": float(np.mean([row["transmitted_documents"] for row in values])),
                    }
                )
    write_csv(output / "method_results.csv", summary)

    comparisons = []
    for dataset in DATASETS:
        ordered = sorted(selected[dataset])
        for reader in READERS:
            table = {
                (row["method"], row["query_id"]): row
                for row in per_query if row["dataset"] == dataset and row["reader"] == reader
            }
            for metric in ("reader_context_complete_at_5", "answer_f1"):
                delta = np.asarray(
                    [
                        table[("ragroute_original_threshold", query_id)][metric]
                        - table[("ragroute_fixed3", query_id)][metric]
                        for query_id in ordered
                    ],
                    dtype=float,
                )
                seed = int(hashlib.sha256(f"{dataset}|{reader}|{metric}".encode()).hexdigest()[:8], 16)
                comparisons.append(
                    {
                        "dataset": dataset,
                        "reader": reader,
                        "comparison": "original_threshold_minus_fixed3",
                        "metric": metric,
                        "queries": len(delta),
                        **bootstrap(delta, seed),
                    }
                )
    write_csv(output / "paired_comparisons.csv", comparisons)
    manifest = {
        "status": "complete_b3f_post_unseal_evaluation",
        "unseal_record_sha256": sha256(record),
        "contexts_sha256": sha256(context_file),
        "prediction_sha256": {reader: sha256(path) for reader, path in prediction_files.items()},
        "rows": len(per_query),
        "metrics": list(METRICS),
        "not_computed": ["sp_f1", "joint_f1"],
        "methods": list(METHODS),
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
