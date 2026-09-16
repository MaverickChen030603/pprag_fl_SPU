#!/usr/bin/env python3
"""Score frozen B0/B1 predictions after validating the unlabeled artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
READERS = ("flan", "unifiedqa")
METRICS = ("answer_f1", "sp_f1", "joint_f1", "answer_em", "sp_em", "joint_em")
CHAIN_METRICS = (
    "client_complete_at_3",
    "local_complete_at_10",
    "transmitted_complete_at_15",
    "merged_complete_at_10",
    "reader_context_complete_at_5",
)


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in values for key in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def paired_bootstrap(delta: np.ndarray, seed: int) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    count = len(delta)
    samples = np.asarray(
        [delta[generator.integers(0, count, count)].mean() for _ in range(5000)]
    )
    low_tail = int((samples <= 0).sum())
    high_tail = int((samples >= 0).sum())
    return {
        "mean_delta": float(delta.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "two_sided_p": min(
            1.0, 2.0 * (min(low_tail, high_tail) + 1) / (len(samples) + 1)
        ),
    }


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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    contexts = list(rows(args.contexts))
    context_map = {
        (row["dataset"], row["method"], row["query_id"]): row for row in contexts
    }
    if len(context_map) != len(contexts) or any(
        row.get("gold_or_answer_used") for row in contexts
    ):
        raise ValueError("invalid unlabeled contexts")
    methods = sorted({row["method"] for row in contexts})
    expected_methods = ["b0_static_top3"] + [
        f"b1_random_top3_seed_{seed:02d}" for seed in range(20)
    ]
    if methods != sorted(expected_methods):
        raise ValueError("method set differs from frozen preregistration")

    predictions = {}
    prediction_hashes = {}
    for reader in READERS:
        path = args.prediction_dir / f"{reader}_unscored.jsonl"
        marker = json.loads(path.with_suffix(".completed.json").read_text())
        values = list(rows(path))
        if (
            len(values) != len(contexts)
            or marker["prediction_sha256"] != sha256(path)
            or marker["contexts_sha256"] != sha256(args.contexts)
            or marker["labels_loaded"]
            or marker["metrics_computed"]
        ):
            raise ValueError(f"prediction firewall failed for {reader}")
        for row in values:
            key = (reader, row["dataset"], row["method"], row["query_id"])
            if key in predictions:
                raise ValueError(f"duplicate prediction key: {key}")
            predictions[key] = row
        prediction_hashes[reader] = sha256(path)
    expected_prediction_count = len(contexts) * len(READERS)
    if len(predictions) != expected_prediction_count:
        raise ValueError("prediction key set is incomplete")

    # Labels are not opened until all contexts and both Reader outputs validate.
    sys.path.insert(0, str(args.v16_eval))
    from eval_common import document_id, official_metrics

    sources = {
        dataset: {
            query_id(row): row
            for row in rows(
                args.sample_root / f"{dataset}_final_test_inputs_n300.jsonl"
            )
        }
        for dataset in DATASETS
    }
    labels = {}
    for dataset in DATASETS:
        selected = set(sources[dataset])
        labels[dataset] = {
            query_id(row): row
            for row in rows(args.label_root / f"{dataset}_final_test_labels.jsonl")
            if query_id(row) in selected
        }
        if set(labels[dataset]) != selected:
            raise ValueError(f"label/sample mismatch for {dataset}")
    assignments = {
        dataset: {
            str(row["doc_id"]): int(row["client_id"])
            for row in rows(
                args.assignment_root / dataset / "topic_silo_m20.jsonl"
            )
        }
        for dataset in DATASETS
    }
    packets = {
        dataset: {
            str(row["query_id"]): row
            for row in rows(args.packet_root / f"{dataset}_probe_packets.jsonl")
        }
        for dataset in DATASETS
    }

    def gold_row(dataset: str, qid: str) -> dict[str, Any]:
        row = dict(sources[dataset][qid])
        label = labels[dataset][qid]
        row.update(label)
        if dataset == "musique":
            indices = {
                int(value["idx"]) for value in label.get("supporting_paragraphs", [])
            }
            row["paragraphs"] = [
                {
                    **paragraph,
                    "is_supporting": int(paragraph.get("idx", index)) in indices,
                }
                for index, paragraph in enumerate(row.get("paragraphs", []))
            ]
        return row

    def gold_documents(dataset: str, row: dict[str, Any]) -> set[str]:
        if dataset == "musique":
            return {
                document_id(
                    dataset,
                    str(paragraph.get("title", "")),
                    str(paragraph.get("paragraph_text", "")),
                )
                for paragraph in row.get("paragraphs", [])
                if paragraph.get("is_supporting")
            }
        facts = row.get("supporting_facts", {})
        titles = (
            facts.get("title", [])
            if isinstance(facts, dict)
            else [value[0] for value in facts]
        )
        return {document_id(dataset, str(title)) for title in titles}

    per_query = []
    for reader in READERS:
        for dataset in DATASETS:
            for qid in sources[dataset]:
                gold = gold_row(dataset, qid)
                support_ids = gold_documents(dataset, gold)
                gold_clients = {
                    assignments[dataset][document]
                    for document in support_ids
                    if document in assignments[dataset]
                }
                packet = packets[dataset][qid]
                for method in expected_methods:
                    context = context_map[(dataset, method, qid)]
                    prediction = predictions[(reader, dataset, method, qid)]
                    official = official_metrics(
                        prediction["predicted_answer"],
                        gold,
                        {tuple(value) for value in prediction["predicted_support"]},
                        dataset,
                    )
                    selected = set(map(int, context["selected_clients"]))
                    local = {
                        str(document["doc_id"])
                        for client in selected
                        for document in packet["local_dense_docs_top10"][str(client)][:10]
                    }

                    def complete(values: set[str]) -> int:
                        return int(bool(support_ids) and support_ids.issubset(values))

                    per_query.append(
                        {
                            "dataset": dataset,
                            "reader": reader,
                            "query_id": qid,
                            "method": method,
                            "client_complete_at_3": int(
                                bool(gold_clients) and gold_clients.issubset(selected)
                            ),
                            "local_complete_at_10": complete(local),
                            "transmitted_complete_at_15": complete(
                                set(context["transmitted_doc_ids"])
                            ),
                            "merged_complete_at_10": complete(
                                set(context["retrieved_doc_ids"])
                            ),
                            "reader_context_complete_at_5": complete(
                                set(context["reader_context_doc_ids"])
                            ),
                            **official,
                        }
                    )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_query:
        grouped[(row["dataset"], row["reader"], row["method"])].append(row)
    main_rows = []
    for (dataset, reader, method), values in sorted(grouped.items()):
        main_rows.append(
            {
                "dataset": dataset,
                "reader": reader,
                "method": method,
                "queries": len(values),
                **{
                    metric: float(np.mean([float(row[metric]) for row in values]))
                    for metric in (*CHAIN_METRICS, *METRICS)
                },
            }
        )

    random_aggregate = []
    for dataset in DATASETS:
        for reader in READERS:
            random_rows = [
                row
                for row in main_rows
                if row["dataset"] == dataset
                and row["reader"] == reader
                and row["method"].startswith("b1_random")
            ]
            payload = {"dataset": dataset, "reader": reader, "seeds": 20}
            for metric in (*CHAIN_METRICS, *METRICS):
                values = [float(row[metric]) for row in random_rows]
                payload[f"{metric}_mean"] = statistics.mean(values)
                payload[f"{metric}_seed_std"] = statistics.stdev(values)
                payload[f"{metric}_seed_min"] = min(values)
                payload[f"{metric}_seed_max"] = max(values)
            random_aggregate.append(payload)

    historical = list(csv.DictReader(args.historical_per_query.open()))
    historical_m2 = {
        (row["reader"], row["dataset"], row["query_id"]): row
        for row in historical
        if row["method"] == "logistic_proberoute"
    }
    comparisons = []
    for dataset in DATASETS:
        for reader in READERS:
            order = list(sources[dataset])
            current = {
                (row["method"], row["query_id"]): row
                for row in per_query
                if row["dataset"] == dataset and row["reader"] == reader
            }
            for metric in ("reader_context_complete_at_5", "answer_f1", "sp_f1", "joint_f1"):
                m2 = np.asarray(
                    [
                        float(historical_m2[(reader, dataset, qid)][metric])
                        for qid in order
                    ]
                )
                b0 = np.asarray(
                    [float(current[("b0_static_top3", qid)][metric]) for qid in order]
                )
                random_mean = np.asarray(
                    [
                        np.mean(
                            [
                                float(
                                    current[
                                        (f"b1_random_top3_seed_{seed:02d}", qid)
                                    ][metric]
                                )
                                for seed in range(20)
                            ]
                        )
                        for qid in order
                    ]
                )
                for baseline, values in (("b0_static_top3", b0), ("b1_random_20seed_mean", random_mean)):
                    seed = int(
                        hashlib.sha256(
                            f"posthoc|{dataset}|{reader}|{baseline}|{metric}".encode()
                        ).hexdigest()[:8],
                        16,
                    )
                    comparisons.append(
                        {
                            "dataset": dataset,
                            "reader": reader,
                            "comparison": f"logistic_proberoute_minus_{baseline}",
                            "metric": metric,
                            "queries": len(order),
                            **paired_bootstrap(m2 - values, seed),
                        }
                    )

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "per_query_results.csv", per_query)
    write_csv(args.output_dir / "method_results.csv", main_rows)
    write_csv(args.output_dir / "random_20seed_aggregate.csv", random_aggregate)
    write_csv(args.output_dir / "m2_posthoc_comparisons.csv", comparisons)
    manifest = {
        "status": "complete_posthoc_evaluation",
        "evidence_role": "retrospective_post_hoc_r5_revealed",
        "claim_restriction": "diagnostic_only_not_fresh_confirmation",
        "contexts_sha256": sha256(args.contexts),
        "prediction_sha256": prediction_hashes,
        "rows": len(per_query),
        "methods": expected_methods,
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    report = [
        "# R5 Revealed Post-hoc B0/B1 Baseline Report",
        "",
        "This report is diagnostic evidence only and is not a fresh confirmation run.",
        "",
        "## M2 minus baseline paired results",
        "",
    ]
    for row in comparisons:
        if row["metric"] == "joint_f1":
            report.append(
                f"- {row['dataset']} / {row['reader']} / {row['comparison']}: "
                f"{row['mean_delta']:+.4f} [{row['ci_low']:+.4f}, "
                f"{row['ci_high']:+.4f}], p={row['two_sided_p']:.4g}"
            )
    (args.output_dir / "posthoc_baseline_report.md").write_text(
        "\n".join(report) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
