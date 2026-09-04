#!/usr/bin/env python3
"""Offline gold evaluator for frozen R2-B0 candidate and retrieval artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def csv_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def normalize(value: str) -> str:
    return " ".join(str(value).lower().split())


def docid(dataset: str, title: str, text: str = "") -> str:
    key = normalize(title) if dataset != "musique" else normalize(title) + "\n" + normalize(text)
    return f"{dataset}:{hashlib.sha1(key.encode()).hexdigest()[:20]}"


def supports(row: dict[str, Any], dataset: str) -> tuple[set[str], set[str]]:
    if dataset == "musique":
        values = [
            (docid(dataset, item.get("title", ""), item.get("paragraph_text", "")), normalize(item.get("title", "")))
            for item in row.get("paragraphs", [])
            if item.get("is_supporting", item.get("is_support", False))
        ]
    else:
        facts = row.get("supporting_facts", {})
        titles = facts.get("title", []) if isinstance(facts, dict) else [item[0] for item in facts if item]
        values = [(docid(dataset, title), normalize(title)) for title in titles]
    return {item[0] for item in values}, {item[1] for item in values}


def complete(gold: set[str], values: list[str]) -> int:
    return int(bool(gold) and gold <= set(values))


def recall(gold: set[str], values: list[str]) -> float:
    return len(gold & set(values)) / max(1, len(gold))


def choose_client_oracle(rows: list[dict[str, Any]], gold_clients: set[int]) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[float, float, tuple[int, ...]]:
        subset = tuple(json.loads(row["subset_clients"]))
        coverage = int(gold_clients <= set(subset))
        client_recall = len(gold_clients & set(subset)) / max(1, len(gold_clients))
        return (-coverage, -client_recall, subset)
    return min(rows, key=key)


def choose_retrieval_oracle(rows: list[dict[str, Any]], gold_docs: set[str], gold_clients: set[int]) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[float, float, float, tuple[int, ...]]:
        subset = tuple(json.loads(row["subset_clients"]))
        merged_docs = json.loads(row["raw_merged_top10_doc_ids"])
        return (-complete(gold_docs, merged_docs), -recall(gold_docs, merged_docs), -int(gold_clients <= set(subset)), subset)
    return min(rows, key=key)


def score(row: dict[str, Any], gold_docs: set[str], gold_titles: set[str], gold_clients: set[int], selection: str) -> dict[str, Any]:
    subset = [int(client) for client in json.loads(row["subset_clients"])]
    result: dict[str, Any] = {
        "dataset": row["dataset"], "query_id": row["query_id"], "candidate_method": row["candidate_method"],
        "selection": selection, "selected_clients": json.dumps(subset),
        "selected_client_complete_coverage_at_3": int(gold_clients <= set(subset)),
        "selected_gold_client_recall_at_3": len(gold_clients & set(subset)) / max(1, len(gold_clients)),
        "gold_client_count_offline_only": len(gold_clients), "gold_support_count_offline_only": len(gold_docs),
    }
    stages = {
        "local_complete_support_at_5": ("local_top5_doc_ids", "local_top5_titles"),
        "local_complete_support_at_10": ("local_top10_doc_ids", "local_top10_titles"),
        "transmitted_complete_support_at_15": ("transmitted_doc_ids", "transmitted_titles"),
        "raw_merged_complete_support_at_5": ("raw_merged_top5_doc_ids", "raw_merged_top5_titles"),
        "raw_merged_complete_support_at_10": ("raw_merged_top10_doc_ids", "raw_merged_top10_titles"),
        "percentile_merged_complete_support_at_5": ("percentile_merged_top5_doc_ids", "percentile_merged_top5_titles"),
        "percentile_merged_complete_support_at_10": ("percentile_merged_top10_doc_ids", "percentile_merged_top10_titles"),
    }
    for name, (doc_field, title_field) in stages.items():
        docs = json.loads(row[doc_field])
        titles = [normalize(value) for value in json.loads(row[title_field])]
        result[name] = complete(gold_docs, docs)
        result[name.replace("complete_support", "support_doc_recall")] = recall(gold_docs, docs)
        result[name.replace("complete_support", "support_title_recall")] = recall(gold_titles, titles)
    return result


def classify(remp_rows: list[dict[str, Any]], gold_docs: set[str], gold_clients: set[int]) -> tuple[str, dict[str, Any]]:
    naive = next(row for row in remp_rows if int(row["is_naive_top3"]) == 1)
    client_oracle = choose_client_oracle(remp_rows, gold_clients)
    top5 = set().union(*(set(json.loads(row["subset_clients"])) for row in remp_rows))
    if len(gold_clients) > 3:
        return "outside_bc3_attainability", client_oracle
    if not gold_clients <= top5:
        return "A_candidate_absent", client_oracle
    if not gold_clients <= set(json.loads(naive["subset_clients"])):
        return "B_compression_error", client_oracle
    if not complete(gold_docs, json.loads(client_oracle["local_top10_doc_ids"])):
        return "C_local_document_missing", client_oracle
    if not complete(gold_docs, json.loads(client_oracle["transmitted_doc_ids"])):
        return "D_transmission_loss", client_oracle
    if not complete(gold_docs, json.loads(client_oracle["raw_merged_top10_doc_ids"])):
        return "E_merge_loss", client_oracle
    return "F_context_complete", client_oracle


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--frozen-retrieval", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    assignment = {str(row["doc_id"]): int(row["client_id"]) for row in jsonl(args.assignment)}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in csv_rows(args.frozen_retrieval):
        grouped[(str(row["query_id"]), row["candidate_method"])].append(row)
    active_query_ids = {query_id for query_id, method in grouped if method == "REMP_rrf_p0_dense_lexical"}
    data = {qid(row): row for row in jsonl(args.split) if qid(row) in active_query_ids}
    if active_query_ids != set(data):
        raise AssertionError("frozen retrieval query ids are not a subset of the declared split")

    selected: list[dict[str, Any]] = []
    per_subset: list[dict[str, Any]] = []
    taxonomy: list[dict[str, Any]] = []
    for query_id, datum in data.items():
        gold_docs, gold_titles = supports(datum, args.dataset)
        gold_clients = {assignment[doc] for doc in gold_docs if doc in assignment}
        for method in ("P0_single_centroid", "REMP_rrf_p0_dense_lexical"):
            candidates = grouped[(query_id, method)]
            if len(candidates) != 10:
                raise AssertionError(f"{query_id} {method}: expected 10 subsets, received {len(candidates)}")
            per_subset.extend(score(item, gold_docs, gold_titles, gold_clients, "all_enumerated") for item in candidates)
            naive = next(item for item in candidates if int(item["is_naive_top3"]) == 1)
            selected.append(score(naive, gold_docs, gold_titles, gold_clients, "naive_top3"))
            if method == "REMP_rrf_p0_dense_lexical":
                client_oracle = choose_client_oracle(candidates, gold_clients)
                retrieval_oracle = choose_retrieval_oracle(candidates, gold_docs, gold_clients)
                selected.append(score(client_oracle, gold_docs, gold_titles, gold_clients, "client_oracle_subset"))
                selected.append(score(retrieval_oracle, gold_docs, gold_titles, gold_clients, "retrieval_oracle_subset"))
                category, audit_oracle = classify(candidates, gold_docs, gold_clients)
                taxonomy.append({
                    "dataset": args.dataset, "query_id": query_id, "failure_taxonomy": category,
                    "gold_clients_offline_only": json.dumps(sorted(gold_clients)), "remp_naive_top3": naive["subset_clients"],
                    "client_oracle_subset": audit_oracle["subset_clients"], "gold_used_only_after_frozen_outputs": True,
                })

    metric_fields = [field for field in selected[0] if field.endswith("_at_3") or field.endswith("_at_5") or field.endswith("_at_10") or field.endswith("_at_15")]
    summary: list[dict[str, Any]] = []
    for method in ("P0_single_centroid", "REMP_rrf_p0_dense_lexical"):
        for selection in ("naive_top3", "client_oracle_subset", "retrieval_oracle_subset"):
            values = [row for row in selected if row["candidate_method"] == method and row["selection"] == selection]
            if values:
                summary.append({"dataset": args.dataset, "candidate_method": method, "selection": selection, "queries": len(values), **{field: sum(float(row[field]) for row in values) / len(values) for field in metric_fields}})

    naive = {row["query_id"]: row for row in selected if row["candidate_method"] == "REMP_rrf_p0_dense_lexical" and row["selection"] == "naive_top3"}
    rescue_harm: list[dict[str, Any]] = []
    for selection in ("client_oracle_subset", "retrieval_oracle_subset"):
        values = [row for row in selected if row["candidate_method"] == "REMP_rrf_p0_dense_lexical" and row["selection"] == selection]
        rescue = sum(int(naive[row["query_id"]]["raw_merged_complete_support_at_10"] == 0 and row["raw_merged_complete_support_at_10"] == 1) for row in values)
        harm = sum(int(naive[row["query_id"]]["raw_merged_complete_support_at_10"] == 1 and row["raw_merged_complete_support_at_10"] == 0) for row in values)
        rescue_harm.append({"dataset": args.dataset, "candidate_method": "REMP_rrf_p0_dense_lexical", "selection": selection, "rescue_raw_merged_complete_support_at_10": rescue, "harm_raw_merged_complete_support_at_10": harm})

    counts = Counter(row["failure_taxonomy"] for row in taxonomy)
    remp_naive = next(row for row in summary if row["candidate_method"] == "REMP_rrf_p0_dense_lexical" and row["selection"] == "naive_top3")
    remp_oracle = next(row for row in summary if row["candidate_method"] == "REMP_rrf_p0_dense_lexical" and row["selection"] == "retrieval_oracle_subset")
    oracle_rh = next(row for row in rescue_harm if row["selection"] == "retrieval_oracle_subset")
    gate = {
        "dataset": args.dataset, "stage": "R2-B0",
        "retrieval_oracle_delta_vs_remp_naive_raw_merged_complete_support_at_10": remp_oracle["raw_merged_complete_support_at_10"] - remp_naive["raw_merged_complete_support_at_10"],
        "compression_error_B_fraction": counts["B_compression_error"] / max(1, len(taxonomy)),
        "retrieval_oracle_rescue": oracle_rh["rescue_raw_merged_complete_support_at_10"],
        "retrieval_oracle_harm": oracle_rh["harm_raw_merged_complete_support_at_10"],
        "candidate_absent_fraction": counts["A_candidate_absent"] / max(1, len(taxonomy)),
        "candidate_present_but_document_missing_fraction": counts["C_local_document_missing"] / max(1, len(taxonomy)),
        "local_or_transmission_dominates": (counts["C_local_document_missing"] + counts["D_transmission_loss"]) > counts["B_compression_error"],
        "gold_used_only_for_offline_evaluation": True, "reader_started": False, "final_test_accessed": False,
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "per_subset_offline_metrics.csv", per_subset)
    write_csv(args.output_root / "per_query_selected_metrics.csv", selected)
    write_csv(args.output_root / "summary.csv", summary)
    write_csv(args.output_root / "rescue_harm.csv", rescue_harm)
    write_csv(args.output_root / "failure_taxonomy.csv", taxonomy)
    (args.output_root / "gate_inputs.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
