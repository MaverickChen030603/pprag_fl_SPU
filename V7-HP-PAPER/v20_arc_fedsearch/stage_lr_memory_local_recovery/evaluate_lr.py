#!/usr/bin/env python3
"""Evaluate frozen LR artifacts; this is the first component that reads gold."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
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


def norm(value: str) -> str:
    return " ".join(str(value).lower().split())


def docid(dataset: str, title: str, text: str = "") -> str:
    key = norm(title) if dataset != "musique" else norm(title) + "\n" + norm(text)
    return f"{dataset}:{hashlib.sha1(key.encode()).hexdigest()[:20]}"


def supports(row: dict[str, Any], dataset: str) -> tuple[set[str], set[str]]:
    if dataset == "musique":
        values = [(docid(dataset, item.get("title", ""), item.get("paragraph_text", "")), norm(item.get("title", "")))
                  for item in row.get("paragraphs", []) if item.get("is_supporting", item.get("is_support", False))]
    else:
        facts = row.get("supporting_facts", {})
        titles = facts.get("title", []) if isinstance(facts, dict) else [item[0] for item in facts if item]
        values = [(docid(dataset, title), norm(title)) for title in titles]
    return {item[0] for item in values}, {item[1] for item in values}


def complete(gold: set[str], values: list[str]) -> int:
    return int(bool(gold) and gold <= set(values))


def recall(gold: set[str], values: list[str]) -> float:
    return len(gold & set(values)) / max(1, len(gold))


def present_in_indexes(gold: set[str], clients: list[int], root: Path) -> set[str]:
    found: set[str] = set()
    for client in clients:
        connection = sqlite3.connect(root / f"client_{client:02d}.sqlite")
        try:
            found.update(str(doc) for doc, in connection.execute(
                "select doc_id from docs where doc_id in (%s)" % ",".join("?" for _ in gold), tuple(gold)
            ))
        finally:
            connection.close()
    return found


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
    parser.add_argument("--local-index-root", type=Path, required=True)
    parser.add_argument("--frozen-outputs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--b0-frozen", type=Path, required=True)
    args = parser.parse_args()

    source = {qid(row): row for row in jsonl(args.split)}
    assignment = {str(row["doc_id"]): int(row["client_id"]) for row in jsonl(args.assignment)}
    rows = list(csv_rows(args.frozen_outputs))
    active = {str(row["query_id"]) for row in rows}
    source = {key: value for key, value in source.items() if key in active}
    scored: list[dict[str, Any]] = []
    c_taxonomy: list[dict[str, Any]] = []
    for frozen in rows:
        query_id = str(frozen["query_id"])
        gold_docs, gold_titles = supports(source[query_id], args.dataset)
        gold_clients = {assignment[doc] for doc in gold_docs if doc in assignment}
        selected = [int(client) for client in json.loads(frozen["selected_clients"])]
        candidate = set(json.loads(frozen["candidate_top5"]))
        values: dict[str, Any] = {
            "dataset": args.dataset, "query_id": query_id, "track": frozen["track"], "method": frozen["method"],
            "selection_gold_only_diagnostic": frozen["selection_gold_only_diagnostic"],
            "selected_client_complete_coverage_at_3": int(gold_clients <= set(selected)),
            "gold_client_recall_at_3": len(gold_clients & set(selected)) / max(1, len(gold_clients)),
        }
        fields = {
            "local_complete_support_at_5": ("local_top5_doc_ids", "local_top5_titles"),
            "local_complete_support_at_10": ("local_top10_doc_ids", "local_top10_titles"),
            "local_complete_support_at_20": ("local_top20_doc_ids", "local_top20_titles"),
            "local_complete_support_at_50": ("local_top50_doc_ids", "local_top50_titles"),
            "transmitted_complete_support_at_15": ("transmitted_doc_ids", "transmitted_titles"),
            "raw_merged_complete_support_at_10": ("raw_merged_top10_doc_ids", "raw_merged_top10_titles"),
        }
        for name, (doc_field, title_field) in fields.items():
            docs = json.loads(frozen[doc_field])
            titles = [norm(value) for value in json.loads(frozen[title_field])]
            values[name] = complete(gold_docs, docs)
            values[name.replace("complete_support", "support_doc_recall")] = recall(gold_docs, docs)
            values[name.replace("complete_support", "support_title_recall")] = recall(gold_titles, titles)
        scored.append(values)
        if frozen["method"] == "L0":
            if not gold_clients <= candidate:
                category = "A_candidate_absent"
            elif not gold_clients <= set(selected):
                category = "B_selection_miss"
            else:
                physical = present_in_indexes(gold_docs, selected, args.local_index_root)
                if not gold_docs <= physical:
                    category = "C0_absent_from_selected_physical_index"
                elif not complete(gold_docs, json.loads(frozen["local_top50_doc_ids"])):
                    category = "C1_not_in_dense_top50"
                elif not complete(gold_docs, json.loads(frozen["local_top10_doc_ids"])):
                    category = "C2_dense_rank_11_to_50"
                elif not complete(gold_docs, json.loads(frozen["transmitted_doc_ids"])):
                    category = "C3_top10_but_top5_transmission_drop"
                elif not complete(gold_docs, json.loads(frozen["raw_merged_top10_doc_ids"])):
                    category = "E_merge_loss"
                else:
                    category = "F_context_complete"
            c_taxonomy.append({"dataset": args.dataset, "query_id": query_id, "track": frozen["track"], "lr0_taxonomy": category,
                               "gold_used_only_after_frozen_outputs": True})

    metric_fields = [field for field in scored[0] if field.endswith(("_at_3", "_at_5", "_at_10", "_at_15", "_at_20", "_at_50"))]
    summary: list[dict[str, Any]] = []
    for track in sorted({row["track"] for row in scored}):
        for method in sorted({row["method"] for row in scored}):
            values = [row for row in scored if row["track"] == track and row["method"] == method]
            summary.append({"dataset": args.dataset, "track": track, "method": method, "queries": len(values),
                            **{field: sum(float(row[field]) for row in values) / len(values) for field in metric_fields}})

    rescue_harm: list[dict[str, Any]] = []
    for track in sorted({row["track"] for row in scored}):
        base = {row["query_id"]: row for row in scored if row["track"] == track and row["method"] == "L0"}
        for method in sorted({row["method"] for row in scored if row["track"] == track and row["method"] != "L0"}):
            values = [row for row in scored if row["track"] == track and row["method"] == method]
            rescue = sum(int(base[row["query_id"]]["raw_merged_complete_support_at_10"] == 0 and row["raw_merged_complete_support_at_10"] == 1) for row in values)
            harm = sum(int(base[row["query_id"]]["raw_merged_complete_support_at_10"] == 1 and row["raw_merged_complete_support_at_10"] == 0) for row in values)
            rescue_harm.append({"dataset": args.dataset, "track": track, "method": method,
                                "rescue_raw_merged_complete_support_at_10": rescue, "harm_raw_merged_complete_support_at_10": harm})

    b0_rows = list(csv_rows(args.b0_frozen))
    b0_lookup = {(str(row["query_id"]), row["candidate_method"], tuple(json.loads(row["subset_clients"]))): row for row in b0_rows}
    l0_rows = [row for row in rows if row["method"] == "L0"]
    replay_matches, replay_total = 0, 0
    for row in l0_rows:
        method = "P0_single_centroid" if row["track"] == "D0_P0_naive_top3" else "REMP_rrf_p0_dense_lexical"
        key = (str(row["query_id"]), method, tuple(json.loads(row["selected_clients"])))
        baseline = b0_lookup.get(key)
        if baseline is None:
            continue
        replay_total += 1
        replay_matches += int(
            json.loads(row["transmitted_doc_ids"]) == json.loads(baseline["transmitted_doc_ids"]) and
            json.loads(row["raw_merged_top10_doc_ids"]) == json.loads(baseline["raw_merged_top10_doc_ids"])
        )
    counts = Counter(row["lr0_taxonomy"] for row in c_taxonomy if row["track"] == "D1_REMP_naive_top3")
    c_counts = {key: counts[key] for key in ("C0_absent_from_selected_physical_index", "C1_not_in_dense_top50", "C2_dense_rank_11_to_50", "C3_top10_but_top5_transmission_drop")}
    c0_dominant = c_counts["C0_absent_from_selected_physical_index"] > max(c_counts["C1_not_in_dense_top50"], c_counts["C2_dense_rank_11_to_50"], c_counts["C3_top10_but_top5_transmission_drop"])
    decision = {"dataset": args.dataset, "stage": "LR-0", "d1_c_taxonomy_counts": c_counts, "c0_dominant": c0_dominant,
                "lr1_permitted_for_dataset": not c0_dominant, "l0_b0_replay_matches": replay_matches,
                "l0_b0_replay_total": replay_total, "l0_b0_replay_exact": replay_matches == replay_total and replay_total > 0,
                "gold_used_only_for_offline_evaluation": True, "reader_started": False, "final_test_accessed": False}

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "per_query_metrics.csv", scored)
    write_csv(args.output_root / "summary.csv", summary)
    write_csv(args.output_root / "rescue_harm.csv", rescue_harm or [{"dataset": args.dataset, "track": "none", "method": "none", "rescue_raw_merged_complete_support_at_10": 0, "harm_raw_merged_complete_support_at_10": 0}])
    write_csv(args.output_root / "lr0_taxonomy.csv", c_taxonomy)
    (args.output_root / "lr0_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
