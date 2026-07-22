#!/usr/bin/env python3
"""Offline pool recall audit; labels never enter retrieval or composer outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def title(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def support_titles(row: dict[str, Any], dataset: str) -> set[str]:
    if dataset == "musique":
        return {title(paragraph.get("title", "")) for paragraph in row.get("paragraphs", []) if paragraph.get("is_supporting", False)}
    facts = row.get("supporting_facts", [])
    if isinstance(facts, dict):
        values = facts.get("title", [])
    else:
        values = [fact[0] for fact in facts if isinstance(fact, (list, tuple)) and fact]
    return {title(value) for value in values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    labels = {qid(row): row for row in read(args.split)}
    rows = []
    for pool in read(args.pool):
        source = labels[str(pool["query_id"])]
        gold = support_titles(source, args.dataset)
        selected = {title(doc["title"]) for doc in pool["pool"]}
        answer = str(source.get("answer", source.get("answer_text", ""))).strip().lower()
        rows.append({
            "query_id": pool["query_id"], "dataset": args.dataset, "pool_size": pool["pool_size"],
            "support_recall": len(gold & selected) / len(gold) if gold else "",
            "all_support_present": int(bool(gold) and gold <= selected),
            "answer_access": int(bool(answer) and any(answer in f"{doc['title']} {doc['text']}".lower() for doc in pool["pool"])),
            "support_count": len(gold), "retrieval_latency_ms": pool.get("retrieval_latency_ms", ""),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    size_label = rows[0]["pool_size"] if rows else "unknown"
    csv_path = args.output_dir / f"{args.dataset}_top{size_label}_pool_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["query_id"])
        writer.writeheader(); writer.writerows(rows)
    numeric_recall = [float(row["support_recall"]) for row in rows if row["support_recall"] != ""]
    summary = {
        "dataset": args.dataset, "queries": len(rows), "pool_size": rows[0]["pool_size"] if rows else None,
        "mean_support_recall": statistics.fmean(numeric_recall) if numeric_recall else None,
        "complete_support_rate": statistics.fmean(row["all_support_present"] for row in rows) if rows else None,
        "answer_access_rate": statistics.fmean(row["answer_access"] for row in rows) if rows else None,
        "label_use": "offline_audit_only",
    }
    (args.output_dir / f"{args.dataset}_top{size_label}_pool_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
