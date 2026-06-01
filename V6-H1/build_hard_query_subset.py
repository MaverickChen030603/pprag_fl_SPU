from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stable hard-query subset from V6-H1 per-query RAG outputs.")
    parser.add_argument("--per-query-root", required=True, help="Root containing per_query_results.jsonl files.")
    parser.add_argument("--output-dir", default="V6-H1/hard_queries")
    parser.add_argument("--baseline-pattern", default="hypernet-v3", help="Only use runs whose path contains this string.")
    parser.add_argument("--min-hard-seeds", type=int, default=2)
    parser.add_argument("--max-queries", type=int, default=0, help="Optional cap; 0 keeps all stable hard queries.")
    parser.add_argument("--suite-name", default="v6h1_hardquery_stable")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def first_gold_rank(retrieval_ids: list[Any], golden_ids: list[Any]) -> int | None:
    golden = {str(item) for item in golden_ids}
    for index, rid in enumerate(retrieval_ids, start=1):
        if str(rid) in golden:
            return index
    return None


def recall_at_k(retrieval_ids: list[Any], golden_ids: list[Any], k: int) -> float:
    if not golden_ids:
        return 0.0
    retrieved = {str(item) for item in retrieval_ids[:k]}
    golden = {str(item) for item in golden_ids}
    return len(retrieved & golden) / len(golden)


def enrich_record(record: dict[str, Any], source_path: Path, root: Path) -> dict[str, Any]:
    retrieval_ids = record.get("retrieval_ids", []) or []
    golden_ids = record.get("golden_ids", []) or []
    metrics = dict(record.get("metrics", {}) or {})
    rank = metrics.get("gold_rank") or first_gold_rank(retrieval_ids, golden_ids)
    rank = int(rank) if rank else None
    mrr = float(metrics.get("mrr_local", 0.0) or (0.0 if rank is None else 1.0 / rank))
    recall_3 = float(metrics.get("recall_3", recall_at_k(retrieval_ids, golden_ids, 3)) or 0.0)
    hard_reasons = []
    if rank is None:
        hard_reasons.append("gold_not_retrieved")
    if rank is not None and rank > 3:
        hard_reasons.append("gold_rank_gt_3")
    if recall_3 == 0:
        hard_reasons.append("recall_3_zero")
    if mrr < 0.5:
        hard_reasons.append("mrr_lt_0.5")
    relative = source_path.relative_to(root)
    parts = relative.parts
    return {
        "query_id": str(record.get("query_id", "")),
        "question": record.get("question", ""),
        "golden_ids": golden_ids,
        "retrieval_ids": retrieval_ids,
        "gold_rank": rank,
        "recall_1": float(metrics.get("recall_1", recall_at_k(retrieval_ids, golden_ids, 1)) or 0.0),
        "recall_3": recall_3,
        "recall_5": float(metrics.get("recall_5", recall_at_k(retrieval_ids, golden_ids, 5)) or 0.0),
        "mrr": mrr,
        "is_hard": bool(hard_reasons),
        "hard_reason": "|".join(hard_reasons),
        "source_file": str(source_path),
        "relative_source": str(relative),
        "suite": parts[0] if len(parts) > 0 else "",
        "task_name": parts[1] if len(parts) > 1 else "",
        "run_name": parts[2] if len(parts) > 2 else "",
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "question",
        "suite",
        "task_name",
        "run_name",
        "golden_ids",
        "retrieval_ids",
        "gold_rank",
        "recall_1",
        "recall_3",
        "recall_5",
        "mrr",
        "is_hard",
        "hard_reason",
        "relative_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), list) else row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    root = Path(args.per_query_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    files = sorted(root.rglob("per_query_results.jsonl"))
    if args.baseline_pattern:
        files = [path for path in files if args.baseline_pattern in str(path)]
    all_rows: list[dict[str, Any]] = []
    hard_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    question_by_id: dict[str, str] = {}
    golden_by_id: dict[str, list[Any]] = {}
    for path in files:
        for record in load_jsonl(path):
            row = enrich_record(record, path, root)
            all_rows.append(row)
            question_by_id[row["query_id"]] = row["question"]
            golden_by_id[row["query_id"]] = row["golden_ids"]
            if row["is_hard"]:
                hard_by_query[row["query_id"]].append(row)

    stable_rows = []
    for query_id, hard_rows in hard_by_query.items():
        hard_votes = len({row["relative_source"] for row in hard_rows})
        if hard_votes >= args.min_hard_seeds:
            reason_counter = Counter(reason for row in hard_rows for reason in row["hard_reason"].split("|") if reason)
            stable_rows.append(
                {
                    "query_id": query_id,
                    "question": question_by_id.get(query_id, ""),
                    "golden_ids": golden_by_id.get(query_id, []),
                    "hard_votes": hard_votes,
                    "hard_reason": "|".join(reason for reason, _ in reason_counter.most_common()),
                    "seed_details": hard_rows,
                }
            )
    stable_rows.sort(key=lambda row: (-int(row["hard_votes"]), str(row["query_id"])))
    if args.max_queries > 0:
        stable_rows = stable_rows[: args.max_queries]

    metadata = {
        "suite_name": args.suite_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "per_query_root": str(root),
        "baseline_pattern": args.baseline_pattern,
        "min_hard_seeds": args.min_hard_seeds,
        "candidate_files": [str(path) for path in files],
        "total_per_query_rows": len(all_rows),
        "stable_hard_query_count": len(stable_rows),
        "hard_rule": "recall_3 == 0 OR gold_rank is None OR gold_rank > 3 OR mrr < 0.5",
    }
    payload = {"metadata": metadata, "queries": stable_rows}
    write_json(output_dir / "stable_hard_queries.json", payload)
    write_json(output_dir / "hard_query_eval_summary.json", {"metadata": metadata, "rows": all_rows})
    write_csv(output_dir / "hard_query_eval_summary.csv", all_rows)
    print(f"Wrote {len(stable_rows)} stable hard queries to {output_dir / 'stable_hard_queries.json'}")


if __name__ == "__main__":
    main()
