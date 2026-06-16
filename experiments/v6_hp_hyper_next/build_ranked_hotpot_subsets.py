from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "experiments" / "v6_hp_hyper_next"
DEFAULT_PER_QUERY = ROOT / "V6-HP1" / "outputs" / "hard_query_builder" / "baseline_per_query_results.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ranked HotpotQA easy/medium/hard query subsets from per-query retrieval logs.")
    parser.add_argument("--per-query-input", default=str(DEFAULT_PER_QUERY))
    parser.add_argument("--output-dir", default=str(EXP_ROOT / "subsets"))
    parser.add_argument("--results-csv", default=str(EXP_ROOT / "results" / "hard_subset_stats.csv"))
    parser.add_argument("--report-md", default=str(EXP_ROOT / "reports" / "hard_subset_stats.md"))
    parser.add_argument("--all-size", type=int, default=1000)
    parser.add_argument("--easy-size", type=int, default=1000)
    parser.add_argument("--medium-size", type=int, default=1000)
    parser.add_argument("--hard-500-size", type=int, default=500)
    parser.add_argument("--hard-1000-size", type=int, default=1000)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(normalize_text(item) for item in value)
    elif isinstance(value, dict):
        value = " ".join(normalize_text(item) for item in value.values())
    else:
        value = str(value)
    value = value.lower()
    value = re.sub(r"[^a-z0-9\\s]", " ", value)
    value = re.sub(r"\\s+", " ", value).strip()
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def coverage(retrieved_ids: list[Any], gold_ids: list[Any], k: int) -> float:
    gold = {str(x) for x in gold_ids}
    if not gold:
        return 0.0
    retrieved = {str(x) for x in retrieved_ids[:k]}
    return len(gold & retrieved) / len(gold)


def hit(retrieved_ids: list[Any], gold_ids: list[Any], k: int) -> float:
    gold = {str(x) for x in gold_ids}
    return 1.0 if gold and any(str(x) in gold for x in retrieved_ids[:k]) else 0.0


def infer_gold_rank(retrieved_ids: list[Any], gold_ids: list[Any], metrics: dict[str, Any]) -> int:
    explicit = safe_int(metrics.get("gold_rank"), 0)
    if explicit > 0:
        return explicit
    gold = {str(x) for x in gold_ids}
    for idx, rid in enumerate(retrieved_ids, start=1):
        if str(rid) in gold:
            return idx
    return 0


def gold_rank_score(rank: int) -> float:
    if rank <= 0:
        return 1.0
    return min(max((rank - 1) / 10.0, 0.0), 1.0)


def answer_coverage(row: dict[str, Any], topk: int = 10) -> float:
    answer = normalize_text(row.get("expected_answer", ""))
    if not answer:
        return 0.0
    contexts = row.get("retrieval_context", []) or []
    retrieved_context = normalize_text(contexts[:topk])
    return 1.0 if answer and answer in retrieved_context else 0.0


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics", {}) or {}
    retrieved_ids = row.get("retrieval_ids", []) or []
    gold_ids = row.get("golden_ids", []) or []
    rank = infer_gold_rank(retrieved_ids, gold_ids, metrics)
    hit1 = safe_float(metrics.get("cos_1", hit(retrieved_ids, gold_ids, 1)))
    hit3 = safe_float(metrics.get("cos_3", hit(retrieved_ids, gold_ids, 3)))
    hit10 = safe_float(metrics.get("cos_10", hit(retrieved_ids, gold_ids, 10)))
    recall3 = safe_float(metrics.get("recall_3", coverage(retrieved_ids, gold_ids, 3)))
    recall10 = safe_float(metrics.get("recall_10", coverage(retrieved_ids, gold_ids, 10)))
    ans_cov = answer_coverage(row, 10)
    support_cov = coverage(retrieved_ids, gold_ids, 10)
    bridge_cov = support_cov
    difficulty = (
        0.35 * gold_rank_score(rank)
        + 0.25 * (1.0 - recall3)
        + 0.20 * (1.0 - ans_cov)
        + 0.10 * (1.0 - support_cov)
        + 0.10 * (1.0 - bridge_cov)
    )
    return {
        "query_id": str(row.get("query_id", "")),
        "question": row.get("question", ""),
        "answer": row.get("expected_answer", ""),
        "gold_doc_ids": gold_ids,
        "retrieved_doc_ids_top10": retrieved_ids[:10],
        "gold_rank": rank,
        "hit@1": hit1,
        "hit@3": hit3,
        "hit@10": hit10,
        "recall@3": recall3,
        "recall@10": recall10,
        "answer_coverage": ans_cov,
        "support_doc_coverage": support_cov,
        "bridge_entity_coverage": bridge_cov,
        "difficulty_score": difficulty,
    }


def take(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    return rows[: min(size, len(rows))]


def write_subset(path: Path, subset_name: str, rows: list[dict[str, Any]], source: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "subset_name": subset_name,
        "source_per_query": str(source),
        "count": len(rows),
        "queries": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def subset_stats(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float:
        return mean([safe_float(row.get(key), 0.0) for row in rows]) if rows else 0.0

    scores = [safe_float(row.get("difficulty_score"), 0.0) for row in rows]
    return {
        "subset": name,
        "num_examples": len(rows),
        "avg_gold_rank": avg("gold_rank"),
        "hit@1": avg("hit@1"),
        "hit@3": avg("hit@3"),
        "hit@10": avg("hit@10"),
        "avg_answer_coverage": avg("answer_coverage"),
        "avg_support_doc_coverage": avg("support_doc_coverage"),
        "difficulty_score_mean": mean(scores) if scores else 0.0,
        "difficulty_score_std": pstdev(scores) if len(scores) > 1 else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, stats: list[dict[str, Any]], warnings: list[str]) -> None:
    lines = ["# HotpotQA Difficulty-Ranked Subset Statistics", ""]
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for item in warnings:
            lines.append(f"- {item}")
        lines.append("")
    lines.append("| subset | num_examples | avg_gold_rank | hit@1 | hit@3 | hit@10 | answer_cov | support_cov | diff_mean | diff_std |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in stats:
        lines.append(
            "| {subset} | {num_examples} | {avg_gold_rank:.4f} | {hit@1:.4f} | {hit@3:.4f} | {hit@10:.4f} | "
            "{avg_answer_coverage:.4f} | {avg_support_doc_coverage:.4f} | {difficulty_score_mean:.4f} | {difficulty_score_std:.4f} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = Path(args.per_query_input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows = [enrich_row(row) for row in load_jsonl(source)]
    if not rows:
        raise SystemExit(f"No per-query rows found in {source}")

    by_hard = sorted(rows, key=lambda row: safe_float(row["difficulty_score"]), reverse=True)
    by_easy = sorted(rows, key=lambda row: safe_float(row["difficulty_score"]))
    medium_start = max((len(rows) - args.medium_size) // 2, 0)
    medium = sorted(rows, key=lambda row: safe_float(row["difficulty_score"]))[medium_start : medium_start + args.medium_size]
    warnings = []
    if len(rows) < args.hard_1000_size:
        warnings.append(f"Only {len(rows)} candidate rows are available; hard_1000 uses all available rows sorted by difficulty.")

    subsets = {
        "hotpot_easy_1000": take(by_easy, args.easy_size),
        "hotpot_medium_1000": take(medium, args.medium_size),
        "hotpot_hard_500": take(by_hard, args.hard_500_size),
        "hotpot_hard_1000": take(by_hard, args.hard_1000_size),
        "hotpot_all_1000": take(rows, args.all_size),
        "hotpot_full_eval": rows,
    }
    for name, subset_rows in subsets.items():
        write_subset(output_dir / f"{name}.json", name, subset_rows, source)

    stats = [subset_stats(name, subset_rows) for name, subset_rows in subsets.items()]
    write_csv(Path(args.results_csv).expanduser().resolve(), stats)
    write_report(Path(args.report_md).expanduser().resolve(), stats, warnings)
    print(f"Wrote {len(subsets)} subsets to {output_dir}")
    print(f"Wrote stats to {args.results_csv} and {args.report_md}")


if __name__ == "__main__":
    main()
