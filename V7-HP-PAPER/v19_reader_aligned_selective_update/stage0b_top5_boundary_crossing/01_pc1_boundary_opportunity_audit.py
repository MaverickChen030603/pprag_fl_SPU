#!/usr/bin/env python3
"""Stage 0B-1: audit whether PC-1 changes move evidence toward Top-5."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm(text: Any) -> str:
    return " ".join(str(text).lower().split())


def support_titles(row: dict[str, Any]) -> set[str]:
    facts = row.get("supporting_facts", {})
    if isinstance(facts, dict):
        return {norm(x) for x in facts.get("title", [])}
    return {norm(x[0]) for x in facts}


def ranked(row: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(row["pool"], key=lambda d: (-float(d.get("hybrid_score", d.get("retrieval_score", 0.0))), str(d["doc_id"])))


def score_of(doc: dict[str, Any]) -> float:
    return float(doc.get("hybrid_score", doc.get("retrieval_score", 0.0)))


def rank_lookup(docs: list[dict[str, Any]]) -> dict[str, tuple[int, dict[str, Any]]]:
    return {norm(doc["title"]): (idx + 1, doc) for idx, doc in enumerate(docs)}


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8f}"
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False)
    return str(value)


def classify(
    base_ranks: list[int | None],
    pc1_ranks: list[int | None],
    top5_changed: bool,
    base_complete: bool,
    pc1_complete: bool,
    margin_5_6: float | None,
    tie_threshold: float,
) -> str:
    if top5_changed and pc1_complete and not base_complete:
        return "E_top5_beneficial_swap"
    if top5_changed and base_complete and not pc1_complete:
        return "F_top5_harmful_swap"
    if top5_changed:
        entered = any((b is None or b > 5) and (p is not None and p <= 5) for b, p in zip(base_ranks, pc1_ranks))
        left = any((b is not None and b <= 5) and (p is None or p > 5) for b, p in zip(base_ranks, pc1_ranks))
        if entered:
            return "E_top5_beneficial_swap"
        if left:
            return "F_top5_harmful_swap"
    if margin_5_6 is not None and margin_5_6 <= tie_threshold:
        return "G_tie_boundary_instability"
    near = any(r is not None and 6 <= r <= 10 for r in base_ranks)
    moved_toward = any(
        b is not None and p is not None and 6 <= b <= 10 and p < b
        for b, p in zip(base_ranks, pc1_ranks)
    )
    moved_away = any(
        b is not None and p is not None and 6 <= b <= 10 and p > b
        for b, p in zip(base_ranks, pc1_ranks)
    )
    if near and moved_toward:
        return "A_support_near_boundary_positive"
    if near and moved_away:
        return "B_support_near_boundary_negative"
    if any(r is None or r > 10 for r in base_ranks):
        return "C_deep_support"
    return "D_irrelevant_reorder"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-pool", type=Path, required=True)
    parser.add_argument("--pc1-pool", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tie-threshold", type=float, default=1e-4)
    parser.add_argument("--boundary-margin-threshold", type=float, default=0.02)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev = {str(row.get("query_id", row.get("id"))): row for row in read_jsonl(args.development)}
    frozen = {str(row["query_id"]): row for row in read_jsonl(args.frozen_pool)}
    pc1 = {str(row["query_id"]): row for row in read_jsonl(args.pc1_pool)}

    per_query: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    margins: list[dict[str, Any]] = []

    for qid, base_row in frozen.items():
        if qid not in pc1 or qid not in dev:
            continue
        gold = sorted(support_titles(dev[qid]))
        base_docs = ranked(base_row)
        pc1_docs = ranked(pc1[qid])
        base_by_title = rank_lookup(base_docs)
        pc1_by_title = rank_lookup(pc1_docs)
        base_ranks = [base_by_title.get(title, (None, None))[0] for title in gold]
        pc1_ranks = [pc1_by_title.get(title, (None, None))[0] for title in gold]
        base_scores = [score_of(base_by_title[title][1]) for title in gold if title in base_by_title]
        pc1_scores = [score_of(pc1_by_title[title][1]) for title in gold if title in pc1_by_title]
        support_delta = (safe_mean(pc1_scores) or 0.0) - (safe_mean(base_scores) or 0.0)

        top5_base = [doc["doc_id"] for doc in base_docs[:5]]
        top5_pc1 = [doc["doc_id"] for doc in pc1_docs[:5]]
        top10_base = [doc["doc_id"] for doc in base_docs[:10]]
        top10_pc1 = [doc["doc_id"] for doc in pc1_docs[:10]]
        top20_base = [doc["doc_id"] for doc in base_docs[:20]]
        top20_pc1 = [doc["doc_id"] for doc in pc1_docs[:20]]
        top5_changed = set(top5_base) != set(top5_pc1)
        top10_changed = set(top10_base) != set(top10_pc1)
        top20_changed = set(top20_base) != set(top20_pc1)
        support_entered_top5 = any((b is None or b > 5) and (p is not None and p <= 5) for b, p in zip(base_ranks, pc1_ranks))
        support_left_top5 = any((b is not None and b <= 5) and (p is None or p > 5) for b, p in zip(base_ranks, pc1_ranks))
        support_entered_6_10 = any((b is None or b > 10) and (p is not None and 6 <= p <= 10) for b, p in zip(base_ranks, pc1_ranks))
        moved_toward = any(b is not None and p is not None and p < b for b, p in zip(base_ranks, pc1_ranks))
        moved_away = any(b is not None and p is not None and p > b for b, p in zip(base_ranks, pc1_ranks))
        base_complete = all(r is not None and r <= 5 for r in base_ranks)
        pc1_complete = all(r is not None and r <= 5 for r in pc1_ranks)
        rank5 = base_docs[4] if len(base_docs) >= 5 else None
        rank6 = base_docs[5] if len(base_docs) >= 6 else None
        margin_5_6 = score_of(rank5) - score_of(rank6) if rank5 and rank6 else None
        best_near_support_gap = None
        near_scores = [score_of(base_by_title[t][1]) for t in gold if t in base_by_title and 6 <= base_by_title[t][0] <= 10]
        if rank5 and near_scores:
            best_near_support_gap = score_of(rank5) - max(near_scores)
        category = classify(
            base_ranks,
            pc1_ranks,
            top5_changed,
            base_complete,
            pc1_complete,
            margin_5_6,
            args.tie_threshold,
        )
        irrelevant_reorder = (top10_changed or top20_changed) and not (moved_toward or moved_away or support_entered_top5 or support_left_top5)
        row = {
            "query_id": qid,
            "category": category,
            "baseline_support_ranks": base_ranks,
            "pc1_support_ranks": pc1_ranks,
            "baseline_top5": top5_base,
            "pc1_top5": top5_pc1,
            "baseline_top10": top10_base,
            "pc1_top10": top10_pc1,
            "baseline_top20": top20_base,
            "pc1_top20": top20_pc1,
            "rank5_doc": rank5["doc_id"] if rank5 else "",
            "rank6_doc": rank6["doc_id"] if rank6 else "",
            "rank5_score": score_of(rank5) if rank5 else None,
            "rank6_score": score_of(rank6) if rank6 else None,
            "rank5_rank6_margin": margin_5_6,
            "best_near_support_gap_to_rank5": best_near_support_gap,
            "support_score_delta": support_delta,
            "support_entered_rank_6_10": support_entered_6_10,
            "support_moved_toward_top5": moved_toward,
            "support_moved_away": moved_away,
            "change_only_irrelevant_documents": irrelevant_reorder,
            "top5_changed": top5_changed,
            "top10_changed": top10_changed,
            "top20_changed": top20_changed,
            "support_entered_top5": support_entered_top5,
            "support_left_top5": support_left_top5,
            "baseline_complete_support_at5": base_complete,
            "pc1_complete_support_at5": pc1_complete,
        }
        per_query.append(row)
        for idx, title in enumerate(gold):
            transitions.append(
                {
                    "query_id": qid,
                    "support_title": title,
                    "baseline_rank": base_ranks[idx],
                    "pc1_rank": pc1_ranks[idx],
                    "rank_delta_pc1_minus_baseline": (
                        pc1_ranks[idx] - base_ranks[idx]
                        if base_ranks[idx] is not None and pc1_ranks[idx] is not None
                        else ""
                    ),
                }
            )
        margins.append(
            {
                "query_id": qid,
                "rank5_rank6_margin": margin_5_6,
                "best_near_support_gap_to_rank5": best_near_support_gap,
                "boundary_opportunity_at5": bool(
                    best_near_support_gap is not None
                    and best_near_support_gap <= args.boundary_margin_threshold
                ),
            }
        )

    per_query_fields = list(per_query[0].keys()) if per_query else []
    with (args.output_dir / "pc1_per_query_categories.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_query_fields)
        writer.writeheader()
        for row in per_query:
            writer.writerow({key: fmt(value) for key, value in row.items()})

    with (args.output_dir / "support_rank_transition.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transitions[0].keys()) if transitions else ["query_id"])
        writer.writeheader()
        writer.writerows(transitions)

    with (args.output_dir / "rank5_rank10_margin.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(margins[0].keys()) if margins else ["query_id"])
        writer.writeheader()
        writer.writerows(margins)

    counts = Counter(row["category"] for row in per_query)
    top10_or_20 = [row for row in per_query if row["top10_changed"] or row["top20_changed"]]
    support_promoting = [row for row in top10_or_20 if row["support_moved_toward_top5"] or row["support_score_delta"] > 0]
    near_boundary = [row for row in per_query if any(r is not None and 6 <= r <= 10 for r in row["baseline_support_ranks"])]
    conversions = [row for row in near_boundary if row["support_entered_top5"]]
    useful_top5 = [row for row in per_query if row["category"] == "E_top5_beneficial_swap"]
    harmful_top5 = [row for row in per_query if row["category"] == "F_top5_harmful_swap"]
    boundary_ops = [row for row in margins if row["boundary_opportunity_at5"]]
    improved = sum(1 for item in transitions if item["rank_delta_pc1_minus_baseline"] != "" and item["rank_delta_pc1_minus_baseline"] < 0)
    worsened = sum(1 for item in transitions if item["rank_delta_pc1_minus_baseline"] != "" and item["rank_delta_pc1_minus_baseline"] > 0)

    report = [
        "# PC-1 Boundary Opportunity Audit",
        "",
        f"Queries audited: {len(per_query)}",
        f"Top-10/20 changed queries: {len(top10_or_20)}",
        f"Support-promoting among Top-10/20 changed: {len(support_promoting)} / {len(top10_or_20)}",
        f"Queries with support already in rank 6-10: {len(near_boundary)}",
        f"BoundaryOpportunity@5 (threshold={args.boundary_margin_threshold}): {len(boundary_ops)} / {len(per_query)}",
        f"BoundaryConversionRate: {len(conversions)} / {len(near_boundary) if near_boundary else 1}",
        f"UsefulTop5ChangeRate: {len(useful_top5)} / {len(per_query)}",
        f"HarmfulTop5ChangeRate: {len(harmful_top5)} / {len(per_query)}",
        f"Support rank improved/worsened documents: {improved} / {worsened}",
        "",
        "## Category Counts",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for key, value in sorted(counts.items()):
        report.append(f"| {key} | {value} |")
    report += [
        "",
        "## Answers Required by Stage 0B-1",
        "",
        f"1. In the Top-10/20 changed set, {len(support_promoting)} of {len(top10_or_20)} queries show support-promoting movement or support-score gain.",
        f"2. {len(near_boundary)} queries have at least one gold/support document at baseline rank 6-10.",
        "3. The rank-5 boundary appears usable only when BoundaryOpportunity@5 is non-trivial; inspect `rank5_rank10_margin.csv` before training.",
        f"4. The two Top-5 change cases are summarized by category: useful={len(useful_top5)}, harmful={len(harmful_top5)}, other={sum(1 for row in per_query if row['top5_changed']) - len(useful_top5) - len(harmful_top5)}.",
        "",
        "No reader gate decision is made by this audit.",
    ]
    (args.output_dir / "pc1_boundary_opportunity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "queries": len(per_query), "category_counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
