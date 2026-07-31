#!/usr/bin/env python3
"""Stage 0B-2: audit PC-1 hard-negative and positive contracts."""

from __future__ import annotations

import argparse
import csv
import json
import re
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


def title_from_doc(text: str) -> str:
    return norm(str(text).split(":", 1)[0])


def tokens(text: Any) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9]+", str(text)) if len(w) > 2}


def support_titles(row: dict[str, Any]) -> set[str]:
    facts = row.get("supporting_facts", {})
    if isinstance(facts, dict):
        return {norm(x) for x in facts.get("title", [])}
    return {norm(x[0]) for x in facts}


def context_titles(row: dict[str, Any]) -> set[str]:
    context = row.get("context", {})
    if isinstance(context, dict):
        return {norm(x) for x in context.get("title", [])}
    return set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--frozen-pool", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = {str(row.get("query_id", row.get("id"))): row for row in read_jsonl(args.train)}
    manifest = read_jsonl(args.manifest)

    pool_by_query: dict[str, dict[str, int]] = {}
    pool_score_by_query: dict[str, dict[str, float]] = {}
    if args.frozen_pool and args.frozen_pool.exists():
        for row in read_jsonl(args.frozen_pool):
            docs = sorted(row["pool"], key=lambda d: (-float(d.get("hybrid_score", d.get("retrieval_score", 0.0))), str(d["doc_id"])))
            pool_by_query[str(row["query_id"])] = {norm(doc["title"]): idx + 1 for idx, doc in enumerate(docs)}
            pool_score_by_query[str(row["query_id"])] = {
                norm(doc["title"]): float(doc.get("hybrid_score", doc.get("retrieval_score", 0.0))) for doc in docs
            }

    contract_rows: list[dict[str, Any]] = []
    fn_rows: list[dict[str, Any]] = []
    provenance = Counter()
    false_negative_risks = 0
    duplicate_negatives = 0
    boundary_negatives = 0
    explicit_second_hop_missing = 0
    partial_hop_as_negative = 0
    inbatch_title_counts = Counter()

    for item in manifest:
        qid = str(item["query_id"])
        q_tokens = tokens(item.get("query", ""))
        row = train.get(qid, {})
        gold = support_titles(row)
        all_context = context_titles(row)
        positive_title = title_from_doc(item.get("positive", ""))
        if len(gold) > 1 and positive_title in gold and len(gold - {positive_title}) > 0:
            explicit_second_hop_missing += 1

        neg_titles_seen = Counter()
        for neg_text, source in zip(item.get("negatives", []), item.get("negative_provenance", [])):
            neg_title = title_from_doc(neg_text)
            neg_titles_seen[neg_title] += 1
            inbatch_title_counts[neg_title] += 1
            provenance[source] += 1
            neg_tokens = tokens(neg_text)
            entity_overlap = len(q_tokens & neg_tokens)
            rank = pool_by_query.get(qid, {}).get(neg_title)
            score = pool_score_by_query.get(qid, {}).get(neg_title)
            if rank is not None and 4 <= rank <= 10:
                boundary_negatives += 1
            is_gold = neg_title in gold
            is_context_non_gold = neg_title in all_context and neg_title not in gold
            if is_gold:
                false_negative_risks += 1
            if is_context_non_gold and entity_overlap > 0:
                partial_hop_as_negative += 1
            contract_rows.append(
                {
                    "query_id": qid,
                    "negative_title": neg_title,
                    "negative_source": source,
                    "baseline_rank": "" if rank is None else rank,
                    "baseline_score": "" if score is None else f"{score:.8f}",
                    "entity_overlap": entity_overlap,
                    "is_gold_support": is_gold,
                    "is_context_non_gold": is_context_non_gold,
                    "rank_5_10_boundary_negative": bool(rank is not None and 4 <= rank <= 10),
                    "positive_title_used": positive_title,
                    "gold_title_count": len(gold),
                    "all_support_titles": json.dumps(sorted(gold), ensure_ascii=False),
                }
            )
        duplicate_negatives += sum(count - 1 for count in neg_titles_seen.values() if count > 1)

    for title, count in inbatch_title_counts.items():
        if count > 1:
            fn_rows.append({"title": title, "times_as_negative": count, "risk_type": "duplicate_in_negative_batches"})
    for row in contract_rows:
        if row["is_gold_support"]:
            fn_rows.append({"title": row["negative_title"], "query_id": row["query_id"], "risk_type": "gold_support_as_negative"})

    with (args.output_dir / "pc1_negative_contract.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contract_rows[0].keys()) if contract_rows else ["query_id"])
        writer.writeheader()
        writer.writerows(contract_rows)

    with (args.output_dir / "false_negative_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in fn_rows for key in row.keys()}) if fn_rows else ["risk_type"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fn_rows)

    ranks = [int(row["baseline_rank"]) for row in contract_rows if row["baseline_rank"] != ""]
    report = [
        "# PC-1 Loss Alignment and Negative Contract Audit",
        "",
        f"Training manifest queries: {len(manifest)}",
        f"Total explicit negatives: {len(contract_rows)}",
        f"Negative source distribution: {dict(provenance)}",
        f"Average baseline rank for negatives with pool rank: {mean(ranks):.2f}" if ranks else "Average baseline rank for negatives with pool rank: unavailable on train-only manifest",
        f"Rank 4-10 boundary negatives: {boundary_negatives} / {len(contract_rows)}",
        f"False-negative risks where a negative is a gold support title: {false_negative_risks}",
        f"Duplicate negative titles within queries: {duplicate_negatives}",
        f"Potential partial-hop/context negatives: {partial_hop_as_negative}",
        f"Queries with multiple gold supports but only one explicit positive used: {explicit_second_hop_missing} / {len(manifest)}",
        "",
        "## Stage 0B-2 Answers",
        "",
        "1. The PC-1 manifest is train-only and mostly entity-overlap negatives; boundary-rank coverage is only measurable when a frozen pool rank is available for the same query.",
        "2. If boundary negatives are near zero, PC-1 optimizes global separation rather than rank-5 boundary crossing.",
        "3. Multi-hop supervision is currently single-positive: the first support is used as the explicit positive even when two gold supports exist.",
        "4. Partial-hop documents are not separately labeled in PC-1, so treating all non-support entity-overlap docs as strong negatives can misalign multi-hop retrieval.",
        "5. This audit does not use calibration or final-test labels.",
    ]
    (args.output_dir / "pc1_loss_alignment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "queries": len(manifest),
                "negatives": len(contract_rows),
                "provenance": dict(provenance),
                "boundary_negatives": boundary_negatives,
                "single_positive_multihop_queries": explicit_second_hop_missing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
