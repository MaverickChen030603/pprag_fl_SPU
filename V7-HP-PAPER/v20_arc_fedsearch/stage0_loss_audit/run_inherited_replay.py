#!/usr/bin/env python3
"""Audit where a frozen federated retrieval pool loses multi-hop evidence.

This is deliberately an *offline* audit.  The executable pipeline never reads
gold fields; this program reads them only after a frozen V17 pool has been
created, to label which stage made a support document unavailable.  It does not
run a reader and cannot be used as a routing or merging component.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def query_id(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def norm(value: str) -> str:
    return " ".join(str(value).lower().split())


def document_id(dataset: str, title: str, text: str = "") -> str:
    identity = norm(title) if dataset != "musique" else norm(title) + "\n" + norm(text)
    return f"{dataset}:{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:20]}"


def support_ids(row: dict[str, Any], dataset: str) -> set[str]:
    if dataset == "musique":
        return {
            document_id(dataset, item.get("title", ""), item.get("paragraph_text", ""))
            for item in row.get("paragraphs", [])
            if item.get("is_supporting", item.get("is_support", False))
        }
    facts = row.get("supporting_facts", [])
    titles = facts.get("title", []) if isinstance(facts, dict) else [item[0] for item in facts if item]
    return {document_id(dataset, title) for title in titles}


def assignment(path: Path) -> dict[str, int]:
    return {str(item["doc_id"]): int(item["client_id"]) for item in rows(path)}


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def complete(support: set[str], docs: list[dict[str, Any]]) -> int:
    return int(bool(support) and support.issubset({str(doc["doc_id"]) for doc in docs}))


def support_recall(support: set[str], docs: list[dict[str, Any]]) -> float:
    return len(support & {str(doc["doc_id"]) for doc in docs}) / max(1, len(support))


def client_groups(pool: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for doc in pool["pool"]:
        key = str(doc["doc_id"])
        if key not in seen:
            grouped[int(doc["client_id"])].append(doc)
            seen.add(key)
    for values in grouped.values():
        values.sort(key=lambda doc: (int(doc.get("local_rank", 10**6)), -float(doc.get("hybrid_score", 0.0)), str(doc["doc_id"])))
    return grouped


def round_robin_transmission(groups: dict[int, list[dict[str, Any]]], clients: list[int], budget: int) -> list[dict[str, Any]]:
    """A label-free source-diverse transmission policy for the audit grid."""
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < budget:
        progressed = False
        for client in clients:
            docs = groups.get(client, [])
            if depth < len(docs):
                selected.append(docs[depth])
                progressed = True
                if len(selected) == budget:
                    break
        if not progressed:
            break
        depth += 1
    return selected


def rank_percentile_merge(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A label-free score-scale baseline: local rank percentile plus tie breaks."""
    counts = Counter(int(doc["client_id"]) for doc in docs)
    decorated = []
    for doc in docs:
        rank = int(doc.get("local_rank", 0))
        client = int(doc["client_id"])
        percentile = 1.0 - rank / max(1, counts[client])
        decorated.append((percentile, float(doc.get("dense_score", 0.0)), -rank, str(doc["doc_id"]), doc))
    return [item[-1] for item in sorted(decorated, reverse=True)]


def raw_merge(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        docs,
        key=lambda doc: (-float(doc.get("hybrid_score", 0.0)), -float(doc.get("dense_score", 0.0)), str(doc["doc_id"])),
    )


def rate(values: list[dict[str, Any]], field: str) -> float:
    return sum(float(item[field]) for item in values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpotqa", "2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--federated-pool", type=Path, required=True)
    parser.add_argument("--centralized-pool", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--actual-local-k", type=int, default=5)
    args = parser.parse_args()

    data = {query_id(row): row for row in rows(args.split)}
    fed = {str(row["query_id"]): row for row in rows(args.federated_pool)}
    central = {str(row["query_id"]): row for row in rows(args.centralized_pool)}
    client_for_doc = assignment(args.assignment)
    common = [qid for qid in fed if qid in central and qid in data]
    if args.max_queries is not None:
        common = common[: args.max_queries]
    if not common:
        raise ValueError("no aligned query IDs")

    per_query: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    merge_rows: list[dict[str, Any]] = []
    for qid in common:
        support = support_ids(data[qid], args.dataset)
        support_clients = sorted({client_for_doc[doc] for doc in support if doc in client_for_doc})
        pool = fed[qid]
        groups = client_groups(pool)
        selected = [int(value) for value in pool["selected_clients"]]
        selected_local = [
            doc for client in selected for doc in groups.get(client, [])[: args.actual_local_k]
        ]
        selected_local_depth = [doc for client in selected for doc in groups.get(client, [])]
        transmitted = round_robin_transmission(groups, selected, 15)
        raw = raw_merge(transmitted)
        calibrated = rank_percentile_merge(transmitted)
        central_docs = list(central[qid]["pool"])

        route_complete = int(set(support_clients).issubset(selected)) if support_clients else 0
        local_complete = complete(support, selected_local)
        local_depth_complete = complete(support, selected_local_depth)
        transmission_complete = complete(support, transmitted)
        raw10, raw15, raw20 = (complete(support, raw[:k]) for k in (10, 15, 20))
        cal10, cal15, cal20 = (complete(support, calibrated[:k]) for k in (10, 15, 20))
        central10, central15, central20 = (complete(support, central_docs[:k]) for k in (10, 15, 20))
        raw5 = complete(support, raw[:5])

        per_query.append({
            "dataset": args.dataset,
            "query_id": qid,
            "support_count": len(support),
            "support_client_count": len(support_clients),
            "oracle_minimum_client_budget": len(support_clients),
            "actual_clients_contacted": len(selected),
            "actual_documents_transmitted": len(transmitted),
            "central_complete_support_at_10": central10,
            "central_complete_support_at_15": central15,
            "central_complete_support_at_20": central20,
            "routing_complete_evidence_clients": route_complete,
            "local_complete_support_selected_clients": local_complete,
            "local_complete_support_selected_clients_at_available_depth": local_depth_complete,
            "support_rescued_by_local_depth": int(not local_complete and local_depth_complete),
            "transmission_complete_support_at_15": transmission_complete,
            "raw_complete_support_at_5": raw5,
            "raw_complete_support_at_10": raw10,
            "raw_complete_support_at_15": raw15,
            "raw_complete_support_at_20": raw20,
            "calibrated_rank_percentile_complete_support_at_10": cal10,
            "calibrated_rank_percentile_complete_support_at_15": cal15,
            "calibrated_rank_percentile_complete_support_at_20": cal20,
            "support_recall_raw_at_10": support_recall(support, raw[:10]),
            "support_recall_calibrated_at_10": support_recall(support, calibrated[:10]),
            "support_rescued_by_calibration_at_10": int(not raw10 and cal10),
            "support_lost_by_raw_merge_at_10": int(transmission_complete and not raw10),
            "support_lost_by_global_truncation_20_to_10": int(raw20 and not raw10),
            "support_lost_by_context_10_to_5": int(raw10 and not raw5),
        })
        local_rows.append({
            "dataset": args.dataset,
            "query_id": qid,
            "support_clients_available": int(bool(support_clients)),
            "oracle_selected_clients_cover_support": int(bool(support_clients)),
            "actual_selected_clients_cover_support": route_complete,
            "actual_local_topk_covers_support": local_complete,
            "available_local_depth_covers_support": local_depth_complete,
            "local_depth_rescue": int(not local_complete and local_depth_complete),
            "local_retrieval_absence_after_correct_route": int(route_complete and not local_complete),
        })
        merge_rows.append({
            "dataset": args.dataset,
            "query_id": qid,
            "support_complete_before_merge": transmission_complete,
            "raw_complete_at_10": raw10,
            "rank_percentile_complete_at_10": cal10,
            "raw_merge_loss": int(transmission_complete and not raw10),
            "rank_percentile_rescue": int(not raw10 and cal10),
            "global_truncation_loss_20_to_10": int(raw20 and not raw10),
        })
        for budget in (1, 2, 3, 4, 5, 20):
            budget_rows.append({
                "dataset": args.dataset,
                "query_id": qid,
                "client_budget": budget,
                "oracle_complete_evidence_client_recall": int(bool(support_clients) and len(support_clients) <= budget),
                "oracle_minimum_client_budget": len(support_clients),
                "actual_topic_router_available": int(budget == len(selected)),
                "actual_topic_router_complete_evidence_client_recall": route_complete if budget == len(selected) else "",
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_query_loss_decomposition.csv", per_query)
    write_csv(args.output_dir / "oracle_client_budget.csv", budget_rows)
    write_csv(args.output_dir / "oracle_local_retrieval.csv", local_rows)
    write_csv(args.output_dir / "oracle_merging.csv", merge_rows)

    stage_fields = [
        ("Centralized reference @20", "central_complete_support_at_20"),
        ("Actual selected clients cover all evidence", "routing_complete_evidence_clients"),
        ("Actual local retrieval within selected clients", "local_complete_support_selected_clients"),
        ("Transmission retention @15", "transmission_complete_support_at_15"),
        ("Raw merge complete support @10", "raw_complete_support_at_10"),
        ("Rank-percentile merge complete support @10", "calibrated_rank_percentile_complete_support_at_10"),
        ("Reader context complete support @5 (raw)", "raw_complete_support_at_5"),
    ]
    lines = [
        "# Stage U0 Loss Waterfall (Inherited Replay)", "",
        "This report is an offline gold-label decomposition over frozen V17 development pools. ",
        "It does not train or evaluate a router, it does not call a reader, and rank-percentile ",
        "merging is a label-free calibration baseline rather than a learned calibration result.", "",
        "| Stage | Complete-support rate | Delta from previous |",
        "|---|---:|---:|",
    ]
    previous = None
    for label, field in stage_fields:
        value = rate(per_query, field)
        delta = "--" if previous is None else f"{value - previous:+.3f}"
        lines.append(f"| {label} | {value:.3f} | {delta} |")
        previous = value
    lines.extend([
        "",
        f"- Queries: {len(per_query)}",
        f"- Routing absence after centralized reference: {rate(per_query, 'central_complete_support_at_20') - rate(per_query, 'routing_complete_evidence_clients'):+.3f}",
        f"- Local retrieval absence after routing: {rate(per_query, 'routing_complete_evidence_clients') - rate(per_query, 'local_complete_support_selected_clients'):+.3f}",
        f"- Raw merge loss at Top-10: {rate(per_query, 'support_lost_by_raw_merge_at_10'):.3f}",
        f"- Rank-percentile rescue at Top-10: {rate(per_query, 'support_rescued_by_calibration_at_10'):.3f}",
        "",
        "## Scope Limitation",
        "The inherited pool records local depth five.  This replay can diagnose the existing `Bc=3, local-k=5` contract, ",
        "but cannot decide a `local-k>5` document-allocation policy.  V20 must materialize a fresh all-client local-k=10 ",
        "pool before Stage U1/U2 conclusions or reader evaluation.",
    ])
    (args.output_dir / "loss_waterfall.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    routing_gap = rate(per_query, "central_complete_support_at_20") - rate(per_query, "routing_complete_evidence_clients")
    local_gap = rate(per_query, "routing_complete_evidence_clients") - rate(per_query, "local_complete_support_selected_clients")
    merge_gap = rate(per_query, "local_complete_support_selected_clients") - rate(per_query, "raw_complete_support_at_10")
    largest = max((routing_gap, "routing"), (local_gap, "local_retrieval"), (merge_gap, "merging"))[1]
    decision = {
        "status": "inherited_replay_complete",
        "dataset": args.dataset,
        "queries": len(per_query),
        "largest_observed_loss": largest,
        "routing_gap": routing_gap,
        "local_retrieval_gap": local_gap,
        "merge_gap": merge_gap,
        "reader_started": False,
        "next_required_artifact": "fresh all-client local-k=10 pool with local score distributions",
        "gold_used_only_for": "offline loss decomposition and oracle rows",
    }
    (args.output_dir / "stage0_go_no_go.md").write_text(
        "# Stage U0 Go/No-Go\n\n```json\n" + json.dumps(decision, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
