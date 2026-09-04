#!/usr/bin/env python3
"""Evaluate REM-P client candidate recall without training or reader access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")
CAP = re.compile(r"(?:[A-Z][A-Za-z0-9'-]*(?:\s+|$)){1,5}")


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as h:
        for line in h:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def did(dataset: str, title: str, text: str = "") -> str:
    key = norm(title) if dataset != "musique" else norm(title) + "\n" + norm(text)
    return f"{dataset}:{hashlib.sha1(key.encode()).hexdigest()[:20]}"


def support_docs(row: dict[str, Any], dataset: str) -> set[str]:
    if dataset == "musique":
        return {
            did(dataset, x.get("title", ""), x.get("paragraph_text", ""))
            for x in row.get("paragraphs", [])
            if x.get("is_supporting", x.get("is_support", False))
        }
    facts = row.get("supporting_facts", {})
    titles = facts.get("title", []) if isinstance(facts, dict) else [x[0] for x in facts if x]
    return {did(dataset, title) for title in titles}


def query_views(question: str) -> dict[str, Any]:
    entities = [x.strip() for x in CAP.findall(question) if x.strip()]
    clauses = [
        x.strip()
        for x in re.split(
            r"\b(?:and|or|but|than|while|that|which|who|where)\b|[;,?]",
            question,
            flags=re.I,
        )
        if len(x.strip()) > 5
    ]
    relation = [
        x.strip()
        for x in re.findall(
            r"(?:of|in|by|from|with|for|between)\s+([A-Za-z][A-Za-z0-9' -]{2,40})",
            question,
            re.I,
        )
    ]
    return {
        "full_query": question,
        "entity_views": entities,
        "clause_views": clauses,
        "relation_views": relation,
        "view_count": 1 + len(entities) + len(clauses) + len(relation),
    }


def tokens(strings: list[str]) -> set[str]:
    return {x.lower() for s in strings for x in TOKEN.findall(s) if len(x) > 2}


def rank_desc(scores: list[float]) -> list[int]:
    return [int(x) for x in np.argsort(-np.asarray(scores, dtype=np.float32))]


def rrf_rank(*rankings: list[int], k: int = 60) -> list[int]:
    rank_maps = [{client: rank for rank, client in enumerate(r)} for r in rankings]
    return sorted(
        range(20),
        key=lambda client: (
            -sum(1.0 / (k + rank_map[client]) for rank_map in rank_maps),
            client,
        ),
    )


def rescue_harm(rows_for_method: list[dict[str, Any]], baseline_by_q: dict[str, int]) -> tuple[int, int]:
    rescue = 0
    harm = 0
    for row in rows_for_method:
        q = str(row["query_id"])
        value = int(row["complete_client_set_recall_at_L"])
        base = baseline_by_q[q]
        rescue += int(base == 0 and value == 1)
        harm += int(base == 1 and value == 0)
    return rescue, harm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--L", default="3,5,8")
    parser.add_argument("--topk-mean", type=int, default=5)
    parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    profile_payload = json.loads(args.profiles.read_text())
    profiles = profile_payload["profiles"]
    assignment = {str(x["doc_id"]): int(x["client_id"]) for x in rows(args.assignment)}
    data = list(rows(args.split))
    model = SentenceTransformer(args.encoder, device=args.device)
    cutoffs = [int(x) for x in args.L.split(",")]

    all_terms = [set(p["lexical_memory"]["term_counts"]) for p in profiles]
    df = {term: sum(term in terms_for_client for terms_for_client in all_terms) for terms_for_client in all_terms for term in terms_for_client}

    per_query = []
    query_view_rows = []
    for row in data:
        query_id = qid(row)
        view = query_views(str(row["question"]))
        strings = [view["full_query"]] + view["entity_views"] + view["clause_views"] + view["relation_views"]
        em = model.encode(
            strings,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        q_terms = tokens(strings)
        gold_docs = support_docs(row, args.dataset)
        gold_clients = sorted({assignment[doc] for doc in gold_docs if doc in assignment})

        scores = {
            "P0_single_centroid": [],
            "REMP_dense_max": [],
            "REMP_dense_topk_mean": [],
            "REMP_lexical": [],
        }
        for profile in profiles:
            centroid = np.asarray(profile["p0_single_centroid"], dtype=np.float32)
            scores["P0_single_centroid"].append(float(em[0] @ centroid))

            units = profile["representative_units"]
            unit_emb = np.asarray([u["embedding"] for u in units], dtype=np.float32)
            sims = em @ unit_emb.T
            per_unit = sims.max(axis=0) if len(sims.shape) == 2 else sims
            sorted_sims = np.sort(per_unit)[::-1]
            scores["REMP_dense_max"].append(float(sorted_sims[0]) if len(sorted_sims) else -1.0)
            topk = sorted_sims[: args.topk_mean]
            scores["REMP_dense_topk_mean"].append(float(topk.mean()) if len(topk) else -1.0)

            counts = profile["lexical_memory"]["term_counts"]
            lexical = 0.0
            for term in q_terms:
                if term in counts:
                    lexical += math.log1p(float(counts[term])) * math.log((21.0) / (1.0 + df.get(term, 0)))
            for entity, freq in profile["lexical_memory"].get("entity_frequency_sketch", {}).items():
                if entity.lower() in " ".join(strings).lower():
                    lexical += 0.5 * math.log1p(float(freq))
            scores["REMP_lexical"].append(float(lexical))

        rankings = {name: rank_desc(value) for name, value in scores.items()}
        rankings["REMP_rrf_dense_lexical"] = rrf_rank(
            rankings["REMP_dense_max"],
            rankings["REMP_lexical"],
        )
        rankings["REMP_rrf_p0_dense_lexical"] = rrf_rank(
            rankings["P0_single_centroid"],
            rankings["REMP_dense_max"],
            rankings["REMP_lexical"],
        )

        query_view_rows.append({"query_id": query_id, **view})
        for method, ranked in rankings.items():
            for cutoff in cutoffs:
                chosen = ranked[:cutoff]
                complete = int(set(gold_clients) <= set(chosen))
                recall = len(set(chosen) & set(gold_clients)) / max(1, len(gold_clients))
                per_query.append(
                    {
                        "dataset": args.dataset,
                        "query_id": query_id,
                        "method": method,
                        "L": cutoff,
                        "candidate_clients": json.dumps(chosen),
                        "gold_clients_offline_only": json.dumps(gold_clients),
                        "gold_client_recall_at_L": recall,
                        "complete_client_set_recall_at_L": complete,
                        "gold_or_answer_used_for_ranking": False,
                    }
                )

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "query_views.jsonl").open("w", encoding="utf-8") as h:
        for item in query_view_rows:
            h.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (args.output_root / "candidate_recall_per_query.csv").open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)

    summary = []
    for method in sorted({x["method"] for x in per_query}):
        for cutoff in cutoffs:
            subset = [x for x in per_query if x["method"] == method and x["L"] == cutoff]
            baseline = {
                str(x["query_id"]): int(x["complete_client_set_recall_at_L"])
                for x in per_query
                if x["method"] == "P0_single_centroid" and x["L"] == cutoff
            }
            rescue, harm = rescue_harm(subset, baseline)
            summary.append(
                {
                    "dataset": args.dataset,
                    "method": method,
                    "L": cutoff,
                    "queries": len(subset),
                    "candidate_gold_client_recall_at_L": sum(float(x["gold_client_recall_at_L"]) for x in subset) / len(subset),
                    "candidate_complete_client_set_recall_at_L": sum(int(x["complete_client_set_recall_at_L"]) for x in subset) / len(subset),
                    "delta_vs_p0_complete": (
                        sum(int(x["complete_client_set_recall_at_L"]) for x in subset) / len(subset)
                        - sum(baseline.values()) / len(baseline)
                    ),
                    "rescue_vs_p0": rescue,
                    "harm_vs_p0": harm,
                    "gold_used_only_for_offline_metrics": True,
                    "reader_started": False,
                }
            )
    with (args.output_root / "candidate_recall.csv").open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    decision = {
        "stage": "REM-P",
        "dataset": args.dataset,
        "queries": len(data),
        "reader_started": False,
        "final_test_accessed": False,
        "methods": sorted({x["method"] for x in per_query}),
        "summary_path": str((args.output_root / "candidate_recall.csv").resolve()),
    }
    (args.output_root / "remp_candidate_gate_manifest.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

