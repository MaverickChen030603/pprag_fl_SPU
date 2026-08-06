#!/usr/bin/env python3
"""Evaluate fixed REMP profiles with one unmodified query embedding per query."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def did(dataset: str, title: str, text: str = "") -> str:
    value = norm(title) if dataset != "musique" else norm(title) + "\n" + norm(text)
    return f"{dataset}:{hashlib.sha1(value.encode()).hexdigest()[:20]}"


def support_docs(row: dict[str, Any], dataset: str) -> set[str]:
    if dataset == "musique":
        return {did(dataset, x.get("title", ""), x.get("paragraph_text", "")) for x in row.get("paragraphs", []) if x.get("is_supporting", x.get("is_support", False))}
    facts = row.get("supporting_facts", {})
    titles = facts.get("title", []) if isinstance(facts, dict) else [item[0] for item in facts if item]
    return {did(dataset, title) for title in titles}


def strata(row: dict[str, Any], dataset: str, gold_client_count: int) -> list[str]:
    names = [f"{min(gold_client_count, 3)}_client_evidence"]
    if gold_client_count > 1:
        names.append("cross_client_evidence")
    kind = str(row.get("type", "")).lower()
    if "comparison" in kind:
        names.append("comparison")
    if dataset == "2wikimultihopqa" and "compositional" in kind:
        names.append("bridge_entity_chain")
    if dataset == "musique" and row.get("question_decomposition"):
        names.append("bridge_entity_chain")
    return names


def pool(values: np.ndarray, name: str) -> float:
    ordered = np.sort(values)[::-1]
    if name == "S0_max":
        return float(ordered[0])
    if name == "S1_top3_mean":
        return float(ordered[:3].mean())
    if name == "S2_logsumexp":
        top = ordered[: min(32, len(ordered))]
        ceiling = float(top[0])
        return float(ceiling + np.log(np.exp(top - ceiling).sum()))
    raise ValueError(name)


def ranked(values: list[float]) -> list[int]:
    return [int(index) for index in np.argsort(-np.asarray(values, dtype=np.float32), kind="stable")]


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("2wikimultihopqa", "musique"), required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--legacy-profiles", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--selected-embeddings", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--best-p", type=int, required=True)
    parser.add_argument("--only-methods", help="comma-separated methods for a frozen Holdout; B0 is added automatically")
    parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    legacy = json.loads(args.legacy_profiles.read_text(encoding="utf-8"))["profiles"]
    unit_rows = list(rows(args.units))
    by_variant_client: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for unit in unit_rows:
        by_variant_client.setdefault((str(unit["variant"]), int(unit["client_id"])), []).append(unit)
    selected_embeddings = np.load(args.selected_embeddings)
    assignment = {str(item["doc_id"]): int(item["client_id"]) for item in rows(args.assignment)}
    data = list(rows(args.split))
    model = SentenceTransformer(args.encoder, device=args.device)
    variants = ("R0_cluster_medoids", "R1_farthest_first", "R2_cross_client_discriminative", "R3_hybrid_16_medoid_16_discriminative")
    poolings = ("S0_max", "S1_top3_mean", "S2_logsumexp")
    only_methods = {"B0_single_centroid"}
    if args.only_methods:
        only_methods.update(item for item in args.only_methods.split(",") if item)
    p0 = np.asarray([profile["p0_single_centroid"] for profile in legacy], dtype=np.float32)
    p1 = [np.asarray([item["centroid"] for item in profile["p1_multi_prototypes"][str(args.best_p)]], dtype=np.float32) for profile in legacy]

    results: list[dict[str, Any]] = []
    rank_records: list[dict[str, Any]] = []
    latency_ms: list[float] = []
    for row in data:
        started = time.perf_counter()
        query = str(row["question"])
        query_embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)
        scores: dict[str, tuple[list[float], dict[int, list[tuple[str, float]]]]] = {}
        scores["B0_single_centroid"] = (list((p0 @ query_embedding).astype(float)), {})
        if not args.only_methods or "B1_kmeans_multi_prototype" in only_methods:
            b1_scores = [float((prototypes @ query_embedding).max()) for prototypes in p1]
            scores["B1_kmeans_multi_prototype"] = (b1_scores, {})
        for variant in variants:
            requested_poolings = poolings if not args.only_methods else tuple(pooling for pooling in poolings if f"{variant}__{pooling}" in only_methods)
            if not requested_poolings:
                continue
            aggregate: list[float] = []
            unit_details: dict[int, list[tuple[str, float]]] = {}
            unit_similarities: dict[int, np.ndarray] = {}
            for client_id in range(20):
                key = f"{variant}__client_{client_id:02d}"
                vectors = np.asarray(selected_embeddings[key], dtype=np.float32)
                units = by_variant_client[(variant, client_id)]
                sims = vectors @ query_embedding
                unit_similarities[client_id] = sims
                unit_details[client_id] = [(units[index]["unit_id"], float(sims[index])) for index in np.argsort(-sims)[:3]]
                aggregate.append(0.0)  # filled separately for each pooling below
            for pooling in requested_poolings:
                aggregate = [pool(unit_similarities[client_id], pooling) for client_id in range(20)]
                scores[f"{variant}__{pooling}"] = (aggregate, unit_details)

        # Rankings are fully determined before gold support fields are read.
        rankings = {method: ranked(values) for method, (values, _details) in scores.items()}
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency_ms.append(elapsed_ms)
        gold_docs = support_docs(row, args.dataset)
        gold_clients = sorted({assignment[doc] for doc in gold_docs if doc in assignment})
        gold_set = set(gold_clients)
        if not gold_clients:
            raise ValueError(f"query {qid(row)} has no assigned gold clients")
        query_strata = strata(row, args.dataset, len(gold_clients))
        oracle_coverage = int(len(gold_clients) <= 3)
        for method, client_rank in rankings.items():
            values, unit_details = scores[method]
            rank_map = {client_id: rank + 1 for rank, client_id in enumerate(client_rank)}
            for cutoff in (3, 5, 8):
                chosen = client_rank[:cutoff]
                complete = int(gold_set.issubset(set(chosen)))
                recall = len(gold_set.intersection(chosen)) / len(gold_set)
                result = {
                    "dataset": args.dataset,
                    "query_id": qid(row),
                    "method": method,
                    "K": cutoff,
                    "strata": json.dumps(query_strata),
                    "gold_client_count": len(gold_set),
                    "gold_clients_offline_only": json.dumps(gold_clients),
                    "candidate_clients": json.dumps(chosen),
                    "gold_client_recall": recall,
                    "complete_client_set_recall": complete,
                    "oracle_coverage_at_3_offline_only": oracle_coverage,
                    "candidate_absence_loss_at_8": oracle_coverage - complete if cutoff == 8 else "",
                    "ranking_used_gold_answer_or_support": False,
                    "reader_started": False,
                }
                results.append(result)
            rank_records.append({
                "dataset": args.dataset,
                "query_id": qid(row),
                "method": method,
                "client_ranking": client_rank,
                "client_scores": [round(float(values[client]), 8) for client in client_rank],
                "top_matching_units": {str(client): [{"unit_id": unit_id, "score": round(score, 8)} for unit_id, score in unit_details.get(client, [])] for client in client_rank[:8]},
                "gold_client_ranks_offline_only": {str(client): rank_map[client] for client in gold_clients},
                "ranking_used_gold_answer_or_support": False,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "recovery_results_per_query.csv", results)
    with (args.output_dir / "per_query_client_ranks.jsonl").open("w", encoding="utf-8") as handle:
        for item in rank_records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    profile_cost = {
        "dataset": args.dataset,
        "queries": len(data),
        "query_profile_comparisons_per_variant": 20 * 32,
        "mean_routing_latency_ms": round(float(np.mean(latency_ms)), 4),
        "p95_routing_latency_ms": round(float(np.quantile(latency_ms, 0.95)), 4),
        "reader_started": False,
    }
    (args.output_dir / "candidate_run_manifest.json").write_text(json.dumps(profile_cost, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(profile_cost, indent=2))


if __name__ == "__main__":
    main()
