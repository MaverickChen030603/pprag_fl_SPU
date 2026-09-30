#!/usr/bin/env python3
"""P0L: read-only warm-cache latency audit for the frozen B3F-M routes.

The harness intentionally re-executes the existing SQLite sparse retrieval plus
BGE dense reranking path.  It never writes a route/context artifact and it
fails closed before timing if recomputed retrieval outputs differ from B3F-M.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import random
import sqlite3
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
METHODS = (
    "b0_static_top3",
    "ragroute_fixed3",
    "m2_logistic_proberoute",
    "ragroute_original_threshold",
    "b4a_all_candidate_top8_high_cost_reference",
)
FEATURES = (
    "dense_top1_score", "dense_top3_mean", "dense_top1_top2_margin",
    "dense_score_std", "dense_score_entropy", "dense_local_rank_percentile",
    "bm25_top1_score", "bm25_top3_mean", "bm25_top1_top2_margin",
    "dense_bm25_top1_same", "dense_bm25_top3_overlap",
    "dense_sparse_rank_correlation", "matched_query_entity_count",
    "matched_query_token_count", "matched_title_token_count",
    "query_title_embedding_similarity", "top3_title_diversity",
    "top3_entity_diversity",
)


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for value in values for key in value}) if values else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Harness:
    def __init__(self, root: Path, b3fm: Path, b3f: Path, device: str):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        import torch
        from sentence_transformers import SentenceTransformer

        self.torch, self.device = torch, device
        torch.set_num_threads(1)
        self.root, self.b3fm, self.b3f = root, b3fm, b3f
        self.asset = root / "experiments/proberoute_submission_baselines_20260915"
        self.arc = root / "V7-HP-PAPER/v20_arc_fedsearch"
        sys.path.insert(0, str(self.arc / "stage_r3_probe_route"))
        from run_probe_audit import (entities, entropy, entity_diversity, query_terms,
                                    rank_correlation, sparse_search, title_diversity)
        self.entities, self.entropy, self.entity_diversity = entities, entropy, entity_diversity
        self.query_terms, self.rank_correlation = query_terms, rank_correlation
        self.sparse_search, self.title_diversity = sparse_search, title_diversity
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
        index_root = self.asset / "inputs/c1_frozen_assets/V7-HP-PAPER/v17_fedaction_rag/retrieval/local_indexes"
        self.connections = {
            dataset: {client: sqlite3.connect(index_root / dataset / "topic_silo" / f"client_{client:02d}.sqlite")
                      for client in range(20)}
            for dataset in DATASETS
        }
        self.centroids = {dataset: np.load(self.asset / "runs/ragroute_b3_r5_posthoc_20260916/centroids" / dataset / "source_centroids.npy").astype(np.float32) for dataset in DATASETS}
        self.m2 = {dataset: pickle.load((self.asset / "inputs/models" / dataset / "logistic_seed_20260807.pkl").open("rb")) for dataset in DATASETS}
        sys.path.insert(0, str(b3f))
        from train_b3f_ragroute import B3FRouter
        self.rag = {}
        for dataset in DATASETS:
            scaler = np.load(b3f / "models" / dataset / "scaler.npz")
            models = []
            for seed in (0, 1, 2):
                checkpoint = torch.load(b3f / "models" / dataset / f"seed_{seed}.pt", map_location=device, weights_only=True)
                model = B3FRouter(int(checkpoint["input_dim"])).to(device)
                model.load_state_dict(checkpoint["state_dict"])
                models.append(model.eval())
            self.rag[dataset] = (models, scaler["mean"].astype(np.float32), scaler["scale"].astype(np.float32))

    def close(self):
        for per_dataset in self.connections.values():
            for connection in per_dataset.values():
                connection.close()

    def timed(self, fn: Callable[[], Any]) -> tuple[Any, float]:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        started = time.perf_counter_ns()
        result = fn()
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        return result, (time.perf_counter_ns() - started) / 1_000_000

    def embed(self, question: str) -> np.ndarray:
        return self.model.encode([question], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)

    def retrieve_client(self, dataset: str, client: int, question: str, q_emb: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        # Matches the frozen packet path: sparse top-100 then BGE dense rerank.
        documents = self.sparse_search(self.connections[dataset][client], question, 100)
        if not documents:
            return [], []
        embeddings = self.model.encode([f"{doc['title']}. {doc['text']}" for doc in documents], normalize_embeddings=True, convert_to_numpy=True, batch_size=256, show_progress_bar=False)
        for document, score in zip(documents, (embeddings @ q_emb).astype(float).tolist()):
            document["dense_score"] = score
        dense = sorted(documents, key=lambda doc: (-float(doc["dense_score"]), str(doc["doc_id"])))[:10]
        sparse = sorted(documents, key=lambda doc: (-float(doc["sparse_score"]), str(doc["doc_id"])))[:10]
        return dense, sparse

    def probe(self, dataset: str, candidates: list[int], question: str, q_emb: np.ndarray) -> tuple[list[dict[str, Any]], list[float]]:
        local, times = {}, []
        for client in candidates:
            result, elapsed = self.timed(lambda c=client: self.retrieve_client(dataset, c, question, q_emb))
            local[client], times = result, times + [elapsed]
        terms, q_entities = self.query_terms(question), self.entities(question)
        title_embeddings = self.model.encode([str(local[c][0]["title"]) if local[c] else "" for c in candidates], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        records = []
        for index, client in enumerate(candidates):
            dense, sparse = local[client]
            dense_scores, sparse_scores = [float(x["dense_score"]) for x in dense], [float(x["sparse_score"]) for x in sparse]
            dense_ids, sparse_ids = {str(x["doc_id"]) for x in dense[:3]}, {str(x["doc_id"]) for x in sparse[:3]}
            dense_rank = {str(x["doc_id"]): rank for rank, x in enumerate(dense)}
            anchor = 1.0 - dense_rank.get(str(sparse[0]["doc_id"]), len(dense)) / max(1, len(dense) - 1) if sparse else 0.0
            titles = [str(x["title"]) for x in dense[:3]]
            records.append({
                "client_id": client, "dense_top1_score": dense_scores[0] if dense_scores else 0.0,
                "dense_top3_mean": float(np.mean(dense_scores[:3])) if dense_scores else 0.0,
                "dense_top1_top2_margin": dense_scores[0] - dense_scores[1] if len(dense_scores) > 1 else 0.0,
                "dense_score_std": float(np.std(dense_scores)) if dense_scores else 0.0,
                "dense_score_entropy": self.entropy(dense_scores), "dense_local_rank_percentile": anchor,
                "bm25_top1_score": sparse_scores[0] if sparse_scores else 0.0,
                "bm25_top3_mean": float(np.mean(sparse_scores[:3])) if sparse_scores else 0.0,
                "bm25_top1_top2_margin": sparse_scores[0] - sparse_scores[1] if len(sparse_scores) > 1 else 0.0,
                "dense_bm25_top1_same": int(bool(dense and sparse and dense[0]["doc_id"] == sparse[0]["doc_id"])),
                "dense_bm25_top3_overlap": len(dense_ids & sparse_ids) / 3.0,
                "dense_sparse_rank_correlation": self.rank_correlation(dense, sparse),
                "matched_query_entity_count": len(q_entities & self.entities(str(dense[0]["title"]))) if dense else 0,
                "matched_query_token_count": len(terms & self.query_terms(" ".join(titles))),
                "matched_title_token_count": len(terms & self.query_terms(str(dense[0]["title"]))) if dense else 0,
                "query_title_embedding_similarity": float(q_emb @ title_embeddings[index]),
                "top3_title_diversity": self.title_diversity(titles), "top3_entity_diversity": self.entity_diversity(titles),
            })
        return records, times

    def rag_scores(self, dataset: str, candidates: list[int], q_emb: np.ndarray) -> np.ndarray:
        models, mean, scale = self.rag[dataset]
        centroids = self.centroids[dataset]
        matrix = np.concatenate((np.repeat(q_emb[None, :], len(candidates), axis=0), centroids[candidates], np.eye(centroids.shape[0], dtype=np.float32)[candidates]), axis=1)
        matrix = (matrix - mean) / np.where(scale == 0, 1, scale)
        tensor = self.torch.from_numpy(matrix.astype(np.float32)).to(self.device)
        with self.torch.inference_mode():
            return np.mean(np.stack([self.torch.sigmoid(model(tensor)).cpu().numpy() for model in models]), axis=0)


def frozen(stage: Path):
    packets, contexts = {}, {}
    for dataset in DATASETS:
        packets[dataset] = {str(row["query_id"]): row for row in rows(stage / "retrieval" / f"{dataset}_probe_packets.jsonl")}
    for row in rows(stage / "contexts/all_contexts_unscored.jsonl"):
        contexts[(row["dataset"], str(row["query_id"]), row["method"])] = row
    return packets, contexts


def route_time(h: Harness, dataset: str, method: str, packet: dict[str, Any], question: str):
    candidates = [int(x) for x in packet["p0_candidate_clients"]]
    components: dict[str, Any] = defaultdict(float)
    deep_docs: dict[int, list[dict[str, Any]]] = {}
    if method == "b0_static_top3":
        q_emb, components["query_embedding_ms"] = h.timed(lambda: h.embed(question))
        components["query_embedding"] = q_emb
        static, components["router_inference_ms"] = h.timed(lambda: sorted(packet["p0_candidate_records"], key=lambda x: int(x["static_candidate_rank"]))[:3])
        components["computed_selection"] = [int(x["client_id"]) for x in static]
    else:
        q_emb, components["query_embedding_ms"] = h.timed(lambda: h.embed(question))
        components["query_embedding"] = q_emb
        if method == "m2_logistic_proberoute":
            _, components["request_serialization_ms"] = h.timed(lambda: json.dumps({"question": question, "candidate_clients": candidates}, separators=(",", ":")))
            (records, probe_times), feature_ms = h.timed(lambda: h.probe(dataset, candidates, question, q_emb))
            # Feature time includes extraction/title embeddings; client times are retained separately.
            components["feature_assembly_ms"] = feature_ms - sum(probe_times)
            components["probe_client_times_ms"] = probe_times
            _, components["probe_aggregation_ms"] = h.timed(lambda: np.asarray([[float(x[k]) for k in FEATURES] for x in records], dtype=np.float32))
            features = np.asarray([[float(x["static_score"]), *[float(x[k]) for k in FEATURES]] for x in packet["p0_candidate_records"]], dtype=np.float32)
            scores, components["router_inference_ms"] = h.timed(lambda: h.m2[dataset]["model"].predict_proba(h.m2[dataset]["scaler"].transform(features))[:, 1])
            computed = [candidates[index] for index in np.argsort(-scores, kind="stable")[:3]]
            components["computed_selection"] = computed
        else:
            _, components["feature_assembly_ms"] = h.timed(lambda: h.centroids[dataset][candidates])
            scores, components["router_inference_ms"] = h.timed(lambda: h.rag_scores(dataset, candidates, q_emb))
            if method == "ragroute_fixed3": computed = sorted(candidates, key=lambda c: (-float(scores[candidates.index(c)]), c))[:3]
            else: computed = [c for index, c in enumerate(candidates) if float(scores[index]) > 0.5]
            components["computed_selection"] = computed
    return components


def run_method(h: Harness, dataset: str, method: str, packet: dict[str, Any], expected: dict[str, Any], repeat_id: int):
    question, selected = str(expected["question"]), [int(x) for x in expected["selected_client_ids"]]
    components = route_time(h, dataset, method, packet, question)
    # Deep retrieval is deliberately separate from probe retrieval: current implementation does not reuse it.
    deep_docs, deep_times = {}, []
    q_emb = components.pop("query_embedding")
    for client in selected:
        value, elapsed = h.timed(lambda c=client, e=q_emb: h.retrieve_client(dataset, c, question, e))
        deep_docs[client], deep_times = value[0], deep_times + [elapsed]
    _, serialization_ms = h.timed(lambda: json.dumps([[str(doc["doc_id"]) for doc in deep_docs[c][:5]] for c in selected], separators=(",", ":")))
    def merge():
        values = [doc for client in selected for doc in deep_docs[client][:5]]
        return sorted(values, key=lambda doc: (-float(doc["dense_score"]), str(doc["doc_id"])))[:10]
    merged, merge_ms = h.timed(merge)
    probe_times = components.get("probe_client_times_ms", [])
    route_ms = sum(float(components.get(key, 0.0)) for key in ("query_embedding_ms", "feature_assembly_ms", "router_inference_ms", "request_serialization_ms", "probe_aggregation_ms"))
    probe_serial, probe_parallel = sum(probe_times), max(probe_times, default=0.0)
    deep_serial, deep_parallel = sum(deep_times), max(deep_times, default=0.0)
    total_serial = route_ms + probe_serial + deep_serial + serialization_ms + merge_ms
    total_parallel = route_ms + probe_parallel + deep_parallel + serialization_ms + merge_ms
    reuse_parallel = route_ms + probe_parallel + serialization_ms + merge_ms if method == "m2_logistic_proberoute" else total_parallel
    return {
        "dataset": dataset, "query_id": str(expected["query_id"]), "method": method, "repeat_id": repeat_id,
        "computed_selection": json.dumps(components.get("computed_selection", selected)), "selected_clients": json.dumps(selected),
        "query_embedding_ms": components.get("query_embedding_ms", 0.0), "feature_assembly_ms": components.get("feature_assembly_ms", 0.0),
        "router_inference_ms": components.get("router_inference_ms", 0.0), "request_serialization_ms": components.get("request_serialization_ms", 0.0),
        "probe_aggregation_ms": components.get("probe_aggregation_ms", 0.0), "probe_client_times_ms": json.dumps(probe_times),
        "probe_serial_ms": probe_serial, "probe_parallel_ms": probe_parallel, "deep_client_times_ms": json.dumps(deep_times),
        "deep_serial_ms": deep_serial, "deep_parallel_ms": deep_parallel, "serialization_ms": serialization_ms, "merge_ms": merge_ms,
        "route_ms": route_ms + probe_serial, "serial_total_ms": total_serial, "parallel_critical_path_ms": total_parallel,
        "reuse_aware_parallel_estimate_ms": reuse_parallel, "aggregate_client_compute_ms": probe_serial + deep_serial,
        "aggregate_deep_compute_ms": deep_serial, "selected_client_count": len(selected), "docs_transmitted": len(selected) * 5,
        "merged_top10_doc_ids": json.dumps([str(doc["doc_id"]) for doc in merged]),
    }


def environment(stage: Path, h: Harness):
    def cmd(value):
        try: return subprocess.check_output(value, shell=True, text=True, stderr=subprocess.STDOUT).strip()
        except subprocess.CalledProcessError as error: return error.output.strip()
    value = {"python": sys.version, "platform": platform.platform(), "kernel": platform.release(), "cpu": cmd("lscpu"), "memory": cmd("free -h"), "gpu": cmd("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"), "cuda_available": h.torch.cuda.is_available(), "torch": h.torch.__version__, "cuda": h.torch.version.cuda, "cpu_affinity": sorted(os.sched_getaffinity(0)), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"), "torch_num_threads": h.torch.get_num_threads(), "index_location": str(h.asset / "inputs/c1_frozen_assets/V7-HP-PAPER/v17_fedaction_rag/retrieval/local_indexes"), "cache_contract": "warm cache: process/model/indexes/SQLite handles loaded before warmup; startup and model/index loading excluded", "latency_scope": "local single-host simulation; not network or distributed deployment latency"}
    (stage / "environment/latency_environment_manifest.json").write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--b3fm", type=Path, required=True)
    parser.add_argument("--b3f", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if args.stage.exists(): raise FileExistsError(args.stage)
    if args.repeats < 3: raise ValueError("P0L minimum is three repeats")
    args.stage.mkdir(parents=True)
    for directory in ("protocol", "environment", "harness", "latency", "statistics", "quality_join", "figures", "tables", "checksums", "reports"):
        (args.stage / directory).mkdir()
    (args.stage / "protocol/p0l_latency_contract.md").write_text("# P0L latency contract\n\nRead-only warm-cache latency microbenchmark over frozen B3F-M routes. Three repeats/query; first 50 queries/dataset/method are warmup only. Serial values are local simulation work paths. Parallel values are ideal-client critical-path estimates, never network latency.\n")
    packets, contexts = frozen(args.b3fm)
    h = Harness(args.root, args.b3fm, args.b3f, args.device)
    try:
        environment(args.stage, h)
        # Fail-closed equivalence audit before any official measurement.
        audit, rng = [], random.Random(args.seed)
        for dataset in DATASETS:
            sample = rng.sample(list(packets[dataset]), 100)
            for query_id in sample:
                packet = packets[dataset][query_id]
                for method in METHODS:
                    expected = contexts[(dataset, query_id, method)]
                    result = run_method(h, dataset, method, packet, expected, -1)
                    expected_local = expected["local_doc_ids"]
                    actual_top10 = json.loads(result["merged_top10_doc_ids"])
                    expected_top10 = [str(x) for x in expected["merged_top10_doc_ids"]]
                    route_match = json.loads(result["computed_selection"]) == [int(x) for x in expected["selected_client_ids"]]
                    retrieval_match = actual_top10 == expected_top10
                    audit.append({"dataset": dataset, "query_id": query_id, "method": method, "candidate_match": [int(x) for x in packet["p0_candidate_clients"]] == [int(x) for x in expected["candidate_client_ids"]], "selected_client_match": route_match, "merged_top10_match": retrieval_match, "status": "PASS" if route_match and retrieval_match else "FAIL", "expected_local_doc_ids": expected_local})
        passed = all(item["status"] == "PASS" for item in audit)
        (args.stage / "protocol/latency_harness_equivalence_audit.json").write_text(json.dumps({"status": "PASS" if passed else "benchmark_harness_mismatch", "sample_queries_per_dataset": 100, "comparisons": len(audit), "rows": audit}, indent=2) + "\n")
        if not passed: return
        order, measured = [], []
        # Per-method warmup under the same loaded model/index contract.
        for dataset in DATASETS:
            ids = list(packets[dataset])[:args.warmup]
            for method in METHODS:
                for query_id in ids: run_method(h, dataset, method, packets[dataset][query_id], contexts[(dataset, query_id, method)], -2)
        for dataset in DATASETS:
            for query_id in packets[dataset]:
                ordered = list(METHODS); rng.shuffle(ordered)
                order.append({"dataset": dataset, "query_id": query_id, "method_order": ordered})
                for repeat_id in range(args.repeats):
                    for method in ordered:
                        measured.append(run_method(h, dataset, method, packets[dataset][query_id], contexts[(dataset, query_id, method)], repeat_id))
        write_csv(args.stage / "latency/per_query_latency.csv", measured)
        medians = []
        numeric = ["route_ms", "probe_serial_ms", "probe_parallel_ms", "deep_serial_ms", "deep_parallel_ms", "aggregate_client_compute_ms", "aggregate_deep_compute_ms", "serial_total_ms", "parallel_critical_path_ms", "reuse_aware_parallel_estimate_ms"]
        grouped = defaultdict(list)
        for row in measured: grouped[(row["dataset"], row["query_id"], row["method"])].append(row)
        for key, values in grouped.items(): medians.append({"dataset": key[0], "query_id": key[1], "method": key[2], **{field: float(np.median([float(x[field]) for x in values])) for field in numeric}, "selected_client_count": values[0]["selected_client_count"], "docs_transmitted": values[0]["docs_transmitted"]})
        write_csv(args.stage / "latency/component_latency.csv", medians)
        write_csv(args.stage / "harness/benchmark_order_manifest.json", order)
        summary = []
        for (dataset, method), values in sorted(defaultdict(list, {(d, m): [x for x in medians if x["dataset"] == d and x["method"] == m] for d in DATASETS for m in METHODS}).items()):
            for metric in numeric:
                data = np.asarray([float(x[metric]) for x in values])
                summary.append({"dataset": dataset, "method": method, "metric": metric, "n": len(data), "mean_ms": float(data.mean()), "p50_ms": float(np.quantile(data, .50)), "p90_ms": float(np.quantile(data, .90)), "p95_ms": float(np.quantile(data, .95)), "p99_ms": float(np.quantile(data, .99)), "std_ms": float(data.std())})
        write_csv(args.stage / "statistics/latency_summary.csv", summary)
        h.close()
        (args.stage / "checksums/p0l_artifact_manifest.json").write_text(json.dumps({"status": "latency_audit_complete_parallel_estimate_only", "repeats": args.repeats, "files": {str(p.relative_to(args.stage)): sha256(p) for p in args.stage.rglob("*.csv")}}, indent=2) + "\n")
    finally:
        h.close()


if __name__ == "__main__": main()
