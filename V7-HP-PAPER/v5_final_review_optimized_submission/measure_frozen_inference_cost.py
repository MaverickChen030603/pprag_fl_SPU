#!/usr/bin/env python3
"""Measure frozen V4/V5 post-retrieval inference without changing predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

import joblib
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
V4 = PROJECT / "V7-HP-PAPER/opportunity_aware_semantic_generation_v4"
V5 = PROJECT / "V7-HP-PAPER/review_driven_revision_v5"
OUT = HERE / "outputs/cost"
FIGURES = HERE / "outputs/figures"
FLAN = "/home/iiserver31/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
SYSTEMS = (
    "frozen_top5_baseline",
    "full_v4",
    "lite_lexical_pair",
    "baseline_truncated_660",
    "recomp_top1",
    "recomp_budgetmatched",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_module(path: Path, name: str) -> Any:
    for parent in (path.parent, V4, V5, PROJECT):
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * q
    low, high = int(index), min(len(ordered) - 1, int(index) + 1)
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_seconds": mean(values),
        "median_seconds": median(values),
        "p95_seconds": percentile(values, 0.95),
    }


def numeric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "p95": percentile(values, 0.95),
    }


def fingerprint(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def context_text(docs: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{rank}] {doc['title']}: {doc['text']}" for rank, doc in enumerate(docs, 1))


def reader_prompt(question: str, docs: list[dict[str, Any]]) -> str:
    return (
        "Answer the question using only the context. Return a short answer.\n\n"
        f"Question: {question}\n\nContext:\n{context_text(docs)[:3200]}\n\nAnswer:"
    )


def prepare_docs(item: dict[str, Any], snapshot: dict[str, Any], v4_common: Any) -> tuple[list[dict[str, Any]], list[str]]:
    baseline = v4_common.context_from_snapshot(snapshot, item["docs"])
    frozen = {str(doc["doc_id"]): doc for doc in baseline}
    docs = [frozen.get(str(doc["doc_id"]), doc) for doc in item["docs"]]
    existing = {str(doc["doc_id"]) for doc in docs}
    docs.extend(doc for doc in baseline if str(doc["doc_id"]) not in existing)
    return docs, [str(doc["doc_id"]) for doc in baseline]


def semantic_cache_from_lexical(
    question: str,
    docs: list[dict[str, Any]],
    baseline_ids: list[str],
    lexical: dict[str, dict[str, Any]],
    question_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    cross_scores: list[float],
    v4_common: Any,
    semantic: Any,
) -> dict[str, Any]:
    cross_normalized = semantic.minmax([float(value) for value in cross_scores])
    by_id = {str(doc["doc_id"]): index for index, doc in enumerate(docs)}
    baseline_indices = [by_id[doc_id] for doc_id in baseline_ids]
    candidate_ids = [str(doc["doc_id"]) for doc in docs if str(doc["doc_id"]) not in set(baseline_ids)]
    baseline_embeddings = doc_embeddings[baseline_indices]
    baseline_query = [semantic.cosine(question_embedding, doc_embeddings[index]) for index in baseline_indices]
    baseline_cross = [cross_normalized[index] for index in baseline_indices]
    baseline_pairs = [semantic.cosine(doc_embeddings[left], doc_embeddings[right]) for left, right in combinations(baseline_indices, 2)]
    details: dict[str, dict[str, float]] = {}
    vectors: dict[str, list[float]] = {}
    for index, doc in enumerate(docs):
        doc_id = str(doc["doc_id"])
        query_semantic = semantic.cosine(question_embedding, doc_embeddings[index])
        baseline_semantics = [semantic.cosine(doc_embeddings[index], value) for value in baseline_embeddings]
        row = lexical[doc_id]
        vector = [
            query_semantic,
            cross_normalized[index],
            float(row["bm25"]),
            float(row["query_overlap"]),
            float(row["title_overlap"]),
            float(row["entity_overlap"]),
            float(row["bridge_entity_match"]),
            float(row["novel_information"]),
            float(row["redundancy"]),
            max(baseline_semantics, default=0.0),
            float(np.mean(baseline_semantics)) if baseline_semantics else 0.0,
            1.0 - max(baseline_semantics, default=0.0),
            float(row["anchor_proxy"]),
            float(doc.get("source_rank", index)) / max(1, len(docs) - 1),
        ]
        vectors[doc_id] = vector
        details[doc_id] = dict(zip(semantic.DOC_FEATURE_NAMES, vector))
    question_entities = v4_common.capitalized_entities(question)
    query_vector = [
        max(baseline_query, default=0.0),
        float(np.mean(baseline_query)) if baseline_query else 0.0,
        min(baseline_query, default=0.0),
        float(np.std(baseline_query)) if baseline_query else 0.0,
        max(baseline_cross, default=0.0),
        float(np.mean(baseline_cross)) if baseline_cross else 0.0,
        float(np.mean(baseline_pairs)) if baseline_pairs else 0.0,
        float(np.mean([lexical[doc_id]["redundancy"] for doc_id in baseline_ids])),
        float(np.mean([lexical[doc_id]["bridge_entity_match"] for doc_id in baseline_ids])),
        float(np.log1p(len(v4_common.tokens(question)))),
        float(np.log1p(len(question_entities))),
    ]
    pair_features: dict[str, list[float]] = {}
    for left_id, right_id in combinations(candidate_ids, 2):
        left_index, right_index = by_id[left_id], by_id[right_id]
        left, right = details[left_id], details[right_id]
        left_entities = v4_common.capitalized_entities(f"{docs[left_index]['title']} {docs[left_index]['text']}")
        right_entities = v4_common.capitalized_entities(f"{docs[right_index]['title']} {docs[right_index]['text']}")
        pair_cosine = semantic.cosine(doc_embeddings[left_index], doc_embeddings[right_index])
        left_prior = 0.45 * left["query_doc_cosine"] + 0.35 * left["cross_encoder_relevance"] + 0.20 * left["bridge_entity_match"]
        right_prior = 0.45 * right["query_doc_cosine"] + 0.35 * right["cross_encoder_relevance"] + 0.20 * right["bridge_entity_match"]
        pair_features[semantic.pair_key(left_id, right_id)] = [
            left_prior,
            right_prior,
            left["query_doc_cosine"] + right["query_doc_cosine"],
            min(left["query_doc_cosine"], right["query_doc_cosine"]),
            left["cross_encoder_relevance"] + right["cross_encoder_relevance"],
            min(left["cross_encoder_relevance"], right["cross_encoder_relevance"]),
            pair_cosine,
            1.0 - pair_cosine,
            v4_common.jaccard(left_entities, right_entities),
            left["bridge_entity_match"] + right["bridge_entity_match"],
            left["novel_information"] + right["novel_information"],
            left["redundancy"] + right["redundancy"],
        ]
    return {
        "question": question,
        "baseline_ids": baseline_ids,
        "candidate_ids": candidate_ids,
        "docs": docs,
        "query_features": query_vector,
        "doc_features": vectors,
        "doc_feature_details": details,
        "pair_features": pair_features,
    }


class CachedClassifier:
    def __init__(self, model: Any, cache: dict[tuple[float, ...], np.ndarray]):
        self.classes_ = model.classes_
        self.cache = cache

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray([self.cache[tuple(np.asarray(row, dtype=np.float32).tolist())] for row in rows])


def cached_model(model: Any, features: list[list[float]]) -> tuple[CachedClassifier, int]:
    if not features:
        return CachedClassifier(model, {}), 0
    array = np.asarray(features, dtype=np.float32)
    values = model.predict_proba(array)
    cache = {tuple(row.tolist()): value for row, value in zip(array, values)}
    return CachedClassifier(model, cache), len(features)


class Benchmark:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.args = args
        self.torch = torch
        self.device = torch.device(args.device)
        self.v4_common = load_module(V4 / "v4_common.py", "final_cost_v4_common")
        self.full_generate = load_module(V4 / "14_generate_frozen_scaleup_actions.py", "final_cost_full_generate")
        self.semantic = load_module(V4 / "semantic_features.py", "final_cost_semantic")
        self.selector = load_module(V4 / "07_train_nested_selector_v4.py", "final_cost_selector")
        self.lite = load_module(V5 / "02_build_lite_generator.py", "final_cost_lite")
        self.recomp = load_module(V5 / "01_recomp_budget_matched.py", "final_cost_recomp")
        self.source = self.v4_common.load_source_examples()
        self.snapshots = self.v4_common.load_context_snapshots()
        self.query_ids = list(self.source)[: args.warmup + args.samples]
        if len(self.query_ids) != args.warmup + args.samples:
            raise AssertionError("Insufficient frozen development queries")
        self.reader_tokenizer = AutoTokenizer.from_pretrained(args.reader_model, local_files_only=True, use_fast=True)
        self.reader = AutoModelForSeq2SeqLM.from_pretrained(
            args.reader_model, local_files_only=True, torch_dtype=torch.float16
        ).to(self.device)
        self.reader.eval()
        self.full_actions = {str(row["action_id"]): row for row in read_jsonl(V4 / "outputs/generated_actions/v4_outer_test_actions.jsonl")}
        self.full_final = {str(row["query_id"]): row for row in read_jsonl(V4 / "outputs/nested_selector/v4_nested_per_query.jsonl")}
        lite_actions = read_jsonl(V5 / "outputs/lite_model/lite_actions_development.jsonl")
        self.lite_actions = {str(row["action_id"]): row for row in lite_actions if row.get("variant") == "lite_lexical_pair"}
        self.lite_final = {
            str(row["query_id"]): row
            for row in read_jsonl(V5 / "outputs/lite_model/lite_nested_per_query.jsonl")
            if row["variant"] == "lite_lexical_pair"
        }
        self.baseline = {
            str(row["query_id"]): row
            for row in self.full_actions.values()
            if row["action_family"] == "fallback"
        }
        self.full_models = {}
        manifest = read_json(V4 / "outputs/semantic_generator/foldwise_generator_models.json")
        for row in manifest["folds"]:
            self.full_models[int(row["fold_id"])] = joblib.load(row["model_path"])
        self.selector_models = {
            fold: joblib.load(V4 / f"outputs/scaleup/selector_models/fold_{fold}_selector.joblib")
            for fold in range(5)
        }
        self.lite_pair_models = {
            fold: joblib.load(V5 / f"outputs/lite_model/pair_models/fold_{fold}_lite_lexical_pair.joblib")
            for fold in range(5)
        }
        self.lite_selector_models = {
            fold: joblib.load(V5 / f"outputs/lite_model/selector_models/fold_{fold}_lite_lexical_pair.joblib")
            for fold in range(5)
        }
        self.extra_models(args.system)

    def extra_models(self, system: str) -> None:
        if system == "full_v4":
            from sentence_transformers import CrossEncoder, SentenceTransformer

            train = load_module(V4 / "03_train_semantic_candidate_generator.py", "final_cost_train")
            self.bi_encoder = SentenceTransformer(train.DEFAULT_BI_ENCODER, device=str(self.device), local_files_only=True)
            self.cross_encoder = CrossEncoder(train.DEFAULT_CROSS_ENCODER, device=str(self.device), local_files_only=True, max_length=512)
        elif system.startswith("recomp_") or system == "baseline_truncated_660":
            from transformers import AutoModel, AutoTokenizer

            model_name = read_json(V5 / "configs/revision_v5.json")["recomp"]["model"]
            self.compressor_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            if system.startswith("recomp_"):
                self.compressor = AutoModel.from_pretrained(model_name, local_files_only=True).to(self.device)
                self.compressor.eval()
            self.official_api = self.recomp.official_module()
            self.official = self.official_api.load_official(self.recomp.DEFAULT_ARROW)
            self.recomp_expected = {
                (str(row["query_id"]), str(row["method"])): row
                for row in read_jsonl(V5 / "outputs/recomp/contexts_development.jsonl")
                if row["method"] in ("recomp_top1", "recomp_budget_660", "baseline_truncated_660")
            }

    def sync(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def timed(self, function: Callable[[], Any]) -> tuple[Any, float]:
        self.sync()
        started = time.perf_counter()
        value = function()
        self.sync()
        return value, time.perf_counter() - started

    def run_reader(self, question: str, docs: list[dict[str, Any]]) -> tuple[str, int]:
        prompt = reader_prompt(question, docs)
        encoded = self.reader_tokenizer(prompt, truncation=True, max_length=1024, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.reader.generate(**encoded, max_new_tokens=32, num_beams=1, do_sample=False)
        prediction = self.reader_tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        context = context_text(docs)[:3200]
        tokens = len(self.reader_tokenizer.backend_tokenizer.encode(context, add_special_tokens=True).ids)
        return prediction, tokens

    def final_docs(self, system: str, query_id: str) -> list[dict[str, Any]]:
        if system == "frozen_top5_baseline":
            return list(self.baseline[query_id]["context_docs"])
        if system == "full_v4":
            return list(self.full_actions[str(self.full_final[query_id]["action_id"])]["context_docs"])
        return list(self.lite_actions[str(self.lite_final[query_id]["action_id"])]["context_docs"])

    def run_baseline(self, query_id: str) -> tuple[dict[str, float], str, int, int]:
        timings = {}
        item = self.source[query_id]
        (docs, _), timings["document_preprocessing"] = self.timed(
            lambda: prepare_docs(item, self.snapshots[query_id], self.v4_common)
        )
        final_docs = self.final_docs("frozen_top5_baseline", query_id)
        _, timings["final_context_serialization"] = self.timed(lambda: context_text(final_docs))
        (prediction, tokens), timings["final_reader"] = self.timed(lambda: self.run_reader(str(item["question"]), final_docs))
        return timings, prediction, tokens, 0

    def run_full(self, query_id: str) -> tuple[dict[str, float], str, int, int]:
        timings = {}
        item = self.source[query_id]
        (docs, baseline_ids), timings["document_preprocessing"] = self.timed(
            lambda: prepare_docs(item, self.snapshots[query_id], self.v4_common)
        )
        lexical, timings["lexical_features"] = self.timed(
            lambda: self.v4_common.lexical_doc_features(str(item["question"]), docs, baseline_ids)
        )
        doc_texts = [f"{doc['title']}. {doc['text']}" for doc in docs]

        def encode() -> tuple[np.ndarray, np.ndarray]:
            question = self.bi_encoder.encode([str(item["question"])], batch_size=1, normalize_embeddings=True, show_progress_bar=False)[0]
            documents = self.bi_encoder.encode(doc_texts, batch_size=len(doc_texts), normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(question), np.asarray(documents)

        (question_embedding, doc_embeddings), timings["mpnet_encoding"] = self.timed(encode)
        cross_scores, timings["cross_encoder_scoring"] = self.timed(
            lambda: [float(value) for value in self.cross_encoder.predict(
                [(str(item["question"]), text) for text in doc_texts], batch_size=len(doc_texts), show_progress_bar=False
            )]
        )
        query, timings["pair_feature_construction"] = self.timed(
            lambda: semantic_cache_from_lexical(
                str(item["question"]), docs, baseline_ids, lexical, question_embedding, doc_embeddings,
                cross_scores, self.v4_common, self.semantic
            )
        )
        fold = int(self.full_final[query_id]["outer_fold"])
        bundle = self.full_models[fold]
        missing_features = [query["query_features"]]
        missing_cache = {}

        def missing_score() -> None:
            wrapped, _ = cached_model(bundle["missing_model"], missing_features)
            missing_cache["model"] = wrapped

        _, timings["missing_hop_prediction"] = self.timed(missing_score)
        doc_features = [query["doc_features"][doc_id] for doc_id in query["candidate_ids"]]
        doc_cache = {}

        def document_score() -> None:
            wrapped, _ = cached_model(bundle["doc_model"], doc_features)
            doc_cache["model"] = wrapped

        _, timings["document_opportunity_scoring"] = self.timed(document_score)
        pair_features = [query["pair_features"][self.semantic.pair_key(left, right)] for left, right in combinations(query["candidate_ids"], 2)]
        pair_cache = {}

        def pair_score() -> None:
            if bundle["pair_model"] is None:
                pair_cache["model"] = None
            else:
                wrapped, _ = cached_model(bundle["pair_model"], pair_features)
                pair_cache["model"] = wrapped

        _, timings["pair_complementarity_scoring"] = self.timed(pair_score)
        cached_bundle = {
            **bundle,
            "missing_model": missing_cache["model"],
            "doc_model": doc_cache["model"],
            "pair_model": pair_cache["model"],
        }
        generated, timings["action_construction"] = self.timed(
            lambda: self.full_generate.generate_actions(query_id, fold, query, cached_bundle, 8)
        )
        effective = [row for row in generated if row["action_family"] != "fallback"]
        selector_bundle = self.selector_models[fold]
        _, timings["safety_head"] = self.timed(
            lambda: self.selector.probabilities(selector_bundle["safety_model"], effective)
        )
        _, timings["positive_utility_head"] = self.timed(
            lambda: self.selector.probabilities(selector_bundle["opportunity_model"], effective)
        )
        final_docs = self.final_docs("full_v4", query_id)
        frozen_ids = [str(doc["doc_id"]) for doc in final_docs]
        context_match = int(any(row["context_doc_ids"] == frozen_ids for row in generated))
        _, timings["final_context_serialization"] = self.timed(lambda: context_text(final_docs))
        (prediction, tokens), timings["final_reader"] = self.timed(lambda: self.run_reader(str(item["question"]), final_docs))
        return timings, prediction, tokens, context_match

    def run_lite(self, query_id: str) -> tuple[dict[str, float], str, int, int]:
        timings = {}
        item = self.source[query_id]
        (docs, baseline_ids), timings["document_preprocessing"] = self.timed(
            lambda: prepare_docs(item, self.snapshots[query_id], self.v4_common)
        )
        query, timings["lexical_and_pair_features"] = self.timed(
            lambda: self.lite.lexical_query_cache(str(item["question"]), docs, baseline_ids, self.v4_common)
        )
        fold = int(self.lite_final[query_id]["outer_fold"])
        generated, timings["pair_scoring_and_action_construction"] = self.timed(
            lambda: self.lite.generate_actions(
                query_id, fold, query, self.lite_pair_models[fold], "lite_lexical_pair", 6
            )
        )
        effective = [row for row in generated if row["action_family"] != "fallback"]
        selector_bundle = self.lite_selector_models[fold]
        _, timings["safety_head"] = self.timed(
            lambda: self.selector.probabilities(selector_bundle["safety_model"], effective)
        )
        _, timings["positive_utility_head"] = self.timed(
            lambda: self.selector.probabilities(selector_bundle["opportunity_model"], effective)
        )
        final_docs = self.final_docs("lite_lexical_pair", query_id)
        frozen_ids = [str(doc["doc_id"]) for doc in final_docs]
        context_match = int(any(row["context_doc_ids"] == frozen_ids for row in generated))
        _, timings["final_context_serialization"] = self.timed(lambda: context_text(final_docs))
        (prediction, tokens), timings["final_reader"] = self.timed(lambda: self.run_reader(str(item["question"]), final_docs))
        return timings, prediction, tokens, context_match

    def run_recomp(self, query_id: str, budget: int) -> tuple[dict[str, float], str, int, int]:
        timings = {}
        item = self.source[query_id]
        baseline_docs = list(self.baseline[query_id]["context_docs"])
        candidates, timings["document_and_sentence_preprocessing"] = self.timed(
            lambda: self.recomp.sentence_candidates(
                self.official[query_id], baseline_docs, self.official_api.normalize_title
            )
        )

        def score() -> list[dict[str, Any]]:
            texts = [str(item["question"])] + [row["text"] for row in candidates]
            encoded = self.compressor_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                hidden = self.compressor(**encoded)[0]
                embeddings = self.recomp.mean_pooling(hidden, encoded["attention_mask"])
                scores = self.torch.mv(embeddings[1:], embeddings[0]).detach().float().cpu().tolist()
            rows = [dict(row) for row in candidates]
            for row, value in zip(rows, scores):
                row["compressor_score"] = float(value)
            return sorted(rows, key=lambda row: (-row["compressor_score"], row["source_order"]))

        ranked, timings["official_compressor_scoring"] = self.timed(score)
        if budget == 1:
            selected, timings["sentence_packing"] = self.timed(lambda: ranked[:1])
            method = "recomp_top1"
        else:
            selected, timings["sentence_packing"] = self.timed(
                lambda: self.recomp.pack_nearest(ranked, budget, self.reader_tokenizer)
            )
            method = "recomp_budget_660"
        final_docs = [{"title": row["title"], "text": row["text"]} for row in selected]
        expected = self.recomp_expected[(query_id, method)]
        expected_signature = [(row["title"], row["text"]) for row in expected["sentences"]]
        context_match = int(expected_signature == [(row["title"], row["text"]) for row in selected])
        _, timings["final_context_serialization"] = self.timed(lambda: context_text(final_docs))
        (prediction, tokens), timings["final_reader"] = self.timed(lambda: self.run_reader(str(item["question"]), final_docs))
        return timings, prediction, tokens, context_match

    def run_baseline_truncated(self, query_id: str) -> tuple[dict[str, float], str, int, int]:
        timings = {}
        item = self.source[query_id]
        baseline_docs = list(self.baseline[query_id]["context_docs"])
        candidates, timings["document_and_sentence_preprocessing"] = self.timed(
            lambda: self.recomp.sentence_candidates(
                self.official[query_id], baseline_docs, self.official_api.normalize_title
            )
        )
        source_order = sorted(candidates, key=lambda row: row["source_order"])
        selected, timings["source_order_sentence_packing"] = self.timed(
            lambda: self.recomp.pack_nearest(source_order, 660, self.reader_tokenizer)
        )
        expected = self.recomp_expected[(query_id, "baseline_truncated_660")]
        expected_signature = [(row["title"], row["text"]) for row in expected["sentences"]]
        context_match = int(expected_signature == [(row["title"], row["text"]) for row in selected])
        final_docs = [{"title": row["title"], "text": row["text"]} for row in selected]
        _, timings["final_context_serialization"] = self.timed(lambda: context_text(final_docs))
        (prediction, tokens), timings["final_reader"] = self.timed(lambda: self.run_reader(str(item["question"]), final_docs))
        return timings, prediction, tokens, context_match

    def run_query(self, query_id: str) -> tuple[dict[str, float], str, int, int]:
        if self.args.system == "frozen_top5_baseline":
            return self.run_baseline(query_id)
        if self.args.system == "full_v4":
            return self.run_full(query_id)
        if self.args.system == "lite_lexical_pair":
            return self.run_lite(query_id)
        if self.args.system == "baseline_truncated_660":
            return self.run_baseline_truncated(query_id)
        return self.run_recomp(query_id, 1 if self.args.system == "recomp_top1" else 660)

    def execute(self) -> dict[str, Any]:
        component_rows: list[dict[str, float]] = []
        predictions: list[str] = []
        token_counts: list[int] = []
        matches: list[int] = []
        self.sync()
        for index, query_id in enumerate(self.query_ids):
            started = time.perf_counter()
            timings, prediction, tokens, match = self.run_query(query_id)
            self.sync()
            timings["end_to_end_post_retrieval"] = time.perf_counter() - started
            if index >= self.args.warmup:
                component_rows.append(timings)
                predictions.append(prediction)
                token_counts.append(tokens)
                matches.append(match)
            if (index + 1) % 25 == 0:
                print(json.dumps({"system": self.args.system, "completed": index + 1, "total": len(self.query_ids)}), flush=True)
        by_component: dict[str, list[float]] = defaultdict(list)
        for row in component_rows:
            for key, value in row.items():
                by_component[key].append(value)
        generator_keys = [
            key for key in by_component
            if key not in {"safety_head", "positive_utility_head", "final_reader", "end_to_end_post_retrieval"}
        ]
        selector_keys = [key for key in ("safety_head", "positive_utility_head") if key in by_component]
        generator = [sum(row.get(key, 0.0) for key in generator_keys) for row in component_rows]
        selector = [sum(row.get(key, 0.0) for key in selector_keys) for row in component_rows]
        reader = [row["final_reader"] for row in component_rows]
        total = [row["end_to_end_post_retrieval"] for row in component_rows]
        if self.args.system == "full_v4":
            encoder_calls, cross_calls, pairs = 2, 10, 10
        elif self.args.system == "lite_lexical_pair":
            encoder_calls, cross_calls, pairs = 0, 0, 10
        elif self.args.system.startswith("recomp_"):
            encoder_calls, cross_calls, pairs = 1, 0, 0
        else:
            encoder_calls, cross_calls, pairs = 0, 0, 0
        payload = {
            "status": "complete",
            "system": self.args.system,
            "device": str(self.device),
            "batch_size": 1,
            "warmup_queries": self.args.warmup,
            "measured_queries": self.args.samples,
            "query_fingerprint": fingerprint(self.query_ids),
            "model_loading_in_latency": False,
            "cuda_synchronize_each_component": True,
            "component_latency": {key: summary(values) for key, values in by_component.items()},
            "generator_only_latency": summary(generator),
            "selector_only_latency": summary(selector),
            "reader_only_latency": summary(reader),
            "end_to_end_post_retrieval_latency": summary(total),
            "throughput_queries_per_second": 1.0 / mean(total),
            "peak_gpu_memory_bytes": int(self.torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0,
            "encoder_calls_per_query": encoder_calls,
            "cross_encoder_document_scores_per_query": cross_calls,
            "pairs_scored_per_query": pairs,
            "final_reader_calls_per_query": 1,
            "context_tokens_per_query": numeric_summary([float(value) for value in token_counts]),
            "frozen_context_match_rate": mean(matches) if matches and self.args.system != "frozen_top5_baseline" else 1.0,
            "prediction_sha256": fingerprint(predictions),
            "prediction_count": len(predictions),
        }
        write_json(OUT / f"frozen_latency_{self.args.system}.json", payload)
        return payload


def combine() -> None:
    systems = {system: read_json(OUT / f"frozen_latency_{system}.json") for system in SYSTEMS}
    fingerprints = {row["query_fingerprint"] for row in systems.values()}
    if len(fingerprints) != 1:
        raise AssertionError("Systems did not use an identical query sample")
    payload = {
        "status": "complete",
        "protocol": read_json(HERE / "configs/final_submission_protocol.json"),
        "environment": {
            "hostname": platform.node(),
            "gpu": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
                text=True,
            ).strip(),
            "python": platform.python_version(),
        },
        "systems": systems,
        "same_query_fingerprint": True,
        "all_reader_calls_per_query_one": all(row["final_reader_calls_per_query"] == 1 for row in systems.values()),
        "all_frozen_context_match": all(row["frozen_context_match_rate"] == 1.0 for row in systems.values()),
    }
    write_json(OUT / "frozen_end_to_end_latency.json", payload)
    component_names = sorted({name for row in systems.values() for name in row["component_latency"]})
    with (OUT / "component_latency_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", "component", "mean_ms", "median_ms", "p95_ms"])
        writer.writeheader()
        for system, row in systems.items():
            for name in component_names:
                value = row["component_latency"].get(name)
                if value:
                    writer.writerow({"system": system, "component": name, "mean_ms": 1000 * value["mean_seconds"], "median_ms": 1000 * value["median_seconds"], "p95_ms": 1000 * value["p95_seconds"]})
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=SYSTEMS)
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--reader-model", default=os.environ.get("V4_FLAN_T5_LARGE", FLAN))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.combine:
        combine()
        return
    if not args.system:
        parser.error("--system is required unless --combine is used")
    benchmark = Benchmark(args)
    if benchmark.device.type == "cuda":
        benchmark.torch.cuda.reset_peak_memory_stats(benchmark.device)
    print(json.dumps(benchmark.execute(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
