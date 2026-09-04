#!/usr/bin/env python3
"""Inference-safe semantic feature construction for v4 generator training and use."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from v4_common import capitalized_entities, jaccard, lexical_doc_features, tokens


DOC_FEATURE_NAMES = [
    "query_doc_cosine", "cross_encoder_relevance", "bm25", "query_overlap", "title_overlap",
    "entity_overlap", "bridge_entity_match", "novel_information", "redundancy",
    "max_baseline_semantic", "mean_baseline_semantic", "semantic_novelty", "anchor_proxy",
    "source_rank_normalized",
]
QUERY_FEATURE_NAMES = [
    "baseline_query_semantic_max", "baseline_query_semantic_mean", "baseline_query_semantic_min",
    "baseline_query_semantic_std", "baseline_cross_max", "baseline_cross_mean",
    "baseline_pair_similarity_mean", "baseline_redundancy_mean", "baseline_bridge_mean",
    "question_token_log_length", "question_entity_log_count",
]
PAIR_FEATURE_NAMES = [
    "left_opportunity_prior", "right_opportunity_prior", "query_semantic_sum", "query_semantic_min",
    "cross_relevance_sum", "cross_relevance_min", "doc_pair_cosine", "semantic_complementarity",
    "entity_chain_overlap", "bridge_sum", "novel_information_sum", "redundancy_sum",
]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def build_query_cache(
    question: str,
    docs: list[dict[str, Any]],
    baseline_ids: list[str],
    question_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    cross_scores: list[float],
) -> dict[str, Any]:
    lexical = lexical_doc_features(question, docs, baseline_ids)
    cross_normalized = minmax([float(value) for value in cross_scores])
    by_id = {str(doc["doc_id"]): index for index, doc in enumerate(docs)}
    baseline_indices = [by_id[doc_id] for doc_id in baseline_ids]
    candidate_ids = [str(doc["doc_id"]) for doc in docs if str(doc["doc_id"]) not in set(baseline_ids)]
    baseline_embeddings = doc_embeddings[baseline_indices]
    baseline_query_cosines = [cosine(question_embedding, doc_embeddings[index]) for index in baseline_indices]
    baseline_cross = [cross_normalized[index] for index in baseline_indices]
    pair_cosines = [cosine(doc_embeddings[left], doc_embeddings[right]) for left, right in combinations(baseline_indices, 2)]

    doc_feature_vectors: dict[str, list[float]] = {}
    doc_feature_details: dict[str, dict[str, float]] = {}
    for index, doc in enumerate(docs):
        doc_id = str(doc["doc_id"])
        query_semantic = cosine(question_embedding, doc_embeddings[index])
        baseline_semantics = [cosine(doc_embeddings[index], value) for value in baseline_embeddings]
        lexical_row = lexical[doc_id]
        vector = [
            query_semantic,
            cross_normalized[index],
            float(lexical_row["bm25"]),
            float(lexical_row["query_overlap"]),
            float(lexical_row["title_overlap"]),
            float(lexical_row["entity_overlap"]),
            float(lexical_row["bridge_entity_match"]),
            float(lexical_row["novel_information"]),
            float(lexical_row["redundancy"]),
            max(baseline_semantics, default=0.0),
            sum(baseline_semantics) / max(1, len(baseline_semantics)),
            1.0 - max(baseline_semantics, default=0.0),
            float(lexical_row["anchor_proxy"]),
            float(doc.get("source_rank", index)) / max(1, len(docs) - 1),
        ]
        doc_feature_vectors[doc_id] = vector
        doc_feature_details[doc_id] = dict(zip(DOC_FEATURE_NAMES, vector))

    question_entities = capitalized_entities(question)
    query_vector = [
        max(baseline_query_cosines, default=0.0),
        float(np.mean(baseline_query_cosines)) if baseline_query_cosines else 0.0,
        min(baseline_query_cosines, default=0.0),
        float(np.std(baseline_query_cosines)) if baseline_query_cosines else 0.0,
        max(baseline_cross, default=0.0),
        float(np.mean(baseline_cross)) if baseline_cross else 0.0,
        float(np.mean(pair_cosines)) if pair_cosines else 0.0,
        float(np.mean([lexical[doc_id]["redundancy"] for doc_id in baseline_ids])),
        float(np.mean([lexical[doc_id]["bridge_entity_match"] for doc_id in baseline_ids])),
        float(np.log1p(len(tokens(question)))),
        float(np.log1p(len(question_entities))),
    ]

    pair_feature_vectors: dict[str, list[float]] = {}
    for left_id, right_id in combinations(candidate_ids, 2):
        left_index, right_index = by_id[left_id], by_id[right_id]
        left, right = doc_feature_details[left_id], doc_feature_details[right_id]
        left_entities = capitalized_entities(f"{docs[left_index]['title']} {docs[left_index]['text']}")
        right_entities = capitalized_entities(f"{docs[right_index]['title']} {docs[right_index]['text']}")
        pair_cosine = cosine(doc_embeddings[left_index], doc_embeddings[right_index])
        left_prior = 0.45 * left["query_doc_cosine"] + 0.35 * left["cross_encoder_relevance"] + 0.20 * left["bridge_entity_match"]
        right_prior = 0.45 * right["query_doc_cosine"] + 0.35 * right["cross_encoder_relevance"] + 0.20 * right["bridge_entity_match"]
        pair_feature_vectors[pair_key(left_id, right_id)] = [
            left_prior,
            right_prior,
            left["query_doc_cosine"] + right["query_doc_cosine"],
            min(left["query_doc_cosine"], right["query_doc_cosine"]),
            left["cross_encoder_relevance"] + right["cross_encoder_relevance"],
            min(left["cross_encoder_relevance"], right["cross_encoder_relevance"]),
            pair_cosine,
            1.0 - pair_cosine,
            jaccard(left_entities, right_entities),
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
        "doc_features": doc_feature_vectors,
        "doc_feature_details": doc_feature_details,
        "pair_features": pair_feature_vectors,
    }


def pair_key(left_id: str, right_id: str) -> str:
    return "|||".join(sorted([str(left_id), str(right_id)]))
