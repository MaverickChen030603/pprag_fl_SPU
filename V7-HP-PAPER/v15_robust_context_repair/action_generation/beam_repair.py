#!/usr/bin/env python3
"""Beam search over complete five-document repair sequences."""

from __future__ import annotations

from typing import Sequence

from .candidate_generation import CandidateAction, Document, _make_action, sequence_score


def _dedupe(sequences: list[tuple[Document, ...]]) -> list[tuple[Document, ...]]:
    seen: set[tuple[str, ...]] = set()
    output = []
    for sequence in sequences:
        key = tuple(doc.doc_id for doc in sequence)
        if len(key) != len(set(key)) or key in seen:
            continue
        seen.add(key)
        output.append(sequence)
    return output


def _expand(sequence: tuple[Document, ...], pool: Sequence[Document]) -> list[tuple[Document, ...]]:
    current = {doc.doc_id for doc in sequence}
    outside = [doc for doc in pool if doc.doc_id not in current]
    children: list[tuple[Document, ...]] = []
    for position in range(len(sequence)):
        for replacement in outside:
            values = list(sequence)
            values[position] = replacement
            children.append(tuple(values))
    # Insert-and-remove: place a new document at each reading position, then
    # remove one existing document. This exposes orders unavailable to replace.
    for inserted in outside:
        for insert_at in range(len(sequence) + 1):
            expanded = list(sequence)
            expanded.insert(insert_at, inserted)
            for remove_at in range(len(expanded)):
                if remove_at != insert_at:
                    children.append(tuple(expanded[:remove_at] + expanded[remove_at + 1:]))
    children.extend([
        tuple(sorted(sequence, key=lambda doc: (-doc.retrieval_score, doc.source_rank, doc.doc_id))),
        tuple(sorted(sequence, key=lambda doc: (-doc.cross_score, -doc.retrieval_score, doc.doc_id))),
        tuple(sorted(sequence, key=lambda doc: (-doc.bridge_score, -doc.anchor_score, -doc.cross_score, doc.doc_id))),
    ])
    return _dedupe(children)


def beam_sequence_repairs(
    pool: Sequence[Document],
    baseline: Sequence[Document],
    beam_width: int,
    depth: int,
    output_k: int = 64,
) -> list[CandidateAction]:
    if len(pool) <= 12:
        raise ValueError("beam repair is reserved for pools larger than twelve")
    if len(baseline) != 5 or not set(doc.doc_id for doc in baseline) <= set(doc.doc_id for doc in pool):
        raise ValueError("beam repair requires an in-pool five-document baseline")
    if beam_width < 1 or depth < 1 or output_k < 1:
        raise ValueError("beam_width, depth, and output_k must be positive")

    baseline_ids = {doc.doc_id for doc in baseline}
    beam = [tuple(baseline)]
    all_sequences = {tuple(doc.doc_id for doc in baseline): tuple(baseline)}
    # An admissible-enough inference heuristic for pruning: current sequence
    # score plus the best possible relevance mass remaining in the pool.
    best_doc_bonus = max((0.30 * doc.retrieval_score + 0.28 * doc.cross_score + 0.24 * doc.bridge_score for doc in pool), default=0.0)
    for step in range(depth):
        expanded = _dedupe([child for sequence in beam for child in _expand(sequence, pool)])
        scored = []
        for sequence in expanded:
            score = sequence_score(sequence, baseline_ids)
            upper_bound = score + max(0, depth - step - 1) * best_doc_bonus
            scored.append((upper_bound, score, sequence))
            all_sequences[tuple(doc.doc_id for doc in sequence)] = sequence
        scored.sort(key=lambda row: (-row[0], -row[1], tuple(doc.doc_id for doc in row[2])))
        beam = [row[2] for row in scored[:beam_width]]

    actions = [_make_action("beam_repair", sequence, baseline) for sequence in all_sequences.values()]
    baseline_action = _make_action("baseline_null", baseline, baseline)
    ranked = sorted(actions, key=lambda row: (-row.cheap_score, row.doc_ids))[:output_k]
    if not any(action.is_baseline for action in ranked):
        ranked = ([baseline_action] if output_k == 1 else ranked[:output_k - 1] + [baseline_action])
    return sorted(ranked, key=lambda row: (not row.is_baseline, -row.cheap_score, row.doc_ids))

