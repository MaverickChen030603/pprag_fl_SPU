#!/usr/bin/env python3
"""Enumerated fixed-budget set repair for pools of at most twelve documents."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str = ""
    text: str = ""
    retrieval_score: float = 0.0
    cross_score: float = 0.0
    bridge_score: float = 0.0
    anchor_score: float = 0.0
    source_rank: int = 0


@dataclass(frozen=True)
class CandidateAction:
    action_id: str
    family: str
    doc_ids: tuple[str, ...]
    cheap_score: float
    added_doc_ids: tuple[str, ...]
    removed_doc_ids: tuple[str, ...]
    is_baseline: bool = False

    def to_dict(self) -> dict:
        row = asdict(self)
        row["doc_ids"] = list(self.doc_ids)
        row["added_doc_ids"] = list(self.added_doc_ids)
        row["removed_doc_ids"] = list(self.removed_doc_ids)
        return row


def stable_action_id(family: str, doc_ids: Sequence[str]) -> str:
    digest = hashlib.sha1((family + "\n" + "\n".join(doc_ids)).encode("utf-8")).hexdigest()[:16]
    return f"{family}:{digest}"


def sequence_score(sequence: Sequence[Document], baseline_ids: set[str]) -> float:
    if not sequence:
        return float("-inf")
    relevance = sum(0.30 * doc.retrieval_score + 0.28 * doc.cross_score for doc in sequence)
    bridge = sum(0.24 * doc.bridge_score for doc in sequence)
    anchors = sum(0.18 * doc.anchor_score for doc in sequence)
    baseline_preservation = len({doc.doc_id for doc in sequence} & baseline_ids) / len(sequence)
    rank_discount = sum((0.04 / (index + 1)) * (doc.cross_score + doc.anchor_score) for index, doc in enumerate(sequence))
    return relevance + bridge + anchors + 0.08 * baseline_preservation + rank_discount


def _orders(subset: Sequence[Document], baseline: Sequence[Document]) -> Iterable[tuple[str, tuple[Document, ...]]]:
    baseline_order = {doc.doc_id: index for index, doc in enumerate(baseline)}
    yield "baseline_preserving", tuple(sorted(
        subset,
        key=lambda doc: (doc.doc_id not in baseline_order, baseline_order.get(doc.doc_id, 10**6), -doc.retrieval_score, doc.source_rank),
    ))
    yield "retrieval_order", tuple(sorted(subset, key=lambda doc: (-doc.retrieval_score, doc.source_rank, doc.doc_id)))
    yield "crossencoder_order", tuple(sorted(subset, key=lambda doc: (-doc.cross_score, -doc.retrieval_score, doc.doc_id)))
    yield "bridge_first", tuple(sorted(subset, key=lambda doc: (-doc.bridge_score, -doc.anchor_score, -doc.cross_score, doc.doc_id)))


def _make_action(family: str, sequence: Sequence[Document], baseline: Sequence[Document]) -> CandidateAction:
    baseline_ids = {doc.doc_id for doc in baseline}
    ids = tuple(doc.doc_id for doc in sequence)
    current = set(ids)
    return CandidateAction(
        action_id=stable_action_id(family, ids),
        family=family,
        doc_ids=ids,
        cheap_score=sequence_score(sequence, baseline_ids),
        added_doc_ids=tuple(value for value in ids if value not in baseline_ids),
        removed_doc_ids=tuple(doc.doc_id for doc in baseline if doc.doc_id not in current),
        is_baseline=ids == tuple(doc.doc_id for doc in baseline),
    )


def enumerate_set_repairs(
    pool: Sequence[Document],
    baseline: Sequence[Document],
    top_k: int,
    context_budget: int = 5,
) -> list[CandidateAction]:
    """Enumerate complete contexts, order them deterministically, and prune.

    All features are inference-safe scores passed in with the documents. The
    exact baseline is retained even when its cheap score is below the Top-K.
    """
    if not 0 < len(pool) <= 12:
        raise ValueError("enumerated repair requires 1 <= pool size <= 12")
    if len(baseline) != context_budget:
        raise ValueError("baseline must exactly match the context budget")
    if not set(doc.doc_id for doc in baseline) <= set(doc.doc_id for doc in pool):
        raise ValueError("baseline documents must be members of the frozen pool")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    unique: dict[tuple[str, ...], CandidateAction] = {}
    for subset in itertools.combinations(pool, context_budget):
        for family, sequence in _orders(subset, baseline):
            action = _make_action(f"enumerated_{family}", sequence, baseline)
            previous = unique.get(action.doc_ids)
            if previous is None or action.cheap_score > previous.cheap_score:
                unique[action.doc_ids] = action

    baseline_action = _make_action("baseline_null", baseline, baseline)
    ranked = sorted(unique.values(), key=lambda row: (-row.cheap_score, row.doc_ids))
    selected = ranked[:top_k]
    if not any(row.is_baseline for row in selected):
        selected = ([baseline_action] if top_k == 1 else selected[:top_k - 1] + [baseline_action])
    return sorted(selected, key=lambda row: (not row.is_baseline, -row.cheap_score, row.doc_ids))

