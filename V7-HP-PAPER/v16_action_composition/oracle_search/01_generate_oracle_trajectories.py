#!/usr/bin/env python3
"""Generate fixed-budget V16 oracle contexts without reader interaction.

Input rows must contain `query_id`, `baseline_doc_ids`, and `pool`, where each
pool item has `doc_id` and may contain retrieval/cross/entity/bridge scores.
The output is a context manifest. Reader outcomes are produced by a separate
offline labeling job so the inference-time composer never calls the reader.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from action_atoms.action_atoms import ActionKind, AtomicAction, ContextState, apply_action, legal_actions, shortest_repair_trajectory


@dataclass(frozen=True)
class Doc:
    doc_id: str
    retrieval_score: float = 0.0
    cross_score: float = 0.0
    bridge_score: float = 0.0


def docs_from_row(row: dict[str, Any], pool_size: int) -> list[Doc]:
    docs = []
    for item in row["pool"][:pool_size]:
        docs.append(Doc(
            doc_id=str(item.get("doc_id", item.get("id", item.get("title")))),
            retrieval_score=float(item.get("retrieval_score", item.get("score", 0.0))),
            cross_score=float(item.get("cross_score", item.get("ce_score", 0.0))),
            bridge_score=float(item.get("bridge_score", item.get("entity_overlap", 0.0))),
        ))
    if len({doc.doc_id for doc in docs}) != len(docs):
        raise ValueError(f"{row['query_id']}: duplicate pool document IDs")
    return docs


def stable_id(query_id: str, context: Sequence[str], depth: int, family: str) -> str:
    payload = f"{query_id}|{depth}|{family}|" + "|".join(context)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def score_context(context: Sequence[str], lookup: dict[str, Doc]) -> float:
    docs = [lookup[doc_id] for doc_id in context]
    relevance = sum(0.34 * doc.retrieval_score + 0.36 * doc.cross_score + 0.30 * doc.bridge_score for doc in docs)
    bridge_pair = max((min(left.bridge_score, right.bridge_score) for left, right in itertools.combinations(docs, 2)), default=0.0)
    return relevance + 0.25 * bridge_pair


def fixed_orders(subset: Sequence[str], baseline: Sequence[str], lookup: dict[str, Doc]) -> list[tuple[str, tuple[str, ...]]]:
    baseline_rank = {doc_id: index for index, doc_id in enumerate(baseline)}
    pool_rank = {doc_id: index for index, doc_id in enumerate(lookup)}
    orders = [
        ("baseline_relative", tuple(sorted(subset, key=lambda doc_id: (baseline_rank.get(doc_id, 100 + pool_rank[doc_id]), pool_rank[doc_id])))),
        ("crossencoder", tuple(sorted(subset, key=lambda doc_id: (-lookup[doc_id].cross_score, pool_rank[doc_id])))),
        ("bridge_first", tuple(sorted(subset, key=lambda doc_id: (-lookup[doc_id].bridge_score, -lookup[doc_id].cross_score, pool_rank[doc_id])))),
    ]
    unique: list[tuple[str, tuple[str, ...]]] = []
    seen: set[tuple[str, ...]] = set()
    for family, order in orders:
        if order not in seen:
            seen.add(order)
            unique.append((family, order))
    return unique


def top10_subsets(query_id: str, baseline: Sequence[str], docs: Sequence[Doc]) -> Iterable[dict[str, Any]]:
    lookup = {doc.doc_id: doc for doc in docs}
    for subset in itertools.combinations([doc.doc_id for doc in docs], 5):
        for order_name, context in fixed_orders(subset, baseline, lookup):
            witness = shortest_repair_trajectory(baseline, context, max_depth=3)
            depth = len(witness) if witness is not None else 5
            yield {
                "query_id": query_id,
                "trajectory_id": stable_id(query_id, context, depth, f"subset_{order_name}"),
                "candidate_type": "exhaustive_reachable_trajectory" if witness is not None else "exhaustive_subset",
                "order_family": order_name,
                "depth": depth,
                "actions": [] if witness is None else [action.to_dict() for action in witness],
                "context_doc_ids": list(context),
                "cheap_score": score_context(context, lookup),
                "is_baseline": tuple(context) == tuple(baseline),
            }


def all_single_edits(query_id: str, baseline: Sequence[str], docs: Sequence[Doc]) -> Iterable[dict[str, Any]]:
    lookup = {doc.doc_id: doc for doc in docs}
    initial = ContextState(context=tuple(baseline), pool=tuple(doc.doc_id for doc in docs))
    yield {
        "query_id": query_id, "trajectory_id": stable_id(query_id, baseline, 0, "baseline"),
        "candidate_type": "trajectory", "depth": 0, "actions": [], "context_doc_ids": list(baseline),
        "cheap_score": score_context(baseline, lookup), "is_baseline": True,
    }
    allowed = {ActionKind.REPLACE, ActionKind.SWAP, ActionKind.MOVE}
    for action in legal_actions(initial, allowed=allowed):
        updated = apply_action(initial, action)
        yield {
            "query_id": query_id, "trajectory_id": stable_id(query_id, updated.context, 1, "single"),
            "candidate_type": "trajectory", "depth": 1, "actions": [action.to_dict()],
            "context_doc_ids": list(updated.context), "cheap_score": score_context(updated.context, lookup) - 0.01,
            "is_baseline": False,
        }


def atomic_beam(query_id: str, baseline: Sequence[str], docs: Sequence[Doc], width: int, depth: int) -> Iterable[dict[str, Any]]:
    lookup = {doc.doc_id: doc for doc in docs}
    initial = ContextState(context=tuple(baseline), pool=tuple(doc.doc_id for doc in docs))
    baseline_row = {
        "query_id": query_id,
        "trajectory_id": stable_id(query_id, baseline, 0, "baseline"),
        "candidate_type": "trajectory",
        "depth": 0,
        "actions": [],
        "context_doc_ids": list(baseline),
        "cheap_score": score_context(baseline, lookup),
        "is_baseline": True,
    }
    yield baseline_row
    beam = [(initial, tuple())]
    seen: set[tuple[str, ...]] = {tuple(baseline)}
    allowed = {ActionKind.REPLACE, ActionKind.SWAP, ActionKind.MOVE, ActionKind.STOP}
    for step in range(1, depth + 1):
        expanded = []
        for state, trajectory in beam:
            for action in legal_actions(state, allowed=allowed):
                updated = apply_action(state, action)
                actions = trajectory + (action,)
                context = updated.context
                if action.kind is not ActionKind.STOP and context in seen:
                    continue
                seen.add(context)
                score = score_context(context, lookup) - 0.01 * len(actions)
                expanded.append((score, updated, actions))
                yield {
                    "query_id": query_id,
                    "trajectory_id": stable_id(query_id, context, step, "atomic"),
                    "candidate_type": "trajectory",
                    "depth": step,
                    "actions": [item.to_dict() for item in actions],
                    "context_doc_ids": list(context),
                    "cheap_score": score,
                    "is_baseline": False,
                    "stopped": updated.stopped,
                }
        expanded.sort(key=lambda row: (-row[0], row[1].context))
        beam = [(state, actions) for _, state, actions in expanded if not state.stopped][:width]
        if not beam:
            break


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, choices=(10, 20), required=True)
    parser.add_argument("--beam-width", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--depth", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--max-queries", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as output:
        for index, row in enumerate(read_jsonl(args.input)):
            if args.max_queries is not None and index >= args.max_queries:
                break
            docs = docs_from_row(row, args.pool_size)
            baseline = [str(value) for value in row["baseline_doc_ids"]]
            generators = [atomic_beam(str(row["query_id"]), baseline, docs, args.beam_width, args.depth)]
            if args.pool_size == 10:
                generators = [all_single_edits(str(row["query_id"]), baseline, docs), top10_subsets(str(row["query_id"]), baseline, docs)]
            for candidate in itertools.chain.from_iterable(generators):
                candidate.update({
                    "dataset": row.get("dataset", "unknown"),
                    "hop_count": row.get("hop_count", "unknown"),
                    "question_type": row.get("question_type", row.get("type", "unknown")),
                    "pool_size": args.pool_size,
                })
                output.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                count += 1
    print(json.dumps({"status": "complete", "contexts": count, "output": str(args.output)}))


if __name__ == "__main__":
    main()
