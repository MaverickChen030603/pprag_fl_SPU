#!/usr/bin/env python3
"""State-dependent atomic edits over an ordered five-document context.

The module intentionally contains no reader labels or support annotations. It
defines the inference-time transition system that is shared by oracle search,
greedy baselines, and learned composers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Sequence


CONTEXT_BUDGET = 5


class ActionKind(str, Enum):
    KEEP = "KEEP"
    REPLACE = "REPLACE"
    SWAP = "SWAP"
    MOVE = "MOVE"
    DROP_ADD = "DROP_ADD"
    STOP = "STOP"


@dataclass(frozen=True)
class AtomicAction:
    kind: ActionKind
    i: int | None = None
    j: int | None = None
    doc_id: str | None = None

    def key(self) -> tuple[str, int, int, str]:
        return (self.kind.value, self.i if self.i is not None else -1, self.j if self.j is not None else -1, self.doc_id or "")

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "i": self.i, "j": self.j, "doc_id": self.doc_id}

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> "AtomicAction":
        return cls(
            kind=ActionKind(str(row["kind"])),
            i=None if row.get("i") is None else int(row["i"]),
            j=None if row.get("j") is None else int(row["j"]),
            doc_id=None if row.get("doc_id") is None else str(row["doc_id"]),
        )


@dataclass(frozen=True)
class ContextState:
    context: tuple[str, ...]
    pool: tuple[str, ...]
    removed: tuple[str, ...] = ()
    history: tuple[AtomicAction, ...] = ()
    stopped: bool = False

    def __post_init__(self) -> None:
        if len(self.context) != CONTEXT_BUDGET:
            raise ValueError(f"context must contain exactly {CONTEXT_BUDGET} documents")
        if len(set(self.context)) != CONTEXT_BUDGET:
            raise ValueError("context documents must be unique")
        if len(set(self.pool)) != len(self.pool):
            raise ValueError("candidate pool documents must be unique")
        if not set(self.context).issubset(self.pool):
            raise ValueError("every context document must occur in the candidate pool")
        if len(self.history) > 3:
            raise ValueError("V16 trajectories are limited to at most three atomic edits")


def _require_index(index: int | None, name: str) -> int:
    if index is None or not 0 <= index < CONTEXT_BUDGET:
        raise ValueError(f"{name} must be in [0, {CONTEXT_BUDGET - 1}]")
    return index


def _require_new_document(state: ContextState, doc_id: str | None) -> str:
    if doc_id is None or doc_id not in state.pool:
        raise ValueError("replacement document must occur in the frozen candidate pool")
    if doc_id in state.context:
        raise ValueError("replacement document is already in the current context")
    return doc_id


def apply_action(state: ContextState, action: AtomicAction) -> ContextState:
    """Apply one legal action and return a new immutable context state."""
    if state.stopped:
        raise ValueError("cannot edit a trajectory after STOP")
    if len(state.history) >= 3:
        raise ValueError("trajectory depth would exceed three")

    values = list(state.context)
    removed = list(state.removed)
    if action.kind is ActionKind.STOP:
        return replace(state, history=state.history + (action,), stopped=True)
    if action.kind is ActionKind.KEEP:
        return replace(state, history=state.history + (action,))
    if action.kind is ActionKind.REPLACE:
        i = _require_index(action.i, "i")
        new_doc = _require_new_document(state, action.doc_id)
        removed.append(values[i])
        values[i] = new_doc
    elif action.kind is ActionKind.SWAP:
        i = _require_index(action.i, "i")
        j = _require_index(action.j, "j")
        if i == j:
            raise ValueError("SWAP positions must differ")
        values[i], values[j] = values[j], values[i]
    elif action.kind is ActionKind.MOVE:
        i = _require_index(action.i, "i")
        j = _require_index(action.j, "j")
        if i == j:
            raise ValueError("MOVE positions must differ")
        doc = values.pop(i)
        values.insert(j, doc)
    elif action.kind is ActionKind.DROP_ADD:
        i = _require_index(action.i, "i")
        new_doc = _require_new_document(state, action.doc_id)
        removed.append(values.pop(i))
        values.append(new_doc)
    else:  # pragma: no cover - Enum makes this defensive
        raise ValueError(f"unsupported action: {action.kind}")

    output = ContextState(
        context=tuple(values),
        pool=state.pool,
        removed=tuple(removed),
        history=state.history + (action,),
        stopped=False,
    )
    return output


def apply_trajectory(initial: ContextState, actions: Sequence[AtomicAction]) -> list[ContextState]:
    """Return the complete state trace, including the T=0 baseline state."""
    if len(actions) > 3:
        raise ValueError("trajectory depth must be in [0, 3]")
    trace = [initial]
    current = initial
    for action in actions:
        current = apply_action(current, action)
        trace.append(current)
        if current.stopped:
            break
    return trace


def legal_actions(
    state: ContextState,
    allowed: Iterable[ActionKind] | None = None,
    include_keep: bool = False,
) -> list[AtomicAction]:
    """Enumerate legal actions for the current state in deterministic order."""
    if state.stopped or len(state.history) >= 3:
        return []
    allowed_set = set(allowed or ActionKind)
    actions: list[AtomicAction] = []
    if include_keep and ActionKind.KEEP in allowed_set:
        actions.append(AtomicAction(ActionKind.KEEP))
    outside = [doc_id for doc_id in state.pool if doc_id not in state.context]
    if ActionKind.REPLACE in allowed_set:
        actions.extend(AtomicAction(ActionKind.REPLACE, i=i, doc_id=doc_id) for i in range(CONTEXT_BUDGET) for doc_id in outside)
    if ActionKind.SWAP in allowed_set:
        actions.extend(AtomicAction(ActionKind.SWAP, i=i, j=j) for i in range(CONTEXT_BUDGET) for j in range(i + 1, CONTEXT_BUDGET))
    if ActionKind.MOVE in allowed_set:
        actions.extend(AtomicAction(ActionKind.MOVE, i=i, j=j) for i in range(CONTEXT_BUDGET) for j in range(CONTEXT_BUDGET) if i != j)
    if ActionKind.DROP_ADD in allowed_set:
        actions.extend(AtomicAction(ActionKind.DROP_ADD, i=i, doc_id=doc_id) for i in range(CONTEXT_BUDGET) for doc_id in outside)
    if ActionKind.STOP in allowed_set:
        actions.append(AtomicAction(ActionKind.STOP))
    return sorted(actions, key=AtomicAction.key)
