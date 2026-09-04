from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def entity_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in re.finditer(r"\b[A-Z][A-Za-z0-9_'-]{2,}\b", text or "")]


@dataclass
class HybridDocument:
    doc_id: str
    title: str
    text: str
    client_id: str = "client_0"
    is_support: bool = False
    support_role: str = ""
    bridge_entities: list[str] = field(default_factory=list)
    rare_tokens: list[str] = field(default_factory=list)
    dense_score_hint: float = 1.0
    soft_weight: float = 1.0

    @property
    def content(self) -> str:
        return f"{self.title}. {self.text}"


class BM25Scorer:
    def __init__(self, docs: Sequence[HybridDocument], k1: float = 1.2, b: float = 0.75) -> None:
        self.docs = list(docs)
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_tokens = [tokenize(d.content) for d in self.docs]
        self.doc_freq: Counter[str] = Counter()
        for toks in self.doc_tokens:
            self.doc_freq.update(set(toks))
        self.avgdl = sum(len(toks) for toks in self.doc_tokens) / max(len(self.doc_tokens), 1)
        self.n = len(self.docs)

    def score(self, query: str, index: int) -> float:
        q_tokens = tokenize(query)
        if not q_tokens:
            return 0.0
        toks = self.doc_tokens[index]
        if not toks:
            return 0.0
        tf = Counter(toks)
        dl = len(toks)
        score = 0.0
        for token in q_tokens:
            df = self.doc_freq.get(token, 0)
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            freq = tf.get(token, 0)
            denom = freq + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            if denom > 0:
                score += idf * freq * (self.k1 + 1.0) / denom
        return score


class HybridSoftRetriever:
    """Dense-sparse retrieval with explicit HP4 soft routing weights."""

    def __init__(
        self,
        docs: Sequence[HybridDocument],
        alpha: float = 0.55,
        dense_weight_mode: str = "identity",
        bridge_boost: float = 1.25,
        weight_temperature: float = 1.0,
    ) -> None:
        self.docs = list(docs)
        self.alpha = max(0.0, min(1.0, float(alpha)))
        self.dense_weight_mode = dense_weight_mode
        self.bridge_boost = float(bridge_boost)
        self.weight_temperature = max(1e-3, float(weight_temperature))
        self.bm25 = BM25Scorer(self.docs)

    @staticmethod
    def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
        aa, bb = set(a), set(b)
        if not aa or not bb:
            return 0.0
        return len(aa & bb) / len(aa | bb)

    def _scale_weight(self, w: float) -> float:
        w = max(0.0, min(1.0, float(w)))
        if self.dense_weight_mode == "sigmoid":
            return 1.0 / (1.0 + math.exp(-8.0 * (w - 0.5)))
        if self.dense_weight_mode in {"temperature", "temp_sigmoid"}:
            # Phase 3 sharpens the soft-routing mask late in training/inference.
            # The centered form keeps low weights suppressive while tau<1
            # polarizes high-advantage blocks toward the top of the pool.
            z = (w - 0.5) / self.weight_temperature
            return 1.0 / (1.0 + math.exp(-z))
        return w

    def dense_score(self, query: str, doc: HybridDocument) -> float:
        # Lightweight deterministic dense proxy. Formal HP4 can replace this
        # with transformer embeddings while preserving the same soft hook.
        lexical = self._jaccard(tokenize(query), tokenize(doc.content))
        bridge = self._jaccard(entity_tokens(query), doc.bridge_entities or entity_tokens(doc.content))
        return max(0.0, float(doc.dense_score_hint)) * (0.70 * lexical + 0.30 * bridge)

    def score(self, query: str, doc: HybridDocument, index: int, weights: Mapping[str, float] | None = None) -> dict[str, float]:
        raw_weight = doc.soft_weight if weights is None else weights.get(doc.doc_id, doc.soft_weight)
        w = self._scale_weight(raw_weight)
        dense = self.dense_score(query, doc) * w
        sparse = self.bm25.score(query, index) * w
        bridge_overlap = self._jaccard(entity_tokens(query), doc.bridge_entities)
        final = self.alpha * dense + (1.0 - self.alpha) * sparse + self.bridge_boost * bridge_overlap * w
        return {
            "final": final,
            "dense": dense,
            "sparse": sparse,
            "weight": w,
            "bridge_overlap": bridge_overlap,
        }

    def rank(
        self,
        query: str,
        weights: Mapping[str, float] | None = None,
        top_k: int = 5,
    ) -> list[tuple[HybridDocument, dict[str, float]]]:
        rows = [(doc, self.score(query, doc, idx, weights=weights)) for idx, doc in enumerate(self.docs)]
        rows.sort(key=lambda item: (item[1]["final"], item[1]["dense"], item[0].doc_id), reverse=True)
        return rows[:top_k]


def overlap_at_k(a: Sequence[str], b: Sequence[str], k: int) -> float:
    aa = set(a[:k])
    bb = set(b[:k])
    if k <= 0:
        return 0.0
    return len(aa & bb) / float(k)


def context_delta_audit(
    docs: Sequence[HybridDocument],
    query: str,
    target_doc_ids: Sequence[str],
    top_k: int = 5,
    alpha: float = 0.55,
    weight_temperature: float = 1.0,
) -> dict[str, object]:
    retriever = HybridSoftRetriever(docs, alpha=alpha, dense_weight_mode="temperature", weight_temperature=weight_temperature)
    low_weights = {doc.doc_id: 1.0 for doc in docs}
    high_weights = {doc.doc_id: 1.0 for doc in docs}
    for doc_id in target_doc_ids:
        low_weights[doc_id] = 0.0
        high_weights[doc_id] = 1.0
    low_rank = [doc.doc_id for doc, _ in retriever.rank(query, weights=low_weights, top_k=top_k)]
    high_rank = [doc.doc_id for doc, _ in retriever.rank(query, weights=high_weights, top_k=top_k)]
    return {
        "low_rank": low_rank,
        "high_rank": high_rank,
        "overlap_at_k": overlap_at_k(low_rank, high_rank, top_k),
        "target_doc_ids": list(target_doc_ids),
        "target_in_low": {doc_id: doc_id in low_rank for doc_id in target_doc_ids},
        "target_in_high": {doc_id: doc_id in high_rank for doc_id in target_doc_ids},
    }


def docs_from_micro_case(case: Mapping[str, object]) -> list[HybridDocument]:
    docs = []
    for raw in case.get("documents", []):
        if not isinstance(raw, Mapping):
            continue
        docs.append(HybridDocument(
            doc_id=str(raw.get("doc_id")),
            title=str(raw.get("title", "")),
            text=str(raw.get("text", "")),
            client_id=str(raw.get("client_id", "client_0")),
            is_support=bool(raw.get("is_support", False)),
            support_role=str(raw.get("support_role", "")),
            bridge_entities=[str(x).lower() for x in raw.get("bridge_entities", [])],
            rare_tokens=[str(x).lower() for x in raw.get("rare_tokens", [])],
            dense_score_hint=float(raw.get("dense_score_hint", 1.0)),
            soft_weight=float(raw.get("soft_weight", 1.0)),
        ))
    return docs
