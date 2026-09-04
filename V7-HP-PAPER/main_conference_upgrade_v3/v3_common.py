#!/usr/bin/env python3
"""Shared, auditable utilities for the V7-HP main-conference v3 pipeline."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("FEDE4RAG_ROOT", HERE.parents[1])).expanduser().resolve()
OUTPUTS = HERE / "outputs"
REPORTS = HERE / "reports"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "in", "into",
    "is", "it", "its", "of", "on", "or", "she", "that", "the", "their", "them", "they",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
}
ARTICLES = {"a", "an", "the"}


def ensure_layout() -> None:
    for rel in [
        "candidate_generation", "action_outcomes", "nested_selector", "official_metrics",
        "multi_reader", "scaleup", "external_dataset", "tables", "figures", "audits",
    ]:
        (OUTPUTS / rel).mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (HERE / "configs").mkdir(parents=True, exist_ok=True)


def first_existing(candidates: Iterable[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("None of the expected artifacts exists:\n" + "\n".join(str(p) for p in candidates))


def source_1000_path() -> Path:
    override = os.environ.get("V3_HOTPOT_1000")
    candidates = [Path(override)] if override else []
    candidates += [
        PROJECT_ROOT / "V7-HP4/data/hotpot_validation_1000.json",
        PROJECT_ROOT / "tmp_submission_v2/remote_snapshot/hotpot_validation_1000.json",
    ]
    return first_existing(candidates)


def context_snapshot_path() -> Path:
    override = os.environ.get("V3_CONTEXT_SNAPSHOT")
    candidates = [Path(override)] if override else []
    candidates += [
        PROJECT_ROOT / "V7-HP-PAPER/high_tier_extension/multi_reader_context_repair/outputs/context_snapshots/final1000_baseline_selected_contexts.jsonl",
        PROJECT_ROOT / "tmp_v3_inputs/final1000_baseline_selected_contexts.jsonl",
    ]
    return first_existing(candidates)


def v2_action_labels_path() -> Path:
    override = os.environ.get("V3_V2_ACTION_LABELS")
    candidates = [Path(override)] if override else []
    candidates += [
        PROJECT_ROOT / "V7-HP-PAPER/selector_v2_3/outputs/labels/action_labels.jsonl",
        PROJECT_ROOT / "tmp_submission_v2/remote_snapshot/selector_v2_3/outputs/labels/action_labels.jsonl",
    ]
    return first_existing(candidates)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_title(value: str) -> str:
    return " ".join(html.unescape(str(value)).lower().split())


def parse_reference(reference: str, query_id: str) -> list[dict[str, Any]]:
    # Use the innermost bracket pair so a malformed distractor such as
    # ``legend[P.P.M. [Supergrass]`` does not swallow the following title.
    matches = list(re.finditer(r"\[([^\[\]]+)\]", reference))
    docs: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(reference)
        docs.append({
            "doc_id": f"{query_id}::doc_{index}",
            "title": html.unescape(match.group(1)).strip(),
            "text": html.unescape(reference[start:end]).strip(),
            "source_rank": index,
        })
    return docs


def load_source_examples() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in read_json(source_1000_path()):
        query_id = str(item.get("_id", item.get("id", "")))
        row = dict(item)
        row["query_id"] = query_id
        row["docs"] = parse_reference(str(item.get("reference", "")), query_id)
        out[query_id] = row
    return out


def load_context_snapshots() -> dict[str, dict[str, Any]]:
    return {str(row["query_id"]): row for row in read_jsonl(context_snapshot_path())}


def tokens(text: str, drop_stopwords: bool = True) -> list[str]:
    values = re.findall(r"[A-Za-z0-9]+", html.unescape(str(text)).lower())
    if drop_stopwords:
        return [value for value in values if value not in STOPWORDS and len(value) > 1]
    return values


def capitalized_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for phrase in re.findall(r"\b(?:[A-Z][A-Za-z0-9'&.-]*)(?:\s+(?:[A-Z][A-Za-z0-9'&.-]*|of|the|and)){0,5}", str(text)):
        value = " ".join(tokens(phrase))
        if value:
            entities.add(value)
            entities.update(token for token in value.split() if len(token) > 2)
    return entities


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left, right = set(a), set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def overlap_ratio(query_tokens: Iterable[str], doc_tokens: Iterable[str]) -> float:
    query, doc = set(query_tokens), set(doc_tokens)
    return len(query & doc) / len(query) if query else 0.0


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def bm25_scores(question: str, docs: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    corpus = [tokens(f"{doc['title']} {doc['text']}") for doc in docs]
    query = tokens(question)
    n_docs = max(1, len(corpus))
    avg_len = sum(len(doc) for doc in corpus) / n_docs or 1.0
    df = Counter(token for doc in corpus for token in set(doc))
    scores: list[float] = []
    for doc in corpus:
        tf = Counter(doc)
        score = 0.0
        for token in query:
            freq = tf[token]
            if not freq:
                continue
            idf = math.log(1.0 + (n_docs - df[token] + 0.5) / (df[token] + 0.5))
            denom = freq + k1 * (1.0 - b + b * len(doc) / avg_len)
            score += idf * freq * (k1 + 1.0) / denom
        scores.append(score)
    return scores


def doc_feature_rows(question: str, docs: list[dict[str, Any]], baseline_ids: list[str]) -> list[dict[str, Any]]:
    query_tokens = tokens(question)
    query_entities = capitalized_entities(question)
    baseline = [doc for doc in docs if doc["doc_id"] in set(baseline_ids)]
    baseline_title_tokens = set(tokens(" ".join(doc["title"] for doc in baseline)))
    baseline_entity_sets = [capitalized_entities(f"{doc['title']} {doc['text']}") for doc in baseline]
    raw_bm25 = bm25_scores(question, docs)
    norm_bm25 = minmax(raw_bm25)
    rows: list[dict[str, Any]] = []
    for index, doc in enumerate(docs):
        doc_tokens = tokens(f"{doc['title']} {doc['text']}")
        title_tokens = tokens(doc["title"])
        entities = capitalized_entities(f"{doc['title']} {doc['text']}")
        entity_overlap = jaccard(query_entities, entities)
        bridge_connection = jaccard(entities, baseline_title_tokens)
        baseline_overlap = max((jaccard(doc_tokens, tokens(f"{b['title']} {b['text']}")) for b in baseline if b["doc_id"] != doc["doc_id"]), default=0.0)
        entity_redundancy = max((jaccard(entities, value) for value in baseline_entity_sets), default=0.0)
        query_overlap = overlap_ratio(query_tokens, doc_tokens)
        title_overlap = overlap_ratio(query_tokens, title_tokens)
        anchor_proxy = 0.45 * norm_bm25[index] + 0.30 * query_overlap + 0.15 * title_overlap + 0.10 * entity_overlap
        rows.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "bm25_raw": raw_bm25[index],
            "bm25_score": norm_bm25[index],
            "query_overlap": query_overlap,
            "title_overlap": title_overlap,
            "entity_overlap": entity_overlap,
            "bridge_connection": bridge_connection,
            "novel_entity_ratio": len(entities - set().union(*baseline_entity_sets)) / max(1, len(entities)),
            "redundancy": max(baseline_overlap, entity_redundancy),
            "anchor_proxy_score": anchor_proxy,
            "baseline_rank": baseline_ids.index(doc["doc_id"]) if doc["doc_id"] in baseline_ids else -1,
        })
    return rows


def normalize_answer(text: Any) -> str:
    def remove_articles(value: str) -> str:
        return " ".join(word for word in value.split() if word not in ARTICLES)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        return "".join(char for char in value if char not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def answer_scores(prediction: Any, gold: Any) -> tuple[float, float]:
    pred, truth = normalize_answer(prediction), normalize_answer(gold)
    em = float(pred == truth)
    pred_tokens, truth_tokens = pred.split(), truth.split()
    if not pred_tokens or not truth_tokens:
        return em, float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    same = sum(common.values())
    if same == 0:
        return em, 0.0
    precision, recall = same / len(pred_tokens), same / len(truth_tokens)
    return em, 2 * precision * recall / (precision + recall)


def title_metrics(pred_titles: Iterable[str], gold_titles: Iterable[str]) -> tuple[float, float]:
    pred = {normalize_title(value) for value in pred_titles}
    gold = {normalize_title(value) for value in gold_titles}
    if not gold:
        return 0.0, 0.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if gold <= pred:
        f1 = max(f1, 1.0)
    return recall, f1


def answer_access(answer: str, docs: Iterable[dict[str, Any]]) -> float:
    needle = normalize_answer(answer)
    if not needle:
        return 0.0
    haystack = normalize_answer(" ".join(f"{doc['title']} {doc['text']}" for doc in docs))
    return float(needle in haystack)


def paired_bootstrap(diffs: list[float], rounds: int = 5000, seed: int = 20260713) -> dict[str, float | int]:
    if not diffs:
        return {"n": 0, "mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_value": 1.0}
    rng = random.Random(seed)
    n = len(diffs)
    samples = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds)]
    samples.sort()
    observed = sum(diffs) / n
    low = samples[int(0.025 * rounds)]
    high = samples[min(rounds - 1, int(0.975 * rounds))]
    p_value = min(1.0, 2.0 * min(sum(value <= 0 for value in samples) / rounds, sum(value >= 0 for value in samples) / rounds))
    return {"n": n, "mean": observed, "ci95_low": low, "ci95_high": high, "p_value": p_value}


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)
