#!/usr/bin/env python3
"""Shared utilities for the fully nested V7-HP semantic-generation study."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("FEDE4RAG_ROOT", HERE.parents[1])).expanduser().resolve()
OUTPUTS = HERE / "outputs"
REPORTS = HERE / "reports"
PAPER = HERE / "paper"
V3_ROOT = PROJECT_ROOT / "V7-HP-PAPER/main_conference_upgrade_v3"
V2_ROOT = PROJECT_ROOT / "V7-HP-PAPER/submission_revision_v2"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "in", "into",
    "is", "it", "its", "of", "on", "or", "she", "that", "the", "their", "them", "they",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
}
ARTICLES = {"a", "an", "the"}
MAIN_V2_EXCLUDED_FAMILY = "keep_top3_insert2_strict"


def ensure_layout() -> None:
    for rel in [
        "audits", "semantic_generator", "generated_actions", "action_outcomes", "opportunity",
        "nested_selector", "official_metrics", "multi_reader", "scaleup", "external_dataset",
        "tables", "figures",
    ]:
        (OUTPUTS / rel).mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    (HERE / "configs").mkdir(parents=True, exist_ok=True)


def first_existing(candidates: Iterable[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("No expected artifact exists:\n" + "\n".join(str(path) for path in candidates))


def source_1000_path() -> Path:
    override = os.environ.get("V4_HOTPOT_1000")
    candidates = [Path(override)] if override else []
    candidates += [
        PROJECT_ROOT / "V7-HP4/data/hotpot_validation_1000.json",
        PROJECT_ROOT / "tmp_submission_v2/remote_snapshot/hotpot_validation_1000.json",
    ]
    return first_existing(candidates)


def context_snapshot_path() -> Path:
    override = os.environ.get("V4_CONTEXT_SNAPSHOT")
    candidates = [Path(override)] if override else []
    candidates += [
        PROJECT_ROOT / "V7-HP-PAPER/high_tier_extension/multi_reader_context_repair/outputs/context_snapshots/final1000_baseline_selected_contexts.jsonl",
        PROJECT_ROOT / "tmp_v3_inputs/final1000_baseline_selected_contexts.jsonl",
    ]
    return first_existing(candidates)


def v2_action_labels_path() -> Path:
    override = os.environ.get("V4_V2_ACTION_LABELS")
    candidates = [Path(override)] if override else []
    candidates += [
        PROJECT_ROOT / "V7-HP-PAPER/selector_v2_3/outputs/labels/action_labels.jsonl",
        PROJECT_ROOT / "tmp_submission_v2/remote_snapshot/selector_v2_3/outputs/labels/action_labels.jsonl",
    ]
    return first_existing(candidates)


def v3_actions_path() -> Path:
    return first_existing([V3_ROOT / "outputs/candidate_generation/v3_candidate_actions.jsonl"])


def v3_outcomes_path() -> Path:
    return first_existing([V3_ROOT / "outputs/action_outcomes/v3_action_reader_outputs.jsonl"])


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


def query_fingerprint(query_ids: Iterable[str]) -> str:
    value = "\n".join(sorted(str(query_id) for query_id in query_ids)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def normalize_title(value: str) -> str:
    return " ".join(html.unescape(str(value)).lower().split())


def parse_reference(reference: str, query_id: str) -> list[dict[str, Any]]:
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


def context_from_snapshot(snapshot: dict[str, Any], source_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_title = {normalize_title(doc["title"]): doc for doc in source_docs}
    baseline: list[dict[str, Any]] = []
    context_rows = snapshot.get("baseline_context", [])
    for index, title in enumerate(snapshot.get("baseline_titles", [])):
        source_doc = by_title.get(normalize_title(title))
        context_row = context_rows[index] if index < len(context_rows) else {}
        text = " ".join(str(value) for value in context_row.get("sentences", []))
        baseline.append({
            "doc_id": source_doc["doc_id"] if source_doc else f"{snapshot['query_id']}::frozen_baseline_{index}",
            "title": str(title),
            "text": text or (source_doc["text"] if source_doc else ""),
            "source_rank": source_doc.get("source_rank", -1) if source_doc else -1,
        })
    if not baseline:
        raise AssertionError(f"No frozen baseline context for {snapshot.get('query_id')}")
    return baseline


def load_v3_merged_rows() -> list[dict[str, Any]]:
    actions = {(str(row["query_id"]), str(row["action_id"])): row for row in read_jsonl(v3_actions_path())}
    merged: list[dict[str, Any]] = []
    for outcome in read_jsonl(v3_outcomes_path()):
        key = (str(outcome["query_id"]), str(outcome["action_id"]))
        row = dict(actions[key])
        row.update(outcome)
        merged.append(row)
    return merged


def v2_main_positive_query_ids() -> set[str]:
    return {
        str(row["query_id"])
        for row in read_jsonl(v2_action_labels_path())
        if row.get("candidate_name") != MAIN_V2_EXCLUDED_FAMILY and bool(row.get("paper_positive"))
    }


def tokens(text: str, drop_stopwords: bool = True) -> list[str]:
    values = re.findall(r"[A-Za-z0-9]+", html.unescape(str(text)).lower())
    if drop_stopwords:
        return [value for value in values if value not in STOPWORDS and len(value) > 1]
    return values


def capitalized_entities(text: str) -> set[str]:
    entities: set[str] = set()
    pattern = r"\b(?:[A-Z][A-Za-z0-9'&.-]*)(?:\s+(?:[A-Z][A-Za-z0-9'&.-]*|of|the|and)){0,5}"
    for phrase in re.findall(pattern, str(text)):
        value = " ".join(tokens(phrase))
        if value:
            entities.add(value)
            entities.update(token for token in value.split() if len(token) > 2)
    return entities


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_ratio(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a) if a else 0.0


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def bm25_scores(question: str, docs: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    corpus = [tokens(f"{doc['title']} {doc['text']}") for doc in docs]
    query = tokens(question)
    n_docs = max(1, len(corpus))
    average_length = sum(len(doc) for doc in corpus) / n_docs or 1.0
    document_frequency = Counter(token for doc in corpus for token in set(doc))
    scores: list[float] = []
    for doc in corpus:
        term_frequency = Counter(doc)
        score = 0.0
        for token in query:
            frequency = term_frequency[token]
            if not frequency:
                continue
            inverse_frequency = math.log(1.0 + (n_docs - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * len(doc) / average_length)
            score += inverse_frequency * frequency * (k1 + 1.0) / denominator
        scores.append(score)
    return scores


def lexical_doc_features(question: str, docs: list[dict[str, Any]], baseline_ids: list[str]) -> dict[str, dict[str, float | str | int]]:
    query_tokens = tokens(question)
    query_entities = capitalized_entities(question)
    baseline = [doc for doc in docs if doc["doc_id"] in set(baseline_ids)]
    baseline_tokens = [tokens(f"{doc['title']} {doc['text']}") for doc in baseline]
    baseline_entities = [capitalized_entities(f"{doc['title']} {doc['text']}") for doc in baseline]
    baseline_title_entities = capitalized_entities(" ".join(doc["title"] for doc in baseline))
    normalized_bm25 = minmax(bm25_scores(question, docs))
    output: dict[str, dict[str, float | str | int]] = {}
    for index, doc in enumerate(docs):
        doc_tokens = tokens(f"{doc['title']} {doc['text']}")
        title_tokens = tokens(doc["title"])
        entities = capitalized_entities(f"{doc['title']} {doc['text']}")
        query_overlap = overlap_ratio(query_tokens, doc_tokens)
        title_overlap = overlap_ratio(query_tokens, title_tokens)
        entity_overlap = jaccard(query_entities, entities)
        bridge = jaccard(entities, baseline_title_entities)
        redundancy = max((jaccard(doc_tokens, value) for value in baseline_tokens if value != doc_tokens), default=0.0)
        entity_redundancy = max((jaccard(entities, value) for value in baseline_entities if value != entities), default=0.0)
        baseline_entity_union = set().union(*baseline_entities) if baseline_entities else set()
        novelty = len(entities - baseline_entity_union) / max(1, len(entities))
        anchor = 0.45 * normalized_bm25[index] + 0.30 * query_overlap + 0.15 * title_overlap + 0.10 * entity_overlap
        output[str(doc["doc_id"])] = {
            "doc_id": str(doc["doc_id"]),
            "bm25": float(normalized_bm25[index]),
            "query_overlap": float(query_overlap),
            "title_overlap": float(title_overlap),
            "entity_overlap": float(entity_overlap),
            "bridge_entity_match": float(bridge),
            "novel_information": float(novelty),
            "redundancy": float(max(redundancy, entity_redundancy)),
            "anchor_proxy": float(anchor),
            "baseline_rank": baseline_ids.index(doc["doc_id"]) if doc["doc_id"] in baseline_ids else -1,
        }
    return output


def build_folds(query_ids: Iterable[str], k: int = 5) -> list[dict[str, Any]]:
    ordered = sorted(set(str(value) for value in query_ids), key=lambda value: int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16))
    folds: list[dict[str, Any]] = []
    for fold_id in range(k):
        test = ordered[fold_id::k]
        test_set = set(test)
        train = [query_id for query_id in ordered if query_id not in test_set]
        folds.append({
            "fold_id": fold_id,
            "n_train": len(train),
            "n_test": len(test),
            "train_query_ids": train,
            "test_query_ids": test,
            "train_fingerprint": query_fingerprint(train),
            "test_fingerprint": query_fingerprint(test),
        })
    return folds


def normalize_answer(text: Any) -> str:
    value = str(text).lower()
    value = "".join(character for character in value if character not in set(string.punctuation))
    value = " ".join(word for word in value.split() if word not in ARTICLES)
    return " ".join(value.split())


def answer_scores(prediction: Any, gold: Any) -> tuple[float, float]:
    pred, truth = normalize_answer(prediction), normalize_answer(gold)
    exact_match = float(pred == truth)
    pred_tokens, truth_tokens = pred.split(), truth.split()
    if not pred_tokens or not truth_tokens:
        return exact_match, float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())
    if not overlap:
        return exact_match, 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(truth_tokens)
    return exact_match, 2 * precision * recall / (precision + recall)


def title_metrics(pred_titles: Iterable[str], gold_titles: Iterable[str]) -> tuple[float, float]:
    pred = {normalize_title(value) for value in pred_titles}
    gold = {normalize_title(value) for value in gold_titles}
    if not gold:
        return 0.0, 0.0
    true_positive = len(pred & gold)
    precision = true_positive / len(pred) if pred else 0.0
    recall = true_positive / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if gold <= pred:
        f1 = max(f1, 1.0)
    return recall, f1


def answer_access(answer: str, docs: Iterable[dict[str, Any]]) -> float:
    needle = normalize_answer(answer)
    haystack = normalize_answer(" ".join(f"{doc['title']} {doc['text']}" for doc in docs))
    return float(bool(needle) and needle in haystack)


def grouped_outcomes(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    return grouped


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)
