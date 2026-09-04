from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        for key in ("data", "examples", "train"):
            if isinstance(payload.get(key), list):
                yield from payload[key]
                return
        raise ValueError(f"Unsupported JSON mapping in {path}")


def normalize_title(value: str) -> str:
    return " ".join(str(value).lower().split())


def query_id(row: dict[str, Any]) -> str:
    for key in ("query_id", "_id", "id", "qid"):
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError("missing query ID")


def documents(row: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    context = row.get("context", [])
    pairs: list[tuple[str, list[str]]] = []
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        pairs = [(str(title), list(sentences[index]) if index < len(sentences) else []) for index, title in enumerate(titles)]
    elif isinstance(context, list):
        for value in context:
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                title, sentences = value[0], value[1]
                pairs.append((str(title), list(sentences) if isinstance(sentences, list) else [str(sentences)]))
            elif isinstance(value, dict):
                title = value.get("title", value.get("name", ""))
                sentences = value.get("sentences", value.get("text", []))
                pairs.append((str(title), list(sentences) if isinstance(sentences, list) else [str(sentences)]))
    output = []
    seen = set()
    for source_rank, (title, sentences) in enumerate(pairs):
        normalized = normalize_title(title)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        text = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())
        doc_id = f"{dataset}:{hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:20]}"
        output.append({"doc_id": doc_id, "title": title, "text": text, "sentences": sentences, "source_rank": source_rank})
    return output


def support_titles(row: dict[str, Any]) -> set[str]:
    facts = row.get("supporting_facts", [])
    if isinstance(facts, dict):
        values = facts.get("title", [])
    else:
        values = [item[0] for item in facts if isinstance(item, (list, tuple)) and item]
    return {normalize_title(value) for value in values}


def lexical_tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9]+", str(value).lower()) if len(token) > 1]


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]
