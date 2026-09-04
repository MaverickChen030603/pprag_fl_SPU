from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TITLE_RE = re.compile(r"\[([^\]]+)\]\s*")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]


def parse_reference(reference: str) -> list[dict[str, Any]]:
    matches = list(TITLE_RE.finditer(reference or ""))
    docs = []
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(reference)
        text = reference[start:end].strip()
        docs.append({"title": title, "text": text, "sentences": split_sentences(text)})
    return docs


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def entity_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in re.finditer(r"\b[A-Z][A-Za-z0-9_'-]{2,}\b", text or "")]


def rare_tokens(text: str, max_tokens: int = 10) -> list[str]:
    toks = [t for t in tokens(text) if len(t) >= 7]
    seen = []
    for token in toks:
        if token not in seen:
            seen.append(token)
        if len(seen) >= max_tokens:
            break
    return seen


def build_case(item: dict[str, Any], idx: int, distractor_limit: int = 8) -> dict[str, Any] | None:
    support_titles = [str(t) for t in item.get("supporting_titles", [])]
    if len(support_titles) != 2:
        return None
    docs = parse_reference(str(item.get("reference", "")))
    by_title = {doc["title"]: doc for doc in docs}
    if not all(title in by_title and by_title[title]["text"] for title in support_titles):
        return None

    support_docs = [by_title[support_titles[0]], by_title[support_titles[1]]]
    distractors = [doc for doc in docs if doc["title"] not in support_titles and doc["text"]][:distractor_limit]
    if len(distractors) < 3:
        return None

    query_entities = entity_tokens(str(item.get("question", "")))
    documents = []
    for role, doc, client_id in [
        ("evidence_a", support_docs[0], "client_x"),
        ("evidence_b", support_docs[1], "client_y"),
    ]:
        documents.append({
            "doc_id": f"{item.get('_id', idx)}::{role}",
            "title": doc["title"],
            "text": doc["text"],
            "client_id": client_id,
            "is_support": True,
            "support_role": role,
            "bridge_entities": sorted(set(query_entities + entity_tokens(doc["title"] + " " + doc["text"])[:12])),
            "rare_tokens": rare_tokens(doc["text"]),
            "dense_score_hint": 3.0,
            "soft_weight": 1.0,
        })

    for didx, doc in enumerate(distractors):
        documents.append({
            "doc_id": f"{item.get('_id', idx)}::distractor_{didx}",
            "title": doc["title"],
            "text": doc["text"],
            "client_id": f"client_d{didx % 3}",
            "is_support": False,
            "support_role": "distractor",
            "bridge_entities": sorted(set(entity_tokens(doc["title"] + " " + doc["text"])[:12])),
            "rare_tokens": rare_tokens(doc["text"]),
            "dense_score_hint": 0.75,
            "soft_weight": 1.0,
        })

    return {
        "id": str(item.get("_id", idx)),
        "question": str(item.get("question", "")),
        "answer": item.get("answer", item.get("company", "")),
        "supporting_titles": support_titles,
        "sharding": {
            "evidence_a_client": "client_x",
            "evidence_b_client": "client_y",
            "rule": "Evidence A and B are isolated on disjoint clients; support docs are suppressed under w=0 and must surface under w=1.",
        },
        "documents": documents,
        "baseline_weights": {doc["doc_id"]: (0.0 if doc["is_support"] else 1.0) for doc in documents},
        "agent_oracle_weights": {doc["doc_id"]: 1.0 for doc in documents},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="V7-HP3/data/hotpot_dev_stratified_300.json")
    parser.add_argument("--output", default="data/v7_hp4_micro_benchmark.json")
    parser.add_argument("--target-size", type=int, default=30)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cases = []
    for idx, item in enumerate(data):
        case = build_case(item, idx)
        if case is not None:
            cases.append(case)
        if len(cases) >= args.target_size:
            break
    if len(cases) < min(args.target_size, 20):
        raise SystemExit(f"Only built {len(cases)} cases; need at least 20")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "source": args.input,
        "target_size": args.target_size,
        "actual_size": len(cases),
        "construction": "two-support HotpotQA cases with Evidence A/B isolated on client_x/client_y",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
