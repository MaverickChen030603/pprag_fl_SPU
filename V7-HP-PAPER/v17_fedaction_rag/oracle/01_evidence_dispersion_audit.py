#!/usr/bin/env python3
"""Offline gold-only audit of evidence dispersion across frozen clients."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def norm(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def qid(row: dict[str, Any]) -> str:
    return str(row.get("query_id", row.get("_id", row.get("id"))))


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def doc_id(dataset: str, title: str, text: str = "") -> str:
    if dataset == "musique":
        identity = norm(title) + "\n" + norm(text)
        return f"musique:{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:20]}"
    return f"{dataset}:{hashlib.sha1(norm(title).encode('utf-8')).hexdigest()[:20]}"


def documents(row: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    if dataset == "musique":
        return [
            {
                "doc_id": doc_id(dataset, paragraph.get("title", ""), paragraph.get("paragraph_text", "")),
                "title": str(paragraph.get("title", "")),
                "text": str(paragraph.get("paragraph_text", "")),
                "support": bool(paragraph.get("is_supporting", paragraph.get("is_support", False))),
            }
            for paragraph in row.get("paragraphs", [])
        ]
    context = row.get("context", [])
    if isinstance(context, dict):
        titles, sentences = context.get("title", []), context.get("sentences", [])
        pairs = [(title, sentences[index] if index < len(sentences) else []) for index, title in enumerate(titles)]
    else:
        pairs = [value[:2] for value in context if isinstance(value, (list, tuple)) and len(value) >= 2]
    supports = row.get("supporting_facts", [])
    if isinstance(supports, dict):
        support_titles = {norm(value) for value in supports.get("title", [])}
    else:
        support_titles = {norm(value[0]) for value in supports if isinstance(value, (list, tuple)) and value}
    output = []
    for title, sentences in pairs:
        text = " ".join(map(str, sentences)) if isinstance(sentences, list) else str(sentences)
        output.append({"doc_id": doc_id(dataset, str(title)), "title": str(title), "text": text, "support": norm(str(title)) in support_titles})
    return output


def load_assignment(path: Path) -> dict[str, int]:
    return {str(row["doc_id"]): int(row["client_id"]) for row in rows(path)}


def lexical_overlap(question: str, doc: dict[str, Any]) -> float:
    query = set(TOKEN_RE.findall(question.lower()))
    tokens = set(TOKEN_RE.findall((doc["title"] + " " + doc["text"]).lower()))
    return len(query & tokens) / max(1, len(query))


def partition_specs(root: Path) -> Iterable[dict[str, Any]]:
    for filename in ("topic_silo_manifest.json", "entity_community_manifest.json", "random_control_manifest.json", "dirichlet_manifest.json"):
        path = root / filename
        if path.exists():
            yield from json.loads(path.read_text(encoding="utf-8")).get("datasets", {}).values()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("evidence_dispersion.csv"))
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("evidence_dispersion_report.md"))
    parser.add_argument("--max-queries", type=int)
    args = parser.parse_args()

    output_rows = []
    summaries: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    entropy_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for spec in partition_specs(args.partition_root):
        dataset, partition = spec["dataset"], spec["partition"]
        assignment = load_assignment(Path(spec["assignment_path"]))
        source = args.data_root / dataset / f"{args.split}.jsonl"
        for index, row in enumerate(rows(source)):
            if args.max_queries is not None and index >= args.max_queries:
                break
            docs = documents(row, dataset)
            supports = [doc for doc in docs if doc["support"] and doc["doc_id"] in assignment]
            gold_supports = [doc for doc in docs if doc["support"]]
            support_clients = [assignment[doc["doc_id"]] for doc in supports]
            unique_support_clients = sorted(set(support_clients))
            counts = Counter(support_clients)
            total = sum(counts.values())
            entropy = -sum((count / total) * math.log((count / total) + 1e-12) for count in counts.values()) if total else 0.0
            answer = norm(row.get("answer", ""))
            answer_clients = sorted({assignment[doc["doc_id"]] for doc in docs if answer and answer in norm(doc["title"] + " " + doc["text"]) and doc["doc_id"] in assignment})
            distractors = [doc for doc in docs if not doc["support"] and doc["doc_id"] in assignment]
            closest = max(distractors, key=lambda doc: lexical_overlap(str(row.get("question", "")), doc), default=None)
            output_rows.append({
                "dataset": dataset,
                "partition": partition,
                "query_id": qid(row),
                "gold_support_documents": len(gold_supports),
                "support_documents": len(supports),
                "indexed_support_fraction": len(supports) / max(1, len(gold_supports)),
                "support_client_count": len(unique_support_clients),
                "all_evidence_one_client": int(len(unique_support_clients) == 1),
                "evidence_two_clients": int(len(unique_support_clients) == 2),
                "evidence_three_plus_clients": int(len(unique_support_clients) >= 3),
                "cross_client_evidence": int(len(unique_support_clients) >= 2),
                "evidence_dispersion_entropy": entropy,
                "answer_client_count": len(answer_clients),
                "closest_distractor_client": assignment.get(closest["doc_id"], -1) if closest else -1,
                "support_clients": "|".join(map(str, unique_support_clients)),
                "support_doc_ids": "|".join(sorted(doc["doc_id"] for doc in supports)),
                "support_doc_client_map": json.dumps(
                    {doc["doc_id"]: assignment[doc["doc_id"]] for doc in supports},
                    sort_keys=True,
                ),
            })
            key = (dataset, partition)
            summaries[key]["queries"] += 1
            summaries[key]["cross"] += int(len(unique_support_clients) >= 2)
            summaries[key]["two"] += int(len(unique_support_clients) == 2)
            summaries[key]["three_plus"] += int(len(unique_support_clients) >= 3)
            summaries[key]["one"] += int(len(unique_support_clients) == 1)
            entropy_values[key].append(entropy)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    lines = [
        "# V17 Federated Evidence Dispersion Audit",
        "",
        "Gold evidence is used only for this offline audit. It is unavailable to partitioning, routing, retrieval, and action generation.",
        "",
        "| Dataset | Partition | N | Cross-client | Two-client | Three-plus | One-client | Mean entropy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(summaries):
        values, n = summaries[key], summaries[key]["queries"]
        lines.append(
            f"| {key[0]} | {key[1]} | {n} | {values['cross']/n:.3f} | {values['two']/n:.3f} | "
            f"{values['three_plus']/n:.3f} | {values['one']/n:.3f} | {sum(entropy_values[key])/n:.3f} |"
        )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(output_rows), "output": str(args.output), "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
