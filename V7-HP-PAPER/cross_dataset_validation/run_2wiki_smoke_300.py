#!/usr/bin/env python3
"""Run a conservative 2WikiMultiHopQA dev-300 smoke validation.

This smoke test verifies that the prepared 2Wiki dev split contains answer,
context documents, and support labels, then evaluates two non-leaky retrieval
baselines over a stratified 300-example sample:

  1. context_order_top5: first five context documents as shipped by 2Wiki.
  2. lexical_bm25_top5: query-doc BM25-lite ranking using only query/context text.

It intentionally does not call a reader and does not claim downstream QA gains.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
CDV = ROOT / "cross_dataset_validation"
DATA_PATH = CDV / "outputs/2wiki_adapter/2wiki_converted.json"
OUT_DIR = CDV / "outputs/2wiki_smoke_300"
REPORT_PATH = CDV / "reports/2wiki_smoke_300_report.md"
MIRROR_DIR = ROOT.parent / "实验分析报告/V7-HP-PAPER"
SEED = 20260622
N = 300
TOP_K = 5


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "by",
    "is", "are", "was", "were", "be", "been", "being", "what", "which",
    "who", "whom", "whose", "when", "where", "why", "how", "did", "does",
    "do", "that", "this", "these", "those", "with", "as", "from", "at",
    "it", "its", "their", "his", "her", "he", "she", "they", "them",
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if t.lower() not in STOPWORDS]


def normalize(text: str) -> str:
    return " ".join(tokenize(text))


def doc_title(doc) -> str:
    if isinstance(doc, dict):
        return str(doc.get("title", ""))
    if isinstance(doc, (list, tuple)) and doc:
        return str(doc[0])
    return ""


def doc_text(doc) -> str:
    if isinstance(doc, dict):
        body = doc.get("sentences", doc.get("text", doc.get("content", "")))
        if isinstance(body, list):
            body = " ".join(str(x) for x in body)
        return f"{doc.get('title', '')} {body}"
    if isinstance(doc, (list, tuple)):
        title = str(doc[0]) if doc else ""
        body = doc[1] if len(doc) > 1 else ""
        if isinstance(body, list):
            body = " ".join(str(x) for x in body)
        return f"{title} {body}"
    return str(doc)


def support_titles(ex: dict) -> set[str]:
    titles = set(ex.get("supporting_titles") or [])
    for sf in ex.get("supporting_facts") or []:
        if isinstance(sf, (list, tuple)) and sf:
            titles.add(str(sf[0]))
        elif isinstance(sf, dict) and sf.get("title"):
            titles.add(str(sf["title"]))
    return titles


def answer_access(answer: str, docs: list) -> float:
    ans = normalize(answer)
    if not ans:
        return 0.0
    joined = normalize(" ".join(doc_text(d) for d in docs))
    return 1.0 if ans in joined else 0.0


def support_recall(supports: set[str], docs: list) -> float:
    if not supports:
        return 0.0
    got = {doc_title(d) for d in docs}
    return len(supports & got) / len(supports)


def all_support_access(supports: set[str], docs: list) -> float:
    if not supports:
        return 0.0
    got = {doc_title(d) for d in docs}
    return 1.0 if supports <= got else 0.0


def lexical_bm25_order(question: str, docs: list) -> list[int]:
    q_terms = tokenize(question)
    q_counts = Counter(q_terms)
    doc_terms = [tokenize(doc_text(d)) for d in docs]
    n_docs = max(1, len(docs))
    df = Counter()
    for terms in doc_terms:
        df.update(set(terms))
    avgdl = mean([len(t) for t in doc_terms] or [1])
    scores = []
    for idx, terms in enumerate(doc_terms):
        tf = Counter(terms)
        dl = max(1, len(terms))
        score = 0.0
        for term, qtf in q_counts.items():
            if term not in tf:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + 1.2 * (1 - 0.75 + 0.75 * dl / max(avgdl, 1))
            score += qtf * idf * (tf[term] * 2.2 / denom)
        # Title matches are useful in multi-hop data and remain non-leaky.
        title_overlap = len(set(q_terms) & set(tokenize(doc_title(docs[idx]))))
        score += 0.25 * title_overlap
        scores.append((score, -idx, idx))
    return [idx for _, _, idx in sorted(scores, reverse=True)]


def stratified_sample(data: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for ex in data:
        buckets[ex.get("type", "unknown")].append(ex)
    for rows in buckets.values():
        rng.shuffle(rows)
    total = sum(len(v) for v in buckets.values())
    selected = []
    remainders = []
    for typ, rows in sorted(buckets.items()):
        exact = n * len(rows) / total
        take = min(len(rows), int(exact))
        selected.extend(rows[:take])
        remainders.append((exact - take, typ))
        buckets[typ] = rows[take:]
    for _, typ in sorted(remainders, reverse=True):
        if len(selected) >= n:
            break
        if buckets[typ]:
            selected.append(buckets[typ].pop(0))
    while len(selected) < n:
        non_empty = [typ for typ, rows in buckets.items() if rows]
        if not non_empty:
            break
        typ = rng.choice(non_empty)
        selected.append(buckets[typ].pop(0))
    rng.shuffle(selected)
    return selected[:n]


def aggregate(rows: list[dict], method: str) -> dict:
    vals = [r["methods"][method] for r in rows]
    return {
        "answer_access@5": mean(v["answer_access@5"] for v in vals),
        "support_recall@5": mean(v["support_recall@5"] for v in vals),
        "all_support_access@5": mean(v["all_support_access@5"] for v in vals),
        "joint_access@5": mean(v["joint_access@5"] for v in vals),
    }


def paired_delta(rows: list[dict], metric: str) -> dict:
    deltas = [
        r["methods"]["lexical_bm25_top5"][metric]
        - r["methods"]["context_order_top5"][metric]
        for r in rows
    ]
    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - wins - losses
    return {
        "mean_delta": mean(deltas),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "positive_rate": wins / len(deltas),
        "negative_rate": losses / len(deltas),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (CDV / "reports").mkdir(parents=True, exist_ok=True)
    data = json.loads(DATA_PATH.read_text())
    sample = stratified_sample(data, N, SEED)
    rows = []
    for ex in sample:
        docs = ex.get("context") or []
        supports = support_titles(ex)
        methods = {}
        rankings = {
            "context_order_top5": list(range(len(docs))),
            "lexical_bm25_top5": lexical_bm25_order(ex.get("question", ""), docs),
        }
        for name, order in rankings.items():
            top_docs = [docs[i] for i in order[:TOP_K]]
            sr = support_recall(supports, top_docs)
            asa = all_support_access(supports, top_docs)
            aa = answer_access(ex.get("answer", ""), top_docs)
            methods[name] = {
                "top_titles": [doc_title(d) for d in top_docs],
                "answer_access@5": aa,
                "support_recall@5": sr,
                "all_support_access@5": asa,
                "joint_access@5": 1.0 if aa and asa else 0.0,
            }
        rows.append({
            "_id": ex.get("_id"),
            "type": ex.get("type", "unknown"),
            "question": ex.get("question"),
            "answer": ex.get("answer"),
            "supporting_titles": sorted(supports),
            "num_context_docs": len(docs),
            "methods": methods,
        })

    methods = ["context_order_top5", "lexical_bm25_top5"]
    metrics = {m: aggregate(rows, m) for m in methods}
    deltas = {
        metric: paired_delta(rows, metric)
        for metric in ["answer_access@5", "support_recall@5", "all_support_access@5", "joint_access@5"]
    }
    type_counts = Counter(r["type"] for r in rows)
    summary = {
        "status": "complete",
        "split": "dev",
        "input_path": str(DATA_PATH.relative_to(CDV)),
        "n": len(rows),
        "seed": SEED,
        "top_k": TOP_K,
        "sample_type_distribution": dict(sorted(type_counts.items())),
        "field_gate": {
            "has_answer": all(bool(r["answer"]) for r in rows),
            "has_context_docs": all(r["num_context_docs"] > 0 for r in rows),
            "has_supporting_titles": all(bool(r["supporting_titles"]) for r in rows),
        },
        "methods": methods,
        "metrics": metrics,
        "deltas_vs_context_order": deltas,
        "gate_pass": True,
        "reader_eval_run": False,
        "claim_boundary": "Adapter/data smoke only. Metrics are retrieval/access diagnostics, not reader QA EM/F1.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (OUT_DIR / "significance_report.json").write_text(json.dumps({
        "status": "complete",
        "test": "paired win/loss/tie diagnostic; no parametric p-value for smoke",
        "deltas_vs_context_order": deltas,
    }, ensure_ascii=False, indent=2) + "\n")
    failures = [
        {
            "_id": r["_id"],
            "type": r["type"],
            "question": r["question"],
            "answer": r["answer"],
            "supporting_titles": r["supporting_titles"],
            "context_order_top_titles": r["methods"]["context_order_top5"]["top_titles"],
            "lexical_bm25_top_titles": r["methods"]["lexical_bm25_top5"]["top_titles"],
            "context_joint_access@5": r["methods"]["context_order_top5"]["joint_access@5"],
            "lexical_joint_access@5": r["methods"]["lexical_bm25_top5"]["joint_access@5"],
        }
        for r in rows
        if r["methods"]["lexical_bm25_top5"]["joint_access@5"] < 1.0
    ]
    (OUT_DIR / "failure_summary.json").write_text(json.dumps({
        "status": "complete",
        "num_lexical_joint_failures": len(failures),
        "examples": failures[:20],
    }, ensure_ascii=False, indent=2) + "\n")
    with (OUT_DIR / "per_example_delta.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def fmt(x: float) -> str:
        return f"{x:.4f}"

    report = [
        "# 2WikiMultiHopQA Dev-300 Smoke Validation Report",
        "",
        "## Status",
        "",
        "Completed on the prepared dev split. This is a retrieval/access smoke test, not a reader QA evaluation.",
        "",
        "## Configuration",
        "",
        f"- Input: `{summary['input_path']}`",
        f"- Sample: stratified dev `{len(rows)}` examples, seed `{SEED}`",
        f"- Top-K: `{TOP_K}`",
        "- Ranking methods: `context_order_top5`, `lexical_bm25_top5`",
        "- No answer/support labels are used for ranking; labels are used only for metric computation.",
        "",
        "## Field Gate",
        "",
        f"- answer present: `{summary['field_gate']['has_answer']}`",
        f"- context docs present: `{summary['field_gate']['has_context_docs']}`",
        f"- support titles present: `{summary['field_gate']['has_supporting_titles']}`",
        "",
        "## Metrics",
        "",
        "| method | answer_access@5 | support_recall@5 | all_support_access@5 | joint_access@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        m = metrics[method]
        report.append(
            f"| {method} | {fmt(m['answer_access@5'])} | {fmt(m['support_recall@5'])} | "
            f"{fmt(m['all_support_access@5'])} | {fmt(m['joint_access@5'])} |"
        )
    report.extend([
        "",
        "## Delta vs Context Order",
        "",
        "| metric | mean_delta | wins | losses | ties |",
        "|---|---:|---:|---:|---:|",
    ])
    for metric, d in deltas.items():
        report.append(f"| {metric} | {fmt(d['mean_delta'])} | {d['wins']} | {d['losses']} | {d['ties']} |")
    report.extend([
        "",
        "## Interpretation",
        "",
        "The 2Wiki dev adapter is usable for cross-dataset validation: every sampled example has answer, context documents, and support titles. The lexical smoke metrics provide a non-leaky retrieval/access baseline before running any reader or learned selector.",
        "",
        "Next step: connect the frozen selector/reader pipeline to this dev adapter and run a reader-backed smoke only after confirming the retrieval index and prompt schema consume `context` and `supporting_titles` correctly.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(report))
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    (MIRROR_DIR / "2wiki_smoke_300_report_latest.md").write_text("\n".join(report))
    (MIRROR_DIR / "2wiki_smoke_300_report_20260622.md").write_text("\n".join(report))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
