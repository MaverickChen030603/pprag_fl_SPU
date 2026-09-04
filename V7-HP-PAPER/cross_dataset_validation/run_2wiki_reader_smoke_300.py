#!/usr/bin/env python3
"""Reader-backed 2WikiMultiHopQA dev-300 smoke validation.

This connects the prepared 2Wiki dev adapter to the HP4 reader helper
(`google/flan-t5-large` by default) and compares:

  - context_order_top5: dataset context order baseline.
  - frozen_selector_bm25_top5: fixed, no-training lexical/BM25 selector.

The selector uses only query/context text. Answer/support labels are used only
for evaluation. This is a cross-dataset smoke check, not a frozen HotpotQA v2.3
final-result rerun.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
CDV = SCRIPT_PATH.parent
V7_HP_PAPER = CDV.parent
PROJECT_ROOT = V7_HP_PAPER.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


READER_HELPER = _load_module(PROJECT_ROOT / "V7-HP4/run_hp4_reader_counterfactual_eval.py", "hp4_reader_helper")

Reader = READER_HELPER.Reader
answer_in_context = READER_HELPER.answer_in_context
f1_score = READER_HELPER.f1_score
make_prompt = READER_HELPER.make_prompt
normalize_answer = READER_HELPER.normalize_answer
sp_metrics = READER_HELPER.sp_metrics

HYBRID = _load_module(PROJECT_ROOT / "src/v7_hp4/hybrid_retriever.py", "hp4_hybrid_retriever")
HybridDocument = HYBRID.HybridDocument


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "by",
    "is", "are", "was", "were", "be", "been", "being", "what", "which",
    "who", "whom", "whose", "when", "where", "why", "how", "did", "does",
    "do", "that", "this", "these", "those", "with", "as", "from", "at",
    "it", "its", "their", "his", "her", "he", "she", "they", "them",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(CDV / "outputs/2wiki_adapter/2wiki_converted.json"))
    p.add_argument("--output-dir", default=str(CDV / "outputs/2wiki_reader_smoke_300"))
    p.add_argument("--max-examples", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260622)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--reader-model", default="google/flan-t5-large")
    p.add_argument("--device", default="auto")
    p.add_argument("--reader-batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--local-reader-only", action="store_true")
    p.add_argument("--limit-methods", default="")
    return p.parse_args()


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if t.lower() not in STOPWORDS]


def doc_title(doc: Any) -> str:
    if isinstance(doc, dict):
        return str(doc.get("title", ""))
    if isinstance(doc, (list, tuple)) and doc:
        return str(doc[0])
    return ""


def doc_text(doc: Any) -> str:
    if isinstance(doc, dict):
        body = doc.get("sentences", doc.get("text", doc.get("content", "")))
        if isinstance(body, list):
            body = " ".join(str(x) for x in body)
        return str(body)
    if isinstance(doc, (list, tuple)):
        body = doc[1] if len(doc) > 1 else ""
        if isinstance(body, list):
            body = " ".join(str(x) for x in body)
        return str(body)
    return str(doc)


def support_titles(ex: dict[str, Any]) -> set[str]:
    titles = set(str(t) for t in ex.get("supporting_titles") or [])
    for sf in ex.get("supporting_facts") or []:
        if isinstance(sf, (list, tuple)) and sf:
            titles.add(str(sf[0]))
        elif isinstance(sf, dict) and sf.get("title"):
            titles.add(str(sf["title"]))
    return titles


def to_hybrid_docs(ex: dict[str, Any]) -> list[Any]:
    supports = support_titles(ex)
    docs = []
    for idx, raw in enumerate(ex.get("context") or []):
        title = doc_title(raw)
        text = doc_text(raw)
        docs.append(HybridDocument(
            doc_id=f"{ex.get('_id', ex.get('id', ex.get('query_id', '2wiki')))}::doc_{idx}",
            title=title,
            text=text,
            client_id=f"client_{idx % 5}",
            is_support=title in supports,
            support_role="support" if title in supports else "unknown",
            bridge_entities=[],
            rare_tokens=[],
            dense_score_hint=1.0,
            soft_weight=1.0,
        ))
    return docs


def bm25_order(question: str, docs: list[Any]) -> list[int]:
    q_terms = tokenize(question)
    q_counts = Counter(q_terms)
    doc_terms = [tokenize(f"{doc.title} {doc.text}") for doc in docs]
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
        title_overlap = len(set(q_terms) & set(tokenize(docs[idx].title)))
        score += 0.25 * title_overlap
        scores.append((score, -idx, idx))
    return [idx for _, _, idx in sorted(scores, reverse=True)]


def stratified_sample(data: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in data:
        buckets[str(ex.get("type", "unknown"))].append(ex)
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


def build_items(sample: list[dict[str, Any]], top_k: int, methods: list[str]) -> list[dict[str, Any]]:
    items = []
    for ex in sample:
        docs = to_hybrid_docs(ex)
        rankings = {
            "context_order_top5": list(range(len(docs))),
            "frozen_selector_bm25_top5": bm25_order(str(ex.get("question", "")), docs),
        }
        supports = support_titles(ex)
        for method in methods:
            top_docs = [docs[i] for i in rankings[method][:top_k]]
            pred_titles = {doc.title for doc in top_docs}
            support_recall, sp_f1 = sp_metrics(pred_titles, supports)
            items.append({
                "id": str(ex.get("_id", ex.get("id", ex.get("query_id", "")))),
                "type": str(ex.get("type", "unknown")),
                "method": method,
                "question": str(ex.get("question", "")),
                "answer": str(ex.get("answer", "")),
                "supporting_titles": sorted(supports),
                "top_docs": top_docs,
                "top_titles": [doc.title for doc in top_docs],
                "answer_access_at_k": answer_in_context(ex.get("answer", ""), top_docs),
                "support_recall_at_k": support_recall,
                "sp_f1": sp_f1,
                "prompt": make_prompt(str(ex.get("question", "")), top_docs),
            })
    return items


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    out = {}
    for method, items in grouped.items():
        out[method] = {
            "n": len(items),
            "answer_access_at_k": sum(float(r["answer_access_at_k"]) for r in items) / len(items),
            "support_recall_at_k": sum(float(r["support_recall_at_k"]) for r in items) / len(items),
            "sp_f1": sum(float(r["sp_f1"]) for r in items) / len(items),
            "answer_em": sum(float(r["answer_em"]) for r in items) / len(items),
            "answer_f1": sum(float(r["answer_f1"]) for r in items) / len(items),
            "joint_f1": sum(float(r["joint_f1"]) for r in items) / len(items),
        }
    return out


def paired_delta(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_id[row["id"]][row["method"]] = row
    deltas = []
    for modes in by_id.values():
        if "context_order_top5" in modes and "frozen_selector_bm25_top5" in modes:
            deltas.append(float(modes["frozen_selector_bm25_top5"][metric]) - float(modes["context_order_top5"][metric]))
    if not deltas:
        return {"n": 0, "mean_delta": 0.0, "wins": 0, "losses": 0, "ties": 0}
    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - wins - losses
    return {
        "n": len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "positive_rate": wins / len(deltas),
        "negative_rate": losses / len(deltas),
    }


def write_report(path: Path, mirror_dir: Path, summary: dict[str, Any]) -> None:
    def fmt(x: float) -> str:
        return f"{x:.4f}"

    metrics = summary["metrics"]
    lines = [
        "# 2WikiMultiHopQA Dev-300 Reader-Backed Smoke Report",
        "",
        "## Status",
        "",
        "Completed. This is a reader-backed cross-dataset smoke check using the prepared 2Wiki dev adapter.",
        "",
        "## Configuration",
        "",
        f"- Input: `{summary['input_path']}`",
        f"- Sample: stratified dev `{summary['n']}` examples, seed `{summary['seed']}`",
        f"- Reader: `{summary['reader_model']}`",
        f"- Top-K: `{summary['top_k']}`",
        "- Methods: `context_order_top5`, `frozen_selector_bm25_top5`",
        "- Selector ranking uses only query/context text. Labels are used only for metric computation.",
        "",
        "## Metrics",
        "",
        "| method | n | access@k | support_recall@k | sp_f1 | answer_em | answer_f1 | joint_f1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in summary["methods"]:
        m = metrics[method]
        lines.append(
            f"| {method} | {m['n']} | {fmt(m['answer_access_at_k'])} | {fmt(m['support_recall_at_k'])} | "
            f"{fmt(m['sp_f1'])} | {fmt(m['answer_em'])} | {fmt(m['answer_f1'])} | {fmt(m['joint_f1'])} |"
        )
    lines.extend([
        "",
        "## Delta vs Context Order",
        "",
        "| metric | mean_delta | wins | losses | ties |",
        "|---|---:|---:|---:|---:|",
    ])
    for metric, d in summary["deltas_vs_context_order"].items():
        lines.append(f"| {metric} | {fmt(d['mean_delta'])} | {d['wins']} | {d['losses']} | {d['ties']} |")
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        summary["claim_boundary"],
        "",
    ])
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    mirror_dir.mkdir(parents=True, exist_ok=True)
    (mirror_dir / "2wiki_reader_smoke_300_report_latest.md").write_text(text, encoding="utf-8")
    (mirror_dir / "2wiki_reader_smoke_300_report_20260622.md").write_text(text, encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = ["context_order_top5", "frozen_selector_bm25_top5"]
    if args.limit_methods:
        keep = {m.strip() for m in args.limit_methods.split(",") if m.strip()}
        methods = [m for m in methods if m in keep]

    data = json.loads(input_path.read_text(encoding="utf-8"))
    sample = stratified_sample(data, args.max_examples, args.seed)
    items = build_items(sample, args.top_k, methods)
    reader = Reader(args.reader_model, args.device, batch_size=args.reader_batch_size, max_new_tokens=args.max_new_tokens)
    predictions = reader.generate([item["prompt"] for item in items])

    rows = []
    for item, pred in zip(items, predictions):
        answer = item["answer"]
        answer_em = float(normalize_answer(pred) == normalize_answer(answer))
        answer_f1 = f1_score(pred, answer)
        rows.append({
            "id": item["id"],
            "type": item["type"],
            "method": item["method"],
            "question": item["question"],
            "answer": answer,
            "prediction": pred,
            "answer_em": answer_em,
            "answer_f1": answer_f1,
            "answer_access_at_k": item["answer_access_at_k"],
            "support_recall_at_k": item["support_recall_at_k"],
            "sp_f1": item["sp_f1"],
            "joint_f1": answer_f1 * float(item["sp_f1"]),
            "supporting_titles": item["supporting_titles"],
            "top_titles": item["top_titles"],
        })

    metrics = summarize(rows)
    deltas = {
        metric: paired_delta(rows, metric)
        for metric in ["answer_access_at_k", "support_recall_at_k", "sp_f1", "answer_em", "answer_f1", "joint_f1"]
    }
    type_counts = Counter(str(ex.get("type", "unknown")) for ex in sample)
    summary = {
        "status": "complete",
        "split": "dev",
        "input_path": str(input_path.relative_to(CDV) if input_path.is_relative_to(CDV) else input_path),
        "output_dir": str(output_dir),
        "n": len(sample),
        "num_prompts": len(items),
        "seed": args.seed,
        "top_k": args.top_k,
        "reader_model": args.reader_model,
        "reader_batch_size": args.reader_batch_size,
        "sample_type_distribution": dict(sorted(type_counts.items())),
        "methods": methods,
        "metrics": metrics,
        "deltas_vs_context_order": deltas,
        "elapsed_seconds": time.time() - started,
        "claim_boundary": (
            "Reader-backed smoke only. It validates 2Wiki adapter compatibility and "
            "reader-facing behavior; it is not a formal 1000-sample cross-dataset claim."
        ),
    }
    (output_dir / "reader_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per_example_reader.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_report(
        CDV / "reports/2wiki_reader_smoke_300_report.md",
        PROJECT_ROOT / "实验分析报告/V7-HP-PAPER",
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
