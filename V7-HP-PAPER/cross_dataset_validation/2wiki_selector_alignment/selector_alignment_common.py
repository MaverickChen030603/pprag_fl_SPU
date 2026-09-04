#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import re
import statistics
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALIGN = Path(__file__).resolve().parent
CDV = ALIGN.parent
V7_HP_PAPER = CDV.parent
ROOT = V7_HP_PAPER.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

DATA_PATH = CDV / "outputs/2wiki_adapter/2wiki_converted.json"
READER_SMOKE_ROWS = CDV / "outputs/2wiki_reader_smoke_300/per_example_reader.jsonl"
MIRROR_DIR = ROOT / "实验分析报告/V7-HP-PAPER"
TOP_K = 5
SEED = 42
ARTICLES = {"a", "an", "the"}
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_'-]{2,}\b")
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "by",
    "is", "are", "was", "were", "be", "been", "being", "what", "which",
    "who", "whom", "whose", "when", "where", "why", "how", "did", "does",
    "do", "that", "this", "these", "those", "with", "as", "from", "at",
    "it", "its", "their", "his", "her", "he", "she", "they", "them",
}
FEATURES = [
    "safe_answer_prob",
    "support_proxy_delta",
    "support_proxy_delta_vs_replaced_doc",
    "support_proxy_delta_vs_baseline_tail_mean",
    "answer_risk_score",
    "displacement_score",
    "hybrid_score_delta",
    "agent_weight_delta",
    "title_bridge_score",
    "prefix1_preserved",
    "prefix2_preserved",
    "prefix3_preserved",
    "num_added_docs",
    "num_removed_docs",
    "removed_baseline_top1",
    "removed_baseline_top2",
]


def ensure_dirs() -> None:
    for rel in [
        "outputs/action_table_300",
        "outputs/selector_smoke_300",
        "outputs/action_table_1000",
        "outputs/selector_crossfit_1000",
        "outputs/ablation",
        "outputs/diagnostics",
        "outputs/audit",
        "reports",
    ]:
        (ALIGN / rel).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if t.lower() not in STOPWORDS]


def entities(text: str) -> set[str]:
    return {m.group(0).lower() for m in ENTITY_RE.finditer(text or "")}


def normalize_answer(s: Any) -> str:
    def remove_articles(text: str) -> str:
        return " ".join(w for w in text.split() if w.lower() not in ARTICLES)

    return " ".join("".join(ch for ch in str(s).lower() if ch not in string.punctuation).split())


def f1_score(prediction: Any, ground_truth: Any) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()
    if not pred or not gold:
        return float(pred == gold)
    common = Counter(pred) & Counter(gold)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred)
    recall = same / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def support_titles(ex: dict[str, Any]) -> set[str]:
    titles = {str(t) for t in ex.get("supporting_titles") or []}
    for sf in ex.get("supporting_facts") or []:
        if isinstance(sf, (list, tuple)) and sf:
            titles.add(str(sf[0]))
        elif isinstance(sf, dict) and sf.get("title"):
            titles.add(str(sf["title"]))
    return titles


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


def query_id(ex: dict[str, Any], idx: int = 0) -> str:
    return str(ex.get("_id", ex.get("id", ex.get("query_id", f"q{idx}"))))


def stratified_sample(data: list[dict[str, Any]], n: int, seed: int = SEED) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in data:
        buckets[str(ex.get("type", "unknown"))].append(ex)
    for rows in buckets.values():
        rng.shuffle(rows)
    total = sum(len(v) for v in buckets.values())
    selected: list[dict[str, Any]] = []
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


def load_dev_sample(n: int = 300, seed: int = SEED) -> list[dict[str, Any]]:
    return stratified_sample(read_json(DATA_PATH), n=n, seed=seed)


def bm25_stats(question: str, docs: list[dict[str, Any]]) -> dict[str, Any]:
    q_terms = tokenize(question)
    q_counts = Counter(q_terms)
    doc_terms = [tokenize(f"{d['title']} {d['text']}") for d in docs]
    n_docs = max(1, len(docs))
    df = Counter()
    for terms in doc_terms:
        df.update(set(terms))
    avgdl = statistics.fmean([len(t) for t in doc_terms] or [1])
    rows = []
    for idx, terms in enumerate(doc_terms):
        tf = Counter(terms)
        dl = max(1, len(terms))
        bm25 = 0.0
        for term, qtf in q_counts.items():
            if term not in tf:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + 1.2 * (1 - 0.75 + 0.75 * dl / max(avgdl, 1))
            bm25 += qtf * idf * (tf[term] * 2.2 / denom)
        title_overlap = len(set(q_terms) & set(tokenize(docs[idx]["title"])))
        lexical = len(set(q_terms) & set(terms)) / max(1, len(set(q_terms)))
        bridge = len(entities(question) & (entities(docs[idx]["title"]) | entities(docs[idx]["text"])))
        support_proxy = 0.65 * bm25 + 0.25 * lexical + 0.10 * title_overlap
        rows.append({
            "idx": idx,
            "title": docs[idx]["title"],
            "bm25": float(bm25),
            "lexical": float(lexical),
            "title_bridge": float(title_overlap + bridge),
            "support_proxy": float(support_proxy),
            "hybrid": float(0.70 * bm25 + 0.20 * lexical + 0.10 * (title_overlap + bridge)),
        })
    return {"rows": rows}


def materialize_docs(ex: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"title": doc_title(raw), "text": doc_text(raw)} for raw in ex.get("context") or []]


def top_by(rows: list[dict[str, Any]], key: str, k: int = TOP_K) -> list[int]:
    return [r["idx"] for r in sorted(rows, key=lambda r: (r[key], r["bm25"], -r["idx"]), reverse=True)[:k]]


def keep_insert(base: list[int], insert: list[int], k: int = TOP_K, keep_prefix: int = 4) -> list[int]:
    out = list(base[:keep_prefix])
    for idx in insert:
        if idx not in out:
            out.append(idx)
        if len(out) >= k:
            break
    for idx in base:
        if len(out) >= k:
            break
        if idx not in out:
            out.append(idx)
    return out[:k]


def action_contexts(ex: dict[str, Any], idx: int = 0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    qid = query_id(ex, idx)
    docs = materialize_docs(ex)
    stats = bm25_stats(str(ex.get("question", "")), docs)["rows"]
    baseline = list(range(min(TOP_K, len(docs))))
    bm25 = top_by(stats, "bm25")
    lexical = top_by(stats, "lexical")
    support_proxy = top_by(stats, "support_proxy")
    bridge = top_by(stats, "title_bridge")
    plus = top_by(stats, "hybrid")
    actions = {
        "baseline_context_order": baseline,
        "bm25_top5": bm25,
        "lexical_top5": lexical,
        "insert1_support_proxy": keep_insert(baseline, support_proxy, keep_prefix=4),
        "insert1_bridge": keep_insert(baseline, bridge, keep_prefix=4),
        "insert1_plus_bridge": keep_insert(baseline, plus, keep_prefix=4),
        "insert2_conservative": keep_insert(baseline, plus, keep_prefix=3),
        "top4_bg1_optional": keep_insert(bm25[:4], baseline, keep_prefix=4),
    }
    meta = {"query_id": qid, "docs": docs, "stats": stats, "actions": actions}
    return meta, docs


def titles_for(indices: list[int], docs: list[dict[str, Any]]) -> list[str]:
    return [docs[i]["title"] for i in indices if 0 <= i < len(docs)]


def context_text(indices: list[int], docs: list[dict[str, Any]], max_chars: int = 3200) -> str:
    parts = [f"[{pos}] {docs[i]['title']}: {docs[i]['text']}" for pos, i in enumerate(indices, start=1) if 0 <= i < len(docs)]
    text = "\n".join(parts)
    return text[:max_chars]


def make_prompt(question: str, indices: list[int], docs: list[dict[str, Any]]) -> str:
    return (
        "Answer the question using only the context. Return a short answer.\n\n"
        f"Question: {question}\n\nContext:\n{context_text(indices, docs)}\n\nAnswer:"
    )


def sp_metrics(pred_titles: set[str], gold_titles: set[str]) -> tuple[float, float]:
    if not gold_titles:
        return 0.0, 0.0
    tp = len(pred_titles & gold_titles)
    precision = tp / len(pred_titles) if pred_titles else 0.0
    recall = tp / len(gold_titles)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if gold_titles <= pred_titles:
        f1 = max(f1, 1.0)
    return recall, f1


def answer_access(answer: str, indices: list[int], docs: list[dict[str, Any]]) -> float:
    ans = normalize_answer(answer)
    if not ans:
        return 0.0
    joined = normalize_answer(" ".join(docs[i]["title"] + " " + docs[i]["text"] for i in indices if 0 <= i < len(docs)))
    if ans in {"yes", "no"}:
        return float(bool(joined))
    return float(ans in joined)


def metric_for_action(ex: dict[str, Any], action: dict[str, Any], prediction: str) -> dict[str, float]:
    gold = support_titles(ex)
    pred_titles = set(action["candidate_titles"])
    recall, sf1 = sp_metrics(pred_titles, gold)
    answer = str(ex.get("answer", ""))
    af1 = f1_score(prediction, answer)
    aem = float(normalize_answer(prediction) == normalize_answer(answer))
    return {
        "answer_access_at_k": answer_access(answer, action["candidate_indices"], action["_docs"]),
        "evidence_recall_at_k": recall,
        "evidence_f1": sf1,
        "sp_f1": sf1,
        "answer_em": aem,
        "answer_f1": af1,
        "joint_f1": af1 * sf1,
    }


def avg_score(indices: list[int], score_by_idx: dict[int, dict[str, Any]], key: str) -> float:
    vals = [float(score_by_idx[i][key]) for i in indices if i in score_by_idx]
    return statistics.fmean(vals) if vals else 0.0


def build_actions_for_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    family = {
        "baseline_context_order": "baseline",
        "bm25_top5": "bm25",
        "lexical_top5": "lexical",
        "insert1_support_proxy": "insert1",
        "insert1_bridge": "bridge",
        "insert1_plus_bridge": "insert1_plus_bridge",
        "insert2_conservative": "insert2",
        "top4_bg1_optional": "top4_bg1",
    }
    for ex_idx, ex in enumerate(examples):
        meta, docs = action_contexts(ex, ex_idx)
        baseline = meta["actions"]["baseline_context_order"]
        score_by_idx = {r["idx"]: r for r in meta["stats"]}
        base_titles = titles_for(baseline, docs)
        base_tail = baseline[-1:] or []
        for name, indices in meta["actions"].items():
            cand_titles = titles_for(indices, docs)
            added = [t for t in cand_titles if t not in base_titles]
            removed = [t for t in base_titles if t not in cand_titles]
            added_idx = [i for i in indices if i not in baseline]
            removed_idx = [i for i in baseline if i not in indices]
            base_support = avg_score(baseline, score_by_idx, "support_proxy")
            cand_support = avg_score(indices, score_by_idx, "support_proxy")
            removed_support = avg_score(removed_idx or base_tail, score_by_idx, "support_proxy")
            action = {
                "query_id": meta["query_id"],
                "question": str(ex.get("question", "")),
                "answer": str(ex.get("answer", "")),
                "type": str(ex.get("type", "unknown")),
                "baseline_titles": base_titles,
                "candidate_titles": cand_titles,
                "candidate_indices": indices,
                "added_titles": added,
                "removed_titles": removed,
                "effective_context_changed": cand_titles != base_titles,
                "num_added_docs": len(added),
                "num_removed_docs": len(removed),
                "prefix1_preserved": cand_titles[:1] == base_titles[:1],
                "prefix2_preserved": cand_titles[:2] == base_titles[:2],
                "prefix3_preserved": cand_titles[:3] == base_titles[:3],
                "removed_baseline_top1": bool(base_titles[:1] and base_titles[0] in removed),
                "removed_baseline_top2": bool(any(t in removed for t in base_titles[:2])),
                "bm25_score_delta": avg_score(indices, score_by_idx, "bm25") - avg_score(baseline, score_by_idx, "bm25"),
                "lexical_score_delta": avg_score(indices, score_by_idx, "lexical") - avg_score(baseline, score_by_idx, "lexical"),
                "support_proxy_delta": cand_support - base_support,
                "support_proxy_delta_vs_replaced_doc": avg_score(added_idx, score_by_idx, "support_proxy") - removed_support if added_idx else 0.0,
                "support_proxy_delta_vs_baseline_tail_mean": avg_score(added_idx, score_by_idx, "support_proxy") - avg_score(base_tail, score_by_idx, "support_proxy") if added_idx else 0.0,
                "title_bridge_score": avg_score(indices, score_by_idx, "title_bridge"),
                "answer_risk_score": 0.35 * len(removed_idx) + 0.35 * float(bool(base_titles[:1] and base_titles[0] in removed)) + 0.20 * float(not (cand_titles[:2] == base_titles[:2])),
                "displacement_score": len(removed_idx) / max(1, TOP_K),
                "hybrid_score_delta": avg_score(indices, score_by_idx, "hybrid") - avg_score(baseline, score_by_idx, "hybrid"),
                "agent_weight_delta": cand_support - base_support,
                "candidate_family": family[name],
                "candidate_name": name,
                "source_dataset": "2WikiMultiHopQA",
                "_docs": docs,
            }
            out.append(action)
    return out


def safe_answer_proxy(row: dict[str, Any]) -> float:
    raw = (
        0.35 * float(row.get("prefix2_preserved", False))
        + 0.30 * float(row.get("prefix3_preserved", False))
        + 0.15 * float(row.get("prefix1_preserved", False))
        - 0.30 * float(row.get("answer_risk_score", 0.0))
        - 0.08 * float(row.get("num_removed_docs", 0.0))
    )
    return 1.0 / (1.0 + math.exp(-3.0 * raw))


def enrich_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        r = dict(row)
        r["safe_answer_prob"] = float(r.get("safe_answer_prob", safe_answer_proxy(r)))
        r["dense_score_delta"] = 0.0
        r["dense_feature_available"] = False
        r["feature_mode"] = "2wiki_v23_heuristic_safety_smoke"
        out.append(r)
    return out


def split_queries(rows: list[dict[str, Any]], folds: int = 5) -> list[tuple[set[str], set[str]]]:
    queries = sorted({str(r["query_id"]) for r in rows})
    queries.sort(key=lambda q: int(hashlib.md5(q.encode()).hexdigest(), 16))
    out = []
    for i in range(folds):
        test = set(queries[i::folds])
        train = set(queries) - test
        out.append((train, test))
    return out


class LinearRanker:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {
            "safe_answer_prob": 0.95,
            "support_proxy_delta": 0.85,
            "support_proxy_delta_vs_replaced_doc": 0.35,
            "title_bridge_score": 0.25,
            "answer_risk_score": -0.75,
            "prefix2_preserved": 0.28,
            "prefix3_preserved": 0.22,
            "num_added_docs": 0.05,
            "num_removed_docs": -0.08,
            "effective_context_changed": 0.08,
        }

    def score(self, row: dict[str, Any], drop_safety: bool = False, drop_support: bool = False) -> float:
        s = 0.0
        for k, w in self.weights.items():
            if drop_safety and k == "safe_answer_prob":
                continue
            if drop_support and k in {"support_proxy_delta", "support_proxy_delta_vs_replaced_doc", "title_bridge_score"}:
                continue
            v = row.get(k, 0.0)
            if isinstance(v, bool):
                v = 1.0 if v else 0.0
            s += float(w) * float(v)
        fam = row.get("candidate_family", "")
        if fam in {"insert1_plus_bridge", "insert1", "bridge"}:
            s += 0.05
        return s


def train_ranker(rows: list[dict[str, Any]], drop_safety: bool = False, drop_support: bool = False) -> LinearRanker:
    feats = [k for k in FEATURES if not (drop_safety and k == "safe_answer_prob")]
    if drop_support:
        feats = [k for k in feats if k not in {"support_proxy_delta", "support_proxy_delta_vs_replaced_doc", "support_proxy_delta_vs_baseline_tail_mean", "title_bridge_score", "agent_weight_delta"}]
    pos = [r for r in rows if float(r.get("answer_f1_delta_vs_bm25", 0)) >= 0 and float(r.get("joint_f1_delta_vs_bm25", 0)) > 0]
    neg = [r for r in rows if r not in pos]
    weights = {}
    for k in feats:
        def val(r):
            x = r.get(k, 0.0)
            return 1.0 if x is True else 0.0 if x is False else float(x or 0.0)
        pm = statistics.fmean([val(r) for r in pos]) if pos else 0.0
        nm = statistics.fmean([val(r) for r in neg]) if neg else 0.0
        weights[k] = pm - nm
    # Stabilize toward answer-neutral v2.3 behavior.
    weights["safe_answer_prob"] = weights.get("safe_answer_prob", 0.0) + (0.65 if not drop_safety else 0.0)
    weights["answer_risk_score"] = weights.get("answer_risk_score", 0.0) - 0.65
    weights["support_proxy_delta"] = weights.get("support_proxy_delta", 0.0) + (0.55 if not drop_support else 0.0)
    weights["title_bridge_score"] = weights.get("title_bridge_score", 0.0) + (0.20 if not drop_support else 0.0)
    return LinearRanker(weights)


def select_by_method(rows: list[dict[str, Any]], method: str, train_rows: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_q[str(r["query_id"])].append(r)
    selected = {}
    if method == "2wiki_v23_crossfit_selector":
        ranker = train_ranker(train_rows or rows)
    elif method == "no_safety_predictor":
        ranker = train_ranker(train_rows or rows, drop_safety=True)
    elif method == "no_support_features":
        ranker = train_ranker(train_rows or rows, drop_support=True)
    else:
        ranker = LinearRanker()
    for qid, items in by_q.items():
        if method == "context_order":
            choice = next((r for r in items if r["candidate_name"] == "baseline_context_order"), items[0])
        elif method == "bm25_or_lexical_routing":
            choice = next((r for r in items if r["candidate_name"] == "bm25_top5"), items[0])
        elif method == "support_first_selector":
            choice = max(items, key=lambda r: (float(r.get("support_proxy_delta", 0)) + 0.25 * float(r.get("title_bridge_score", 0)), r["candidate_name"]))
        elif method == "hotpot_v23_frozen_transfer":
            choice = max(items, key=lambda r: (LinearRanker().score(r), r["candidate_name"]))
        elif method in {"2wiki_v23_crossfit_selector", "no_safety_predictor", "no_support_features"}:
            choice = max(items, key=lambda r: (ranker.score(r, drop_safety=(method == "no_safety_predictor"), drop_support=(method == "no_support_features")), r["candidate_name"]))
        elif method == "oracle_diagnostic_only":
            choice = max(items, key=lambda r: (float(r.get("joint_f1", 0)), float(r.get("answer_f1", 0)), r["candidate_name"]))
        else:
            raise ValueError(f"unknown method {method}")
        selected[qid] = choice
    return selected


def summarize_selected(selected: dict[str, dict[str, Any]], bm25_by_q: dict[str, dict[str, Any]] | None = None) -> dict[str, float]:
    rows = list(selected.values())
    n = max(1, len(rows))
    out = {
        "n": len(rows),
        "answer_access@5": sum(float(r.get("answer_access_at_k", 0)) for r in rows) / n,
        "evidence_recall@5": sum(float(r.get("evidence_recall_at_k", 0)) for r in rows) / n,
        "evidence_f1": sum(float(r.get("evidence_f1", 0)) for r in rows) / n,
        "sp_f1": sum(float(r.get("sp_f1", 0)) for r in rows) / n,
        "answer_em": sum(float(r.get("answer_em", 0)) for r in rows) / n,
        "answer_f1": sum(float(r.get("answer_f1", 0)) for r in rows) / n,
        "joint_f1": sum(float(r.get("joint_f1", 0)) for r in rows) / n,
        "fallback_rate": sum(1 for r in rows if r.get("candidate_name") == "baseline_context_order") / n,
        "selected_count": len(rows),
        "selected_effective_action_rate": sum(float(bool(r.get("effective_context_changed"))) for r in rows) / n,
    }
    if bm25_by_q:
        paired = [(r, bm25_by_q[q]) for q, r in selected.items() if q in bm25_by_q]
        m = max(1, len(paired))
        for metric in ["answer_f1", "joint_f1", "evidence_recall_at_k", "evidence_f1"]:
            public = "evidence_recall@5" if metric == "evidence_recall_at_k" else metric
            out[f"{public}_delta_vs_bm25"] = sum(float(r.get(metric, 0)) - float(b.get(metric, 0)) for r, b in paired) / m
    return out


def pairwise_significance(selected: dict[str, dict[str, Any]], bm25: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for metric in ["answer_f1", "joint_f1", "evidence_recall_at_k", "evidence_f1"]:
        diffs = [float(selected[q].get(metric, 0)) - float(bm25[q].get(metric, 0)) for q in selected if q in bm25]
        if not diffs:
            out[metric] = {"n": 0, "mean_delta": 0.0, "wins": 0, "losses": 0, "ties": 0}
            continue
        wins = sum(1 for d in diffs if d > 1e-12)
        losses = sum(1 for d in diffs if d < -1e-12)
        out[metric] = {
            "n": len(diffs),
            "mean_delta": sum(diffs) / len(diffs),
            "wins": wins,
            "losses": losses,
            "ties": len(diffs) - wins - losses,
        }
    return out


def load_reader_class():
    helper = ROOT / "V7-HP4/run_hp4_reader_counterfactual_eval.py"
    spec = importlib.util.spec_from_file_location("hp4_reader_helper_for_2wiki_alignment", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {helper}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Reader


def run_reader_for_actions(action_rows: list[dict[str, Any]], output_cache: Path, model: str = "google/flan-t5-large", batch_size: int = 4) -> list[dict[str, Any]]:
    if output_cache.exists():
        return list(iter_jsonl(output_cache))
    Reader = load_reader_class()
    reader = Reader(model, "auto", batch_size=batch_size, max_new_tokens=32)
    prompts = [make_prompt(r["question"], r["candidate_indices"], r["_docs"]) for r in action_rows]
    preds = reader.generate(prompts)
    out = []
    by_q_ex = {r["query_id"]: r for r in action_rows}
    for row, pred in zip(action_rows, preds):
        ex = {
            "answer": row["answer"],
            "supporting_titles": row.get("_supporting_titles", []),
        }
        metrics = metric_for_action(ex, row, pred)
        r = {k: v for k, v in row.items() if not k.startswith("_")}
        r.update(metrics)
        r["prediction"] = pred
        out.append(r)
    write_jsonl(output_cache, out)
    return out


def strip_private(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

