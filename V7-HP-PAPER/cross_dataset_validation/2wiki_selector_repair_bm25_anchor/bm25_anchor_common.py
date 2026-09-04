#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPAIR = Path(__file__).resolve().parent
CDV = REPAIR.parent
ALIGN = CDV / "2wiki_selector_alignment"
ROOT = CDV.parents[1]
MIRROR = ROOT / "实验分析报告/V7-HP-PAPER"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ALIGN))

import selector_alignment_common as base  # noqa: E402


TOP_K = 5
SEED = 42
METHODS = [
    "bm25_or_lexical_routing",
    "previous_2wiki_v23_crossfit_selector",
    "bm25_anchor_support_first",
    "bm25_anchor_safety_first",
    "bm25_anchor_positive_selector",
    "bm25_anchor_answer_neutral_selector",
    "no_safety_predictor",
    "no_support_features",
    "oracle_diagnostic_only",
]


def ensure_dirs() -> None:
    for rel in [
        "outputs/oracle_gap_300",
        "outputs/action_table_300",
        "outputs/safety_predictor",
        "outputs/selector_smoke_300",
        "outputs/ablation",
        "outputs/diagnostics",
        "outputs/audit",
        "reports",
    ]:
        (REPAIR / rel).mkdir(parents=True, exist_ok=True)


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
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def row_key(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return str(row["query_id"]), tuple(row.get("candidate_titles") or row.get("selected_titles") or [])


def alignment_outcomes() -> list[dict[str, Any]]:
    return list(iter_jsonl(ALIGN / "outputs/selector_smoke_300/action_reader_outcomes_300_labeled.jsonl"))


def bm25_by_q(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["query_id"]: r for r in rows if r.get("candidate_name") == "bm25_top5"}


def previous_selector_by_q() -> dict[str, dict[str, Any]]:
    rows = list(iter_jsonl(ALIGN / "outputs/selector_smoke_300/per_example_delta.jsonl"))
    return {r["query_id"]: r for r in rows if r.get("method") == "2wiki_v23_crossfit_selector"}


def load_examples() -> list[dict[str, Any]]:
    return base.load_dev_sample(300, base.SEED)


def build_anchor_actions() -> list[dict[str, Any]]:
    examples = load_examples()
    out = []
    for ex_idx, ex in enumerate(examples):
        meta, docs = base.action_contexts(ex, ex_idx)
        stats = meta["stats"]
        score_by_idx = {r["idx"]: r for r in stats}
        bm25 = meta["actions"]["bm25_top5"]
        qid = meta["query_id"]
        bm25_titles = base.titles_for(bm25, docs)
        top3 = bm25[:3]
        tail = bm25[3:]
        candidates = [r["idx"] for r in sorted(stats, key=lambda r: (r["support_proxy"], r["title_bridge"], r["bm25"]), reverse=True) if r["idx"] not in bm25]
        bridge = [r["idx"] for r in sorted(stats, key=lambda r: (r["title_bridge"], r["support_proxy"], r["bm25"]), reverse=True) if r["idx"] not in bm25]

        def replace_slot(base_idx: list[int], slot: int, pool: list[int]) -> list[int]:
            cur = list(base_idx)
            for cand in pool:
                if cand not in cur and slot < len(cur):
                    cur[slot] = cand
                    break
            return cur[:TOP_K]

        action_defs = {
            "bm25_fallback": bm25,
            "bm25_no_change_control": bm25,
            "bm25_keep_top3_insert1_slot5": replace_slot(bm25, 4, candidates),
            "bm25_keep_top3_insert1_slot4": replace_slot(bm25, 3, candidates),
            "bm25_keep_top2_bridge_insert1": replace_slot(bm25, 4, bridge),
            "bm25_tail_swap_evidence": replace_slot(bm25, 4 if tail else 3, candidates),
        }
        families = {
            "bm25_fallback": "bm25_fallback",
            "bm25_no_change_control": "bm25_no_change_control",
            "bm25_keep_top3_insert1_slot5": "replace_slot5_only",
            "bm25_keep_top3_insert1_slot4": "replace_slot4_or_5",
            "bm25_keep_top2_bridge_insert1": "bridge_insert",
            "bm25_tail_swap_evidence": "tail_swap_evidence",
        }
        for name, indices in action_defs.items():
            titles = base.titles_for(indices, docs)
            added = [t for t in titles if t not in bm25_titles]
            removed = [t for t in bm25_titles if t not in titles]
            added_idx = [i for i in indices if i not in bm25]
            removed_idx = [i for i in bm25 if i not in indices]
            support_delta = base.avg_score(indices, score_by_idx, "support_proxy") - base.avg_score(bm25, score_by_idx, "support_proxy")
            evidence_delta = base.avg_score(indices, score_by_idx, "hybrid") - base.avg_score(bm25, score_by_idx, "hybrid")
            row = {
                "query_id": qid,
                "question": str(ex.get("question", "")),
                "answer": str(ex.get("answer", "")),
                "bm25_titles": bm25_titles,
                "candidate_titles": titles,
                "candidate_indices": indices,
                "added_titles": added,
                "removed_titles": removed,
                "effective_context_changed": titles != bm25_titles,
                "num_added_docs": len(added),
                "num_removed_docs": len(removed),
                "bm25_top1_preserved": bm25_titles[:1] == titles[:1],
                "bm25_top2_preserved": bm25_titles[:2] == titles[:2],
                "bm25_top3_preserved": bm25_titles[:3] == titles[:3],
                "prefix2_preserved": bm25_titles[:2] == titles[:2],
                "prefix3_preserved": bm25_titles[:3] == titles[:3],
                "support_proxy_delta_vs_bm25": support_delta,
                "evidence_proxy_delta_vs_bm25": evidence_delta,
                "title_bridge_score": base.avg_score(indices, score_by_idx, "title_bridge"),
                "answer_risk_score": 0.45 * float(not (bm25_titles[:2] == titles[:2])) + 0.25 * len(removed_idx),
                "bm25_score_delta": base.avg_score(indices, score_by_idx, "bm25") - base.avg_score(bm25, score_by_idx, "bm25"),
                "lexical_score_delta": base.avg_score(indices, score_by_idx, "lexical") - base.avg_score(bm25, score_by_idx, "lexical"),
                "candidate_family": families[name],
                "candidate_name": name,
                "source_dataset": "2WikiMultiHopQA",
                "_docs": docs,
                "_supporting_titles": sorted(base.support_titles(ex)),
            }
            out.append(row)
    return out


def metric_delta(row: dict[str, Any], bm25: dict[str, Any]) -> dict[str, float]:
    return {
        "answer_f1_delta_vs_bm25": float(row.get("answer_f1", 0)) - float(bm25.get("answer_f1", 0)),
        "joint_f1_delta_vs_bm25": float(row.get("joint_f1", 0)) - float(bm25.get("joint_f1", 0)),
        "evidence_recall_delta_vs_bm25": float(row.get("evidence_recall_at_k", 0)) - float(bm25.get("evidence_recall_at_k", 0)),
        "evidence_f1_delta_vs_bm25": float(row.get("evidence_f1", 0)) - float(bm25.get("evidence_f1", 0)),
    }


def attach_metrics(action_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    old = alignment_outcomes()
    lookup = {row_key(r): r for r in old}
    bm25_lookup = bm25_by_q(old)
    rows = []
    missing = []
    for r in action_rows:
        key = row_key(r)
        if key in lookup:
            rr = dict(r)
            oldr = lookup[key]
            for k in ["prediction", "answer_access_at_k", "evidence_recall_at_k", "evidence_f1", "sp_f1", "answer_em", "answer_f1", "joint_f1"]:
                rr[k] = oldr.get(k, 0)
            rows.append(rr)
        else:
            missing.append(r)
    if missing:
        cache = REPAIR / "outputs/selector_smoke_300/new_action_reader_outcomes.jsonl"
        new = base.run_reader_for_actions(missing, cache)
        by_new = {row_key(r): r for r in new}
        for r in missing:
            rr = dict(r)
            newr = by_new[row_key(r)]
            for k in ["prediction", "answer_access_at_k", "evidence_recall_at_k", "evidence_f1", "sp_f1", "answer_em", "answer_f1", "joint_f1"]:
                rr[k] = newr.get(k, 0)
            rows.append(rr)
    bm25_rows = {r["query_id"]: r for r in rows if r["candidate_name"] in {"bm25_fallback", "bm25_no_change_control"}}
    # Prefer old BM25 metrics because they are the formal strong baseline.
    for r in rows:
        b = bm25_lookup.get(r["query_id"]) or bm25_rows.get(r["query_id"], {})
        r.update(metric_delta(r, b))
        r["answer_safe"] = int(r["answer_f1_delta_vs_bm25"] >= 0)
        r["paper_positive_vs_bm25"] = int(r["answer_f1_delta_vs_bm25"] >= 0 and r["joint_f1_delta_vs_bm25"] > 0 and r["evidence_f1_delta_vs_bm25"] >= 0)
    return rows


def split_queries(rows: list[dict[str, Any]], folds: int = 5) -> list[tuple[set[str], set[str]]]:
    queries = sorted({r["query_id"] for r in rows})
    queries.sort(key=lambda q: int(hashlib.md5(q.encode()).hexdigest(), 16))
    out = []
    for i in range(folds):
        test = set(queries[i::folds])
        train = set(queries) - test
        out.append((train, test))
    return out


def fit_linear_classifier(rows: list[dict[str, Any]], target: str, drop_safety: bool = False, drop_support: bool = False) -> dict[str, float]:
    feats = [
        "support_proxy_delta_vs_bm25",
        "evidence_proxy_delta_vs_bm25",
        "title_bridge_score",
        "answer_risk_score",
        "bm25_top3_preserved",
        "prefix2_preserved",
        "prefix3_preserved",
        "num_added_docs",
        "num_removed_docs",
        "bm25_score_delta",
        "lexical_score_delta",
    ]
    if drop_support:
        feats = [f for f in feats if f not in {"support_proxy_delta_vs_bm25", "evidence_proxy_delta_vs_bm25", "title_bridge_score"}]
    pos = [r for r in rows if int(r.get(target, 0)) == 1]
    neg = [r for r in rows if int(r.get(target, 0)) != 1]
    weights = {}
    for feat in feats:
        def val(r):
            x = r.get(feat, 0)
            return 1.0 if x is True else 0.0 if x is False else float(x or 0.0)
        pm = statistics.fmean([val(r) for r in pos]) if pos else 0.0
        nm = statistics.fmean([val(r) for r in neg]) if neg else 0.0
        weights[feat] = pm - nm
    weights["answer_risk_score"] = weights.get("answer_risk_score", 0.0) - 0.5
    weights["bias"] = math.log((len(pos) + 1) / (len(neg) + 1))
    return weights


def score_with(weights: dict[str, float], row: dict[str, Any], drop_support: bool = False) -> float:
    s = float(weights.get("bias", 0.0))
    for k, w in weights.items():
        if k == "bias":
            continue
        if drop_support and k in {"support_proxy_delta_vs_bm25", "evidence_proxy_delta_vs_bm25", "title_bridge_score"}:
            continue
        x = row.get(k, 0.0)
        x = 1.0 if x is True else 0.0 if x is False else float(x or 0.0)
        s += float(w) * x
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, s))))


def build_crossfit_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preds = []
    for fold, (train_q, test_q) in enumerate(split_queries(rows), start=1):
        train = [r for r in rows if r["query_id"] in train_q]
        test = [r for r in rows if r["query_id"] in test_q]
        safe_w = fit_linear_classifier(train, "answer_safe")
        pos_w = fit_linear_classifier(train, "paper_positive_vs_bm25")
        for r in test:
            rr = dict(r)
            rr["fold"] = fold
            rr["safe_answer_prob"] = score_with(safe_w, r)
            rr["positive_action_prob"] = score_with(pos_w, r)
            preds.append(rr)
    return preds


def selected_by_method(rows: list[dict[str, Any]], method: str, selected_fraction: float = 0.20, safe_threshold: float = 0.50, positive_threshold: float = 0.05, preserve_top3: bool = True) -> dict[str, dict[str, Any]]:
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_q[r["query_id"]].append(r)
    selected = {}
    for qid, items in by_q.items():
        bm25 = next((r for r in items if r["candidate_name"] == "bm25_fallback"), items[0])
        effective = [r for r in items if r.get("effective_context_changed")]
        legal = [r for r in effective if r.get("bm25_top1_preserved") and r.get("bm25_top2_preserved") and int(r.get("num_added_docs", 0)) <= 1 and int(r.get("num_removed_docs", 0)) <= 1]
        if preserve_top3:
            legal = [r for r in legal if r.get("bm25_top3_preserved")]
        if method == "bm25_or_lexical_routing":
            selected[qid] = bm25
            continue
        if method == "previous_2wiki_v23_crossfit_selector":
            prev = previous_selector_by_q().get(qid)
            selected[qid] = prev or bm25
            continue
        if method == "oracle_diagnostic_only":
            selected[qid] = max(items, key=lambda r: (float(r.get("paper_positive_vs_bm25", 0)), float(r.get("joint_f1_delta_vs_bm25", 0)), float(r.get("answer_f1_delta_vs_bm25", 0))))
            continue
        if not legal:
            selected[qid] = bm25
            continue
        if method == "bm25_anchor_support_first":
            choice = max(legal, key=lambda r: (float(r.get("evidence_proxy_delta_vs_bm25", 0)) + float(r.get("support_proxy_delta_vs_bm25", 0)), -float(r.get("answer_risk_score", 0))))
            selected[qid] = choice if float(choice.get("evidence_proxy_delta_vs_bm25", 0)) > 0 else bm25
            continue
        if method == "bm25_anchor_safety_first":
            legal2 = [r for r in legal if float(r.get("safe_answer_prob", 0)) >= safe_threshold]
            selected[qid] = max(legal2, key=lambda r: (float(r.get("safe_answer_prob", 0)), float(r.get("evidence_proxy_delta_vs_bm25", 0)))) if legal2 else bm25
            continue
        if method == "bm25_anchor_positive_selector":
            legal2 = [r for r in legal if float(r.get("positive_action_prob", 0)) >= positive_threshold]
            selected[qid] = max(legal2, key=lambda r: (float(r.get("positive_action_prob", 0)), float(r.get("evidence_proxy_delta_vs_bm25", 0)))) if legal2 else bm25
            continue
        drop_safety = method == "no_safety_predictor"
        drop_support = method == "no_support_features"
        scored = []
        for r in legal:
            score = (
                (0 if drop_safety else 0.35 * float(r.get("positive_action_prob", 0)))
                + (0 if drop_safety else 0.25 * float(r.get("safe_answer_prob", 0)))
                + (0 if drop_support else 0.20 * float(r.get("evidence_proxy_delta_vs_bm25", 0)))
                + (0 if drop_support else 0.10 * float(r.get("title_bridge_score", 0)))
                + (0 if drop_support else 0.10 * float(r.get("support_proxy_delta_vs_bm25", 0)))
                - 0.20 * float(r.get("answer_risk_score", 0))
            )
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        choice = scored[0][1]
        if (
            method == "bm25_anchor_answer_neutral_selector"
            and (float(choice.get("safe_answer_prob", 0)) < safe_threshold or float(choice.get("positive_action_prob", 0)) < positive_threshold)
        ):
            choice = bm25
        selected[qid] = choice
    return selected


def summarize(selected: dict[str, dict[str, Any]], bm25_by_query: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(selected.values())
    n = max(1, len(rows))
    out = {
        "n": len(rows),
        "answer_f1": sum(float(r.get("answer_f1", 0)) for r in rows) / n,
        "evidence_recall@5": sum(float(r.get("evidence_recall_at_k", 0)) for r in rows) / n,
        "evidence_f1": sum(float(r.get("evidence_f1", 0)) for r in rows) / n,
        "joint_f1": sum(float(r.get("joint_f1", 0)) for r in rows) / n,
        "selected_effective_action_rate": sum(float(bool(r.get("effective_context_changed"))) for r in rows) / n,
        "fallback_rate": sum(1 for r in rows if r.get("candidate_name") in {"bm25_fallback", "bm25_no_change_control"}) / n,
        "answer_drop_selected_count": sum(1 for r in rows if float(r.get("answer_f1_delta_vs_bm25", 0)) < 0),
        "positive_vs_bm25_recall": sum(1 for r in rows if r.get("paper_positive_vs_bm25")) / n,
    }
    paired = [(r, bm25_by_query.get(q, {})) for q, r in selected.items()]
    for metric, out_key in [
        ("answer_f1", "answer_f1_delta_vs_bm25"),
        ("joint_f1", "joint_f1_delta_vs_bm25"),
        ("evidence_recall_at_k", "evidence_recall_delta_vs_bm25"),
        ("evidence_f1", "evidence_f1_delta_vs_bm25"),
    ]:
        out[out_key] = sum(float(r.get(metric, 0)) - float(b.get(metric, 0)) for r, b in paired) / max(1, len(paired))
    return out


def significance(selected: dict[str, dict[str, Any]], bm25_by_query: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for metric in ["answer_f1", "joint_f1", "evidence_recall_at_k", "evidence_f1"]:
        diffs = [float(r.get(metric, 0)) - float(bm25_by_query.get(q, {}).get(metric, 0)) for q, r in selected.items()]
        wins = sum(1 for d in diffs if d > 1e-12)
        losses = sum(1 for d in diffs if d < -1e-12)
        out[metric] = {"n": len(diffs), "mean_delta": sum(diffs) / max(1, len(diffs)), "wins": wins, "losses": losses, "ties": len(diffs) - wins - losses}
    return out

