from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.v7_hp4.agent_continuous import ContinuousUploadPolicy
from src.v7_hp4.hybrid_retriever import HybridDocument, HybridSoftRetriever, entity_tokens, tokenize
from src.v7_hp4.policy_gradient import block_state_from_doc

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for V7-HP-PAPER predictor_v2") from exc


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


READER = _load("V7-HP4/run_hp4_reader_counterfactual_eval.py", "hp4_reader")


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")


def log(message: str) -> None:
    print(f"[paper-selector] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def toks(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text or "")}


def ents(text: str) -> set[str]:
    return {t.lower() for t in entity_tokens(text or "")}


def minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sanitize_docs(docs: list[HybridDocument]) -> list[HybridDocument]:
    clean = []
    for idx, doc in enumerate(docs):
        clean.append(replace(
            doc,
            client_id=f"client_{idx % 5}",
            support_role="unknown",
            bridge_entities=[],
            rare_tokens=[],
            dense_score_hint=1.0,
            soft_weight=1.0,
        ))
    return clean


def load_policy(path: Path, device: str) -> tuple[ContinuousUploadPolicy, torch.device]:
    runtime = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path, map_location=runtime)
    model = ContinuousUploadPolicy(
        input_dim=len(payload.get("feature_names", ContinuousUploadPolicy.feature_names)),
        hidden_dim=int(payload.get("hidden_dim", 48)),
        init_bias=-0.4,
    ).to(runtime)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, runtime


def weights_for_docs(question: str, docs: list[HybridDocument], model: ContinuousUploadPolicy, device: torch.device) -> dict[str, float]:
    states = [block_state_from_doc(question, doc, mode="natural") for doc in docs]
    x = ContinuousUploadPolicy.tensor_from_states(states, device=str(device))
    with torch.no_grad():
        weights = model(x).detach().cpu().tolist()
    return {doc.doc_id: float(max(0.0, min(1.0, w))) for doc, w in zip(docs, weights)}


def hardgate(weights: dict[str, float], gate: int, floor: float) -> dict[str, float]:
    keep = {doc_id for doc_id, _ in sorted(weights.items(), key=lambda item: item[1], reverse=True)[:gate]}
    return {doc_id: (weight if doc_id in keep else weight * floor) for doc_id, weight in weights.items()}


def rank_docs(
    question: str,
    docs: list[HybridDocument],
    weights: dict[str, float],
    *,
    alpha: float,
    tau: float = 1.0,
    gate: int | None = None,
    gate_floor: float = 0.01,
    top_k: int = 5,
    pool: int | None = None,
) -> list[tuple[HybridDocument, dict[str, float]]]:
    use_weights = weights
    if gate is not None:
        use_weights = hardgate(weights, gate, gate_floor)
    retriever = HybridSoftRetriever(
        docs,
        alpha=alpha,
        dense_weight_mode="temperature" if tau != 1.0 or gate is not None else "identity",
        weight_temperature=tau,
    )
    return retriever.rank(question, weights=use_weights, top_k=pool or top_k)


def score_all(question: str, docs: list[HybridDocument], weights: dict[str, float], alpha: float) -> dict[str, dict[str, float]]:
    retriever = HybridSoftRetriever(docs, alpha=alpha)
    out = {}
    for idx, doc in enumerate(docs):
        out[doc.doc_id] = retriever.score(question, doc, idx, weights=weights)
    return out


def support_proxy(
    question: str,
    doc: HybridDocument,
    score: dict[str, float],
    dense_n: dict[str, float],
    sparse_n: dict[str, float],
    weight_n: dict[str, float],
    title_bridge: float,
) -> float:
    q_entities = ents(question)
    doc_entities = ents(doc.content) | toks(doc.title)
    entity_overlap = jaccard(q_entities, doc_entities)
    return (
        0.30 * weight_n.get(doc.doc_id, 0.0)
        + 0.25 * sparse_n.get(doc.doc_id, 0.0)
        + 0.20 * dense_n.get(doc.doc_id, 0.0)
        + 0.15 * entity_overlap
        + 0.10 * title_bridge
    )


def title_bridge_scores(question: str, docs: list[HybridDocument]) -> dict[str, float]:
    q = toks(question) | ents(question)
    titles = {doc.doc_id: toks(doc.title) | ents(doc.title) for doc in docs}
    out = {}
    for doc in docs:
        own = titles[doc.doc_id]
        query_overlap = jaccard(q, own)
        other_overlap = max((jaccard(own, t) for did, t in titles.items() if did != doc.doc_id), default=0.0)
        out[doc.doc_id] = 0.65 * query_overlap + 0.35 * other_overlap
    return out


def low_value_tail_doc(question: str, tail: list[HybridDocument], scores: dict[str, dict[str, float]]) -> str | None:
    if not tail:
        return None
    q = toks(question)
    ranked = []
    dense_n = minmax({d.doc_id: scores[d.doc_id]["dense"] for d in tail})
    sparse_n = minmax({d.doc_id: scores[d.doc_id]["sparse"] for d in tail})
    for doc in tail:
        lexical = jaccard(q, toks(doc.content))
        value = 0.40 * sparse_n.get(doc.doc_id, 0.0) + 0.35 * dense_n.get(doc.doc_id, 0.0) + 0.25 * lexical
        ranked.append((value, doc.doc_id))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][1]


def unique_docs(docs: list[HybridDocument], top_k: int = 5) -> list[HybridDocument]:
    out, seen = [], set()
    for doc in docs:
        if doc.doc_id in seen:
            continue
        out.append(doc)
        seen.add(doc.doc_id)
        if len(out) >= top_k:
            break
    return out


def generate_candidates(
    question: str,
    docs: list[HybridDocument],
    weights: dict[str, float],
    *,
    alpha: float,
    top_k: int,
) -> list[dict[str, Any]]:
    doc_by_id = {doc.doc_id: doc for doc in docs}
    base_rank = rank_docs(question, docs, {d.doc_id: 1.0 for d in docs}, alpha=alpha, top_k=top_k)
    baseline = [doc for doc, _ in base_rank]
    baseline_ids = [doc.doc_id for doc in baseline]

    all_scores = score_all(question, docs, weights, alpha)
    dense_n = minmax({doc_id: s["dense"] for doc_id, s in all_scores.items()})
    sparse_n = minmax({doc_id: s["sparse"] for doc_id, s in all_scores.items()})
    weight_n = minmax(weights)
    bridge_n = title_bridge_scores(question, docs)
    proxy = {
        doc.doc_id: support_proxy(question, doc, all_scores[doc.doc_id], dense_n, sparse_n, weight_n, bridge_n.get(doc.doc_id, 0.0))
        for doc in docs
    }
    proxy_ranked = sorted(docs, key=lambda d: (proxy[d.doc_id], all_scores[d.doc_id]["final"], d.doc_id), reverse=True)

    def pack(mode: str, selected: list[HybridDocument], ordering: str) -> dict[str, Any]:
        selected = unique_docs(selected, top_k)
        if len(selected) < top_k:
            selected = unique_docs(selected + baseline + proxy_ranked, top_k)
        return {
            "mode": mode,
            "ordering": ordering,
            "top_docs": selected,
            "top_doc_ids": [d.doc_id for d in selected],
            "top_titles": [d.title for d in selected],
            "support_proxy_mean": mean([proxy.get(d.doc_id, 0.0) for d in selected]),
            "support_proxy_max": max([proxy.get(d.doc_id, 0.0) for d in selected], default=0.0),
        }

    candidates = [pack("baseline", baseline, "baseline_order")]

    gate4 = [doc for doc, _ in rank_docs(question, docs, weights, alpha=alpha, tau=0.7, gate=4, top_k=4, pool=4)]
    bg_pool = [doc for doc, _ in rank_docs(question, docs, weights, alpha=alpha, tau=1.0, top_k=16, pool=min(len(docs), 16))]
    bg = [doc for doc in bg_pool if doc.doc_id not in {d.doc_id for d in gate4}]
    candidates.append(pack("top4_bg1_balanced", gate4 + bg[:1], "balanced"))

    support3 = proxy_ranked[:3]
    q = toks(question)
    anchors = sorted(
        [doc for doc in docs if doc.doc_id not in {d.doc_id for d in support3}],
        key=lambda d: (
            0.40 * all_scores[d.doc_id]["sparse"]
            + 0.30 * all_scores[d.doc_id]["dense"]
            + 0.20 * jaccard(q, toks(d.content))
            + 0.10 * jaccard(q, toks(d.title)),
            d.doc_id,
        ),
        reverse=True,
    )[:2]
    candidates.append(pack("support3_anchor2", support3 + anchors, "support_first_anchor_fill"))

    for insert_n, mode in [(1, "baseline_prefix_preserve_insert1"), (2, "baseline_prefix_preserve_insert2")]:
        prefix_len = 2 if insert_n == 1 else 3
        prefix = baseline[:prefix_len]
        current = list(baseline)
        insert_docs = [doc for doc in proxy_ranked if doc.doc_id not in {d.doc_id for d in current}][:insert_n]
        for ins in insert_docs:
            replace_id = low_value_tail_doc(question, current[prefix_len:], all_scores)
            if replace_id is None:
                break
            current = [ins if d.doc_id == replace_id else d for d in current]
        # Keep baseline anchors stable; support insertions occupy the first tail slots.
        ordered = prefix + [d for d in insert_docs if d.doc_id in {x.doc_id for x in current}]
        ordered += [d for d in baseline[prefix_len:] if d.doc_id in {x.doc_id for x in current} and d.doc_id not in {x.doc_id for x in ordered}]
        ordered += [d for d in current if d.doc_id not in {x.doc_id for x in ordered}]
        candidates.append(pack(mode, ordered, "prefix_preserve"))

    bridge_docs = sorted(
        [doc for doc in docs if doc.doc_id not in set(baseline_ids[:2])],
        key=lambda d: (bridge_n.get(d.doc_id, 0.0), proxy.get(d.doc_id, 0.0), d.doc_id),
        reverse=True,
    )
    bridge_insert = baseline[:2] + bridge_docs[:1] + [doc for doc in baseline[2:] if doc.doc_id not in {d.doc_id for d in bridge_docs[:1]}]
    candidates.append(pack("bridge_title_insert", bridge_insert, "prefix_bridge_insert"))

    return candidates


def kendall_similarity(base_ids: list[str], cand_ids: list[str]) -> float:
    common = [doc_id for doc_id in base_ids if doc_id in cand_ids]
    if len(common) < 2:
        return 0.0
    pos = {doc_id: idx for idx, doc_id in enumerate(cand_ids)}
    total = concord = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            total += 1
            if pos[common[i]] < pos[common[j]]:
                concord += 1
    return concord / max(total, 1)


def context_features(question: str, docs_by_id: dict[str, HybridDocument], baseline: dict[str, Any], cand: dict[str, Any]) -> dict[str, float]:
    base_ids = list(map(str, baseline["top_doc_ids"]))
    cand_ids = list(map(str, cand["top_doc_ids"]))
    base_set, cand_set = set(base_ids), set(cand_ids)
    added = [doc_id for doc_id in cand_ids if doc_id not in base_set]
    removed = [doc_id for doc_id in base_ids if doc_id not in cand_set]
    q = toks(question)
    qe = ents(question)

    def doc_text(doc_id: str) -> str:
        doc = docs_by_id.get(doc_id)
        return doc.content if doc else ""

    def lex(ids: list[str]) -> list[float]:
        return [jaccard(q, toks(doc_text(doc_id))) for doc_id in ids]

    def title_lex(ids: list[str]) -> list[float]:
        return [jaccard(q, toks(docs_by_id[doc_id].title)) for doc_id in ids if doc_id in docs_by_id]

    def entity_cov(ids: list[str]) -> float:
        ctx = set()
        for doc_id in ids:
            ctx |= ents(doc_text(doc_id))
        return jaccard(qe, ctx)

    base_lex, cand_lex = lex(base_ids), lex(cand_ids)
    removed_lex = lex(removed)
    added_lex = lex(added)
    pos = {doc_id: idx for idx, doc_id in enumerate(base_ids)}
    disp = [abs(idx - pos[doc_id]) / max(len(base_ids) - 1, 1) for idx, doc_id in enumerate(cand_ids) if doc_id in pos]
    baseline_anchor_retention = sum(1 for doc_id in base_ids[:3] if doc_id in cand_set)
    overlap = len(base_set & cand_set) / max(len(base_set), 1)
    prefix1 = float(base_ids[:1] == cand_ids[:1])
    prefix2 = float(base_ids[:2] == cand_ids[:2])
    prefix3 = float(base_ids[:3] == cand_ids[:3])
    base_entity = entity_cov(base_ids)
    cand_entity = entity_cov(cand_ids)
    query_overlap_loss = max(0.0, mean(base_lex) - mean(cand_lex))
    entity_loss = max(0.0, base_entity - cand_entity)
    removed_high_bm25_anchor = float(any(doc_id in removed for doc_id in base_ids[:2]))
    answer_risk = (
        0.25 * (1.0 - prefix2)
        + 0.20 * removed_high_bm25_anchor
        + 0.20 * removed_high_bm25_anchor
        + 0.15 * query_overlap_loss
        + 0.10 * entity_loss
        + 0.10 * mean(disp)
    )
    stability = (
        0.30 * overlap
        + 0.25 * prefix2
        + 0.15 * prefix3
        + 0.15 * (baseline_anchor_retention / 3.0)
        + 0.15 * kendall_similarity(base_ids, cand_ids)
    )
    return {
        "overlap_at5": overlap,
        "prefix1_same": prefix1,
        "prefix2_same": prefix2,
        "prefix3_same": prefix3,
        "number_added_docs": float(len(added)),
        "number_removed_docs": float(len(removed)),
        "average_displacement": mean(disp),
        "max_displacement": max(disp, default=0.0),
        "kendall_tau_order_similarity": kendall_similarity(base_ids, cand_ids),
        "baseline_anchor_retention_count": float(baseline_anchor_retention),
        "query_context_token_overlap": mean(cand_lex),
        "query_title_overlap": mean(title_lex(cand_ids)),
        "named_entity_overlap_between_query_and_context": cand_entity,
        "title_diversity": len({docs_by_id[d].title for d in cand_ids if d in docs_by_id}) / max(len(cand_ids), 1),
        "context_entity_coverage": cand_entity,
        "baseline_removed_entity_loss": entity_loss,
        "candidate_added_entity_gain": max(0.0, cand_entity - base_entity),
        "removed_high_bm25_anchor": removed_high_bm25_anchor,
        "removed_high_dense_anchor": removed_high_bm25_anchor,
        "query_overlap_loss": query_overlap_loss,
        "entity_coverage_loss": entity_loss,
        "context_stability_score": stability,
        "answer_risk_score": answer_risk,
        "added_lex_mean": mean(added_lex),
        "removed_lex_mean": mean(removed_lex),
        "baseline_mean_lex": mean(base_lex),
        "candidate_mean_lex": mean(cand_lex),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        "n": len(rows),
        "answer_access_at_k": mean([float(r["answer_access_at_k"]) for r in rows]),
        "support_recall_at_k": mean([float(r["support_recall_at_k"]) for r in rows]),
        "sp_f1": mean([float(r["sp_f1"]) for r in rows]),
        "answer_em": mean([float(r["answer_em"]) for r in rows]),
        "answer_f1": mean([float(r["answer_f1"]) for r in rows]),
        "joint_f1": mean([float(r["joint_f1"]) for r in rows]),
    }


def train_predictor(train: list[dict[str, Any]], feature_names: list[str], label: str, model_type: str):
    y = np.array([int(r[label]) for r in train], dtype=np.int64)
    if len(set(y.tolist())) < 2:
        return None, None
    x = np.array([[float(r["features"].get(name, 0.0)) for name in feature_names] for r in train], dtype=np.float32)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=17)
    elif model_type == "gb":
        model = GradientBoostingClassifier(random_state=17, max_depth=3)
    else:
        model = RandomForestClassifier(
            n_estimators=180,
            max_depth=5,
            min_samples_leaf=3,
            random_state=17,
            class_weight="balanced",
        )
    model.fit(x_scaled, y)
    return scaler, model


def query_level_predict(samples: list[dict[str, Any]], feature_names: list[str], model_type: str) -> list[dict[str, Any]]:
    out = []
    for qid in sorted({s["id"] for s in samples}):
        train = [s for s in samples if s["id"] != qid]
        test = [s for s in samples if s["id"] == qid]
        scaler, model = train_predictor(train, feature_names, "label_safe_answer", model_type)
        if model is None:
            continue
        x = np.array([[float(r["features"].get(name, 0.0)) for name in feature_names] for r in test], dtype=np.float32)
        probs = model.predict_proba(scaler.transform(x))[:, 1]
        for row, prob in zip(test, probs):
            cp = dict(row)
            cp["safe_answer_prob"] = float(prob)
            out.append(cp)
    return out


def predictor_quality(preds: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    y = np.array([int(r["label_safe_answer"]) for r in preds])
    yhat = np.array([int(float(r["safe_answer_prob"]) >= threshold) for r in preds])
    p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    unsafe = 1 - y
    unsafe_hat = 1 - yhat
    up, ur, uf1, _ = precision_recall_fscore_support(unsafe, unsafe_hat, average="binary", zero_division=0)
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, yhat)),
        "precision_safe": float(p),
        "recall_safe": float(r),
        "f1_safe": float(f1),
        "precision_unsafe": float(up),
        "recall_unsafe": float(ur),
        "f1_unsafe": float(uf1),
        "safe_rate": float(y.mean()) if len(y) else 0.0,
        "pred_safe_rate": float(yhat.mean()) if len(yhat) else 0.0,
    }


def selector_score(row: dict[str, Any], variant: str) -> float:
    f = row["features"]
    support_delta = float(row.get("support_proxy_delta", 0.0))
    agent_delta = float(row.get("agent_weight_delta_vs_baseline", 0.0))
    hybrid_delta = float(row.get("hybrid_score_delta_vs_baseline", 0.0))
    stability = float(f.get("context_stability_score", 0.0))
    safe = float(row.get("safe_answer_prob", 0.5))
    risk = float(f.get("answer_risk_score", 0.0))
    if variant == "without_predictor":
        safe = 0.5
    if variant == "predictor_only":
        return safe - 0.15 * risk
    if variant == "support_proxy_only":
        return support_delta + 0.35 * agent_delta + 0.15 * hybrid_delta
    if variant == "support_proxy_answer_risk":
        return support_delta + 0.25 * agent_delta + 0.15 * hybrid_delta + 0.15 * stability - 0.45 * risk
    return 0.35 * support_delta + 0.20 * agent_delta + 0.15 * hybrid_delta + 0.15 * stability + 0.15 * safe - 0.30 * risk


def run_selector(
    preds: list[dict[str, Any]],
    baseline_by_id: dict[str, dict[str, Any]],
    *,
    threshold: float,
    risk_threshold: float,
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in preds:
        if row["mode"] != "baseline":
            by_q[row["id"]].append(row)
    chosen = []
    for qid, rows in by_q.items():
        eligible = rows
        if variant not in {"without_predictor", "support_proxy_only"}:
            eligible = [r for r in eligible if float(r.get("safe_answer_prob", 0.0)) >= threshold]
        if variant not in {"predictor_only", "support_proxy_only"}:
            eligible = [r for r in eligible if float(r["features"].get("answer_risk_score", 1.0)) <= risk_threshold]
        if not eligible:
            base = baseline_by_id[qid]
            chosen.append({**base, "selected_mode": "baseline_fallback", "used_candidate": False, "selector_score": 0.0, "safe_answer_prob": 0.0})
            continue
        best = max(eligible, key=lambda r: (selector_score(r, variant), float(r.get("safe_answer_prob", 0.0)), r["mode"]))
        chosen.append({**best["candidate_metrics"], "selected_mode": best["mode"], "used_candidate": True, "selector_score": selector_score(best, variant), "safe_answer_prob": best.get("safe_answer_prob", 0.0), "features": best["features"]})
    metrics = summarize_rows(chosen)
    n = len(chosen)
    metrics.update({
        "variant": variant,
        "threshold": threshold,
        "risk_threshold": risk_threshold,
        "fallback_rate": sum(1 for r in chosen if not r.get("used_candidate")) / max(n, 1),
        "average_added_docs": mean([float(r.get("features", {}).get("number_added_docs", 0.0)) for r in chosen]),
        "average_removed_docs": mean([float(r.get("features", {}).get("number_removed_docs", 0.0)) for r in chosen]),
        "prefix2_preserve_rate": mean([float(r.get("features", {}).get("prefix2_same", 1.0)) for r in chosen]),
        "prefix3_preserve_rate": mean([float(r.get("features", {}).get("prefix3_same", 1.0)) for r in chosen]),
        "selected_candidate_distribution": dict(Counter([r.get("selected_mode", r.get("mode")) for r in chosen])),
    })
    return chosen, metrics


def failure_label(base: dict[str, Any], selected: dict[str, Any]) -> str:
    da = float(selected["answer_f1"]) - float(base["answer_f1"])
    dj = float(selected["joint_f1"]) - float(base["joint_f1"])
    dr = float(selected["support_recall_at_k"]) - float(base["support_recall_at_k"])
    feats = selected.get("features", {})
    if selected.get("selected_mode") == "baseline_fallback":
        return "baseline_already_optimal"
    if da < -1e-9 and feats.get("query_overlap_loss", 0.0) > 0.02:
        return "answer_text_missing_proxy"
    if da < -1e-9 and feats.get("average_displacement", 0.0) > 0.25:
        return "evidence_order_issue"
    if da < -1e-9 and feats.get("number_removed_docs", 0.0) > 0:
        return "context_replacement_loss"
    if dr > 1e-9 and da < -1e-9:
        return "support_gain_reader_interference"
    if selected.get("safe_answer_prob", 1.0) >= 0.7 and da < -1e-9:
        return "predictor_false_safe"
    if selected.get("safe_answer_prob", 0.0) < 0.7 and dj > 1e-9:
        return "predictor_false_unsafe"
    if dr <= 1e-9 and dj <= 1e-9:
        return "insufficient_support_gain"
    return "baseline_already_optimal" if abs(da) < 1e-12 and abs(dj) < 1e-12 else "other"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# V7-HP-PAPER support_insertion_selector_v1 Report",
        "",
        "## Experiment Purpose",
        "",
        "V7-HP-PAPER targets the HP4 Phase 3 bottleneck: routing improves support exposure, but aggressive context replacement can lower answer_f1. The selector therefore chooses no-leak support insertions only when a query-level split reader-safety predictor and explicit answer-risk features consider them safe.",
        "",
        "## Strict No-Leak Design",
        "",
        "- Inference features exclude gold support titles, gold answer strings, answer presence, and current-query reader outcomes.",
        "- Reader outcome labels are used only for training/evaluating predictor_v2 with leave-one-query-out splits.",
        "- Candidate selection uses query/document lexical, dense/BM25 proxy, agent weights, context stability, entity/title overlap, and predictor probability.",
        "",
        "## 100-Sample Metrics",
        "",
        "| mode | n | access@5 | recall@5 | sp_f1 | answer_em | answer_f1 | joint_f1 | d_answer | d_joint | d_recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline = payload["baseline_summary"]
    for row in payload["metric_table"]:
        lines.append(
            f"| {row['mode']} | {row['n']} | {row['answer_access_at_k']:.4f} | {row['support_recall_at_k']:.4f} | "
            f"{row['sp_f1']:.4f} | {row['answer_em']:.4f} | {row['answer_f1']:.4f} | {row['joint_f1']:.4f} | "
            f"{row['answer_f1'] - baseline['answer_f1']:+.4f} | {row['joint_f1'] - baseline['joint_f1']:+.4f} | "
            f"{row['support_recall_at_k'] - baseline['support_recall_at_k']:+.4f} |"
        )
    lines.extend([
        "",
        "## Selector Ablation",
        "",
        "| variant | answer_f1 | joint_f1 | recall@5 | sp_f1 | fallback | prefix2 | selected distribution | gate_pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ])
    for row in payload["ablation"]:
        lines.append(
            f"| {row['variant']} | {row['answer_f1']:.4f} | {row['joint_f1']:.4f} | {row['support_recall_at_k']:.4f} | "
            f"{row['sp_f1']:.4f} | {row['fallback_rate']:.4f} | {row['prefix2_preserve_rate']:.4f} | "
            f"{json.dumps(row['selected_candidate_distribution'], ensure_ascii=False)} | {str(row['gate_pass'])} |"
        )
    best = payload["best_selector"]
    lines.extend([
        "",
        "## Gate Decision",
        "",
        f"- best_selector: `{best['variant']}`",
        f"- answer_f1_delta: {best['answer_f1'] - baseline['answer_f1']:+.4f}",
        f"- joint_f1_delta: {best['joint_f1'] - baseline['joint_f1']:+.4f}",
        f"- support_recall_delta: {best['support_recall_at_k'] - baseline['support_recall_at_k']:+.4f}",
        f"- sp_f1_delta: {best['sp_f1'] - baseline['sp_f1']:+.4f}",
        f"- gate_pass: {payload['gate_pass']}",
        "",
    ])
    if payload["gate_pass"]:
        lines.append("The 100-sample gate passed. The same selector can proceed to 1000 validation under identical no-leak constraints.")
    else:
        lines.append("The 100-sample gate did not pass. Do not start 1000 validation; inspect failure_summary and improve predictor/selector calibration first.")
    lines.extend([
        "",
        "## Failure Summary",
        "",
        "```json",
        json.dumps(payload["failure_summary"], ensure_ascii=False, indent=2),
        "```",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", default="V7-HP4/data/hotpot_validation_1000.json")
    parser.add_argument("--policy-a", default="V7-HP4/outputs/hp4_phase3_100_batch/policy_a/hp4_policy_reinforce.pt")
    parser.add_argument("--output-root", default="V7-HP-PAPER/outputs")
    parser.add_argument("--report-dir", default="V7-HP-PAPER/reports")
    parser.add_argument("--reader-model", default="google/flan-t5-large")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reader-batch-size", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--predictor-model", choices=["rf", "gb", "logreg"], default="rf")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--risk-threshold", type=float, default=0.38)
    args = parser.parse_args()

    random.seed(17)
    out_root = Path(args.output_root)
    selector_dir = out_root / "selector_v1_100"
    predictor_dir = out_root / "predictor_v2"
    ablation_dir = out_root / "ablation_100"
    selector_dir.mkdir(parents=True, exist_ok=True)
    predictor_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir.mkdir(parents=True, exist_ok=True)

    log(f"materializing {args.sample_size} HotpotQA validation cases")
    examples = READER.materialize_dev(Path(args.validation), args.sample_size)
    log(f"loading policy from {args.policy_a}")
    policy, policy_device = load_policy(Path(args.policy_a), args.device)

    candidate_items: list[dict[str, Any]] = []
    feature_samples: list[dict[str, Any]] = []
    baseline_by_id: dict[str, dict[str, Any]] = {}
    docs_meta: dict[str, dict[str, Any]] = {}

    for idx, (case, raw_docs) in enumerate(examples, start=1):
        docs = sanitize_docs(raw_docs)
        question = str(case.get("question", ""))
        qid = str(case.get("id", case.get("_id", "")))
        docs_by_id = {doc.doc_id: doc for doc in docs}
        docs_meta[qid] = {"question": question, "docs_by_id": docs_by_id, "case": case}
        weights = weights_for_docs(question, docs, policy, policy_device)
        candidates = generate_candidates(question, docs, weights, alpha=args.alpha, top_k=args.top_k)
        baseline = next(c for c in candidates if c["mode"] == "baseline")
        base_proxy = baseline["support_proxy_mean"]
        base_weight = mean([weights.get(doc_id, 0.0) for doc_id in baseline["top_doc_ids"]])
        base_hybrid = mean([score_all(question, docs, weights, args.alpha)[doc_id]["final"] for doc_id in baseline["top_doc_ids"]])
        for cand in candidates:
            gold = {str(t) for t in case.get("supporting_titles", [])}
            pred = {doc.title for doc in cand["top_docs"]}
            support_recall, sp_f1 = READER.sp_metrics(pred, gold)
            prompt = READER.make_prompt(question, cand["top_docs"])
            row = {
                "id": qid,
                "mode": cand["mode"],
                "ordering": cand["ordering"],
                "case": case,
                "prompt": prompt,
                "answer_access_at_k": READER.answer_in_context(case.get("answer", ""), cand["top_docs"]),
                "support_recall_at_k": support_recall,
                "sp_f1": sp_f1,
                "top_doc_ids": cand["top_doc_ids"],
                "top_titles": cand["top_titles"],
                "support_proxy_mean": cand["support_proxy_mean"],
                "support_proxy_max": cand["support_proxy_max"],
            }
            candidate_items.append(row)
        if idx % 25 == 0 or idx == len(examples):
            log(f"prepared candidate contexts {idx}/{len(examples)}")

    log(f"loading reader {args.reader_model} on {args.device}; prompts={len(candidate_items)}")
    reader = READER.Reader(args.reader_model, args.device, args.reader_batch_size)
    predictions = reader.generate([item["prompt"] for item in candidate_items])
    candidate_rows = []
    for item, prediction in zip(candidate_items, predictions):
        case = item.pop("case")
        item.pop("prompt")
        answer = case.get("answer", "")
        answer_f1 = READER.f1_score(prediction, answer)
        row = {
            **item,
            "prediction": prediction,
            "answer": answer,
            "answer_em": float(READER.normalize_answer(prediction) == READER.normalize_answer(answer)),
            "answer_f1": answer_f1,
            "joint_f1": answer_f1 * float(item["sp_f1"]),
        }
        candidate_rows.append(row)
        if row["mode"] == "baseline":
            baseline_by_id[row["id"]] = row

    by_id_mode = {(r["id"], r["mode"]): r for r in candidate_rows}
    for row in candidate_rows:
        if row["mode"] == "baseline":
            continue
        qid = row["id"]
        meta = docs_meta[qid]
        base = baseline_by_id[qid]
        features = context_features(meta["question"], meta["docs_by_id"], base, row)
        scores = score_all(meta["question"], list(meta["docs_by_id"].values()), weights_for_docs(meta["question"], list(meta["docs_by_id"].values()), policy, policy_device), args.alpha)
        base_hybrid = mean([scores[d]["final"] for d in base["top_doc_ids"] if d in scores])
        cand_hybrid = mean([scores[d]["final"] for d in row["top_doc_ids"] if d in scores])
        base_weight = mean([scores[d]["weight"] for d in base["top_doc_ids"] if d in scores])
        cand_weight = mean([scores[d]["weight"] for d in row["top_doc_ids"] if d in scores])
        features.update({
            "mean_dense_score": mean([scores[d]["dense"] for d in row["top_doc_ids"] if d in scores]),
            "max_dense_score": max([scores[d]["dense"] for d in row["top_doc_ids"] if d in scores], default=0.0),
            "mean_bm25_score": mean([scores[d]["sparse"] for d in row["top_doc_ids"] if d in scores]),
            "max_bm25_score": max([scores[d]["sparse"] for d in row["top_doc_ids"] if d in scores], default=0.0),
            "dense_delta_vs_baseline": 0.0,
            "bm25_delta_vs_baseline": 0.0,
            "hybrid_score_delta_vs_baseline": cand_hybrid - base_hybrid,
            "mean_agent_weight": cand_weight,
            "max_agent_weight": max([scores[d]["weight"] for d in row["top_doc_ids"] if d in scores], default=0.0),
            "top_agent_weight_count": float(sum(1 for d in row["top_doc_ids"] if scores.get(d, {}).get("weight", 0.0) >= 0.5)),
            "agent_weight_delta_vs_baseline": cand_weight - base_weight,
            "gated_doc_count": float(sum(1 for d in row["top_doc_ids"] if scores.get(d, {}).get("weight", 0.0) >= 0.5)),
        })
        d_answer = float(row["answer_f1"]) - float(base["answer_f1"])
        d_joint = float(row["joint_f1"]) - float(base["joint_f1"])
        sample = {
            "id": qid,
            "mode": row["mode"],
            "features": features,
            "support_proxy_delta": float(row.get("support_proxy_mean", 0.0)) - float(base.get("support_proxy_mean", 0.0)),
            "agent_weight_delta_vs_baseline": features["agent_weight_delta_vs_baseline"],
            "hybrid_score_delta_vs_baseline": features["hybrid_score_delta_vs_baseline"],
            "label_safe_answer": int(d_answer >= -1e-12),
            "label_safe_joint": int(d_joint >= -1e-12),
            "delta_answer_f1": d_answer,
            "delta_joint_f1": d_joint,
            "candidate_metrics": row,
            "baseline_metrics": base,
        }
        feature_samples.append(sample)

    feature_names = sorted(feature_samples[0]["features"]) if feature_samples else []
    preds = query_level_predict(feature_samples, feature_names, args.predictor_model)
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    quality = [predictor_quality(preds, t) for t in thresholds]

    baseline_summary = summarize_rows(list(baseline_by_id.values()))
    metric_table = []
    for mode in sorted({r["mode"] for r in candidate_rows}):
        rows = [r for r in candidate_rows if r["mode"] == mode]
        m = summarize_rows(rows)
        m["mode"] = mode
        metric_table.append(m)

    ablation = []
    selected_by_variant = {}
    for variant in ["without_predictor", "predictor_only", "support_proxy_only", "support_proxy_answer_risk", "full_selector"]:
        chosen, metrics = run_selector(preds, baseline_by_id, threshold=args.threshold, risk_threshold=args.risk_threshold, variant=variant)
        metrics["gate_pass"] = (
            metrics["answer_f1"] + 1e-12 >= baseline_summary["answer_f1"]
            and metrics["joint_f1"] > baseline_summary["joint_f1"] + 1e-12
            and metrics["support_recall_at_k"] > baseline_summary["support_recall_at_k"] + 1e-12
            and metrics["sp_f1"] + 1e-12 >= baseline_summary["sp_f1"]
            and metrics["fallback_rate"] < 1.0
        )
        ablation.append(metrics)
        selected_by_variant[variant] = chosen

    best = max(ablation, key=lambda r: (r["gate_pass"], r["answer_f1"] - baseline_summary["answer_f1"], r["joint_f1"] - baseline_summary["joint_f1"], r["support_recall_at_k"] - baseline_summary["support_recall_at_k"]))
    best_rows = selected_by_variant[best["variant"]]
    gate_pass = bool(best["gate_pass"])
    per_example = []
    failure_cases = []
    for row in best_rows:
        qid = row["id"]
        base = baseline_by_id[qid]
        added = [d for d in row.get("top_doc_ids", []) if d not in set(base.get("top_doc_ids", []))]
        removed = [d for d in base.get("top_doc_ids", []) if d not in set(row.get("top_doc_ids", []))]
        delta = {
            "id": qid,
            "question": docs_meta[qid]["question"],
            "selected_mode": row.get("selected_mode", row.get("mode")),
            "baseline_titles": base.get("top_titles", []),
            "selected_titles": row.get("top_titles", []),
            "added_titles": [docs_meta[qid]["docs_by_id"][d].title for d in added if d in docs_meta[qid]["docs_by_id"]],
            "removed_titles": [docs_meta[qid]["docs_by_id"][d].title for d in removed if d in docs_meta[qid]["docs_by_id"]],
            "answer_f1_delta": float(row["answer_f1"]) - float(base["answer_f1"]),
            "joint_f1_delta": float(row["joint_f1"]) - float(base["joint_f1"]),
            "support_recall_delta": float(row["support_recall_at_k"]) - float(base["support_recall_at_k"]),
            "selector_score": float(row.get("selector_score", 0.0)),
            "safe_probability": float(row.get("safe_answer_prob", 0.0)),
            "answer_risk_score": float(row.get("features", {}).get("answer_risk_score", 0.0)),
        }
        label = failure_label(base, row)
        delta["failure_label"] = label
        per_example.append(delta)
        if delta["answer_f1_delta"] < -1e-12 or delta["joint_f1_delta"] < -1e-12 or delta["support_recall_delta"] <= 1e-12:
            failure_cases.append(delta)
    failure_summary = {
        "n_failure_cases": len(failure_cases),
        "label_counts": dict(Counter([r["failure_label"] for r in failure_cases])),
    }

    selector_summary = {
        "sample_size": len(examples),
        "baseline_summary": baseline_summary,
        "metric_table": metric_table,
        "predictor_quality": quality,
        "ablation": ablation,
        "best_selector": best,
        "gate_pass": gate_pass,
        "failure_summary": failure_summary,
        "strict_no_leak": {
            "forbidden_inference_features": ["gold_supporting_facts", "gold_answer_string", "answer_presence", "current_query_reader_outcome", "oracle_delta"],
            "split": "leave-one-query-out",
            "feature_names": feature_names,
        },
    }

    save_json(selector_dir / "selector_summary.json", selector_summary)
    write_jsonl(selector_dir / "per_example_delta.jsonl", per_example)
    write_jsonl(selector_dir / "failure_cases.jsonl", failure_cases)
    save_json(selector_dir / "failure_summary.json", failure_summary)
    save_json(ablation_dir / "ablation_summary.json", {"baseline": baseline_summary, "ablation": ablation})
    save_json(predictor_dir / "predictor_v2_summary.json", {"feature_names": feature_names, "quality": quality, "model": args.predictor_model})
    save_json(predictor_dir / "candidate_rows.json", candidate_rows)
    save_json(predictor_dir / "predictor_samples.json", feature_samples)
    save_json(predictor_dir / "predictor_predictions.json", preds)
    write_report(Path(args.report_dir) / "v7_hp_paper_selector_v1_report.md", selector_summary)
    log(json.dumps({"gate_pass": gate_pass, "best_selector": best, "report": str(Path(args.report_dir) / "v7_hp_paper_selector_v1_report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
