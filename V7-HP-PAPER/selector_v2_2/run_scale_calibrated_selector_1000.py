from __future__ import annotations

import importlib.util
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    LogisticRegression = None
    StandardScaler = None


ROOT = Path("V7-HP-PAPER")
V2 = ROOT / "selector_v2"
V21 = ROOT / "selector_v2_1"
V22 = ROOT / "selector_v2_2"
RAW = V21 / "outputs" / "final_1000" / "raw_candidate_eval"
OUT = V22 / "outputs"


def _load_v2():
    path = V2 / "run_selector_v2_100.py"
    spec = importlib.util.spec_from_file_location("selector_v2_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V2H = _load_v2()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "n": len(rows),
        "answer_access_at_k": mean([float(r.get("answer_access_at_k", 0.0)) for r in rows]),
        "support_recall_at_k": mean([float(r.get("support_recall_at_k", 0.0)) for r in rows]),
        "sp_f1": mean([float(r.get("sp_f1", 0.0)) for r in rows]),
        "answer_em": mean([float(r.get("answer_em", 0.0)) for r in rows]),
        "answer_f1": mean([float(r.get("answer_f1", 0.0)) for r in rows]),
        "joint_f1": mean([float(r.get("joint_f1", 0.0)) for r in rows]),
    }


def family_of(mode: str) -> str:
    if "insert2" in mode:
        return "insert2"
    if "bridge" in mode:
        return "bridge"
    if "top4" in mode:
        return "top4_bg1"
    return "insert1"


def load_action_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows = load_json(RAW / "predictor_v2" / "candidate_rows.json")
    preds = V2H.enrich_predictions(load_json(RAW / "predictor_v2" / "predictor_predictions.json"))
    v21_per = []
    p = V21 / "outputs" / "final_1000" / "per_example_delta.jsonl"
    if p.exists():
        v21_per = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return candidate_rows, preds, v21_per


def build_action_table(candidate_rows: list[dict[str, Any]], preds: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    baseline_by_id = {str(r["id"]): r for r in candidate_rows if r.get("mode") == "baseline"}
    rows = []
    for p in preds:
        qid = str(p["id"])
        base = baseline_by_id.get(qid)
        cand = p.get("candidate_metrics", {})
        if not base or not cand:
            continue
        base_ids = [str(x) for x in base.get("top_doc_ids", [])]
        cand_ids = [str(x) for x in cand.get("top_doc_ids", [])]
        base_titles = [str(x) for x in base.get("top_titles", [])]
        cand_titles = [str(x) for x in cand.get("top_titles", [])]
        added_titles = [t for t in cand_titles if t not in set(base_titles)]
        removed_titles = [t for t in base_titles if t not in set(cand_titles)]
        f = p.get("features", {})
        effective = (base_ids != cand_ids) or (base_titles != cand_titles)
        mode = str(p.get("v2_mode", p.get("mode", "")))
        row = {
            "query_id": qid,
            "question": "",
            "candidate_name": mode,
            "baseline_titles": base_titles,
            "candidate_titles": cand_titles,
            "baseline_doc_ids": base_ids,
            "candidate_doc_ids": cand_ids,
            "added_titles": added_titles,
            "removed_titles": removed_titles,
            "effective_context_changed": bool(effective),
            "num_added_docs": float(f.get("number_added_docs", len(added_titles))),
            "num_removed_docs": float(f.get("number_removed_docs", len(removed_titles))),
            "prefix1_preserved": bool(float(f.get("prefix1_same", 0.0)) >= 1.0),
            "prefix2_preserved": bool(float(f.get("prefix2_same", 0.0)) >= 1.0),
            "prefix3_preserved": bool(float(f.get("prefix3_same", 0.0)) >= 1.0),
            "removed_baseline_top1": bool(base_titles[:1] and base_titles[0] in removed_titles),
            "removed_baseline_top2": bool(any(t in removed_titles for t in base_titles[:2])),
            "safe_answer_prob": float(p.get("safe_answer_prob", 0.0)),
            "support_proxy_delta": float(p.get("support_proxy_delta", 0.0)),
            "support_proxy_delta_vs_replaced_doc": float(p.get("support_proxy_delta_vs_replaced_doc", p.get("support_proxy_delta", 0.0))),
            "support_proxy_delta_vs_baseline_tail_mean": float(p.get("support_proxy_delta_vs_baseline_tail_mean", p.get("support_proxy_delta", 0.0))),
            "answer_risk_score": float(p.get("answer_risk_score_v2", f.get("answer_risk_score", 0.0))),
            "displacement_score": float(f.get("average_displacement", 0.0)),
            "hybrid_score_delta": float(p.get("hybrid_score_delta_vs_baseline", 0.0)),
            "agent_weight_delta": float(p.get("agent_weight_delta_vs_baseline", 0.0)),
            "title_bridge_score": float(p.get("title_bridge_score", f.get("candidate_added_entity_gain", 0.0))),
            "candidate_family": family_of(mode),
            "baseline_answer_f1": float(base.get("answer_f1", 0.0)),
            "candidate_answer_f1": float(cand.get("answer_f1", 0.0)),
            "baseline_joint_f1": float(base.get("joint_f1", 0.0)),
            "candidate_joint_f1": float(cand.get("joint_f1", 0.0)),
            "baseline_support_recall": float(base.get("support_recall_at_k", 0.0)),
            "candidate_support_recall": float(cand.get("support_recall_at_k", 0.0)),
            "baseline_sp_f1": float(base.get("sp_f1", 0.0)),
            "candidate_sp_f1": float(cand.get("sp_f1", 0.0)),
            "baseline_answer_access_at_k": float(base.get("answer_access_at_k", 0.0)),
            "candidate_answer_access_at_k": float(cand.get("answer_access_at_k", 0.0)),
            "baseline_answer_em": float(base.get("answer_em", 0.0)),
            "candidate_answer_em": float(cand.get("answer_em", 0.0)),
        }
        row["answer_f1_delta"] = row["candidate_answer_f1"] - row["baseline_answer_f1"]
        row["joint_f1_delta"] = row["candidate_joint_f1"] - row["baseline_joint_f1"]
        row["support_recall_delta"] = row["candidate_support_recall"] - row["baseline_support_recall"]
        row["sp_f1_delta"] = row["candidate_sp_f1"] - row["baseline_sp_f1"]
        rows.append(row)
    return rows, baseline_by_id


def action_audit(rows: list[dict[str, Any]], v21_per: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = {str(r.get("id")) for r in v21_per if r.get("budget_selected")}
    selected = [r for r in rows if r["query_id"] in selected_ids]
    fam = defaultdict(list)
    for r in rows:
        fam[r["candidate_family"]].append(r)
    return {
        "total_queries": len({r["query_id"] for r in rows}),
        "total_candidate_actions": len(rows),
        "available_actions": len(rows),
        "effective_actions": sum(1 for r in rows if r["effective_context_changed"]),
        "ineffective_actions": sum(1 for r in rows if not r["effective_context_changed"]),
        "effective_action_rate": mean([float(r["effective_context_changed"]) for r in rows]),
        "selected_actions_in_v2_1": len(selected_ids),
        "selected_effective_actions_in_v2_1": sum(1 for r in selected if r["effective_context_changed"]),
        "selected_ineffective_actions_in_v2_1": sum(1 for r in selected if not r["effective_context_changed"]),
        "candidate_family_effective_rate": {
            k: mean([float(x["effective_context_changed"]) for x in v]) for k, v in fam.items()
        },
        "candidate_family_avg_delta_if_available": {
            k: {
                "answer_f1_delta": mean([x["answer_f1_delta"] for x in v]),
                "joint_f1_delta": mean([x["joint_f1_delta"] for x in v]),
                "support_recall_delta": mean([x["support_recall_delta"] for x in v]),
                "sp_f1_delta": mean([x["sp_f1_delta"] for x in v]),
            }
            for k, v in fam.items()
        },
    }


def family_ok(row: dict[str, Any], family: str) -> bool:
    cf = row["candidate_family"]
    if family == "insert1_only":
        return cf == "insert1"
    if family == "insert1_plus_bridge":
        return cf in {"insert1", "bridge"}
    if family == "all_effective_conservative":
        return cf in {"insert1", "bridge", "insert2"}
    return True


def base_eligible(row: dict[str, Any], cfg: dict[str, Any], effective_filter: bool = True) -> bool:
    if effective_filter and not row["effective_context_changed"]:
        return False
    if not row["prefix2_preserved"]:
        return False
    if row["removed_baseline_top1"] or row["removed_baseline_top2"]:
        return False
    if row["num_added_docs"] > 2 or row["num_removed_docs"] > 2:
        return False
    if row["safe_answer_prob"] < cfg["safe_threshold"]:
        return False
    gain = cfg.get("support_gain_threshold")
    if gain is not None and row["support_proxy_delta"] < gain:
        return False
    if not family_ok(row, cfg["candidate_family"]):
        return False
    return True


def score(row: dict[str, Any], cfg: dict[str, Any], model: Any = None, scaler: Any = None) -> float:
    util = cfg["utility"]
    if util == "calibrated_linear" and model is not None and scaler is not None:
        x = np.array([[row[k] for k in MODEL_FEATURES]], dtype=np.float32)
        return float(model.predict_proba(scaler.transform(x))[0, 1])
    if util == "safety_support":
        return 0.40 * row["safe_answer_prob"] + 0.35 * row["support_proxy_delta"] + 0.15 * row["title_bridge_score"] + 0.10 * row["agent_weight_delta"]
    if util == "support_first":
        return 0.45 * row["support_proxy_delta"] + 0.25 * row["safe_answer_prob"] + 0.15 * row["title_bridge_score"] + 0.15 * row["agent_weight_delta"] - 0.10 * row["answer_risk_score"]
    if util == "predictor_first":
        return 0.60 * row["safe_answer_prob"] + 0.20 * row["support_proxy_delta"] + 0.10 * row["hybrid_score_delta"] + 0.10 * row["title_bridge_score"]
    return (
        0.35 * row["safe_answer_prob"]
        + 0.30 * row["support_proxy_delta"]
        + 0.15 * row["title_bridge_score"]
        + 0.10 * row["agent_weight_delta"]
        + 0.10 * row["hybrid_score_delta"]
        - float(cfg.get("risk_penalty_weight", 0.10)) * row["answer_risk_score"]
    )


MODEL_FEATURES = [
    "safe_answer_prob", "support_proxy_delta", "answer_risk_score",
    "num_added_docs", "num_removed_docs", "hybrid_score_delta",
    "agent_weight_delta", "title_bridge_score", "displacement_score",
]


def train_linear(rows: list[dict[str, Any]]):
    if LogisticRegression is None or StandardScaler is None:
        return None, None
    y = np.array([int(r["answer_f1_delta"] >= -1e-12 and r["joint_f1_delta"] > 1e-12) for r in rows], dtype=np.int64)
    if len(set(y.tolist())) < 2:
        return None, None
    x = np.array([[float(r[k]) for k in MODEL_FEATURES] for r in rows], dtype=np.float32)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=17)
    model.fit(x, y)
    return model, scaler


def select_for_queries(
    rows: list[dict[str, Any]],
    query_ids: set[str],
    baseline_by_id: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    train_rows_for_model: list[dict[str, Any]] | None = None,
    effective_filter: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model = scaler = None
    if cfg["utility"] == "calibrated_linear":
        model, scaler = train_linear(train_rows_for_model or rows)
    by_q = defaultdict(list)
    for r in rows:
        if r["query_id"] not in query_ids:
            continue
        if base_eligible(r, cfg, effective_filter=effective_filter):
            rr = dict(r)
            rr["utility_score"] = score(rr, cfg, model=model, scaler=scaler)
            by_q[rr["query_id"]].append(rr)
    actions = []
    for qid, vals in by_q.items():
        best = max(vals, key=lambda x: (x["utility_score"], x["safe_answer_prob"], x["support_proxy_delta"], x["candidate_name"]))
        actions.append(best)
    actions.sort(key=lambda x: (x["utility_score"], x["safe_answer_prob"], x["support_proxy_delta"], x["query_id"]), reverse=True)
    budget = int(round(float(cfg["selected_fraction"]) * len(query_ids)))
    selected_ids = {r["query_id"] for r in actions[:budget]}
    action_by_id = {r["query_id"]: r for r in actions}
    chosen = []
    for qid in sorted(query_ids):
        base = baseline_by_id[qid]
        if qid in selected_ids:
            a = action_by_id[qid]
            rank = actions.index(a) + 1
            chosen.append({
                "id": qid,
                "mode": "selector_v2_2",
                "selected_mode": a["candidate_name"],
                "budget_selected": True,
                "effective_context_changed": a["effective_context_changed"],
                "answer_access_at_k": a["candidate_answer_access_at_k"],
                "support_recall_at_k": a["candidate_support_recall"],
                "sp_f1": a["candidate_sp_f1"],
                "answer_em": a["candidate_answer_em"],
                "answer_f1": a["candidate_answer_f1"],
                "joint_f1": a["candidate_joint_f1"],
                "baseline_answer_access_at_k": a["baseline_answer_access_at_k"],
                "baseline_support_recall_at_k": a["baseline_support_recall"],
                "baseline_sp_f1": a["baseline_sp_f1"],
                "baseline_answer_em": base.get("answer_em", 0.0),
                "baseline_answer_f1": a["baseline_answer_f1"],
                "baseline_joint_f1": a["baseline_joint_f1"],
                "answer_f1_delta": a["answer_f1_delta"],
                "joint_f1_delta": a["joint_f1_delta"],
                "support_recall_delta": a["support_recall_delta"],
                "sp_f1_delta": a["sp_f1_delta"],
                "baseline_titles": a["baseline_titles"],
                "selected_titles": a["candidate_titles"],
                "added_titles": a["added_titles"],
                "removed_titles": a["removed_titles"],
                "safe_answer_prob": a["safe_answer_prob"],
                "support_proxy_delta": a["support_proxy_delta"],
                "answer_risk_score": a["answer_risk_score"],
                "utility_score": a["utility_score"],
                "budget_rank": rank,
                "candidate_family": a["candidate_family"],
            })
        else:
            chosen.append({
                "id": qid,
                "mode": "baseline_fallback",
                "selected_mode": "baseline_fallback",
                "budget_selected": False,
                "effective_context_changed": False,
                "answer_access_at_k": float(base.get("answer_access_at_k", 0.0)),
                "support_recall_at_k": float(base.get("support_recall_at_k", 0.0)),
                "sp_f1": float(base.get("sp_f1", 0.0)),
                "answer_em": float(base.get("answer_em", 0.0)),
                "answer_f1": float(base.get("answer_f1", 0.0)),
                "joint_f1": float(base.get("joint_f1", 0.0)),
                "baseline_answer_access_at_k": float(base.get("answer_access_at_k", 0.0)),
                "baseline_support_recall_at_k": float(base.get("support_recall_at_k", 0.0)),
                "baseline_sp_f1": float(base.get("sp_f1", 0.0)),
                "baseline_answer_em": float(base.get("answer_em", 0.0)),
                "baseline_answer_f1": float(base.get("answer_f1", 0.0)),
                "baseline_joint_f1": float(base.get("joint_f1", 0.0)),
                "answer_f1_delta": 0.0,
                "joint_f1_delta": 0.0,
                "support_recall_delta": 0.0,
                "sp_f1_delta": 0.0,
                "baseline_titles": base.get("top_titles", []),
                "selected_titles": base.get("top_titles", []),
                "added_titles": [],
                "removed_titles": [],
                "safe_answer_prob": 0.0,
                "support_proxy_delta": 0.0,
                "answer_risk_score": 0.0,
                "utility_score": 0.0,
                "budget_rank": None,
                "candidate_family": "fallback",
            })
    return chosen, actions


def summarize_selection(chosen: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = summarize_metric_rows(chosen)
    baseline = {
        "answer_access_at_k": mean([float(r["baseline_answer_access_at_k"]) for r in chosen]),
        "support_recall_at_k": mean([float(r["baseline_support_recall_at_k"]) for r in chosen]),
        "sp_f1": mean([float(r["baseline_sp_f1"]) for r in chosen]),
        "answer_em": mean([float(r["baseline_answer_em"]) for r in chosen]),
        "answer_f1": mean([float(r["baseline_answer_f1"]) for r in chosen]),
        "joint_f1": mean([float(r["baseline_joint_f1"]) for r in chosen]),
    }
    selected = [r for r in chosen if r["budget_selected"]]
    metrics.update({
        "answer_f1_delta": metrics["answer_f1"] - baseline["answer_f1"],
        "joint_f1_delta": metrics["joint_f1"] - baseline["joint_f1"],
        "support_recall_delta": metrics["support_recall_at_k"] - baseline["support_recall_at_k"],
        "sp_f1_delta": metrics["sp_f1"] - baseline["sp_f1"],
        "fallback_rate": mean([0.0 if r["budget_selected"] else 1.0 for r in chosen]),
        "selected_count": len(selected),
        "effective_selected_count": sum(1 for r in selected if r["effective_context_changed"]),
        "selected_effective_action_rate": mean([float(r["effective_context_changed"]) for r in selected]) if selected else 0.0,
        "selected_candidate_distribution": dict(Counter(r["selected_mode"] for r in chosen)),
        "baseline": baseline,
    })
    metrics["gate_pass"] = (
        metrics["answer_f1_delta"] >= -1e-12
        and metrics["joint_f1_delta"] > 1e-12
        and metrics["support_recall_delta"] > 1e-12
        and metrics["sp_f1_delta"] >= -1e-12
        and metrics["selected_effective_action_rate"] >= 0.95
        and metrics["fallback_rate"] <= 0.85
    )
    return metrics


def config_grid(utilities: list[str] | None = None, families: list[str] | None = None) -> list[dict[str, Any]]:
    out = []
    utilities = utilities or ["safety_support", "safety_support_risk_soft", "support_first", "predictor_first"]
    families = families or ["insert1_plus_bridge", "all_effective_conservative"]
    for frac in [0.20, 0.30, 0.40, 0.50]:
        for safe in [0.55, 0.65, 0.75]:
            for gain in [None, 0.00, 0.04]:
                for risk in [0.00, 0.10]:
                    for fam in families:
                        for util in utilities:
                            out.append({
                                "selected_fraction": frac,
                                "safe_threshold": safe,
                                "support_gain_threshold": gain,
                                "risk_penalty_weight": risk,
                                "candidate_family": fam,
                                "utility": util,
                            })
    return out


def choose_config(train_rows: list[dict[str, Any]], train_qids: set[str], baseline_by_id: dict[str, dict[str, Any]], grid: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = []
    for cfg in grid:
        chosen, _ = select_for_queries(train_rows, train_qids, baseline_by_id, cfg, train_rows_for_model=train_rows)
        m = summarize_selection(chosen)
        item = {**cfg, **{k: v for k, v in m.items() if k != "baseline"}}
        scored.append(item)
    feasible = [
        s for s in scored
        if s["answer_f1_delta"] >= -1e-12
        and s["support_recall_delta"] > 1e-12
        and s["sp_f1_delta"] >= -1e-12
        and 0.50 <= s["fallback_rate"] <= 0.80
        and s["selected_effective_action_rate"] >= 0.95
    ]
    pool = feasible or scored
    best = max(pool, key=lambda s: (
        s.get("gate_pass", False),
        0.50 <= s["fallback_rate"] <= 0.80,
        s["answer_f1_delta"] >= 0,
        s["joint_f1_delta"],
        s["support_recall_delta"],
        -abs(s["fallback_rate"] - 0.65),
    ))
    cfg = {k: best[k] for k in ["selected_fraction", "safe_threshold", "support_gain_threshold", "risk_penalty_weight", "candidate_family", "utility"]}
    return cfg, sorted(scored, key=lambda s: (s.get("gate_pass", False), s["joint_f1_delta"], s["support_recall_delta"]), reverse=True)[:20]


def crossfit(rows: list[dict[str, Any]], baseline_by_id: dict[str, dict[str, Any]], grid: list[dict[str, Any]], n_folds: int = 5, effective_filter: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qids = sorted(baseline_by_id)
    folds = [set(qids[i::n_folds]) for i in range(n_folds)]
    all_chosen = []
    fold_configs = []
    for fold_id, test_qids in enumerate(folds):
        train_qids = set(qids) - test_qids
        train_rows = [r for r in rows if r["query_id"] in train_qids]
        cfg, top = choose_config(train_rows, train_qids, baseline_by_id, grid)
        chosen, actions = select_for_queries(rows, test_qids, baseline_by_id, cfg, train_rows_for_model=train_rows, effective_filter=effective_filter)
        for r in chosen:
            r["fold_id"] = fold_id
            r["fold_config"] = cfg
        all_chosen.extend(chosen)
        fold_configs.append({
            "fold_id": fold_id,
            "n_test": len(test_qids),
            "config": cfg,
            "heldout_summary": summarize_selection(chosen),
            "top_train_configs": top[:5],
        })
    return all_chosen, fold_configs


def bootstrap(rows: list[dict[str, Any]], samples: int = 2000) -> dict[str, Any]:
    random.seed(17)
    keys = {
        "answer_f1": "answer_f1_delta",
        "joint_f1": "joint_f1_delta",
        "support_recall@5": "support_recall_delta",
        "sp_f1": "sp_f1_delta",
    }
    out = {"n": len(rows), "num_bootstrap_samples": samples, "metrics": {}}
    for name, key in keys.items():
        vals = [float(r.get(key, 0.0)) for r in rows]
        obs = mean(vals)
        boots = []
        for _ in range(samples):
            boots.append(mean([random.choice(vals) for __ in vals]) if vals else 0.0)
        boots.sort()
        lo = boots[int(0.025 * samples)]
        hi = boots[max(0, int(0.975 * samples) - 1)]
        p = sum(1 for b in boots if b <= 0) / samples if obs >= 0 else sum(1 for b in boots if b >= 0) / samples
        out["metrics"][name] = {"mean_delta": obs, "ci95": [lo, hi], "p_value": p}
    return out


def failure_label(row: dict[str, Any], actions_by_qid: dict[str, list[dict[str, Any]]]) -> str:
    if row["budget_selected"]:
        if not row["effective_context_changed"]:
            return "ineffective_action_selected"
        if row["answer_f1_delta"] < -1e-12:
            return "under_abstention_answer_drop"
        if row["support_recall_delta"] > 0 and row["joint_f1_delta"] <= 0:
            return "support_gain_no_reader_gain"
        if row["answer_f1_delta"] > 0 and row["support_recall_delta"] <= 0:
            return "answer_gain_no_support_gain"
        if row["joint_f1_delta"] <= 0:
            return "wrong_action_selected"
        return "baseline_already_optimal"
    positives = [a for a in actions_by_qid.get(row["id"], []) if a["answer_f1_delta"] >= -1e-12 and a["joint_f1_delta"] > 1e-12]
    if positives:
        return "positive_action_rejected_by_budget"
    if actions_by_qid.get(row["id"]):
        return "candidate_pool_no_positive_action"
    return "over_abstention"


def diagnostics(rows: list[dict[str, Any]], action_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions_by_qid = defaultdict(list)
    for a in action_rows:
        if a["effective_context_changed"]:
            actions_by_qid[a["query_id"]].append(a)
    cases = []
    for r in rows:
        label = failure_label(r, actions_by_qid)
        rec = {
            "query_id": r["id"],
            "question": "",
            "baseline_titles": r["baseline_titles"],
            "selected_titles": r["selected_titles"],
            "candidate_name": r["selected_mode"],
            "added_titles": r["added_titles"],
            "removed_titles": r["removed_titles"],
            "effective_context_changed": r["effective_context_changed"],
            "answer_f1_delta": r["answer_f1_delta"],
            "joint_f1_delta": r["joint_f1_delta"],
            "support_recall_delta": r["support_recall_delta"],
            "sp_f1_delta": r["sp_f1_delta"],
            "safe_answer_prob": r["safe_answer_prob"],
            "support_proxy_delta": r["support_proxy_delta"],
            "answer_risk_score": r["answer_risk_score"],
            "utility_score": r["utility_score"],
            "budget_rank": r["budget_rank"],
            "fold_id": r.get("fold_id"),
            "fold_config": r.get("fold_config"),
            "failure_label": label,
        }
        cases.append(rec)
    failures = [c for c in cases if c["failure_label"] != "baseline_already_optimal"]
    summary = {
        "n_cases": len(cases),
        "n_failure_cases": len(failures),
        "label_counts": dict(Counter(c["failure_label"] for c in failures)),
        "selected_but_answer_drop_count": sum(1 for c in cases if c["candidate_name"] != "baseline_fallback" and c["answer_f1_delta"] < -1e-12),
        "selected_and_joint_gain_count": sum(1 for c in cases if c["candidate_name"] != "baseline_fallback" and c["joint_f1_delta"] > 1e-12),
        "selected_and_support_gain_count": sum(1 for c in cases if c["candidate_name"] != "baseline_fallback" and c["support_recall_delta"] > 1e-12),
        "fallback_but_positive_action_exists_count": sum(1 for c in cases if c["candidate_name"] == "baseline_fallback" and actions_by_qid.get(c["query_id"]) and any(a["answer_f1_delta"] >= -1e-12 and a["joint_f1_delta"] > 1e-12 for a in actions_by_qid[c["query_id"]])),
        "positive_action_rejected_count": sum(1 for c in cases if c["failure_label"] == "positive_action_rejected_by_budget"),
        "ineffective_action_selected_count": sum(1 for c in cases if c["failure_label"] == "ineffective_action_selected"),
        "candidate_pool_positive_rate": mean([float(any(a["answer_f1_delta"] >= -1e-12 and a["joint_f1_delta"] > 1e-12 for a in actions_by_qid.get(qid, []))) for qid in {c["query_id"] for c in cases}]),
    }
    return cases, summary


def oracle_gap(action_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_q = defaultdict(list)
    for r in action_rows:
        if r["effective_context_changed"]:
            by_q[r["query_id"]].append(r)
    selected = {r["id"]: r for r in selected_rows}
    best_joint = []
    best_safe = []
    best_support = []
    selector_hits = 0
    positive_q = 0
    for qid, vals in by_q.items():
        bj = max(vals, key=lambda x: x["joint_f1_delta"])
        bs = max(vals, key=lambda x: (x["answer_f1_delta"] >= -1e-12 and x["joint_f1_delta"] > 0, x["joint_f1_delta"]))
        bp = max(vals, key=lambda x: x["support_recall_delta"])
        best_joint.append(bj["joint_f1_delta"])
        best_safe.append(bs["joint_f1_delta"] if bs["answer_f1_delta"] >= -1e-12 else 0.0)
        best_support.append(bp["support_recall_delta"])
        pos = [v for v in vals if v["answer_f1_delta"] >= -1e-12 and v["joint_f1_delta"] > 1e-12]
        if pos:
            positive_q += 1
            sr = selected.get(qid)
            if sr and sr["selected_mode"] != "baseline_fallback" and sr["joint_f1_delta"] > 1e-12 and sr["answer_f1_delta"] >= -1e-12:
                selector_hits += 1
    n = max(len(by_q), 1)
    return {
        "queries_with_actions": len(by_q),
        "oracle_best_joint_delta": mean(best_joint),
        "oracle_best_answer_safe_joint_delta": mean(best_safe),
        "oracle_support_delta": mean(best_support),
        "positive_candidate_rate": positive_q / n,
        "answer_safe_positive_candidate_rate": positive_q / n,
        "selector_recall_of_positive_candidates": selector_hits / max(positive_q, 1),
        "diagnostic_only": True,
    }


def run_ablation(action_rows: list[dict[str, Any]], baseline_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    one = lambda utility, family="all_effective_conservative", frac=0.30, risk=0.0: [{
        "selected_fraction": frac,
        "safe_threshold": 0.65,
        "support_gain_threshold": None,
        "risk_penalty_weight": risk,
        "candidate_family": family,
        "utility": utility,
    }]
    specs = {
        "ablation_no_effective_filter": (one("safety_support", "all_effective", 0.30), False),
        "ablation_transfer_100_budget": ([{"selected_fraction": 0.08, "safe_threshold": 0.65, "support_gain_threshold": None, "risk_penalty_weight": 0.0, "candidate_family": "insert1_only", "utility": "safety_support_risk_soft"}], True),
        "ablation_global_oracle_like_calibration": (config_grid(["safety_support", "support_first"], ["all_effective"]), True),
        "ablation_cv_no_risk_term": (one("safety_support", "all_effective_conservative", 0.30, 0.0), True),
        "ablation_cv_soft_risk": (one("safety_support_risk_soft", "all_effective_conservative", 0.30, 0.10), True),
        "ablation_cv_support_first": (one("support_first", "all_effective_conservative", 0.30, 0.10), True),
        "ablation_cv_predictor_first": (one("predictor_first", "all_effective_conservative", 0.30, 0.0), True),
        "ablation_cv_insert1_only": (one("safety_support", "insert1_only", 0.30, 0.0), True),
        "ablation_cv_all_effective_conservative": (one("safety_support", "all_effective_conservative", 0.30, 0.0), True),
    }
    out = {}
    for name, (grid, effective_filter) in specs.items():
        chosen, folds = crossfit(action_rows, baseline_by_id, grid, effective_filter=effective_filter)
        summary = summarize_selection(chosen)
        summary["fold_config_distribution"] = dict(Counter(json.dumps(f["config"], sort_keys=True) for f in folds))
        summary["diagnostic_only"] = name == "ablation_global_oracle_like_calibration"
        out[name] = summary
    return out


def write_report(path: Path, payload: dict[str, Any]) -> None:
    s = payload["final_summary"]
    lines = [
        "# V7-HP-PAPER selector_v2.2 Scale-Calibrated Budget Report",
        "",
        "## Purpose",
        "",
        "v2.2 reuses the completed 1000 raw candidate reader outputs and performs query-level cross-fitted scale calibration. It filters ineffective actions and evaluates only held-out fold decisions.",
        "",
        "## Main Result",
        "",
        f"- gate_pass: {s['gate_pass']}",
        f"- answer_f1_delta: {s['answer_f1_delta']:+.4f}",
        f"- joint_f1_delta: {s['joint_f1_delta']:+.4f}",
        f"- support_recall_delta: {s['support_recall_delta']:+.4f}",
        f"- sp_f1_delta: {s['sp_f1_delta']:+.4f}",
        f"- fallback_rate: {s['fallback_rate']:.4f}",
        f"- selected_effective_action_rate: {s['selected_effective_action_rate']:.4f}",
        "",
        "## Audit",
        "",
        "```json",
        json.dumps(payload["audit_summary"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Significance",
        "",
        "```json",
        json.dumps(payload["significance"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Diagnosis",
        "",
        "```json",
        json.dumps(payload["failure_summary"], ensure_ascii=False, indent=2),
        "```",
    ]
    if "oracle_gap" in payload:
        lines += ["", "## Oracle Gap Diagnostic", "", "```json", json.dumps(payload["oracle_gap"], ensure_ascii=False, indent=2), "```"]
    lines += [
        "",
        "## Interpretation",
        "",
        "If gate passes, this is the first paper-ready no-leak selector result because thresholds are selected on train folds and evaluated on held-out queries. If it fails while the oracle gap is positive, the candidate space contains useful contexts but no-leak selection remains the bottleneck.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    candidate_rows, preds, v21_per = load_action_inputs()
    action_rows, baseline_by_id = build_action_table(candidate_rows, preds)
    audit = action_audit(action_rows, v21_per)
    save_json(OUT / "audit" / "action_audit_summary.json", audit)
    write_jsonl(OUT / "audit" / "action_audit_rows.jsonl", action_rows)
    action_summary = {
        "n_actions": len(action_rows),
        "n_queries": len(baseline_by_id),
        "effective_action_rate": mean([float(r["effective_context_changed"]) for r in action_rows]),
        "positive_action_rate": mean([float(r["answer_f1_delta"] >= -1e-12 and r["joint_f1_delta"] > 1e-12) for r in action_rows]),
    }
    save_json(OUT / "action_table" / "action_table_summary.json", action_summary)
    write_jsonl(OUT / "action_table" / "effective_action_table.jsonl", action_rows)

    grid = config_grid()
    chosen, folds = crossfit(action_rows, baseline_by_id, grid)
    final_summary = summarize_selection(chosen)
    final_summary["fold_config_distribution"] = dict(Counter(json.dumps(f["config"], sort_keys=True) for f in folds))
    save_json(OUT / "cv_calibration" / "cv_threshold_budget_summary.json", {"fold_configs": folds, "grid_size": len(grid)})
    save_json(OUT / "final_1000" / "final_1000_crossfit_summary.json", final_summary)
    write_jsonl(OUT / "final_1000" / "per_example_delta.jsonl", chosen)
    significance = bootstrap(chosen)
    save_json(OUT / "final_1000" / "significance_report.json", significance)

    ablation = run_ablation(action_rows, baseline_by_id)
    save_json(OUT / "ablation" / "ablation_summary.json", ablation)
    cases, failure_summary = diagnostics(chosen, action_rows)
    write_jsonl(OUT / "diagnostics" / "failure_cases.jsonl", cases)
    save_json(OUT / "diagnostics" / "failure_summary.json", failure_summary)

    payload = {
        "audit_summary": audit,
        "action_table_summary": action_summary,
        "final_summary": final_summary,
        "significance": significance,
        "ablation": ablation,
        "failure_summary": failure_summary,
    }
    if not final_summary["gate_pass"]:
        oracle = oracle_gap(action_rows, chosen)
        save_json(OUT / "diagnostics" / "oracle_gap_summary.json", oracle)
        payload["oracle_gap"] = oracle
    write_report(V22 / "reports" / "v7_hp_paper_selector_v2_2_report.md", payload)
    print(json.dumps({"gate_pass": final_summary["gate_pass"], "final_summary": final_summary, "report": str(V22 / "reports" / "v7_hp_paper_selector_v2_2_report.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
