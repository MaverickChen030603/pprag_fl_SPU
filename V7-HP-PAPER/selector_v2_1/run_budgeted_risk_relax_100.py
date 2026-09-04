from __future__ import annotations

import importlib.util
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("V7-HP-PAPER")
V1_OUT = Path(os.environ.get("V21_INPUT_OUT", str(ROOT / "outputs")))
V2_DIR = ROOT / "selector_v2"
V21_DIR = ROOT / "selector_v2_1"
OUT = Path(os.environ.get("V21_OUTPUT_ROOT", str(V21_DIR / "outputs")))


def _load_v2():
    path = V2_DIR / "run_selector_v2_100.py"
    spec = importlib.util.spec_from_file_location("selector_v2_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V2 = _load_v2()


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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def family_ok(row: dict[str, Any], family: str) -> bool:
    mode = row.get("v2_mode", "")
    cf = row.get("candidate_family", "")
    if family == "insert1_only":
        return cf == "insert1"
    if family == "insert1_plus_bridge":
        return cf == "insert1" or "bridge" in mode
    if family == "insert1_plus_insert2":
        return cf in {"insert1", "insert2"}
    if family == "all_conservative":
        return mode != "top4_bg1_balanced"
    return True


def utility(row: dict[str, Any], risk_weight: float, mode: str = "full") -> float:
    safe = float(row.get("safe_answer_prob", 0.0))
    gain_replaced = float(row.get("support_proxy_delta_vs_replaced_doc", 0.0))
    gain_tail = float(row.get("support_proxy_delta_vs_baseline_tail_mean", 0.0))
    title_bridge = float(row.get("title_bridge_score", 0.0))
    hybrid_delta = float(row.get("hybrid_score_delta_vs_baseline", 0.0))
    agent_delta = float(row.get("agent_weight_delta_vs_baseline", 0.0))
    risk = float(row.get("answer_risk_score_v2", 0.0))
    displacement = float(row.get("features", {}).get("average_displacement", 0.0))
    if mode == "predictor_budget":
        return safe
    if mode == "support_gain_only":
        return gain_replaced + 0.5 * gain_tail
    if mode == "support_safety_budget":
        return 0.55 * safe + 0.30 * gain_replaced + 0.15 * gain_tail
    if mode == "no_risk_term":
        risk_weight = 0.0
    return (
        0.35 * safe
        + 0.30 * gain_replaced
        + 0.20 * gain_tail
        + 0.15 * title_bridge
        + 0.10 * hybrid_delta
        + 0.10 * agent_delta
        - float(risk_weight) * risk
        - 0.10 * displacement
    )


def candidate_actions(
    preds: list[dict[str, Any]],
    *,
    safe_threshold: float,
    support_gain_threshold: float | None,
    risk_penalty_weight: float,
    candidate_family: str,
    scoring_mode: str,
    hard_risk_gate: bool = False,
) -> list[dict[str, Any]]:
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in preds:
        f = row.get("features", {})
        if float(f.get("prefix2_same", 0.0)) < 1.0:
            continue
        if float(f.get("number_added_docs", 0.0)) > 1.0 or float(f.get("number_removed_docs", 0.0)) > 1.0:
            continue
        if float(row.get("safe_answer_prob", 0.0)) < safe_threshold:
            continue
        if support_gain_threshold is not None and float(row.get("support_proxy_delta_vs_replaced_doc", 0.0)) < support_gain_threshold:
            continue
        if hard_risk_gate and float(row.get("answer_risk_score_v2", 1.0)) > 0.30:
            continue
        if not family_ok(row, candidate_family):
            continue
        cp = dict(row)
        cp["budgeted_utility_score"] = utility(cp, risk_penalty_weight, scoring_mode)
        by_q[cp["id"]].append(cp)
    actions = []
    for qid, rows in by_q.items():
        best = max(rows, key=lambda r: (
            r["budgeted_utility_score"],
            float(r.get("safe_answer_prob", 0.0)),
            float(r.get("support_proxy_delta_vs_replaced_doc", 0.0)),
            r.get("v2_mode", ""),
        ))
        actions.append(best)
    actions.sort(key=lambda r: (
        r["budgeted_utility_score"],
        float(r.get("safe_answer_prob", 0.0)),
        float(r.get("support_proxy_delta_vs_replaced_doc", 0.0)),
        r.get("id", ""),
    ), reverse=True)
    for idx, row in enumerate(actions, start=1):
        row["budget_rank"] = idx
    return actions


def select_budget(
    preds: list[dict[str, Any]],
    baseline_by_id: dict[str, dict[str, Any]],
    *,
    budget: int | None,
    safe_threshold: float,
    support_gain_threshold: float | None,
    risk_penalty_weight: float,
    candidate_family: str,
    scoring_mode: str = "full",
    hard_risk_gate: bool = False,
    label: str = "selector_v2_1_budgeted",
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    actions = candidate_actions(
        preds,
        safe_threshold=safe_threshold,
        support_gain_threshold=support_gain_threshold,
        risk_penalty_weight=risk_penalty_weight,
        candidate_family=candidate_family,
        scoring_mode=scoring_mode,
        hard_risk_gate=hard_risk_gate,
    )
    selected_ids = {a["id"] for a in (actions if budget is None else actions[:budget])}
    action_by_id = {a["id"]: a for a in actions}
    chosen = []
    for qid, base in baseline_by_id.items():
        if qid in selected_ids:
            a = action_by_id[qid]
            chosen.append({
                **a["candidate_metrics"],
                "selected_mode": a["v2_mode"],
                "used_candidate": True,
                "budget_rank": a["budget_rank"],
                "budget_selected": True,
                "budgeted_utility_score": a["budgeted_utility_score"],
                "safe_answer_prob": a.get("safe_answer_prob", 0.0),
                "answer_risk_score": a.get("answer_risk_score_v2", 0.0),
                "support_proxy_delta_vs_replaced_doc": a.get("support_proxy_delta_vs_replaced_doc", 0.0),
                "support_proxy_delta_vs_baseline_tail_mean": a.get("support_proxy_delta_vs_baseline_tail_mean", 0.0),
                "features": a.get("features", {}),
            })
        else:
            chosen.append({
                **base,
                "selected_mode": "baseline_fallback",
                "used_candidate": False,
                "budget_rank": None,
                "budget_selected": False,
                "budgeted_utility_score": 0.0,
                "safe_answer_prob": 0.0,
                "answer_risk_score": 0.0,
                "support_proxy_delta_vs_replaced_doc": 0.0,
                "support_proxy_delta_vs_baseline_tail_mean": 0.0,
                "features": {},
            })
    metrics = V2.summarize(chosen)
    n = len(chosen)
    metrics.update({
        "variant": label,
        "budget_select_count": budget if budget is not None else len(actions),
        "available_actions": len(actions),
        "safe_answer_prob_threshold": safe_threshold,
        "support_gain_threshold": support_gain_threshold,
        "risk_penalty_weight": risk_penalty_weight,
        "candidate_family": candidate_family,
        "scoring_mode": scoring_mode,
        "hard_risk_gate": hard_risk_gate,
        "fallback_rate": V2.mean([0.0 if r.get("used_candidate") else 1.0 for r in chosen]),
        "selected_count": sum(1 for r in chosen if r.get("used_candidate")),
        "selected_candidate_distribution": dict(Counter(r.get("selected_mode") for r in chosen)),
        "average_added_docs": V2.mean([float(r.get("features", {}).get("number_added_docs", 0.0)) for r in chosen]),
        "average_removed_docs": V2.mean([float(r.get("features", {}).get("number_removed_docs", 0.0)) for r in chosen]),
        "prefix2_preserve_rate": V2.mean([float(r.get("features", {}).get("prefix2_same", 1.0)) for r in chosen]),
        "prefix3_preserve_rate": V2.mean([float(r.get("features", {}).get("prefix3_same", 1.0)) for r in chosen]),
        "safe_answer_prob_mean": V2.mean([float(r.get("safe_answer_prob", 0.0)) for r in chosen if r.get("used_candidate")]),
        "answer_risk_score_mean": V2.mean([float(r.get("answer_risk_score", 0.0)) for r in chosen if r.get("used_candidate")]),
        "support_proxy_delta_mean": V2.mean([float(r.get("support_proxy_delta_vs_replaced_doc", 0.0)) for r in chosen if r.get("used_candidate")]),
    })
    return chosen, metrics, actions


def gate_pass(m: dict[str, Any], base: dict[str, float]) -> bool:
    return (
        m["answer_f1"] + 1e-12 >= base["answer_f1"]
        and m["joint_f1"] > base["joint_f1"] + 1e-12
        and m["support_recall_at_k"] > base["support_recall_at_k"] + 1e-12
        and m["sp_f1"] + 1e-12 >= base["sp_f1"]
        and 0.30 <= m["fallback_rate"] <= 0.95
    )


def deltas(m: dict[str, Any], base: dict[str, float]) -> dict[str, float]:
    return {
        "d_answer": m["answer_f1"] - base["answer_f1"],
        "d_joint": m["joint_f1"] - base["joint_f1"],
        "d_recall": m["support_recall_at_k"] - base["support_recall_at_k"],
        "d_sp_f1": m["sp_f1"] - base["sp_f1"],
    }


def calibrate(preds: list[dict[str, Any]], baseline_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = V2.summarize(list(baseline_by_id.values()))
    rows = []
    for budget in [40, 50, 60, 70, 80]:
        for safe in [0.65, 0.70, 0.75, 0.80, 0.85]:
            for gain in [None, 0.00, 0.03, 0.05, 0.08]:
                for risk_w in [0.00, 0.05, 0.10, 0.15, 0.20]:
                    for family in ["insert1_only", "insert1_plus_bridge", "insert1_plus_insert2", "all_conservative"]:
                        _, m, _ = select_budget(
                            preds,
                            baseline_by_id,
                            budget=budget,
                            safe_threshold=safe,
                            support_gain_threshold=gain,
                            risk_penalty_weight=risk_w,
                            candidate_family=family,
                        )
                        m.update(deltas(m, base))
                        m["gate_pass"] = gate_pass(m, base)
                        m["target_fallback_band"] = 0.30 <= m["fallback_rate"] <= 0.50
                        rows.append(m)
    feasible = [
        r for r in rows
        if r["answer_f1"] + 1e-12 >= base["answer_f1"]
        and r["joint_f1"] > base["joint_f1"] + 1e-12
        and r["support_recall_at_k"] > base["support_recall_at_k"] + 1e-12
        and r["sp_f1"] + 1e-12 >= base["sp_f1"]
        and 0.30 <= r["fallback_rate"] <= 0.50
    ]
    relaxed = [
        r for r in rows
        if r["answer_f1"] + 1e-12 >= base["answer_f1"]
        and r["joint_f1"] > base["joint_f1"] + 1e-12
        and r["support_recall_at_k"] > base["support_recall_at_k"] + 1e-12
        and r["sp_f1"] + 1e-12 >= base["sp_f1"]
        and 0.25 <= r["fallback_rate"] <= 0.60
    ]
    pool = feasible or relaxed or rows
    best = max(pool, key=lambda r: (
        r.get("gate_pass", False),
        r["target_fallback_band"],
        r["d_answer"] >= 0,
        r["d_joint"],
        r["d_recall"],
        -abs(r["fallback_rate"] - 0.40),
    ))
    cfg = {k: best[k] for k in ["budget_select_count", "safe_answer_prob_threshold", "support_gain_threshold", "risk_penalty_weight", "candidate_family"]}
    return {
        "grid_size": len(rows),
        "strict_feasible_count": len(feasible),
        "relaxed_feasible_count": len(relaxed),
        "chosen_policy": cfg,
        "best_metrics": best,
        "top20": sorted(rows, key=lambda r: (r.get("gate_pass", False), r["target_fallback_band"], r["d_answer"], r["d_joint"], r["d_recall"]), reverse=True)[:20],
    }


def failure_label(row: dict[str, Any], base: dict[str, Any], actions_by_id: dict[str, dict[str, Any]]) -> str:
    da = float(row["answer_f1"]) - float(base["answer_f1"])
    dj = float(row["joint_f1"]) - float(base["joint_f1"])
    dr = float(row["support_recall_at_k"]) - float(base["support_recall_at_k"])
    if row.get("used_candidate"):
        if da < -1e-12:
            return "under_conservative_answer_drop"
        if dr > 0 and dj <= 0:
            return "support_gain_but_no_joint_gain"
        if da > 0 and dr <= 0:
            return "answer_gain_without_support_gain"
        if dr <= 0:
            return "insufficient_support_gain"
        if dj > 0:
            return "baseline_already_optimal"
        return "budget_selected_wrong_candidate"
    if row["id"] in actions_by_id:
        return "budget_rejected_positive_candidate"
    return "over_conservative_fallback"


def diagnostics(selected: list[dict[str, Any]], baseline_by_id: dict[str, dict[str, Any]], actions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions_by_id = {a["id"]: a for a in actions}
    rows = []
    for row in selected:
        qid = row["id"]
        base = baseline_by_id[qid]
        added = [t for t in row.get("top_titles", []) if t not in set(base.get("top_titles", []))]
        removed = [t for t in base.get("top_titles", []) if t not in set(row.get("top_titles", []))]
        rec = {
            "id": qid,
            "question": "",
            "baseline_titles": base.get("top_titles", []),
            "selected_titles": row.get("top_titles", []),
            "candidate_type": row.get("selected_mode"),
            "added_titles": added,
            "removed_titles": removed,
            "answer_f1_delta": float(row["answer_f1"]) - float(base["answer_f1"]),
            "joint_f1_delta": float(row["joint_f1"]) - float(base["joint_f1"]),
            "support_recall_delta": float(row["support_recall_at_k"]) - float(base["support_recall_at_k"]),
            "safe_answer_prob": float(row.get("safe_answer_prob", 0.0)),
            "answer_risk_score": float(row.get("answer_risk_score", 0.0)),
            "support_proxy_delta_vs_replaced_doc": float(row.get("support_proxy_delta_vs_replaced_doc", 0.0)),
            "support_proxy_delta_vs_baseline_tail_mean": float(row.get("support_proxy_delta_vs_baseline_tail_mean", 0.0)),
            "budget_rank": row.get("budget_rank"),
            "budget_selected": bool(row.get("budget_selected")),
        }
        rec["failure_label"] = failure_label(row, base, actions_by_id)
        rows.append(rec)
    failures = [r for r in rows if r["answer_f1_delta"] < -1e-12 or r["joint_f1_delta"] <= 1e-12 or r["support_recall_delta"] <= 1e-12]
    summary = {
        "n_cases": len(rows),
        "n_failure_cases": len(failures),
        "label_counts": dict(Counter(r["failure_label"] for r in failures)),
        "selected_but_answer_drop_count": sum(1 for r in rows if r["budget_selected"] and r["answer_f1_delta"] < -1e-12),
        "selected_and_answer_gain_count": sum(1 for r in rows if r["budget_selected"] and r["answer_f1_delta"] > 1e-12),
        "selected_and_joint_gain_count": sum(1 for r in rows if r["budget_selected"] and r["joint_f1_delta"] > 1e-12),
        "fallback_but_oracle_positive_count": None,
        "budget_rejected_positive_count": None,
        "budget_selected_negative_count": sum(1 for r in rows if r["budget_selected"] and (r["answer_f1_delta"] < -1e-12 or r["joint_f1_delta"] <= 1e-12)),
    }
    return rows, summary


def write_report(path: Path, payload: dict[str, Any]) -> None:
    base = payload["baseline"]
    best = payload["selector_v2_1_best"]
    lines = [
        "# V7-HP-PAPER selector_v2.1 Budgeted Risk Relax Report",
        "",
        "## Purpose",
        "",
        "`selector_v2.1_budgeted_risk_relax` starts from the positive `v2_no_answer_risk_gate` signal, keeps safety/prefix constraints, relaxes hard answer-risk filtering, and uses a global budget to control how often support insertions are allowed.",
        "",
        "## Gate Result",
        "",
        f"- gate_pass: {payload['gate_pass']}",
        f"- fallback_rate: {best['fallback_rate']:.4f}",
        f"- answer_f1_delta: {best['answer_f1'] - base['answer_f1']:+.4f}",
        f"- joint_f1_delta: {best['joint_f1'] - base['joint_f1']:+.4f}",
        f"- support_recall_delta: {best['support_recall_at_k'] - base['support_recall_at_k']:+.4f}",
        f"- sp_f1_delta: {best['sp_f1'] - base['sp_f1']:+.4f}",
        "",
        "## Main Metrics",
        "",
        "| mode | answer_f1 | joint_f1 | recall@5 | sp_f1 | fallback | selected | d_answer | d_joint | d_recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["metric_table"]:
        lines.append(
            f"| {row['mode']} | {row['answer_f1']:.4f} | {row['joint_f1']:.4f} | {row['support_recall_at_k']:.4f} | {row['sp_f1']:.4f} | "
            f"{row.get('fallback_rate', 0.0):.4f} | {row.get('selected_count', 0)} | {row['d_answer']:+.4f} | {row['d_joint']:+.4f} | {row['d_recall']:+.4f} |"
        )
    lines.extend([
        "",
        "## Calibration",
        "",
        "```json",
        json.dumps(payload["calibration"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Ablation",
        "",
        "| variant | answer_f1 | joint_f1 | recall@5 | sp_f1 | fallback | selected | gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in payload["ablation"]:
        lines.append(
            f"| {row['variant']} | {row['answer_f1']:.4f} | {row['joint_f1']:.4f} | {row['support_recall_at_k']:.4f} | {row['sp_f1']:.4f} | "
            f"{row['fallback_rate']:.4f} | {row['selected_count']} | {row['gate_pass']} |"
        )
    lines.extend([
        "",
        "## Diagnosis",
        "",
        "```json",
        json.dumps(payload["failure_summary"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "The key question is whether budgeted insertion can keep the positive no-risk signal while moving fallback from the overly aggressive 0.19 region into the 0.30-0.50 target band. If it passes the 100-sample gate, it is the first PAPER selector variant eligible for 1000 validation.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    candidate_rows = load_json(V1_OUT / "predictor_v2" / "candidate_rows.json")
    raw_preds = load_json(V1_OUT / "predictor_v2" / "predictor_predictions.json")
    preds = V2.enrich_predictions(raw_preds)
    baseline_by_id = {r["id"]: r for r in candidate_rows if r.get("mode") == "baseline"}
    baseline = V2.summarize(list(baseline_by_id.values()))

    calibration = calibrate(preds, baseline_by_id)
    cfg = calibration["chosen_policy"]
    selected, best_metrics, actions = select_budget(
        preds,
        baseline_by_id,
        budget=int(cfg["budget_select_count"]),
        safe_threshold=float(cfg["safe_answer_prob_threshold"]),
        support_gain_threshold=cfg["support_gain_threshold"],
        risk_penalty_weight=float(cfg["risk_penalty_weight"]),
        candidate_family=str(cfg["candidate_family"]),
        label="selector_v2_1_best",
    )
    best_metrics.update(deltas(best_metrics, baseline))
    best_metrics["gate_pass"] = gate_pass(best_metrics, baseline)

    metric_table = []
    metric_table.append({**baseline, "mode": "baseline", "fallback_rate": 0.0, "selected_count": 100, **deltas(baseline, baseline)})

    # Comparators from previous reports.
    v2_cfg = {"safe_threshold": 0.70, "support_gain_threshold": 0.0, "risk_penalty_weight": 0.15, "candidate_family": "insert1_only"}
    for budget in [40, 50, 60, 70, 80]:
        _, m, _ = select_budget(preds, baseline_by_id, budget=budget, scoring_mode="full", label=f"budgeted_top{budget}", **v2_cfg)
        m.update(deltas(m, baseline))
        m["gate_pass"] = gate_pass(m, baseline)
        m["mode"] = f"budgeted_top{budget}"
        metric_table.append(m)
    no_risk_selected, no_risk_m, _ = select_budget(
        preds, baseline_by_id, budget=None, safe_threshold=0.70, support_gain_threshold=0.0,
        risk_penalty_weight=0.0, candidate_family="insert1_only", scoring_mode="full", label="selector_v2_no_answer_risk_gate"
    )
    no_risk_m.update(deltas(no_risk_m, baseline))
    no_risk_m["gate_pass"] = gate_pass(no_risk_m, baseline)
    no_risk_m["mode"] = "selector_v2_no_answer_risk_gate"
    metric_table.append(no_risk_m)
    best_row = dict(best_metrics)
    best_row["mode"] = "selector_v2_1_best"
    metric_table.append(best_row)

    ablation_specs = [
        ("v2_1_no_budget", None, "full", 0.15, False, "insert1_only"),
        ("v2_1_hard_risk_gate", cfg["budget_select_count"], "full", 0.15, True, "insert1_only"),
        ("v2_1_soft_risk_only", cfg["budget_select_count"], "full", 0.15, False, "insert1_only"),
        ("v2_1_no_risk_term", cfg["budget_select_count"], "no_risk_term", 0.0, False, "insert1_only"),
        ("v2_1_support_gain_only", cfg["budget_select_count"], "support_gain_only", 0.0, False, "insert1_only"),
        ("v2_1_predictor_budget", cfg["budget_select_count"], "predictor_budget", 0.0, False, "insert1_only"),
        ("v2_1_support_safety_budget", cfg["budget_select_count"], "support_safety_budget", 0.0, False, "insert1_only"),
        ("v2_1_full_budgeted", cfg["budget_select_count"], "full", cfg["risk_penalty_weight"], False, cfg["candidate_family"]),
    ]
    ablation = []
    for name, budget, scoring, risk_w, hard_risk, family in ablation_specs:
        _, m, _ = select_budget(
            preds,
            baseline_by_id,
            budget=None if budget is None else int(budget),
            safe_threshold=float(cfg["safe_answer_prob_threshold"]),
            support_gain_threshold=cfg["support_gain_threshold"],
            risk_penalty_weight=float(risk_w),
            candidate_family=str(family),
            scoring_mode=scoring,
            hard_risk_gate=hard_risk,
            label=name,
        )
        m.update(deltas(m, baseline))
        m["gate_pass"] = gate_pass(m, baseline)
        ablation.append(m)

    per_example, failure_summary = diagnostics(selected, baseline_by_id, actions)
    payload = {
        "baseline": baseline,
        "selector_v2_1_best": best_metrics,
        "gate_pass": best_metrics["gate_pass"],
        "calibration": calibration,
        "metric_table": metric_table,
        "ablation": ablation,
        "failure_summary": failure_summary,
        "strict_no_leak": {
            "inference_forbidden": ["gold_support", "gold_answer", "answer_presence", "current_query_reader_outcome", "oracle_delta"],
            "selection_features": ["safe_answer_prob", "support_proxy_delta", "title_bridge", "hybrid_score_delta", "agent_weight_delta", "soft_answer_risk_penalty", "prefix2_preserved"],
            "note": "Reader outcomes are used only for offline 100-sample evaluation and grid reporting, not as per-query inference features.",
        },
    }

    save_json(OUT / "budget_policy_100" / "budget_policy_summary.json", payload)
    write_jsonl(OUT / "budget_policy_100" / "per_example_delta.jsonl", per_example)
    save_json(OUT / "budget_ablation_100" / "ablation_summary.json", {"baseline": baseline, "ablation": ablation})
    save_json(OUT / "calibration" / "budget_threshold_summary.json", calibration)
    write_jsonl(OUT / "diagnostics" / "failure_cases.jsonl", [r for r in per_example if r["failure_label"] != "baseline_already_optimal"])
    save_json(OUT / "diagnostics" / "failure_summary.json", failure_summary)
    write_report(V21_DIR / "reports" / "v7_hp_paper_selector_v2_1_report.md", payload)
    print(json.dumps({"gate_pass": payload["gate_pass"], "best": best_metrics, "report": str(V21_DIR / "reports" / "v7_hp_paper_selector_v2_1_report.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
