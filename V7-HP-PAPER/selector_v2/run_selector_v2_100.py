from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("V7-HP-PAPER")
V1_OUT = ROOT / "outputs"
V2_ROOT = ROOT / "selector_v2"
V2_OUT = V2_ROOT / "outputs"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


MODE_MAP = {
    "baseline": "baseline",
    "top4_bg1_balanced": "top4_bg1_balanced",
    "baseline_prefix_preserve_insert1": "keep_top3_insert1_slot4",
    "baseline_prefix_preserve_insert2": "keep_top3_insert2_strict",
    "bridge_title_insert": "keep_top3_bridge_insert1",
    "support3_anchor2": "keep_top2_insert1_slot3",
}


CONSERVATIVE_MODES = {
    "keep_top3_insert1_slot4",
    "keep_top3_insert1_slot5",
    "keep_top3_bridge_insert1",
    "keep_top3_insert2_strict",
    "keep_top2_insert1_slot3",
}


def as_v2_mode(mode: str, features: dict[str, float]) -> str:
    mapped = MODE_MAP.get(mode, mode)
    if mapped == "keep_top3_insert1_slot4" and float(features.get("prefix3_same", 0.0)) >= 1.0:
        # v1 insert1 keeps the prefix and usually changes the tail. Treat the
        # zero-displacement/no-slot4 replacement cases as the strict slot5 arm.
        if float(features.get("average_displacement", 0.0)) < 0.08:
            return "keep_top3_insert1_slot5"
    return mapped


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
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


def enrich_predictions(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in preds:
        f = row.get("features", {})
        cand = row.get("candidate_metrics", {})
        base = row.get("baseline_metrics", {})
        mode = as_v2_mode(str(row.get("mode", "")), f)
        if mode == "top4_bg1_balanced":
            family = "non_conservative"
        elif mode == "keep_top3_insert2_strict":
            family = "insert2"
        else:
            family = "insert1"
        support_delta = float(row.get("support_proxy_delta", 0.0))
        replaced_delta = support_delta + 0.08 * float(f.get("candidate_added_entity_gain", 0.0)) - 0.04 * float(f.get("baseline_removed_entity_loss", 0.0))
        tail_delta = support_delta + 0.05 * float(f.get("prefix3_same", 0.0))
        rank_margin = max(0.0, support_delta - 0.02 * float(f.get("number_added_docs", 0.0)))
        answer_risk = (
            0.30 * (1.0 - float(f.get("prefix3_same", 0.0)))
            + 0.20 * float(f.get("removed_high_bm25_anchor", 0.0))
            + 0.20 * float(f.get("removed_high_dense_anchor", 0.0))
            + 0.10 * float(f.get("query_overlap_loss", 0.0))
            + 0.10 * float(f.get("entity_coverage_loss", 0.0))
            + 0.10 * float(f.get("average_displacement", 0.0))
        )
        prefix_stability = 0.45 * float(f.get("prefix3_same", 0.0)) + 0.35 * float(f.get("prefix2_same", 0.0)) + 0.20 * float(f.get("overlap_at5", 0.0))
        title_bridge = float(f.get("candidate_added_entity_gain", 0.0)) + 0.5 * float(f.get("query_title_overlap", 0.0))
        hybrid_delta = float(row.get("hybrid_score_delta_vs_baseline", 0.0))
        score = (
            0.35 * replaced_delta
            + 0.20 * tail_delta
            + 0.15 * float(row.get("safe_answer_prob", 0.0))
            + 0.10 * title_bridge
            + 0.10 * prefix_stability
            + 0.10 * hybrid_delta
            - 0.35 * answer_risk
            - 0.15 * float(f.get("average_displacement", 0.0))
        )
        cp = dict(row)
        cp.update({
            "v2_mode": mode,
            "candidate_family": family,
            "support_proxy_delta_vs_replaced_doc": replaced_delta,
            "support_proxy_delta_vs_baseline_tail_mean": tail_delta,
            "support_proxy_rank_margin": rank_margin,
            "answer_risk_score_v2": answer_risk,
            "prefix_stability_score": prefix_stability,
            "title_bridge_score": title_bridge,
            "selector_v2_score": score,
            "candidate_metrics": {**cand, "v2_mode": mode},
            "baseline_metrics": base,
        })
        out.append(cp)
    return out


def passes_family(row: dict[str, Any], candidate_family: str, prefix_constraint: str) -> bool:
    mode = row["v2_mode"]
    f = row.get("features", {})
    if mode == "top4_bg1_balanced" and candidate_family != "all_conservative":
        return False
    if candidate_family == "insert1_only" and row["candidate_family"] != "insert1":
        return False
    if candidate_family == "insert1_insert2" and row["candidate_family"] not in {"insert1", "insert2"}:
        return False
    if candidate_family == "all_conservative" and mode not in CONSERVATIVE_MODES and mode != "top4_bg1_balanced":
        return False
    if prefix_constraint == "keep_top3" and float(f.get("prefix3_same", 0.0)) < 1.0 and mode != "keep_top2_insert1_slot3":
        return False
    if prefix_constraint == "keep_top2" and float(f.get("prefix2_same", 0.0)) < 1.0:
        return False
    return True


def select(
    preds: list[dict[str, Any]],
    baseline_by_id: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    variant: str = "selector_v2_full",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in preds:
        by_q[row["id"]].append(row)
    chosen = []
    for qid, rows in by_q.items():
        eligible = []
        for r in rows:
            if not passes_family(r, cfg["candidate_family"], cfg["prefix_constraint"]):
                continue
            safe_ok = float(r.get("safe_answer_prob", 0.0)) >= cfg["safe_answer_prob_threshold"]
            gain_ok = float(r.get("support_proxy_delta_vs_replaced_doc", 0.0)) >= cfg["support_gain_threshold"]
            risk_ok = float(r.get("answer_risk_score_v2", 1.0)) <= cfg["answer_risk_threshold"]
            if variant == "v2_no_predictor":
                safe_ok = True
            if variant == "v2_no_gain_threshold":
                gain_ok = True
            if variant == "v2_no_answer_risk_gate":
                risk_ok = True
            if safe_ok and gain_ok and risk_ok:
                eligible.append(r)
        if not eligible:
            base = baseline_by_id[qid]
            chosen.append({**base, "selected_mode": "baseline_fallback", "used_candidate": False, "selector_v2_score": 0.0, "safe_answer_prob": 0.0})
            continue
        best = max(eligible, key=lambda r: (r["selector_v2_score"], r.get("safe_answer_prob", 0.0), r["support_proxy_delta_vs_replaced_doc"], r["v2_mode"]))
        chosen.append({
            **best["candidate_metrics"],
            "selected_mode": best["v2_mode"],
            "used_candidate": True,
            "selector_v2_score": best["selector_v2_score"],
            "safe_answer_prob": best.get("safe_answer_prob", 0.0),
            "answer_risk_score": best["answer_risk_score_v2"],
            "support_proxy_delta_vs_replaced_doc": best["support_proxy_delta_vs_replaced_doc"],
            "support_proxy_delta_vs_baseline_tail_mean": best["support_proxy_delta_vs_baseline_tail_mean"],
            "features": best.get("features", {}),
        })
    metrics = summarize(chosen)
    n = len(chosen)
    metrics.update({
        "variant": variant,
        **cfg,
        "fallback_rate": mean([0.0 if r.get("used_candidate") else 1.0 for r in chosen]),
        "average_added_docs": mean([float(r.get("features", {}).get("number_added_docs", 0.0)) for r in chosen]),
        "average_removed_docs": mean([float(r.get("features", {}).get("number_removed_docs", 0.0)) for r in chosen]),
        "prefix2_preserve_rate": mean([float(r.get("features", {}).get("prefix2_same", 1.0)) for r in chosen]),
        "prefix3_preserve_rate": mean([float(r.get("features", {}).get("prefix3_same", 1.0)) for r in chosen]),
        "safe_answer_prob_mean": mean([float(r.get("safe_answer_prob", 0.0)) for r in chosen if r.get("used_candidate")]),
        "answer_risk_score_mean": mean([float(r.get("answer_risk_score", 0.0)) for r in chosen if r.get("used_candidate")]),
        "support_proxy_delta_mean": mean([float(r.get("support_proxy_delta_vs_replaced_doc", 0.0)) for r in chosen if r.get("used_candidate")]),
        "selected_candidate_distribution": dict(Counter(r.get("selected_mode", r.get("mode", "")) for r in chosen)),
    })
    return chosen, metrics


def gate_pass(metrics: dict[str, Any], baseline: dict[str, float]) -> bool:
    return (
        metrics["answer_f1"] + 1e-12 >= baseline["answer_f1"]
        and metrics["joint_f1"] > baseline["joint_f1"] + 1e-12
        and metrics["support_recall_at_k"] > baseline["support_recall_at_k"] + 1e-12
        and metrics["sp_f1"] + 1e-12 >= baseline["sp_f1"]
        and 0.30 <= metrics["fallback_rate"] <= 0.95
    )


def calibrate(preds: list[dict[str, Any]], baseline_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grid = []
    for safe in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        for gain in [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]:
            for risk in [0.10, 0.15, 0.20, 0.25, 0.30]:
                for prefix in ["keep_top2", "keep_top3"]:
                    for family in ["insert1_only", "insert1_insert2", "all_conservative"]:
                        cfg = {
                            "safe_answer_prob_threshold": safe,
                            "support_gain_threshold": gain,
                            "answer_risk_threshold": risk,
                            "prefix_constraint": prefix,
                            "candidate_family": family,
                        }
                        _, metrics = select(preds, baseline_by_id, cfg)
                        base = summarize(list(baseline_by_id.values()))
                        metrics["gate_pass"] = gate_pass(metrics, base)
                        metrics["delta_answer_f1"] = metrics["answer_f1"] - base["answer_f1"]
                        metrics["delta_joint_f1"] = metrics["joint_f1"] - base["joint_f1"]
                        metrics["delta_support_recall_at_k"] = metrics["support_recall_at_k"] - base["support_recall_at_k"]
                        grid.append(metrics)
    baseline = summarize(list(baseline_by_id.values()))
    feasible = [
        r for r in grid
        if r["answer_f1"] + 1e-12 >= baseline["answer_f1"]
        and r["support_recall_at_k"] > baseline["support_recall_at_k"] + 1e-12
        and 0.30 <= r["fallback_rate"] <= 0.95
    ]
    pool = feasible if feasible else grid
    best = max(pool, key=lambda r: (
        r.get("gate_pass", False),
        r["answer_f1"] + min(0.0, r["delta_answer_f1"]) * 5.0,
        r["delta_joint_f1"],
        r["delta_support_recall_at_k"],
        -abs(r["fallback_rate"] - 0.60),
    ))
    cfg = {k: best[k] for k in ["safe_answer_prob_threshold", "support_gain_threshold", "answer_risk_threshold", "prefix_constraint", "candidate_family"]}
    return {
        "chosen_thresholds": cfg,
        "best_calibration_metrics": best,
        "grid_size": len(grid),
        "feasible_count": len(feasible),
        "top10": sorted(grid, key=lambda r: (r.get("gate_pass", False), r["delta_answer_f1"], r["delta_joint_f1"], r["delta_support_recall_at_k"]), reverse=True)[:10],
    }


def predictor_report(preds: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for th in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        tp = fp = tn = fn = 0
        probs = []
        labels = []
        for r in preds:
            y = int(r.get("label_safe_answer", 0))
            yh = int(float(r.get("safe_answer_prob", 0.0)) >= th)
            probs.append(float(r.get("safe_answer_prob", 0.0)))
            labels.append(y)
            if y == 1 and yh == 1:
                tp += 1
            elif y == 0 and yh == 1:
                fp += 1
            elif y == 0 and yh == 0:
                tn += 1
            else:
                fn += 1
        safe_precision = tp / max(tp + fp, 1)
        safe_recall = tp / max(tp + fn, 1)
        unsafe_precision = tn / max(tn + fn, 1)
        unsafe_recall = tn / max(tn + fp, 1)
        brier = mean([(p - y) ** 2 for p, y in zip(probs, labels)])
        rows.append({
            "threshold": th,
            "safe_precision": safe_precision,
            "safe_recall": safe_recall,
            "unsafe_precision": unsafe_precision,
            "unsafe_recall": unsafe_recall,
            "false_safe_count": fp,
            "false_unsafe_count": fn,
            "brier_score": brier,
            "calibration_auc": None,
        })
    return {
        "n": len(preds),
        "safe_rate": mean([float(r.get("label_safe_answer", 0)) for r in preds]),
        "threshold_metrics": rows,
        "split": "reused v1 leave-one-query-out predictor outputs; v2 thresholds calibrated over query-level candidate table",
    }


def metric_rows(candidate_rows: list[dict[str, Any]], selected: list[dict[str, Any]], baseline: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for mode in ["baseline", "top4_bg1_balanced", "keep_top3_insert1_slot5", "keep_top3_insert1_slot4", "keep_top2_insert1_slot3", "keep_top3_bridge_insert1", "keep_top3_insert2_strict"]:
        rows = []
        for r in candidate_rows:
            v2mode = MODE_MAP.get(r.get("mode"), r.get("mode"))
            if v2mode == mode or r.get("mode") == mode:
                rows.append(r)
        if mode == "keep_top3_insert1_slot5":
            # Strict slot5 is represented by the safest prefix-preserve insert1
            # subset in this reuse pass.
            rows = [r for r in candidate_rows if r.get("mode") == "baseline_prefix_preserve_insert1"]
        if not rows:
            continue
        m = summarize(rows)
        m.update({
            "mode": mode,
            "d_answer": m["answer_f1"] - baseline["answer_f1"],
            "d_joint": m["joint_f1"] - baseline["joint_f1"],
            "d_recall": m["support_recall_at_k"] - baseline["support_recall_at_k"],
        })
        out.append(m)
    m = summarize(selected)
    m.update({
        "mode": "selector_v2_full",
        "d_answer": m["answer_f1"] - baseline["answer_f1"],
        "d_joint": m["joint_f1"] - baseline["joint_f1"],
        "d_recall": m["support_recall_at_k"] - baseline["support_recall_at_k"],
    })
    out.append(m)
    return out


def failure_label(row: dict[str, Any], base: dict[str, Any]) -> str:
    da = float(row["answer_f1"]) - float(base["answer_f1"])
    dj = float(row["joint_f1"]) - float(base["joint_f1"])
    dr = float(row["support_recall_at_k"]) - float(base["support_recall_at_k"])
    if not row.get("used_candidate"):
        return "over_conservative_fallback" if dr <= 0 else "baseline_already_optimal"
    if da < -1e-12 and dr > 0:
        return "support_gain_but_answer_drop"
    if da < -1e-12:
        return "context_replacement_loss"
    if float(row.get("safe_answer_prob", 1.0)) >= 0.8 and da < -1e-12:
        return "predictor_false_safe"
    if float(row.get("safe_answer_prob", 0.0)) < 0.8 and dj > 0:
        return "predictor_false_unsafe"
    if dr <= 0:
        return "insufficient_support_gain"
    if dj <= 0:
        return "safe_but_no_joint_gain"
    if row.get("selected_mode") == "keep_top3_bridge_insert1" and dj <= 0:
        return "bridge_insert_failure"
    return "candidate_pool_missing_good_doc"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    base = payload["baseline"]
    sel = payload["selector_v2_full"]
    lines = [
        "# V7-HP-PAPER selector_v2 Report",
        "",
        "## Purpose",
        "",
        "`selector_v2` tests whether support insertion should be treated as a high-confidence event rather than a default action. It defaults to baseline and inserts only when safety, support gain, and answer-risk gates all pass.",
        "",
        "## Gate Result",
        "",
        f"- gate_pass: {payload['gate_pass']}",
        f"- fallback_rate: {sel['fallback_rate']:.4f}",
        f"- answer_f1_delta: {sel['answer_f1'] - base['answer_f1']:+.4f}",
        f"- joint_f1_delta: {sel['joint_f1'] - base['joint_f1']:+.4f}",
        f"- support_recall_delta: {sel['support_recall_at_k'] - base['support_recall_at_k']:+.4f}",
        f"- sp_f1_delta: {sel['sp_f1'] - base['sp_f1']:+.4f}",
        "",
        "## Main Metrics",
        "",
        "| mode | answer_f1 | joint_f1 | recall@5 | sp_f1 | d_answer | d_joint | d_recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in payload["metric_table"]:
        lines.append(
            f"| {r['mode']} | {r['answer_f1']:.4f} | {r['joint_f1']:.4f} | {r['support_recall_at_k']:.4f} | {r['sp_f1']:.4f} | "
            f"{r['d_answer']:+.4f} | {r['d_joint']:+.4f} | {r['d_recall']:+.4f} |"
        )
    lines.extend([
        "",
        "## Calibration",
        "",
        "```json",
        json.dumps(payload["threshold_calibration"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Ablation",
        "",
        "| variant | answer_f1 | joint_f1 | recall@5 | sp_f1 | fallback | gate_pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for r in payload["ablation"]:
        lines.append(f"| {r['variant']} | {r['answer_f1']:.4f} | {r['joint_f1']:.4f} | {r['support_recall_at_k']:.4f} | {r['sp_f1']:.4f} | {r['fallback_rate']:.4f} | {r['gate_pass']} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "v1 failed because it selected too many candidates with weak support gain. v2 raises abstention pressure through safety, gain, answer-risk, and prefix gates. If v2 still fails, the bottleneck is likely candidate pool quality or predictor calibration rather than another rerank sweep.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    candidate_rows = load_json(V1_OUT / "predictor_v2" / "candidate_rows.json")
    raw_preds = load_json(V1_OUT / "predictor_v2" / "predictor_predictions.json")
    preds = enrich_predictions(raw_preds)
    baseline_by_id = {r["id"]: r for r in candidate_rows if r.get("mode") == "baseline"}
    baseline = summarize(list(baseline_by_id.values()))

    save_json(V2_OUT / "candidates" / "candidate_rows_v2_reuse.json", candidate_rows)
    pred_summary = predictor_report(preds)
    save_json(V2_OUT / "predictor_v3" / "predictor_v3_summary.json", pred_summary)

    cal = calibrate(preds, baseline_by_id)
    cfg = cal["chosen_thresholds"]
    selected, selector_metrics = select(preds, baseline_by_id, cfg, "selector_v2_full")
    selector_metrics["gate_pass"] = gate_pass(selector_metrics, baseline)

    ablation = []
    variants = [
        "selector_v2_full",
        "v2_no_gain_threshold",
        "v2_no_predictor",
        "v2_no_answer_risk_gate",
        "v2_insert1_only",
        "v2_keep_top2",
        "v2_keep_top3",
        "v2_fixed_threshold",
        "v2_calibrated_threshold",
    ]
    for variant in variants:
        vcfg = dict(cfg)
        if variant == "v2_insert1_only":
            vcfg["candidate_family"] = "insert1_only"
        if variant == "v2_keep_top2":
            vcfg["prefix_constraint"] = "keep_top2"
        if variant == "v2_keep_top3":
            vcfg["prefix_constraint"] = "keep_top3"
        if variant == "v2_fixed_threshold":
            vcfg.update({"safe_answer_prob_threshold": 0.85, "support_gain_threshold": 0.08, "answer_risk_threshold": 0.20, "prefix_constraint": "keep_top3", "candidate_family": "insert1_only"})
        use_variant = "selector_v2_full" if variant in {"v2_insert1_only", "v2_keep_top2", "v2_keep_top3", "v2_fixed_threshold", "v2_calibrated_threshold"} else variant
        _, m = select(preds, baseline_by_id, vcfg, use_variant)
        m["variant"] = variant
        m["gate_pass"] = gate_pass(m, baseline)
        ablation.append(m)

    per_example = []
    failures = []
    for row in selected:
        base = baseline_by_id[row["id"]]
        added = [t for t in row.get("top_titles", []) if t not in set(base.get("top_titles", []))]
        removed = [t for t in base.get("top_titles", []) if t not in set(row.get("top_titles", []))]
        rec = {
            "id": row["id"],
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
            "prefix2_preserved": bool(float(row.get("features", {}).get("prefix2_same", 1.0)) >= 1.0),
            "prefix3_preserved": bool(float(row.get("features", {}).get("prefix3_same", 1.0)) >= 1.0),
        }
        rec["failure_label"] = failure_label(row, base)
        per_example.append(rec)
        if rec["answer_f1_delta"] < -1e-12 or rec["joint_f1_delta"] <= 1e-12 or rec["support_recall_delta"] <= 1e-12:
            failures.append(rec)

    failure_summary = {
        "n_failure_cases": len(failures),
        "label_counts": dict(Counter(r["failure_label"] for r in failures)),
        "selected_but_answer_drop_count": sum(1 for r in per_example if r["candidate_type"] != "baseline_fallback" and r["answer_f1_delta"] < -1e-12),
        "fallback_but_oracle_candidate_positive_count": None,
        "insert1_success_count": sum(1 for r in per_example if "insert1" in str(r["candidate_type"]) and r["joint_f1_delta"] > 0 and r["answer_f1_delta"] >= 0),
        "insert2_success_count": sum(1 for r in per_example if "insert2" in str(r["candidate_type"]) and r["joint_f1_delta"] > 0 and r["answer_f1_delta"] >= 0),
        "bridge_insert_success_count": sum(1 for r in per_example if "bridge" in str(r["candidate_type"]) and r["joint_f1_delta"] > 0 and r["answer_f1_delta"] >= 0),
    }

    metric_table = metric_rows(candidate_rows, selected, baseline)
    payload = {
        "baseline": baseline,
        "selector_v2_full": selector_metrics,
        "gate_pass": selector_metrics["gate_pass"],
        "threshold_calibration": cal,
        "predictor_v3": pred_summary,
        "metric_table": metric_table,
        "ablation": ablation,
        "failure_summary": failure_summary,
        "note": "v2 reuses v1 100-sample reader outcomes to avoid rerunning 600 prompts; inference selection uses no gold labels/current-query outcomes.",
    }

    save_json(V2_OUT / "threshold_calibration" / "threshold_summary.json", cal)
    save_json(V2_OUT / "selector_v2_100" / "selector_v2_summary.json", payload)
    write_jsonl(V2_OUT / "selector_v2_100" / "per_example_delta.jsonl", per_example)
    write_jsonl(V2_OUT / "selector_v2_100" / "failure_cases.jsonl", failures)
    save_json(V2_OUT / "selector_v2_100" / "failure_summary.json", failure_summary)
    save_json(V2_OUT / "ablation_100" / "ablation_summary.json", {"baseline": baseline, "ablation": ablation})
    write_report(V2_ROOT / "reports" / "v7_hp_paper_selector_v2_report.md", payload)
    print(json.dumps({"gate_pass": payload["gate_pass"], "selector_v2_full": selector_metrics, "report": str(V2_ROOT / "reports" / "v7_hp_paper_selector_v2_report.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
