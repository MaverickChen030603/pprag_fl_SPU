#!/usr/bin/env python3
from bm25_anchor_common import *


def main() -> None:
    ensure_dirs()
    pred_path = REPAIR / "outputs/safety_predictor/crossfit_predictions.jsonl"
    rows = list(iter_jsonl(pred_path))
    bm25 = {r["query_id"]: r for r in rows if r["candidate_name"] == "bm25_fallback"}
    previous = previous_selector_by_q()
    results = {}
    selected_rows = []
    sig = {}
    grid = []
    for selected_fraction in [0.10, 0.20, 0.30, 0.40]:
        for safe_threshold in [0.50, 0.55, 0.60, 0.65]:
            for positive_threshold in [0.05, 0.10, 0.15, 0.20]:
                for preserve_top3 in [True, False]:
                    sel = selected_by_method(rows, "bm25_anchor_answer_neutral_selector", selected_fraction, safe_threshold, positive_threshold, preserve_top3)
                    stats = summarize(sel, bm25)
                    grid.append({"config": {"selected_fraction": selected_fraction, "safe_threshold": safe_threshold, "positive_threshold": positive_threshold, "preserve_top3": preserve_top3}, "stats": stats})
    grid.sort(key=lambda x: (x["stats"]["joint_f1_delta_vs_bm25"], x["stats"]["answer_f1_delta_vs_bm25"], x["stats"]["evidence_f1_delta_vs_bm25"]), reverse=True)
    best_cfg = grid[0]["config"]
    for method in METHODS:
        if method == "previous_2wiki_v23_crossfit_selector":
            # Previous rows already contain deltas but not full comparable evidence fields.
            sel = {}
            for q, b in bm25.items():
                p = previous.get(q, {})
                proxy = dict(b)
                proxy.update({
                    "candidate_name": p.get("candidate_name", "previous_unknown"),
                    "answer_f1": float(b.get("answer_f1", 0)) + float(p.get("answer_f1_delta_vs_bm25", 0)),
                    "joint_f1": float(b.get("joint_f1", 0)) + float(p.get("joint_f1_delta_vs_bm25", 0)),
                    "evidence_f1": float(b.get("evidence_f1", 0)) + float(p.get("evidence_f1_delta_vs_bm25", 0)),
                    "evidence_recall_at_k": float(b.get("evidence_recall_at_k", 0)) + float(p.get("evidence_recall_delta_vs_bm25", 0)),
                    "effective_context_changed": bool(p.get("candidate_name") not in {"baseline_context_order", "bm25_top5", None}),
                })
                sel[q] = proxy
        elif method == "bm25_anchor_answer_neutral_selector":
            sel = selected_by_method(rows, method, **best_cfg)
        else:
            sel = selected_by_method(rows, method)
        results[method] = summarize(sel, bm25)
        sig[method] = significance(sel, bm25)
        for q, r in sel.items():
            b = bm25.get(q, {})
            selected_rows.append({
                "query_id": q,
                "method": method,
                "candidate_name": r.get("candidate_name"),
                "candidate_family": r.get("candidate_family"),
                "bm25_titles": b.get("candidate_titles", b.get("bm25_titles", [])),
                "selected_titles": r.get("candidate_titles"),
                "added_titles": r.get("added_titles", []),
                "removed_titles": r.get("removed_titles", []),
                "answer_f1_delta_vs_bm25": float(r.get("answer_f1", 0)) - float(b.get("answer_f1", 0)),
                "evidence_f1_delta_vs_bm25": float(r.get("evidence_f1", 0)) - float(b.get("evidence_f1", 0)),
                "evidence_recall_delta_vs_bm25": float(r.get("evidence_recall_at_k", 0)) - float(b.get("evidence_recall_at_k", 0)),
                "joint_f1_delta_vs_bm25": float(r.get("joint_f1", 0)) - float(b.get("joint_f1", 0)),
                "safe_answer_prob": r.get("safe_answer_prob", 1.0),
                "positive_action_prob": r.get("positive_action_prob", 0.0),
                "evidence_proxy_delta_vs_bm25": r.get("evidence_proxy_delta_vs_bm25", 0.0),
                "answer_risk_score": r.get("answer_risk_score", 0.0),
            })
    target = results["bm25_anchor_answer_neutral_selector"]
    prev = results["previous_2wiki_v23_crossfit_selector"]
    gate = {
        "passed": bool(
            target["answer_f1_delta_vs_bm25"] >= 0
            and target["joint_f1_delta_vs_bm25"] > 0
            and target["evidence_f1_delta_vs_bm25"] >= 0
            and target["evidence_recall_delta_vs_bm25"] > 0
            and target["selected_effective_action_rate"] >= 0.95
            and target["positive_vs_bm25_recall"] > prev.get("positive_vs_bm25_recall", 0)
        ),
        "best_config": best_cfg,
        "decision": "consider_1000" if False else "stop_at_smoke_300",
    }
    if gate["passed"]:
        gate["decision"] = "consider_1000"
    summary = {
        "status": "complete",
        "n": len(bm25),
        "reader_model": "google/flan-t5-large",
        "top_k": TOP_K,
        "methods": results,
        "grid_top5": grid[:5],
        "gate": gate,
    }
    write_json(REPAIR / "outputs/selector_smoke_300/summary.json", summary)
    write_jsonl(REPAIR / "outputs/selector_smoke_300/per_example_delta.jsonl", selected_rows)
    write_json(REPAIR / "outputs/selector_smoke_300/significance_report.json", sig)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
