#!/usr/bin/env python3
from bm25_anchor_common import *


def main() -> None:
    ensure_dirs()
    rows = alignment_outcomes()
    bm25 = bm25_by_q(rows)
    by_q = defaultdict(list)
    for r in rows:
        if r["query_id"] not in bm25:
            continue
        rr = dict(r)
        rr.update(metric_delta(rr, bm25[rr["query_id"]]))
        rr["positive_vs_bm25"] = int(rr["answer_f1_delta_vs_bm25"] >= 0 and rr["joint_f1_delta_vs_bm25"] > 0 and rr["evidence_f1_delta_vs_bm25"] >= 0)
        by_q[rr["query_id"]].append(rr)
    out_rows = []
    family_pos = Counter()
    family_total = Counter()
    selector = previous_selector_by_q()
    recall_hits = 0
    for qid, items in by_q.items():
        best = max(items, key=lambda r: (r["positive_vs_bm25"], r["joint_f1_delta_vs_bm25"], r["answer_f1_delta_vs_bm25"]))
        positives = [r for r in items if r["positive_vs_bm25"]]
        sel = selector.get(qid, {})
        if positives and sel.get("candidate_name") in {r.get("candidate_name") for r in positives}:
            recall_hits += 1
        for r in items:
            family_total[r.get("candidate_family", r.get("candidate_name", "unknown"))] += 1
            if r["positive_vs_bm25"]:
                family_pos[r.get("candidate_family", r.get("candidate_name", "unknown"))] += 1
        out_rows.append({
            "query_id": qid,
            "positive_vs_bm25": bool(positives),
            "num_positive_actions": len(positives),
            "best_candidate_name": best.get("candidate_name"),
            "best_candidate_family": best.get("candidate_family"),
            "best_answer_safe_joint_delta_vs_bm25": max([r["joint_f1_delta_vs_bm25"] for r in positives], default=0.0),
            "best_evidence_delta_vs_bm25": max(r["evidence_f1_delta_vs_bm25"] for r in items),
            "best_joint_delta_vs_bm25": max(r["joint_f1_delta_vs_bm25"] for r in items),
            "best_answer_f1_delta_vs_bm25": max(r["answer_f1_delta_vs_bm25"] for r in items),
        })
    n = max(1, len(out_rows))
    pos_count = sum(1 for r in out_rows if r["positive_vs_bm25"])
    summary = {
        "status": "complete",
        "num_queries": len(out_rows),
        "num_queries_with_positive_vs_bm25": pos_count,
        "positive_vs_bm25_rate": pos_count / n,
        "oracle_best_answer_delta_vs_bm25": sum(r["best_answer_f1_delta_vs_bm25"] for r in out_rows) / n,
        "oracle_best_evidence_delta_vs_bm25": sum(r["best_evidence_delta_vs_bm25"] for r in out_rows) / n,
        "oracle_best_joint_delta_vs_bm25": sum(r["best_joint_delta_vs_bm25"] for r in out_rows) / n,
        "candidate_family_positive_rate_vs_bm25": {k: family_pos[k] / family_total[k] for k in sorted(family_total)},
        "selector_recall_of_positive_vs_bm25": recall_hits / max(1, pos_count),
        "decision": "continue_bm25_anchor_repair" if pos_count / n >= 0.10 else "stop_candidate_pool_limitation" if pos_count / n < 0.05 else "borderline_manual_check",
        "claim_boundary": "Oracle is diagnostic only; it is not an inference-time method.",
    }
    write_jsonl(REPAIR / "outputs/oracle_gap_300/oracle_gap_rows.jsonl", out_rows)
    write_json(REPAIR / "outputs/oracle_gap_300/oracle_gap_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
