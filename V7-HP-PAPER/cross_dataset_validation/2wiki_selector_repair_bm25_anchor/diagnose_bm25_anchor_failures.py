#!/usr/bin/env python3
from bm25_anchor_common import *


def main() -> None:
    ensure_dirs()
    rows = [r for r in iter_jsonl(REPAIR / "outputs/selector_smoke_300/per_example_delta.jsonl") if r["method"] == "bm25_anchor_answer_neutral_selector"]
    oracle = {r["query_id"]: r for r in iter_jsonl(REPAIR / "outputs/oracle_gap_300/oracle_gap_rows.jsonl")}
    cases = []
    for r in rows:
        labels = []
        o = oracle.get(r["query_id"], {})
        if not o.get("positive_vs_bm25"):
            labels.append("candidate_pool_no_positive_vs_bm25")
        if r["candidate_name"] in {"bm25_fallback", "bm25_no_change_control"} and o.get("positive_vs_bm25"):
            labels.append("positive_vs_bm25_available_but_not_selected")
        if float(r["answer_f1_delta_vs_bm25"]) < 0:
            labels.append("answer_drop_selected")
        if float(r["joint_f1_delta_vs_bm25"]) < 0:
            labels.append("selector_underperforms_bm25")
        if r["added_titles"] == [] and r["removed_titles"] == []:
            labels.append("bm25_already_optimal" if not o.get("positive_vs_bm25") else "ineffective_action_selected")
        if labels:
            rr = dict(r)
            rr["failure_label"] = labels[0]
            rr["failure_labels"] = labels
            cases.append(rr)
    summary = {
        "status": "complete",
        "num_failures": len(cases),
        "failure_distribution": dict(Counter(c["failure_label"] for c in cases)),
    }
    write_jsonl(REPAIR / "outputs/diagnostics/failure_cases.jsonl", cases)
    write_json(REPAIR / "outputs/diagnostics/failure_summary.json", summary)
    write_json(REPAIR / "outputs/selector_smoke_300/failure_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
