#!/usr/bin/env python3
from selector_alignment_common import *


def main() -> None:
    ensure_dirs()
    summary = read_json(ALIGN / "outputs/selector_smoke_300/summary.json")
    rows = list(iter_jsonl(ALIGN / "outputs/selector_smoke_300/per_example_delta.jsonl"))
    target_rows = [r for r in rows if r["method"] == "2wiki_v23_crossfit_selector"]
    failures = []
    for r in target_rows:
        labels = []
        if r["candidate_name"] == "bm25_top5":
            labels.append("bm25_already_strong")
        if float(r["answer_f1_delta_vs_bm25"]) < 0:
            labels.append("answer_drop_selected")
        if float(r["joint_f1_delta_vs_bm25"]) < 0:
            labels.append("selector_underperforms_bm25")
        if float(r["evidence_recall_delta_vs_bm25"]) > 0 and float(r["joint_f1_delta_vs_bm25"]) <= 0:
            labels.append("support_positive_but_joint_negative")
        if not labels and float(r["joint_f1_delta_vs_bm25"]) <= 0:
            labels.append("positive_action_available_but_not_selected")
        if labels:
            failures.append({
                "query_id": r["query_id"],
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "baseline_titles": r.get("baseline_titles", []),
                "bm25_titles": r.get("bm25_titles", []),
                "selected_titles": r.get("selected_titles", []),
                "candidate_name": r.get("candidate_name", ""),
                "added_titles": r.get("added_titles", []),
                "removed_titles": r.get("removed_titles", []),
                "answer_f1_delta_vs_bm25": r["answer_f1_delta_vs_bm25"],
                "joint_f1_delta_vs_bm25": r["joint_f1_delta_vs_bm25"],
                "evidence_recall_delta_vs_bm25": r["evidence_recall_delta_vs_bm25"],
                "evidence_f1_delta_vs_bm25": r["evidence_f1_delta_vs_bm25"],
                "failure_label": labels[0],
                "failure_labels": labels,
            })
    counts = Counter(f["failure_label"] for f in failures)
    payload = {
        "status": "complete",
        "method": "2wiki_v23_crossfit_selector",
        "num_failures": len(failures),
        "failure_distribution": dict(counts),
        "gate": summary.get("gate", {}),
        "known_missing_checks": {
            "candidate_pool_no_positive_action": "manual_check_required",
            "adapter_metric_mismatch": False,
            "evidence_label_missing": False,
        },
    }
    write_jsonl(ALIGN / "outputs/diagnostics/failure_cases.jsonl", failures)
    write_json(ALIGN / "outputs/diagnostics/failure_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
