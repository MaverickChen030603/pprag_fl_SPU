#!/usr/bin/env python3
from selector_alignment_common import *


METHODS = [
    "context_order",
    "bm25_or_lexical_routing",
    "support_first_selector",
    "hotpot_v23_frozen_transfer",
    "2wiki_v23_crossfit_selector",
    "no_safety_predictor",
    "no_support_features",
    "oracle_diagnostic_only",
]


def add_reader_outcomes() -> list[dict[str, Any]]:
    feature_rows = list(iter_jsonl(ALIGN / "outputs/action_table_300/2wiki_v23_feature_table_300.jsonl"))
    # Rebuild private docs for prompts while preserving public feature rows.
    examples = load_dev_sample(300, SEED)
    private = build_actions_for_examples(examples)
    public_by_key = {(r["query_id"], r["candidate_name"]): r for r in feature_rows}
    merged = []
    for row in private:
        key = (row["query_id"], row["candidate_name"])
        pub = dict(public_by_key[key])
        pub["_docs"] = row["_docs"]
        pub["candidate_indices"] = row["candidate_indices"]
        pub["_supporting_titles"] = sorted(support_titles(next(ex for ex in examples if query_id(ex) == row["query_id"])))
        merged.append(pub)
    cache = ALIGN / "outputs/selector_smoke_300/action_reader_outcomes_300.jsonl"
    outcomes = run_reader_for_actions(merged, cache)
    bm25_by_q = {r["query_id"]: r for r in outcomes if r["candidate_name"] == "bm25_top5"}
    enriched = []
    for r in outcomes:
        b = bm25_by_q.get(r["query_id"], {})
        rr = dict(r)
        rr["answer_f1_delta_vs_bm25"] = float(rr.get("answer_f1", 0)) - float(b.get("answer_f1", 0))
        rr["joint_f1_delta_vs_bm25"] = float(rr.get("joint_f1", 0)) - float(b.get("joint_f1", 0))
        rr["evidence_recall_delta_vs_bm25"] = float(rr.get("evidence_recall_at_k", 0)) - float(b.get("evidence_recall_at_k", 0))
        rr["evidence_f1_delta_vs_bm25"] = float(rr.get("evidence_f1", 0)) - float(b.get("evidence_f1", 0))
        rr["answer_safe"] = int(rr["answer_f1_delta_vs_bm25"] >= 0)
        rr["paper_positive"] = int(rr["answer_f1_delta_vs_bm25"] >= 0 and rr["joint_f1_delta_vs_bm25"] > 0)
        enriched.append(rr)
    write_jsonl(ALIGN / "outputs/selector_smoke_300/action_reader_outcomes_300_labeled.jsonl", enriched)
    return enriched


def crossfit_select(rows: list[dict[str, Any]], method: str) -> dict[str, dict[str, Any]]:
    all_selected = {}
    if method not in {"2wiki_v23_crossfit_selector", "no_safety_predictor", "no_support_features"}:
        return select_by_method(rows, method)
    for train_q, test_q in split_queries(rows, folds=5):
        train = [r for r in rows if r["query_id"] in train_q]
        test = [r for r in rows if r["query_id"] in test_q]
        all_selected.update(select_by_method(test, method, train_rows=train))
    return all_selected


def main() -> None:
    ensure_dirs()
    outcomes = add_reader_outcomes()
    bm25 = select_by_method(outcomes, "bm25_or_lexical_routing")
    summary = {
        "status": "complete",
        "split": "dev",
        "n": len({r["query_id"] for r in outcomes}),
        "seed": SEED,
        "reader_model": "google/flan-t5-large",
        "top_k": TOP_K,
        "methods": {},
        "gate": {},
        "claim_boundary": "Selector smoke 300 with query-level cross-fit for 2Wiki-trained variants. Main baseline is BM25/lexical, not raw context order.",
    }
    per_rows = []
    sig = {}
    for method in METHODS:
        selected = crossfit_select(outcomes, method)
        stats = summarize_selected(selected, bm25_by_q=bm25)
        positives = [r for r in outcomes if r["query_id"] in selected and r.get("paper_positive")]
        q_with_positive = {r["query_id"] for r in positives}
        selected_positive = sum(1 for q, r in selected.items() if r.get("paper_positive"))
        stats["positive_candidate_recall"] = len(q_with_positive) / max(1, len(selected))
        stats["answer_safe_positive_candidate_recall"] = sum(1 for q, r in selected.items() if r.get("paper_positive") and r.get("answer_safe")) / max(1, len(selected))
        stats["selected_paper_positive_rate"] = selected_positive / max(1, len(selected))
        summary["methods"][method] = stats
        sig[method] = pairwise_significance(selected, bm25)
        for q, r in selected.items():
            b = bm25.get(q, {})
            per_rows.append({
                "query_id": q,
                "method": method,
                "candidate_name": r.get("candidate_name"),
                "candidate_family": r.get("candidate_family"),
                "answer_f1_delta_vs_bm25": float(r.get("answer_f1", 0)) - float(b.get("answer_f1", 0)),
                "joint_f1_delta_vs_bm25": float(r.get("joint_f1", 0)) - float(b.get("joint_f1", 0)),
                "evidence_recall_delta_vs_bm25": float(r.get("evidence_recall_at_k", 0)) - float(b.get("evidence_recall_at_k", 0)),
                "evidence_f1_delta_vs_bm25": float(r.get("evidence_f1", 0)) - float(b.get("evidence_f1", 0)),
                "selected_titles": r.get("candidate_titles"),
                "bm25_titles": b.get("candidate_titles"),
                "answer": r.get("answer"),
                "prediction": r.get("prediction"),
            })
    target = summary["methods"]["2wiki_v23_crossfit_selector"]
    gate_pass = (
        target.get("answer_f1_delta_vs_bm25", -1) >= 0
        and (target.get("evidence_recall@5_delta_vs_bm25", 0) > 0 or target.get("evidence_f1_delta_vs_bm25", 0) > 0)
        and target.get("joint_f1_delta_vs_bm25", 0) > 0
        and target.get("selected_effective_action_rate", 0) >= 0.95
    )
    summary["gate"] = {
        "passed": bool(gate_pass),
        "answer_f1_delta_vs_bm25": target.get("answer_f1_delta_vs_bm25", 0),
        "evidence_recall_delta_vs_bm25": target.get("evidence_recall@5_delta_vs_bm25", 0),
        "evidence_f1_delta_vs_bm25": target.get("evidence_f1_delta_vs_bm25", 0),
        "joint_f1_delta_vs_bm25": target.get("joint_f1_delta_vs_bm25", 0),
        "selected_effective_action_rate": target.get("selected_effective_action_rate", 0),
        "decision": "run_1000" if gate_pass else "stop_at_smoke_300",
    }
    write_json(ALIGN / "outputs/selector_smoke_300/summary.json", summary)
    write_jsonl(ALIGN / "outputs/selector_smoke_300/per_example_delta.jsonl", per_rows)
    write_json(ALIGN / "outputs/selector_smoke_300/significance_report.json", sig)
    print(json.dumps(summary["gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
