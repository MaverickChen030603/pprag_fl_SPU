#!/usr/bin/env python3
from selector_alignment_common import *


def main() -> None:
    ensure_dirs()
    src = ALIGN / "outputs/action_table_300/2wiki_action_table_300.jsonl"
    rows = list(iter_jsonl(src))
    rows = enrich_features(rows)
    write_jsonl(ALIGN / "outputs/action_table_300/2wiki_v23_feature_table_300.jsonl", rows)
    n = max(1, len(rows))
    aligned = [
        "support_proxy_delta",
        "support_proxy_delta_vs_replaced_doc",
        "answer_risk_score",
        "title_bridge_score",
        "prefix2_preserved",
        "prefix3_preserved",
        "num_added_docs",
        "num_removed_docs",
        "candidate_family",
        "candidate_name",
        "effective_context_changed",
        "safe_answer_prob",
    ]
    summary = {
        "status": "complete",
        "num_actions": len(rows),
        "num_queries": len({r["query_id"] for r in rows}),
        "aligned_features": aligned,
        "dense_feature_available": False,
        "safe_answer_prob_mode": "heuristic_smoke_only",
        "safe_answer_prob_mean": sum(float(r["safe_answer_prob"]) for r in rows) / n,
        "feature_null_counts": {k: sum(1 for r in rows if k not in r or r[k] is None) for k in aligned},
        "claim_boundary": "safe_answer_prob is heuristic at smoke stage. Formal 2Wiki selector training must use query-level cross-fitted train-fold outcome labels.",
    }
    write_json(ALIGN / "outputs/action_table_300/feature_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
