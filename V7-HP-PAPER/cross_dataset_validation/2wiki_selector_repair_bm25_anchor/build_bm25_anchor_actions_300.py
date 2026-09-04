#!/usr/bin/env python3
from bm25_anchor_common import *


def main() -> None:
    ensure_dirs()
    rows = build_anchor_actions()
    public = base.strip_private(rows)
    write_jsonl(REPAIR / "outputs/action_table_300/bm25_anchor_action_table_300.jsonl", public)
    n = max(1, len(public))
    hard_violations = [r for r in public if not (r["bm25_top1_preserved"] and r["bm25_top2_preserved"]) or r["num_added_docs"] > 1 or r["num_removed_docs"] > 1]
    summary = {
        "status": "complete",
        "num_queries": len({r["query_id"] for r in public}),
        "num_actions": len(public),
        "actions_per_query": len(public) / max(1, len({r["query_id"] for r in public})),
        "effective_action_rate": sum(float(r["effective_context_changed"]) for r in public) / n,
        "candidate_family_distribution": dict(Counter(r["candidate_family"] for r in public)),
        "avg_added_docs": sum(float(r["num_added_docs"]) for r in public) / n,
        "avg_removed_docs": sum(float(r["num_removed_docs"]) for r in public) / n,
        "bm25_top1_preserve_rate": sum(float(r["bm25_top1_preserved"]) for r in public) / n,
        "bm25_top2_preserve_rate": sum(float(r["bm25_top2_preserved"]) for r in public) / n,
        "bm25_top3_preserve_rate": sum(float(r["bm25_top3_preserved"]) for r in public) / n,
        "hard_rule_violations": len(hard_violations),
    }
    write_json(REPAIR / "outputs/action_table_300/action_table_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
