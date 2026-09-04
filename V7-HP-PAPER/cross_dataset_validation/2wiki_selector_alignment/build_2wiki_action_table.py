#!/usr/bin/env python3
from selector_alignment_common import *


def main() -> None:
    ensure_dirs()
    examples = load_dev_sample(300, SEED)
    actions = build_actions_for_examples(examples)
    for r in actions:
        r["_supporting_titles"] = sorted(support_titles(next(ex for ex in examples if query_id(ex) == r["query_id"])))
    public = strip_private(actions)
    write_jsonl(ALIGN / "outputs/action_table_300/2wiki_action_table_300.jsonl", public)
    n = max(1, len(public))
    qids = {r["query_id"] for r in public}
    summary = {
        "status": "complete",
        "split": "dev",
        "seed": SEED,
        "num_queries": len(qids),
        "num_actions": len(public),
        "actions_per_query": len(public) / max(1, len(qids)),
        "effective_action_rate": sum(float(r["effective_context_changed"]) for r in public) / n,
        "candidate_family_distribution": dict(Counter(r["candidate_family"] for r in public)),
        "candidate_name_distribution": dict(Counter(r["candidate_name"] for r in public)),
        "avg_added_docs": sum(float(r["num_added_docs"]) for r in public) / n,
        "avg_removed_docs": sum(float(r["num_removed_docs"]) for r in public) / n,
        "prefix2_preserve_rate": sum(float(r["prefix2_preserved"]) for r in public) / n,
        "prefix3_preserve_rate": sum(float(r["prefix3_preserved"]) for r in public) / n,
        "dense_feature_available": False,
        "note": "Features use query/document lexical, BM25, title bridge, support proxy, and prefix preservation only. No gold answer/support labels are used as inference features.",
    }
    write_json(ALIGN / "outputs/action_table_300/action_table_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
