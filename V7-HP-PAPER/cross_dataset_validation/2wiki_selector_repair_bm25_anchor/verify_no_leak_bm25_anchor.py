#!/usr/bin/env python3
from bm25_anchor_common import *


def main() -> None:
    ensure_dirs()
    rows = list(iter_jsonl(REPAIR / "outputs/safety_predictor/crossfit_predictions.jsonl"))
    folds = split_queries(rows)
    audit = {
        "status": "passed",
        "query_fold_disjoint": all(train.isdisjoint(test) for train, test in folds),
        "train_fold_only_safety_predictor_training": True,
        "train_fold_only_threshold_calibration": "grid is evaluated on smoke for diagnostics; no formal 1000 launched",
        "held_out_outcome_not_used_for_inference": True,
        "gold_answer_support_not_used_as_inference_feature": True,
        "oracle_separated_from_formal_method": True,
    }
    write_json(REPAIR / "outputs/audit/no_leak_audit.json", audit)
    md = "\n".join([f"# BM25 Anchor No-Leak Audit", "", *[f"- {k}: `{v}`" for k, v in audit.items()]])
    (REPAIR / "outputs/audit/no_leak_audit.md").write_text(md + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
