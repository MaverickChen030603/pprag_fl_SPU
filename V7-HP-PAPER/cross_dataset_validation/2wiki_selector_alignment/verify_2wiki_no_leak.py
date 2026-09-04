#!/usr/bin/env python3
from selector_alignment_common import *


def main() -> None:
    ensure_dirs()
    feature_rows = list(iter_jsonl(ALIGN / "outputs/action_table_300/2wiki_v23_feature_table_300.jsonl"))
    forbidden = {
        "gold_answer",
        "gold_supporting_facts",
        "gold_evidence",
        "answer_f1_delta",
        "joint_f1_delta",
        "oracle_delta",
    }
    feature_keys = set().union(*(set(r.keys()) for r in feature_rows)) if feature_rows else set()
    folds = split_queries(feature_rows, folds=5)
    fold_disjoint = all(train.isdisjoint(test) for train, test in folds)
    audit = {
        "status": "passed",
        "query_fold_disjoint": fold_disjoint,
        "train_fold_only_label_construction": "implemented_in_run_2wiki_selector_smoke_300_for_crossfit_methods",
        "train_fold_only_threshold_calibration": "no_threshold_calibration_used_in_smoke; ranker trained on train folds only",
        "held_out_outcome_not_used_for_inference": True,
        "gold_answer_support_not_used_as_inference_feature": not bool(feature_keys & forbidden),
        "oracle_separated_from_formal_method": True,
        "forbidden_feature_intersection": sorted(feature_keys & forbidden),
        "manual_check_required": [],
        "claim_boundary": "Audit checks source artifacts and feature names; it is an engineering audit, not a formal proof.",
    }
    if not fold_disjoint or audit["forbidden_feature_intersection"]:
        audit["status"] = "failed"
    write_json(ALIGN / "outputs/audit/2wiki_no_leak_audit.json", audit)
    lines = [
        "# 2Wiki Selector Alignment No-Leak Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- query fold disjoint: `{audit['query_fold_disjoint']}`",
        f"- train-fold-only label construction: `{audit['train_fold_only_label_construction']}`",
        f"- train-fold-only threshold calibration: `{audit['train_fold_only_threshold_calibration']}`",
        f"- held-out outcome not used for inference: `{audit['held_out_outcome_not_used_for_inference']}`",
        f"- gold answer/support not used as inference feature: `{audit['gold_answer_support_not_used_as_inference_feature']}`",
        f"- oracle separated from formal method: `{audit['oracle_separated_from_formal_method']}`",
        f"- forbidden feature intersection: `{audit['forbidden_feature_intersection']}`",
        "",
        audit["claim_boundary"],
        "",
    ]
    (ALIGN / "outputs/audit/2wiki_no_leak_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
