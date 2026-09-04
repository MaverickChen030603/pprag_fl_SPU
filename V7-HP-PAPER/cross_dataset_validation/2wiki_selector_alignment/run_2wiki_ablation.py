#!/usr/bin/env python3
from selector_alignment_common import *


def main() -> None:
    ensure_dirs()
    summary = read_json(ALIGN / "outputs/selector_smoke_300/summary.json")
    methods = summary.get("methods", {})
    target = methods.get("2wiki_v23_crossfit_selector", {})
    payload = {
        "status": "complete",
        "baseline": "bm25_or_lexical_routing",
        "answer_neutral_effect": {
            "with_safety_answer_f1_delta_vs_bm25": target.get("answer_f1_delta_vs_bm25", 0),
            "no_safety_answer_f1_delta_vs_bm25": methods.get("no_safety_predictor", {}).get("answer_f1_delta_vs_bm25", 0),
            "with_safety_joint_f1_delta_vs_bm25": target.get("joint_f1_delta_vs_bm25", 0),
            "no_safety_joint_f1_delta_vs_bm25": methods.get("no_safety_predictor", {}).get("joint_f1_delta_vs_bm25", 0),
        },
        "support_feature_effect": {
            "with_support_evidence_f1_delta_vs_bm25": target.get("evidence_f1_delta_vs_bm25", 0),
            "no_support_evidence_f1_delta_vs_bm25": methods.get("no_support_features", {}).get("evidence_f1_delta_vs_bm25", 0),
            "with_support_joint_f1_delta_vs_bm25": target.get("joint_f1_delta_vs_bm25", 0),
            "no_support_joint_f1_delta_vs_bm25": methods.get("no_support_features", {}).get("joint_f1_delta_vs_bm25", 0),
        },
    }
    write_json(ALIGN / "outputs/ablation/ablation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
