#!/usr/bin/env python3
from bm25_anchor_common import *


def main() -> None:
    ensure_dirs()
    summary = read_json(REPAIR / "outputs/selector_smoke_300/summary.json")
    m = summary["methods"]
    payload = {
        "status": "complete",
        "top3_preservation_necessary": "manual_check_required",
        "real_safety_predictor_vs_no_safety": {
            "answer_neutral_joint_delta": m["bm25_anchor_answer_neutral_selector"]["joint_f1_delta_vs_bm25"],
            "no_safety_joint_delta": m["no_safety_predictor"]["joint_f1_delta_vs_bm25"],
            "answer_neutral_answer_delta": m["bm25_anchor_answer_neutral_selector"]["answer_f1_delta_vs_bm25"],
            "no_safety_answer_delta": m["no_safety_predictor"]["answer_f1_delta_vs_bm25"],
        },
        "support_features_effect": {
            "answer_neutral_evidence_delta": m["bm25_anchor_answer_neutral_selector"]["evidence_f1_delta_vs_bm25"],
            "no_support_evidence_delta": m["no_support_features"]["evidence_f1_delta_vs_bm25"],
            "support_first_evidence_delta": m["bm25_anchor_support_first"]["evidence_f1_delta_vs_bm25"],
        },
        "selector_fallback_behavior": {
            "answer_neutral_fallback_rate": m["bm25_anchor_answer_neutral_selector"]["fallback_rate"],
            "answer_neutral_effective_rate": m["bm25_anchor_answer_neutral_selector"]["selected_effective_action_rate"],
        },
        "all_methods": m,
    }
    write_json(REPAIR / "outputs/ablation/ablation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
