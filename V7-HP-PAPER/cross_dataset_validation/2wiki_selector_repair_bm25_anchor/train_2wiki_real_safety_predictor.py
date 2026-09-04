#!/usr/bin/env python3
from bm25_anchor_common import *


def auc(rows: list[tuple[float, int]]) -> float:
    pos = [s for s, y in rows if y == 1]
    neg = [s for s, y in rows if y == 0]
    if not pos or not neg:
        return 0.0
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def main() -> None:
    ensure_dirs()
    actions = build_anchor_actions()
    rows = attach_metrics(actions)
    preds = build_crossfit_predictions(rows)
    write_jsonl(REPAIR / "outputs/safety_predictor/crossfit_predictions.jsonl", base.strip_private(preds))
    safe_pairs = [(float(r["safe_answer_prob"]), int(r["answer_safe"])) for r in preds]
    pos_pairs = [(float(r["positive_action_prob"]), int(r["paper_positive_vs_bm25"])) for r in preds]
    false_safe = [r for r in preds if r["safe_answer_prob"] >= 0.5 and not r["answer_safe"]]
    false_neg = [r for r in preds if r["safe_answer_prob"] < 0.5 and r["answer_safe"]]
    summary = {
        "status": "complete",
        "num_actions": len(preds),
        "num_queries": len({r["query_id"] for r in preds}),
        "answer_safe_auc": auc(safe_pairs),
        "paper_positive_auc": auc(pos_pairs),
        "false_safe_rate": len(false_safe) / max(1, len(preds)),
        "false_negative_rate": len(false_neg) / max(1, len(preds)),
        "calibration_summary": {
            "safe_mean": sum(r["safe_answer_prob"] for r in preds) / max(1, len(preds)),
            "positive_mean": sum(r["positive_action_prob"] for r in preds) / max(1, len(preds)),
            "answer_safe_rate": sum(r["answer_safe"] for r in preds) / max(1, len(preds)),
            "paper_positive_rate": sum(r["paper_positive_vs_bm25"] for r in preds) / max(1, len(preds)),
        },
        "stability_note": "2Wiki dev-300 is small; safety predictor is cross-fitted but should be treated as smoke-level calibration.",
    }
    write_json(REPAIR / "outputs/safety_predictor/safety_predictor_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
