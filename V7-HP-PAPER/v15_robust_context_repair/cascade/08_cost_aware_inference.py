#!/usr/bin/env python3
"""Run independent cheap-gate -> robust action-scorer inference with fallback."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "action_scorer"))

from model import DirectMultiReaderScorer  # noqa: E402
from scorer_common import read_jsonl, validate_feature_names, vectorize  # noqa: E402


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def robust_score(values: np.ndarray, beta: float) -> np.ndarray:
    return values.mean(axis=1) - beta * values.std(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--cheap-gate", type=Path, required=True)
    parser.add_argument("--action-scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma-cost", type=float, default=0.0)
    parser.add_argument("--nonbaseline-cost-ms", type=float, default=1.0)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--harm-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = read_jsonl(args.actions)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["query_id"])].append(index)
    cheap = joblib.load(args.cheap_gate)
    checkpoint = load_checkpoint(args.action_scorer)
    feature_names = validate_feature_names(checkpoint["feature_names"])
    x = (vectorize(rows, feature_names) - checkpoint["mean"]) / checkpoint["std"]
    model = DirectMultiReaderScorer(len(feature_names), len(checkpoint["readers"]), checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()

    selections, scorer_ms = [], []
    invoked = 0
    for query_id in sorted(groups):
        indices = groups[query_id]
        baseline_index = next(index for index in indices if float(rows[index]["features"].get("is_baseline", 0.0)) > 0.5)
        baseline = rows[baseline_index]
        cheap_x = np.asarray([[float(baseline["features"].get(name, 0.0)) for name in cheap["feature_names"]]])
        opportunity_probability = float(cheap["model"].predict_proba(cheap_x)[0, 1])
        selected_index, reason = baseline_index, "cheap_fallback"
        predicted_utility, predicted_harm = 0.0, 0.0
        elapsed_ms = 0.0
        if opportunity_probability >= float(cheap["threshold"]):
            invoked += 1
            started = time.perf_counter()
            with torch.no_grad():
                output = model(torch.as_tensor(x[indices], device=args.device)).cpu().numpy()
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            scorer_ms.append(elapsed_ms)
            utility = robust_score(output[:, :, 2], args.beta)
            harm = 1.0 / (1.0 + np.exp(-output[:, :, 3:5]))
            worst_harm = harm.max(axis=(1, 2))
            costs = np.asarray([
                0.0 if float(rows[index]["features"].get("is_baseline", 0.0)) > 0.5 else args.nonbaseline_cost_ms
                for index in indices
            ])
            objective = utility - args.gamma_cost * costs
            feasible = np.where((utility > args.utility_threshold) & (worst_harm < args.harm_threshold))[0]
            if feasible.size:
                local = int(feasible[np.argmax(objective[feasible])])
                selected_index = indices[local]
                predicted_utility = float(utility[local])
                predicted_harm = float(worst_harm[local])
                reason = "robust_action"
            else:
                reason = "risk_fallback"
        selections.append(
            {
                "query_id": query_id,
                "action_id": rows[selected_index]["action_id"],
                "is_baseline": selected_index == baseline_index,
                "decision": reason,
                "opportunity_probability": opportunity_probability,
                "predicted_robust_utility": predicted_utility,
                "predicted_worst_harm": predicted_harm,
                "action_scorer_ms": elapsed_ms,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in selections), encoding="utf-8")
    summary = {
        "status": "complete",
        "queries": len(selections),
        "expensive_stage_invocation_rate": invoked / max(1, len(selections)),
        "intervention_rate": sum(not row["is_baseline"] for row in selections) / max(1, len(selections)),
        "mean_action_scorer_ms_when_invoked": float(np.mean(scorer_ms)) if scorer_ms else 0.0,
        "p95_action_scorer_ms_when_invoked": float(np.percentile(scorer_ms, 95)) if scorer_ms else 0.0,
        "beta": args.beta,
        "gamma_cost": args.gamma_cost,
        "utility_threshold": args.utility_threshold,
        "harm_threshold": args.harm_threshold,
        "reader_labels_used_at_inference": False,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

