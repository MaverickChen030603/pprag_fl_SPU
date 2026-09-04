#!/usr/bin/env python3
"""Audit whether one action transfers across all frozen readers.

This diagnostic never feeds reader outcomes into inference. Gold outcomes are used
only after an action has been selected, or to compute an explicitly labelled
oracle upper bound.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "action_scorer"))

from model import DirectMultiReaderScorer  # noqa: E402
from scorer_common import label_tensor, read_jsonl, validate_feature_names, vectorize  # noqa: E402


def robust_score(values: np.ndarray, beta: float) -> np.ndarray:
    """Mean-reader utility penalized by cross-reader standard deviation."""
    return values.mean(axis=-1) - beta * values.std(axis=-1)


def summarize_selection(
    rows: list[dict],
    groups: dict[str, list[int]],
    actual_joint: np.ndarray,
    selected: dict[str, int],
    readers: list[str],
) -> dict:
    chosen = np.asarray([selected[query_id] for query_id in sorted(groups)])
    values = actual_joint[chosen]
    is_baseline = np.asarray([float(rows[index]["features"].get("is_baseline", 0.0)) for index in chosen])
    per_reader = {
        reader: {
            "mean_joint_delta": float(values[:, reader_index].mean()),
            "positive_rate": float((values[:, reader_index] > 1e-9).mean()),
            "harm_rate": float((values[:, reader_index] < -1e-9).mean()),
        }
        for reader_index, reader in enumerate(readers)
    }
    mean_delta = values.mean(axis=1)
    min_delta = values.min(axis=1)
    return {
        "queries": len(chosen),
        "intervention_rate": float((is_baseline < 0.5).mean()),
        "mean_reader_joint_delta": float(mean_delta.mean()),
        "minimum_reader_joint_delta": float(min_delta.mean()),
        "both_readers_positive_rate": float((values > 1e-9).all(axis=1).mean()),
        "any_reader_harm_rate": float((values < -1e-9).any(axis=1).mean()),
        "per_reader": per_reader,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# V15 Cross-Reader Robust Utility Diagnostic",
        "",
        "> Reader labels are used only for retrospective evaluation and the explicitly labelled oracle. Learned selections use inference-safe features only.",
        "",
        "## Reader Agreement",
        "",
        f"- Joint-delta Spearman: {report['reader_agreement']['spearman']:.4f}",
        f"- Joint-delta Pearson: {report['reader_agreement']['pearson']:.4f}",
        f"- Non-zero sign disagreement: {report['reader_agreement']['sign_disagreement_rate']:.4f}",
        "",
        "## Robust Selection",
        "",
        "| Selector | Intervention | Mean-reader Joint delta | Min-reader Joint delta | Both positive | Any harm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["selections"].items():
        lines.append(
            f"| {name} | {values['intervention_rate']:.4f} | {values['mean_reader_joint_delta']:+.4f} | "
            f"{values['minimum_reader_joint_delta']:+.4f} | {values['both_readers_positive_rate']:.4f} | "
            f"{values['any_reader_harm_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The oracle rows measure action-set opportunity, not deployable performance. The learned rows are the valid checkpoint test: a positive robust delta must be realized by one label-free action across both readers.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = read_jsonl(args.data)
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # torch<2.0 does not expose weights_only.
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    readers = list(checkpoint["readers"])
    if len(readers) < 2:
        raise ValueError("robust utility diagnostic requires at least two frozen readers")
    feature_names = validate_feature_names(checkpoint["feature_names"])
    x = (vectorize(rows, feature_names) - checkpoint["mean"]) / checkpoint["std"]
    labels = label_tensor(rows, readers)

    model = DirectMultiReaderScorer(len(feature_names), len(readers), checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()
    with torch.no_grad():
        predictions = model(torch.as_tensor(x, device=args.device)).cpu().numpy()

    actual_joint = labels[:, :, 2]
    predicted_joint = predictions[:, :, 2]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["query_id"])].append(index)

    selections: dict[str, dict] = {}
    for beta in (0.0, 0.25, 0.5, 1.0):
        selected = {
            query_id: indices[int(np.argmax(robust_score(predicted_joint[indices], beta)))]
            for query_id, indices in groups.items()
        }
        selections[f"learned_beta_{beta:g}"] = summarize_selection(rows, groups, actual_joint, selected, readers)

        oracle = {
            query_id: indices[int(np.argmax(robust_score(actual_joint[indices], beta)))]
            for query_id, indices in groups.items()
        }
        selections[f"oracle_beta_{beta:g}"] = summarize_selection(rows, groups, actual_joint, oracle, readers)

    baseline = {
        query_id: next(index for index in indices if float(rows[index]["features"].get("is_baseline", 0.0)) > 0.5)
        for query_id, indices in groups.items()
    }
    selections["exact_fallback"] = summarize_selection(rows, groups, actual_joint, baseline, readers)

    left, right = actual_joint[:, 0], actual_joint[:, 1]
    active = (np.abs(left) > 1e-9) | (np.abs(right) > 1e-9)
    report = {
        "status": "complete",
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "queries": len(groups),
        "actions": len(rows),
        "readers": readers,
        "reader_agreement": {
            "spearman": float(spearmanr(left, right).statistic),
            "pearson": float(pearsonr(left, right).statistic),
            "sign_disagreement_rate": float(((left * right < 0) & active).sum() / max(1, active.sum())),
            "active_actions": int(active.sum()),
        },
        "selections": selections,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": "complete", "queries": len(groups), "actions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
