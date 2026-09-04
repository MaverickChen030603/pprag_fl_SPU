#!/usr/bin/env python3
"""Evaluate direct deltas, within-query ranking, and harm calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from model import DirectMultiReaderScorer
from scorer_common import TARGET_NAMES, label_tensor, read_jsonl, validate_feature_names, vectorize


def safe_metric(function, truth, score) -> float:
    try:
        return float(function(truth, score))
    except ValueError:
        return float("nan")


def ece(truth: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    for low in np.linspace(0.0, 1.0, bins, endpoint=False):
        high = low + 1.0 / bins
        mask = (probability >= low) & (probability < high if high < 1.0 else probability <= high)
        if mask.any():
            total += mask.mean() * abs(probability[mask].mean() - truth[mask].mean())
    return float(total)


def ndcg(actual: list[float], predicted: list[float]) -> float:
    if len(actual) < 2:
        return 1.0
    relevance = np.asarray(actual) - min(actual)
    order = np.argsort(-np.asarray(predicted))
    ideal = np.argsort(-relevance)
    discount = 1.0 / np.log2(np.arange(len(actual)) + 2.0)
    score = float(np.sum((2.0 ** relevance[order] - 1.0) * discount))
    best = float(np.sum((2.0 ** relevance[ideal] - 1.0) * discount))
    return score / best if best > 1e-12 else 1.0


def ranking_metrics(rows, actual, predicted) -> dict[str, float]:
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["query_id"])].append(index)
    correct = pairs = 0
    ndcgs, top_deltas = [], []
    for indices in groups.values():
        truth = [float(actual[index]) for index in indices]
        score = [float(predicted[index]) for index in indices]
        ndcgs.append(ndcg(truth, score))
        top_deltas.append(truth[int(np.argmax(score))])
        for offset, left in enumerate(indices):
            for right in indices[offset + 1:]:
                truth_diff = actual[left] - actual[right]
                if abs(truth_diff) < 1e-9:
                    continue
                pairs += 1
                correct += int((predicted[left] - predicted[right]) * truth_diff > 0)
    return {"pairwise_accuracy": correct / pairs if pairs else float("nan"), "ndcg_over_actions": float(np.mean(ndcgs)), "top_action_realized_delta": float(np.mean(top_deltas)), "queries": len(groups), "pairs": pairs}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    rows = read_jsonl(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_names = validate_feature_names(checkpoint["feature_names"])
    if feature_names != checkpoint["feature_names"]:
        raise AssertionError("checkpoint feature order is not canonical")
    x = (vectorize(rows, feature_names) - checkpoint["mean"]) / checkpoint["std"]
    y = label_tensor(rows, checkpoint["readers"])
    model = DirectMultiReaderScorer(len(feature_names), len(checkpoint["readers"]), checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()
    with torch.no_grad():
        output = model(torch.as_tensor(x, device=args.device)).cpu().numpy()
    deltas, ranking, harms = [], [], []
    for reader_index, reader in enumerate(checkpoint["readers"]):
        for metric_index, metric in enumerate(TARGET_NAMES[:3]):
            truth, score = y[:, reader_index, metric_index], output[:, reader_index, metric_index]
            rho = spearmanr(truth, score)
            tau = kendalltau(truth, score)
            deltas.append({"reader": reader, "target": metric, "spearman": float(rho.statistic), "kendall_tau": float(tau.statistic), "mae": float(np.mean(np.abs(truth - score))), "rmse": float(np.sqrt(np.mean((truth - score) ** 2)))})
            rank = ranking_metrics(rows, truth, score)
            ranking.append({"reader": reader, "target": metric, **rank})
        for target_index, target in ((3, "answer_drop"), (4, "joint_drop")):
            truth = y[:, reader_index, target_index]
            probability = 1.0 / (1.0 + np.exp(-output[:, reader_index, target_index]))
            harms.append({"reader": reader, "target": target, "brier": brier_score_loss(truth, probability), "ece": ece(truth, probability), "auroc": safe_metric(roc_auc_score, truth, probability), "auprc": safe_metric(average_precision_score, truth, probability), "prevalence": float(truth.mean())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "action_scorer_metrics.csv", deltas + harms)
    write_csv(args.output_dir / "within_query_ranking_metrics.csv", ranking)
    (args.output_dir / "delta_correlation_report.md").write_text("# Direct-Delta Correlation\n\n" + "\n".join(f"- {row['reader']} / {row['target']}: Spearman={row['spearman']:.4f}, Kendall={row['kendall_tau']:.4f}." for row in deltas) + "\n", encoding="utf-8")
    (args.output_dir / "harm_calibration_report.md").write_text("# Harm Calibration\n\n" + "\n".join(f"- {row['reader']} / {row['target']}: Brier={row['brier']:.4f}, ECE={row['ece']:.4f}, AUROC={row['auroc']:.4f}, AUPRC={row['auprc']:.4f}." for row in harms) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(rows), "readers": checkpoint["readers"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
