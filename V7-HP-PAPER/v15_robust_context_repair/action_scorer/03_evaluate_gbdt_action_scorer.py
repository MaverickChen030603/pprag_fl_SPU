#!/usr/bin/env python3
"""Evaluate GBDT direct scorer with the same V15 metric contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from scorer_common import TARGET_NAMES, label_tensor, read_jsonl, validate_feature_names, vectorize


def ranking(rows, truth, score):
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row["query_id"]), []).append(index)
    correct = pairs = 0
    top = []
    for indices in groups.values():
        top.append(float(truth[indices[int(np.argmax(score[indices]))]]))
        for offset, left in enumerate(indices):
            for right in indices[offset + 1:]:
                delta = truth[left] - truth[right]
                if abs(delta) > 1e-9:
                    pairs += 1
                    correct += int((score[left] - score[right]) * delta > 0)
    return correct / pairs if pairs else float("nan"), float(np.mean(top))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.data)
    checkpoint = joblib.load(args.checkpoint)
    features = validate_feature_names(checkpoint["feature_names"])
    x = vectorize(rows, features)
    y = label_tensor(rows, checkpoint["readers"])
    output = []
    for reader_index, reader in enumerate(checkpoint["readers"]):
        for target_index, target in enumerate(TARGET_NAMES):
            model = checkpoint["models"][(reader, target)]
            truth = y[:, reader_index, target_index]
            if target_index < 3:
                score = model.predict(x)
                pairwise, top = ranking(rows, truth, score)
                output.append({"reader": reader, "target": target, "spearman": float(spearmanr(truth, score).statistic), "kendall_tau": float(kendalltau(truth, score).statistic), "pairwise_accuracy": pairwise, "top_action_realized_delta": top})
            else:
                probability = model.predict_proba(x)[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(x))
                output.append({"reader": reader, "target": target, "brier": brier_score_loss(truth, probability), "auroc": roc_auc_score(truth, probability) if len(set(truth)) > 1 else float("nan"), "auprc": average_precision_score(truth, probability) if len(set(truth)) > 1 else float("nan")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in output for key in row})
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(output)
    print(json.dumps({"status": "complete", "rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

