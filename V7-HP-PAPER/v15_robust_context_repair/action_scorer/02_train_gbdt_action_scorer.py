#!/usr/bin/env python3
"""Train the preregistered simple GBDT direct-delta/harm baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from scorer_common import TARGET_NAMES, label_tensor, read_jsonl, validate_feature_names, vectorize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    rows = read_jsonl(args.train)
    feature_names = validate_feature_names(rows[0]["features"].keys())
    readers = sorted(rows[0]["reader_labels"])
    x, y = vectorize(rows, feature_names), label_tensor(rows, readers)
    models = {}
    for reader_index, reader in enumerate(readers):
        for target_index, target in enumerate(TARGET_NAMES):
            if target_index < 3:
                model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=args.seed)
            else:
                model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, class_weight="balanced", random_state=args.seed)
            model.fit(x, y[:, reader_index, target_index])
            models[(reader, target)] = model
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"feature_names": feature_names, "readers": readers, "models": models, "seed": args.seed}, args.checkpoint)
    print(json.dumps({"status": "complete", "rows": len(rows), "features": len(feature_names), "readers": readers, "checkpoint": str(args.checkpoint)}, indent=2))


if __name__ == "__main__":
    main()

