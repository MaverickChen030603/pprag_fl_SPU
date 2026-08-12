#!/usr/bin/env python3
"""Stage-C evaluator; refuses gold access until Stage B is complete and frozen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


EXPECTED_QUERIES = 4200
EXPECTED_PREDICTIONS = 8400
METHODS = ("federated_baseline", "logistic_proberoute")
SEED = 20260812
BOOTSTRAPS = 5000


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(delta: np.ndarray):
    rng = np.random.default_rng(SEED)
    means = np.empty(BOOTSTRAPS, dtype=np.float64)
    for start in range(0, BOOTSTRAPS, 100):
        size = min(100, BOOTSTRAPS - start)
        means[start : start + size] = delta[rng.integers(0, len(delta), size=(size, len(delta)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    left = (1 + int(np.sum(means <= 0))) / (BOOTSTRAPS + 1)
    right = (1 + int(np.sum(means >= 0))) / (BOOTSTRAPS + 1)
    return float(delta.mean()), float(low), float(high), min(1.0, 2 * min(left, right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--eval-common", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    expected_queries = 4 if args.synthetic else EXPECTED_QUERIES
    expected_predictions = expected_queries * len(METHODS)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("evaluation output directory is not empty")
    if not args.completion.is_file():
        raise PermissionError("Stage B completion marker missing; gold access denied")
    complete = json.loads(args.completion.read_text())
    if complete.get("status") != "stage_b_predictions_complete_unscored":
        raise PermissionError("invalid Stage B completion state")
    if sha256(args.predictions) != complete.get("output_sha256"):
        raise PermissionError("prediction checksum mismatch; gold access denied")
    predictions = list(rows(args.predictions))
    keys = [(str(row["query_id"]), str(row["method"])) for row in predictions]
    if len(predictions) != expected_predictions or len(set(keys)) != expected_predictions:
        raise PermissionError("predictions incomplete or duplicated; gold access denied")
    selected = {query_id for query_id, _ in keys}
    if len(selected) != expected_queries:
        raise PermissionError("query cardinality mismatch; gold access denied")

    import sys

    sys.path.insert(0, str(args.eval_common.parent))
    from eval_common import official_metrics

    gold = {}
    for row in rows(args.gold):
        query_id = str(row.get("query_id", row.get("_id", row.get("id")))).strip().lower()
        if query_id in selected:
            gold[query_id] = row
    if set(gold) != selected:
        raise ValueError("sealed gold does not cover exactly the frozen split")
    per_query = []
    for row in predictions:
        query_id = str(row["query_id"])
        metric = official_metrics(
            str(row["predicted_answer"]),
            gold[query_id],
            {tuple(value) for value in row["predicted_support"]},
            "hotpotqa",
        )
        per_query.append({"query_id": query_id, "method": row["method"], **metric})
    by_method = {method: {row["query_id"]: row for row in per_query if row["method"] == method} for method in METHODS}
    order = sorted(selected)
    metrics = ("answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1")
    comparisons = {}
    for metric in metrics:
        delta = np.asarray(
            [by_method["logistic_proberoute"][query_id][metric] - by_method["federated_baseline"][query_id][metric] for query_id in order],
            dtype=np.float64,
        )
        mean, low, high, p = bootstrap(delta)
        comparisons[metric] = {
            "mean_delta": mean,
            "ci_low": low,
            "ci_high": high,
            "two_sided_p_plus_one": p,
            "win": int(np.sum(delta > 0)),
            "tie": int(np.sum(delta == 0)),
            "loss": int(np.sum(delta < 0)),
        }
    primary = comparisons["joint_f1"]
    if primary["ci_high"] < 0:
        decision = "R5_C1_HARM"
    elif primary["mean_delta"] > 0 and primary["ci_low"] > 0:
        decision = "R5_C1_CONFIRMED"
    elif primary["mean_delta"] > 0 and primary["ci_low"] <= 0 <= primary["ci_high"]:
        decision = "R5_C1_POSITIVE_INCONCLUSIVE"
    else:
        decision = "R5_C1_NULL_OR_MIXED"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "decision": decision,
        "primary": "logistic_proberoute minus federated_baseline query-level Joint F1",
        "comparisons": comparisons,
        "bootstrap_repetitions": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "predictions_sha256": sha256(args.predictions),
        "answer_tradeoff_must_be_reported": comparisons["answer_f1"]["mean_delta"] < 0,
        "synthetic": args.synthetic,
    }
    temp = args.output_dir / "r5_c1_results.tmp"
    final = args.output_dir / "r5_c1_results.json"
    temp.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    os.replace(temp, final)


if __name__ == "__main__":
    main()
