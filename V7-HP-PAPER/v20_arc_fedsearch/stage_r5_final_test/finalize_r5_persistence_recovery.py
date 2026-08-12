#!/usr/bin/env python3
"""Finalize R5 from already-frozen CSVs after a JSON persistence-only failure."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def csv_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    decision_path = run / "reports/r5_final_decision.json"
    if decision_path.exists():
        raise FileExistsError(decision_path)

    required = {
        "retrieval": (run / "retrieval/final_test_retrieval_results.csv", 3600),
        "per_query": (run / "statistics/per_query_results.csv", 7200),
        "main": (run / "statistics/main_final_test_results.csv", 24),
        "bootstrap": (run / "statistics/paired_bootstrap.csv", 54),
        "consistency": (run / "statistics/r4_r5_consistency.csv", 6),
        "transitions": (run / "mechanism/support_transition_analysis.csv", 44),
    }
    hashes = {}
    for name, (path, expected) in required.items():
        count = len(csv_rows(path))
        if count != expected:
            raise ValueError(f"{name}: {count} rows, expected {expected}")
        hashes[name] = {"path": str(path), "rows": count, "sha256": sha256(path)}

    unseal = json.loads((run / "protocol/label_unseal_record.json").read_text())
    if not unseal.get("all_predictions_complete_before_unseal"):
        raise RuntimeError("original unseal gate did not pass")
    pre_unseal = json.loads((run / "checksums/pre_unseal_prediction_manifest.json").read_text())
    if pre_unseal.get("labels_opened") is not False:
        raise RuntimeError("invalid pre-unseal manifest")

    comparisons = csv_rows(required["bootstrap"][0])
    for row in comparisons:
        for key in ("mean_delta", "ci_low", "ci_high", "two_sided_p"):
            row[key] = float(row[key])
        for key in ("paired_win", "paired_tie", "paired_loss"):
            row[key] = int(row[key])
    primary = [row for row in comparisons if row["method_a"] == "logistic_proberoute" and row["method_b"] == "federated_baseline" and row["metric"] == "joint_f1"]
    if len(primary) != 6:
        raise ValueError(f"expected six primary cells, found {len(primary)}")

    by_dataset = defaultdict(list)
    for row in primary:
        by_dataset[row["dataset"]].append(row)
    positive_datasets = int(sum(np.mean([value["mean_delta"] for value in values]) > 0 for values in by_dataset.values()))
    macro_joint = float(np.mean([row["mean_delta"] for row in primary]))
    clearly_negative = any(all(row["ci_high"] < 0 for row in values) for values in by_dataset.values())

    transitions = csv_rows(required["transitions"][0])
    primary_transitions = [row for row in transitions if row["comparison"].startswith("logistic")]
    rescue = [row for row in primary_transitions if row["support_transition"] == "T1_rescue"]
    harm = [row for row in primary_transitions if row["support_transition"] == "T3_harm"]
    rescue_n = sum(int(row["n"]) for row in rescue if row["reader"] == "flan")
    harm_n = sum(int(row["n"]) for row in harm if row["reader"] == "flan")
    rescue_joint = float(np.mean([float(row["delta_joint_f1"]) for row in rescue])) if rescue else 0.0

    answer_primary = [row for row in comparisons if row["method_a"] == "logistic_proberoute" and row["method_b"] == "federated_baseline" and row["metric"] == "answer_f1"]
    systematic_answer_harm = bool(np.mean([row["mean_delta"] for row in answer_primary]) < 0 and sum(row["mean_delta"] < 0 for row in answer_primary) >= 4)
    if positive_datasets >= 2 and macro_joint > 0 and not clearly_negative and rescue_n > harm_n and rescue_joint > 0 and not systematic_answer_harm:
        decision = "final_test_strongly_confirmed"
    elif positive_datasets >= 2 and macro_joint > 0 and not clearly_negative and not systematic_answer_harm:
        decision = "final_test_partially_confirmed"
    elif macro_joint < 0 and sum(all(row["mean_delta"] < 0 for row in values) for values in by_dataset.values()) >= 2:
        decision = "final_test_contradiction"
    else:
        decision = "final_test_mixed_generalization"

    consistency = csv_rows(required["consistency"][0])
    r4_macro = float(np.mean([float(row["r4_delta_joint"]) for row in consistency]))
    final = {
        "status": decision,
        "lifecycle_status": "v20_empirical_evaluation_complete",
        "r5_macro_joint_delta": macro_joint,
        "r4_macro_joint_delta": r4_macro,
        "positive_datasets": positive_datasets,
        "support_rescue_count": int(rescue_n),
        "support_harm_count": int(harm_n),
        "support_rescue_mean_joint_delta": rescue_joint,
        "systematic_answer_harm": systematic_answer_harm,
        "final_test_kind": "V17 train-derived untouched held-out split",
        "method_development_closed": True,
        "persistence_recovery_only": True,
    }
    decision_path.write_text(json.dumps(final, indent=2) + "\n")

    lines = ["# V20 R5 One-Shot Final-Test Report", "", f"**Decision:** `{decision}`", "", f"Labels were unsealed once at `{unseal['label_unseal_timestamp']}` after all 7,200 predictions passed checksum validation.", "", "## Primary Joint F1", ""]
    for row in primary:
        lines.append(f"- {row['dataset']} / {row['reader']}: {row['mean_delta']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}], p={row['two_sided_p']:.4g}, W/T/L={row['paired_win']}/{row['paired_tie']}/{row['paired_loss']}")
    lines += ["", f"R4 macro delta: {r4_macro:+.4f}; R5 macro delta: {macro_joint:+.4f}.", f"Support rescue/harm: {rescue_n}/{harm_n}; rescue mean Joint delta: {rescue_joint:+.4f}.", "", "SP uses the shared frozen V16 support predictor and is context-level, not an independent cross-reader replication.", "", "The first evaluator invocation completed all frozen CSVs but failed while serializing a NumPy integer into the final JSON. This report was recovered exclusively from those frozen CSVs without reopening labels or recomputing metrics."]
    (run / "reports/r5_final_test_report.md").write_text("\n".join(lines) + "\n")

    claims = f"""# V20 Paper Claim Freeze

## Tier 1 - Fully Supported

- ProbeRoute improves federated resource selection under the frozen Bc=3 and 15-document budget.
- ProbeRoute improves complete multi-hop evidence access.
- ProbeRoute improves downstream Joint F1 only to the extent supported by the frozen R4/R5 results.

## Tier 2 - Qualified

- Final confirmation status: `{decision}`; R5 is a train-derived untouched held-out confirmation, not an official hidden-test result.
- Logistic is a lightweight supervised enhancement; its advantage over label-free ProbeRoute must be stated only where directly supported.

## Tier 3 - Unsupported

- ProbeRoute is always better or guarantees no harm.
- Centralized retrieval is an upper bound.
- Logistic is significantly better than label-free everywhere.
- ProbeRoute has zero extra cost or formal privacy/security guarantees.

V20 method development is permanently closed. Only paper writing, visualization, re-statistics of frozen results, and reproducibility packaging remain.
"""
    (run / "reports/paper_claim_freeze.md").write_text(claims)

    error_log = run / "logs/final_evaluation.log"
    recovery = {
        "status": "persistence_recovery_complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": "NumPy int64 was not JSON serializable after all scientific CSVs had been written",
        "labels_reopened": False,
        "metrics_recomputed": False,
        "frozen_inputs": hashes,
        "original_error_log_sha256": sha256(error_log),
        "label_unseal_record_sha256": sha256(run / "protocol/label_unseal_record.json"),
    }
    (run / "protocol/r5_persistence_recovery_audit.json").write_text(json.dumps(recovery, indent=2) + "\n")

    manifest = {}
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.name != "artifact_checksum_manifest.json":
            manifest[str(path.relative_to(run))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (run / "checksums/artifact_checksum_manifest.json").write_text(json.dumps({"status": "v20_empirical_evaluation_complete", "files": manifest}, indent=2) + "\n")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
