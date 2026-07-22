#!/usr/bin/env python3
"""Exploratory V15 lower-bound probe for V16-reachable context synergy.

This reclassifies previously reader-evaluated V15 contexts by their minimum
number of V16 atomic edits from the baseline. It does not contain the complete
single-edit action set and therefore cannot support a V16 Go/No-Go decision.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from oracle_search.oracle_landscape import aggregate, summarize_query


def permutation_distance(source: tuple[str, ...], target: tuple[str, ...]) -> int:
    """Minimum of unrestricted SWAP distance and MOVE distance."""
    if set(source) != set(target):
        raise ValueError("permutation distance requires equal sets")
    target_index = {value: index for index, value in enumerate(target)}
    permutation = [target_index[value] for value in source]
    cycles, seen = 0, set()
    for start in range(len(permutation)):
        if start in seen:
            continue
        cycles += 1
        current = start
        while current not in seen:
            seen.add(current)
            current = permutation[current]
    swap_distance = len(source) - cycles
    # Minimum arbitrary moves equals n minus longest common subsequence.
    dp = [[0] * (len(target) + 1) for _ in range(len(source) + 1)]
    for i, left in enumerate(source, 1):
        for j, right in enumerate(target, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if left == right else max(dp[i - 1][j], dp[i][j - 1])
    move_distance = len(source) - dp[-1][-1]
    return min(swap_distance, move_distance)


def minimum_atomic_depth(baseline: Iterable[str], target: Iterable[str], max_depth: int = 3) -> int | None:
    baseline_tuple, target_tuple = tuple(baseline), tuple(target)
    if baseline_tuple == target_tuple:
        return 0
    removed = tuple(value for value in baseline_tuple if value not in target_tuple)
    added = tuple(value for value in target_tuple if value not in baseline_tuple)
    if len(removed) != len(added) or len(removed) > max_depth:
        return None
    best = max_depth + 1
    if not removed:
        best = permutation_distance(baseline_tuple, target_tuple)
    else:
        # Every shortest membership repair removes each baseline-only document
        # and adds each target-only document exactly once. Try REPLACE and
        # DROP_ADD variants; ordering is repaired with SWAP/MOVE afterwards.
        for add_order in itertools.permutations(added):
            for mode_bits in itertools.product(("replace", "drop_add"), repeat=len(removed)):
                values = list(baseline_tuple)
                valid = True
                for old_doc, new_doc, mode in zip(removed, add_order, mode_bits):
                    if old_doc not in values:
                        valid = False
                        break
                    index = values.index(old_doc)
                    if mode == "replace":
                        values[index] = new_doc
                    else:
                        values.pop(index)
                        values.append(new_doc)
                if valid:
                    best = min(best, len(removed) + permutation_distance(tuple(values), target_tuple))
    return best if best <= max_depth else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flan", type=Path, required=True)
    parser.add_argument("--unifiedqa", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "v15_exploratory_probe")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    transformed: list[dict[str, Any]] = []
    unreachable = 0
    for path, reader in ((args.flan, "flan"), (args.unifiedqa, "unifiedqa")):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(path):
            grouped[str(row["query_id"])].append(row)
        for query_id, rows in grouped.items():
            baseline = next(row for row in rows if row.get("is_baseline"))
            baseline_docs = baseline["context_doc_ids"]
            for row in rows:
                depth = minimum_atomic_depth(baseline_docs, row["context_doc_ids"])
                if depth is None:
                    unreachable += 1
                    continue
                transformed.append({
                    **row,
                    "dataset": "hotpotqa_v15_exploratory",
                    "reader": reader,
                    "depth": depth,
                    "trajectory_id": row["action_id"],
                })
    by_query: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in transformed:
        by_query[(str(row["reader"]), str(row["query_id"]))].append(row)
    per_query = [summarize_query(rows) for rows in by_query.values()]
    summary = aggregate(per_query, args.bootstrap_samples, 20260722)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "v15_exploratory_per_query.csv", per_query)
    write_csv(args.output_dir / "v15_exploratory_synergy_summary.csv", summary)
    lines = [
        "# V15 Exploratory Atomic-Reachability Synergy Probe",
        "",
        "**Protocol warning:** these are 100 previously inspected HotpotQA development queries and only 16 reader-evaluated V15 contexts per query. The legal single-edit denominator is incomplete, so strict synergy can be overestimated. This is a prioritization diagnostic, not V16 evidence.",
        "",
        "| Reader | N | Single positive | Composed positive | Composition-only | Mean observed StrictSyn Joint | 95% bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["metric"] == "joint":
            lines.append(f"| {row['reader']} | {row['queries']} | {row['positive_single_rate']:.3f} | {row['positive_composed_rate']:.3f} | {row['composition_only_positive_rate']:.3f} | {row['mean_strict_synergy']:+.4f} | [{row['strict_synergy_ci_low']:+.4f}, {row['strict_synergy_ci_high']:+.4f}] |")
    lines += ["", f"Contexts not reachable within three V16 edits: {unreachable}.", "", "Decision use: a positive result accelerates Phase-A oracle labeling; a negative result does not stop V16 because V15 did not expose the complete V16 trajectory space.", ""]
    (args.output_dir / "v15_exploratory_synergy_probe.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "per_query": len(per_query), "unreachable": unreachable, "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
