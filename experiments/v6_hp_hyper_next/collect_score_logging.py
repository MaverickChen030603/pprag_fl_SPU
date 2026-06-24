from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "v6_hp_hyper_next"
DEFAULT_RAW = EXP_DIR / "results" / "score_logging_raw.jsonl"
DEFAULT_SUMMARY = EXP_DIR / "results" / "score_logging_summary.csv"
DEFAULT_REPORT = EXP_DIR / "reports" / "score_logging_report.md"


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _as_float(value) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _block_set(row: dict) -> set[str]:
    return set(str(item) for item in _parse_list(row.get("selected_block_ids")) if item != "__ALL__")


def _normalize_subset(value: str) -> str:
    text = str(value or "")
    for name in ["hotpot_hard_1000", "hotpot_all_1000", "hotpot_hard_500", "hotpot_all_50", "hotpot_hard_50"]:
        if name in text:
            return name
    return text


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def _js_divergence(values_a: list[float], values_b: list[float], bins: int = 20) -> float | None:
    if not values_a or not values_b:
        return None
    lo = min(values_a + values_b)
    hi = max(values_a + values_b)
    if hi <= lo:
        return 0.0
    width = (hi - lo) / bins

    def hist(values: list[float]) -> list[float]:
        counts = [0.0] * bins
        for value in values:
            idx = min(int((value - lo) / width), bins - 1)
            counts[idx] += 1.0
        total = sum(counts) or 1.0
        return [count / total for count in counts]

    pa = hist(values_a)
    pb = hist(values_b)
    pm = [(a + b) / 2.0 for a, b in zip(pa, pb)]

    def kl(p: list[float], q: list[float]) -> float:
        return sum(pi * math.log((pi + 1e-12) / (qi + 1e-12)) for pi, qi in zip(p, q))

    return 0.5 * kl(pa, pm) + 0.5 * kl(pb, pm)


def discover_score_logs(raw_csvs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for csv_path in raw_csvs:
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                run_dir = Path(row.get("run_dir", ""))
                score_path = run_dir / "score_logging_raw.jsonl"
                if score_path.exists():
                    paths.append(score_path)
    return sorted(set(paths))


def summarize(rows: list[dict], anchor_method: str) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("method", ""), row.get("seed", ""), _normalize_subset(row.get("subset", "")))].append(row)

    anchor_blocks_by_key: dict[tuple, set[str]] = {}
    anchor_scores_by_key: dict[tuple, list[float]] = {}
    for (method, seed, subset), group in grouped.items():
        if method == anchor_method:
            key = (seed, subset)
            blocks: set[str] = set()
            scores: list[float] = []
            for row in group:
                blocks |= _block_set(row)
                scores.extend(float(v) for v in _parse_list(row.get("selected_block_scores")) if isinstance(v, (int, float)))
            anchor_blocks_by_key[key] = blocks
            anchor_scores_by_key[key] = scores

    summaries: list[dict] = []
    for (method, seed, subset), group in sorted(grouped.items()):
        all_blocks: set[str] = set()
        selected_scores: list[float] = []
        margins = []
        entropies = []
        pooler_counts = 0
        layer8_counts = 0
        selected_total = 0
        for row in group:
            blocks = _block_set(row)
            all_blocks |= blocks
            selected_total += int(row.get("num_selected_blocks") or len(blocks))
            pooler_counts += int(row.get("pooler_selected_count") or 0)
            layer8_counts += int(row.get("encoder_layer8_selected_count") or 0)
            score_values = _parse_list(row.get("selected_block_scores"))
            selected_scores.extend(float(v) for v in score_values if isinstance(v, (int, float)))
            margin = _as_float(row.get("score_margin_selected_vs_next"))
            entropy = _as_float(row.get("layer_distribution_entropy"))
            if margin is not None:
                margins.append(margin)
            if entropy is not None:
                entropies.append(entropy)
        anchor_key = (seed, subset)
        anchor_blocks = anchor_blocks_by_key.get(anchor_key, set())
        anchor_scores = anchor_scores_by_key.get(anchor_key, [])
        summaries.append(
            {
                "method": method,
                "seed": seed,
                "subset": subset,
                "num_score_records": len(group),
                "unique_selected_blocks": ";".join(sorted(all_blocks)),
                "selected_block_count": len(all_blocks),
                "selected_block_jaccard_vs_anchor": _jaccard(all_blocks, anchor_blocks) if anchor_blocks else "",
                "score_distribution_js_divergence_vs_anchor": _js_divergence(selected_scores, anchor_scores) if anchor_scores else "",
                "avg_score_margin_selected_vs_next": mean(margins) if margins else "",
                "avg_layer_distribution_entropy": mean(entropies) if entropies else "",
                "pooler_selected_ratio": pooler_counts / max(selected_total, 1),
                "encoder_layer8_selected_ratio": layer8_counts / max(selected_total, 1),
                "avg_selected_score": mean(selected_scores) if selected_scores else "",
            }
        )
    return summaries


def write_report(path: Path, rows: list[dict], summaries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Score Logging Report",
        "",
        "## Scope",
        "",
        f"- Score records: {len(rows)}",
        f"- Summary groups: {len(summaries)}",
        "",
    ]
    if not rows:
        lines += [
            "## Current Status",
            "",
            "No score-distribution records were found yet. This means existing B1/B2 runs can confirm selected-block identity, but cannot answer whether V4/V5/V6 had different internal score distributions before top-k selection.",
            "",
            "Score logging has to be enabled in a rerun before RQ1/RQ3 can be answered.",
            "",
        ]
    else:
        lines += [
            "## Summary Table",
            "",
            "| method | subset | records | jaccard vs anchor | JS divergence | avg margin | pooler ratio | layer8 ratio | entropy |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in summaries:
            lines.append(
                "| {method} | {subset} | {num_score_records} | {selected_block_jaccard_vs_anchor} | "
                "{score_distribution_js_divergence_vs_anchor} | {avg_score_margin_selected_vs_next} | "
                "{pooler_selected_ratio:.4f} | {encoder_layer8_selected_ratio:.4f} | {avg_layer_distribution_entropy} |".format(**row)
            )
        lines += [
            "",
            "## Diagnostic Answers",
            "",
            "1. Candidate score distribution difference: see JS divergence against the anchor after score-log reruns.",
            "2. Identical top blocks despite different scores would indicate a large selected-vs-next score margin or topk bottleneck.",
            "3. Large positive margins indicate selector collapse caused by dominant high-score blocks.",
            "4. Pooler and encoder.layer.8 dominance is measured by selected ratios.",
            "5. Layerwise-budget effect is measured by entropy and block-set diversity across ablations.",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect V6-HP-hyper score logging outputs.")
    parser.add_argument("--raw-csv", action="append", type=Path, default=[])
    parser.add_argument("--score-log", action="append", type=Path, default=[])
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--anchor-method", default="hypernet_v6")
    args = parser.parse_args()

    score_logs = list(args.score_log)
    score_logs.extend(discover_score_logs(args.raw_csv))
    rows: list[dict] = []
    for path in sorted(set(score_logs)):
        rows.extend(_read_jsonl(path))

    summaries = summarize(rows, args.anchor_method)
    _write_jsonl(args.output_jsonl, rows)
    _write_csv(args.output_summary, summaries)
    write_report(args.output_report, rows, summaries)
    print(args.output_jsonl)
    print(args.output_summary)
    print(args.output_report)


if __name__ == "__main__":
    main()
