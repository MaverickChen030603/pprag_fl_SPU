from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "v6_hp_hyper_next"
DEFAULT_RAW = EXP_DIR / "results" / "method_identity_audit_after_ablation_raw.jsonl"
DEFAULT_SUMMARY = EXP_DIR / "results" / "method_identity_audit_after_ablation_summary.csv"
DEFAULT_REPORT = EXP_DIR / "reports" / "method_identity_audit_after_ablation_report.md"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def block_layer(block: str) -> str:
    if block == "pooler":
        return "pooler"
    parts = block.split(".")
    if len(parts) >= 3 and parts[0] == "encoder" and parts[1] == "layer":
        return f"encoder.layer.{parts[2]}"
    return block


def entropy(labels: list[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return -sum((count / total) * math.log(count / total + 1e-12) for count in counts.values())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def js_divergence(values_a: list[float], values_b: list[float], bins: int = 20) -> float | None:
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
            counts[idx] += 1
        total = sum(counts) or 1.0
        return [count / total for count in counts]

    pa = hist(values_a)
    pb = hist(values_b)
    pm = [(a + b) / 2.0 for a, b in zip(pa, pb)]

    def kl(p: list[float], q: list[float]) -> float:
        return sum(pi * math.log((pi + 1e-12) / (qi + 1e-12)) for pi, qi in zip(p, q))

    return 0.5 * kl(pa, pm) + 0.5 * kl(pb, pm)


def read_metrics(raw_csvs: list[Path]) -> dict[str, dict]:
    out = {}
    for path in raw_csvs:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                out[row.get("method", "")] = row
    return out


def summarize(score_rows: list[dict], metrics: dict[str, dict], anchor_method: str) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in score_rows:
        grouped[row.get("method", "")].append(row)

    if anchor_method not in grouped and "V6_HP_hyper_anchor" in grouped:
        anchor_method = "V6_HP_hyper_anchor"
    anchor_group = grouped.get(anchor_method, [])
    anchor_blocks = set()
    anchor_layers = set()
    anchor_scores = []
    anchor_metric = metrics.get(anchor_method, {})
    for row in anchor_group:
        blocks = set(str(item) for item in parse_list(row.get("selected_block_ids")) if item != "__ALL__")
        anchor_blocks |= blocks
        anchor_layers |= {block_layer(block) for block in blocks}
        anchor_scores.extend(float(v) for v in parse_list(row.get("selected_block_scores")) if isinstance(v, (int, float)))

    raw_rows = []
    summaries = []
    for method, group in sorted(grouped.items()):
        blocks = set()
        layers = set()
        scores = []
        entropies = []
        margins = []
        pooler_count = 0
        layer8_count = 0
        selected_total = 0
        for row in group:
            selected = [str(item) for item in parse_list(row.get("selected_block_ids")) if item != "__ALL__"]
            layer_ids = [block_layer(block) for block in selected]
            blocks |= set(selected)
            layers |= set(layer_ids)
            scores.extend(float(v) for v in parse_list(row.get("selected_block_scores")) if isinstance(v, (int, float)))
            entropies.append(float(row.get("layer_distribution_entropy") or entropy(layer_ids)))
            if row.get("score_margin_selected_vs_next") not in {None, ""}:
                margins.append(float(row["score_margin_selected_vs_next"]))
            pooler_count += int(row.get("pooler_selected_count") or selected.count("pooler"))
            layer8_count += int(row.get("encoder_layer8_selected_count") or sum(1 for block in selected if block_layer(block) == "encoder.layer.8"))
            selected_total += int(row.get("num_selected_blocks") or len(selected))
            raw_rows.append(row)
        metric = metrics.get(method, {})
        def f(name: str) -> float:
            try:
                return float(metric.get(name, ""))
            except (TypeError, ValueError):
                return float("nan")
        def delta(name: str) -> str:
            base = anchor_metric.get(name, "")
            try:
                return f"{float(metric.get(name, 'nan')) - float(base):.6f}"
            except (TypeError, ValueError):
                return ""
        summaries.append(
            {
                "method": method,
                "num_score_records": len(group),
                "selected_block_jaccard_vs_anchor": jaccard(blocks, anchor_blocks) if anchor_blocks else "",
                "selected_layer_jaccard_vs_anchor": jaccard(layers, anchor_layers) if anchor_layers else "",
                "layer_distribution_entropy": mean(entropies) if entropies else "",
                "pooler_selected_ratio": pooler_count / max(selected_total, 1),
                "encoder_layer8_selected_ratio": layer8_count / max(selected_total, 1),
                "score_distribution_js_divergence": js_divergence(scores, anchor_scores) if anchor_scores else "",
                "avg_score_margin": mean(margins) if margins else "",
                "MRR": metric.get("MRR", ""),
                "F1": metric.get("F1", ""),
                "Recall@3": metric.get("Recall@3", ""),
                "payload_ratio": metric.get("payload_ratio", ""),
                "metric_delta_mrr_vs_anchor": delta("MRR"),
                "metric_delta_f1_vs_anchor": delta("F1"),
                "payload_delta_vs_anchor": delta("payload_ratio"),
                "meaningful_selection_diversity": (
                    (jaccard(blocks, anchor_blocks) <= 0.85 if anchor_blocks else False)
                    or ((pooler_count / max(selected_total, 1)) <= 0.8 * 0.5)
                ),
            }
        )
    return raw_rows, summaries


def write_report(path: Path, summaries: list[dict]) -> None:
    lines = [
        "# Method Identity Audit After Ablation",
        "",
        "## Scope",
        "",
        f"- Audited methods: {len(summaries)}",
        "",
    ]
    if not summaries:
        lines += [
            "No score logging records were found. Run `run_scorelog_anchor_hard1000.sh` or `run_selection_diversity_ablation.sh` first.",
            "",
        ]
    else:
        lines += [
            "| method | block Jaccard | layer Jaccard | entropy | pooler ratio | layer8 ratio | JS divergence | MRR delta | F1 delta | payload delta | diverse? |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in summaries:
            lines.append(
                "| {method} | {selected_block_jaccard_vs_anchor} | {selected_layer_jaccard_vs_anchor} | "
                "{layer_distribution_entropy} | {pooler_selected_ratio:.4f} | {encoder_layer8_selected_ratio:.4f} | "
                "{score_distribution_js_divergence} | {metric_delta_mrr_vs_anchor} | {metric_delta_f1_vs_anchor} | "
                "{payload_delta_vs_anchor} | {meaningful_selection_diversity} |".format(**row)
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit method identity after V6-HP-hyper ablations.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Compatibility option; defaults are still used.")
    parser.add_argument("--score-log", action="append", type=Path, default=[EXP_DIR / "results" / "score_logging_raw.jsonl"])
    parser.add_argument("--raw-csv", action="append", type=Path, default=[EXP_DIR / "results" / "selection_diversity_ablation_raw.csv"])
    parser.add_argument("--anchor-method", default="v6_anchor_layerwise_on")
    parser.add_argument("--output-raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--report-md", type=Path, default=None)
    args = parser.parse_args()
    if args.report_md is not None:
        args.output_report = args.report_md
    score_rows = []
    for path in args.score_log:
        score_rows.extend(read_jsonl(path))
    metrics = read_metrics(args.raw_csv)
    raw_rows, summaries = summarize(score_rows, metrics, args.anchor_method)
    write_jsonl(args.output_raw, raw_rows)
    write_csv(args.output_summary, summaries)
    write_report(args.output_report, summaries)
    print(args.output_raw)
    print(args.output_summary)
    print(args.output_report)


if __name__ == "__main__":
    main()
