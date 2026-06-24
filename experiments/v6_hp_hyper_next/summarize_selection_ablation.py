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


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def as_float(row: dict, key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def aggregate_score_logs(score_rows: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in score_rows:
        method = str(row.get("subset", "")).split("_hotpot_")[0]
        if not method:
            method = str(row.get("method", ""))
        grouped[method].append(row)
    out = {}
    for method, rows in grouped.items():
        blocks = set()
        layers = set()
        margins = []
        entropies = []
        pooler_count = 0
        layer8_count = 0
        selected_total = 0
        for row in rows:
            selected = [str(item) for item in parse_list(row.get("selected_block_ids")) if item != "__ALL__"]
            layer_ids = [block_layer(block) for block in selected]
            blocks |= set(selected)
            layers |= set(layer_ids)
            selected_total += int(row.get("num_selected_blocks") or len(selected))
            pooler_count += int(row.get("pooler_selected_count") or selected.count("pooler"))
            layer8_count += int(row.get("encoder_layer8_selected_count") or sum(1 for block in selected if block_layer(block) == "encoder.layer.8"))
            if row.get("score_margin_selected_vs_next") not in {None, ""}:
                margins.append(float(row["score_margin_selected_vs_next"]))
            if row.get("layer_distribution_entropy") not in {None, ""}:
                entropies.append(float(row["layer_distribution_entropy"]))
        out[method] = {
            "selected_blocks": blocks,
            "selected_layers": layers,
            "selected_block_signature": ";".join(sorted(blocks)),
            "selected_layer_signature": ";".join(sorted(layers)),
            "pooler_selected_ratio": pooler_count / max(selected_total, 1),
            "encoder_layer8_selected_ratio": layer8_count / max(selected_total, 1),
            "layer_distribution_entropy": mean(entropies) if entropies else "",
            "score_margin_selected_vs_next": mean(margins) if margins else "",
        }
    return out


def build_summary(metric_rows: list[dict], score_info: dict[str, dict], anchor_method: str) -> list[dict]:
    anchor_metrics = next((row for row in metric_rows if row.get("method") == anchor_method), metric_rows[0] if metric_rows else {})
    anchor_score = score_info.get(anchor_method, {})
    anchor_blocks = set(anchor_score.get("selected_blocks", set()))
    anchor_layers = set(anchor_score.get("selected_layers", set()))
    rows = []
    for metric in metric_rows:
        method = metric.get("method", "")
        score = score_info.get(method, {})
        blocks = set(score.get("selected_blocks", set()))
        layers = set(score.get("selected_layers", set()))
        mrr = as_float(metric, "MRR")
        f1 = as_float(metric, "F1")
        payload = as_float(metric, "payload_ratio")
        anchor_mrr = as_float(anchor_metrics, "MRR")
        anchor_f1 = as_float(anchor_metrics, "F1")
        anchor_payload = as_float(anchor_metrics, "payload_ratio")
        block_j = jaccard(blocks, anchor_blocks) if anchor_blocks else ""
        pooler_ratio = score.get("pooler_selected_ratio", "")
        entropy = score.get("layer_distribution_entropy", "")
        anchor_pooler = anchor_score.get("pooler_selected_ratio", "")
        anchor_entropy = anchor_score.get("layer_distribution_entropy", "")
        pooler_drop = ""
        entropy_gain = ""
        if isinstance(pooler_ratio, float) and isinstance(anchor_pooler, float):
            pooler_drop = anchor_pooler - pooler_ratio
        if isinstance(entropy, float) and isinstance(anchor_entropy, float) and anchor_entropy:
            entropy_gain = (entropy - anchor_entropy) / anchor_entropy
        metric_delta_mrr = (mrr - anchor_mrr) if mrr is not None and anchor_mrr is not None else ""
        metric_delta_f1 = (f1 - anchor_f1) if f1 is not None and anchor_f1 is not None else ""
        payload_delta = (payload - anchor_payload) if payload is not None and anchor_payload is not None else ""
        meaningful_diversity = (
            (isinstance(block_j, float) and block_j <= 0.85)
            or (isinstance(pooler_drop, float) and pooler_drop >= 0.20 * max(float(anchor_pooler), 1e-12))
            or (isinstance(entropy_gain, float) and entropy_gain >= 0.15)
        )
        rows.append(
            {
                "method": method,
                "run_id": metric.get("run_dir", ""),
                "seed": metric.get("seed", ""),
                "subset": metric.get("subset", ""),
                "payload_ratio": metric.get("payload_ratio", ""),
                "MRR": metric.get("MRR", ""),
                "NDCG": metric.get("NDCG", ""),
                "F1": metric.get("F1", ""),
                "EM": metric.get("EM", ""),
                "Recall@3": metric.get("Recall@3", ""),
                "Hit@10": metric.get("Hit@10", ""),
                "selected_block_signature": score.get("selected_block_signature", ""),
                "selected_layer_signature": score.get("selected_layer_signature", ""),
                "selected_block_jaccard_vs_anchor": block_j,
                "selected_layer_jaccard_vs_anchor": jaccard(layers, anchor_layers) if anchor_layers else "",
                "pooler_selected_ratio": pooler_ratio,
                "encoder_layer8_selected_ratio": score.get("encoder_layer8_selected_ratio", ""),
                "layer_distribution_entropy": entropy,
                "score_margin_selected_vs_next": score.get("score_margin_selected_vs_next", ""),
                "metric_delta_mrr_vs_anchor": metric_delta_mrr,
                "metric_delta_f1_vs_anchor": metric_delta_f1,
                "payload_delta_vs_anchor": payload_delta,
                "meaningful_selection_diversity": meaningful_diversity,
                "metric_acceptable": (
                    (not isinstance(metric_delta_mrr, float) or metric_delta_mrr >= -0.003)
                    and (not isinstance(metric_delta_f1, float) or metric_delta_f1 >= -0.003)
                ),
                "metric_promising": (
                    (isinstance(metric_delta_mrr, float) and metric_delta_mrr >= 0.005)
                    or (isinstance(metric_delta_f1, float) and metric_delta_f1 >= 0.005)
                ),
            }
        )
    return rows


def write_report(path: Path, title: str, summary_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- Methods: {len(summary_rows)}",
        "",
    ]
    if summary_rows:
        lines += [
            "| method | MRR | F1 | payload | block Jaccard | pooler ratio | layer8 ratio | entropy | MRR delta | F1 delta | diverse? | acceptable? |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for row in summary_rows:
            lines.append(
                "| {method} | {MRR} | {F1} | {payload_ratio} | {selected_block_jaccard_vs_anchor} | "
                "{pooler_selected_ratio} | {encoder_layer8_selected_ratio} | {layer_distribution_entropy} | "
                "{metric_delta_mrr_vs_anchor} | {metric_delta_f1_vs_anchor} | {meaningful_selection_diversity} | {metric_acceptable} |".format(**row)
            )
    lines += [
        "",
        "## Decision Rules",
        "",
        "- Meaningful selection diversity: block Jaccard <= 0.85, pooler ratio decreases by >= 20%, or entropy increases by >= 15%.",
        "- Metric acceptable: hard_1000 MRR/F1 drop <= 0.003.",
        "- Promising: hard_1000 MRR or F1 improves >= 0.005.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize V6-HP-hyper selection-diversity ablation.")
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--score-log", type=Path, default=EXP_DIR / "results" / "score_logging_raw.jsonl")
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--anchor-method", default="")
    parser.add_argument("--title", default="Selection Diversity Ablation Report")
    args = parser.parse_args()
    metric_rows = read_csv(args.metrics_csv)
    score_info = aggregate_score_logs(read_jsonl(args.score_log))
    anchor = args.anchor_method or (metric_rows[0].get("method", "") if metric_rows else "")
    summary = build_summary(metric_rows, score_info, anchor)
    write_csv(args.output_summary, summary)
    write_report(args.output_report, args.title, summary)
    print(args.output_summary)
    print(args.output_report)


if __name__ == "__main__":
    main()
