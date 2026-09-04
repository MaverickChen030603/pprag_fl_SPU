from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit method selection identity from same-payload runs.")
    parser.add_argument("--raw-csv", action="append", required=True, help="Benchmark raw CSV. Can be repeated.")
    parser.add_argument("--output-jsonl", default="experiments/v6_hp_hyper_next/results/method_identity_audit_raw.jsonl")
    parser.add_argument("--summary-csv", default="experiments/v6_hp_hyper_next/results/method_identity_audit_summary.csv")
    parser.add_argument("--report-md", default="experiments/v6_hp_hyper_next/reports/method_identity_audit_report.md")
    return parser.parse_args()


def _read_csv_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    seen = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = (row.get("method"), row.get("seed"), row.get("subset"), row.get("run_dir"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _block_to_layer(block: str) -> str:
    if block == "pooler":
        return "pooler"
    prefix = "encoder.layer."
    if block.startswith(prefix):
        rest = block[len(prefix) :].split(".", 1)[0]
        return f"encoder.layer.{rest}"
    return block.split(".", 1)[0]


def _selection_details(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _round_entries(run_dir: Path) -> list[dict]:
    json_logs = _load_json(run_dir / "round_logs.json")
    if isinstance(json_logs, list):
        return json_logs
    csv_path = run_dir / "round_logs.csv"
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["selection_details"] = _selection_details(row.get("selection_details"))
    return rows


def _flatten_run(row: dict) -> tuple[list[dict], dict]:
    run_dir = Path(row["run_dir"])
    config = _load_json(run_dir / "upstream_config.json") or {}
    flat: list[dict] = []
    layer_counter: Counter[str] = Counter()
    block_counter: Counter[str] = Counter()
    fingerprints: list[str] = []
    round_payloads: list[float] = []
    for round_log in _round_entries(run_dir):
        round_id = int(round_log.get("round", round_log.get("round_id", -1)))
        try:
            round_payloads.append(float(round_log.get("avg_payload_ratio", 0.0)))
        except Exception:
            pass
        for detail in _selection_details(round_log.get("selection_details")):
            client_id = int(detail.get("client_id", -1))
            blocks = list(detail.get("upload_blocks", []) or [])
            layers = [_block_to_layer(block) for block in blocks]
            layer_counter.update(layers)
            block_counter.update(blocks)
            fingerprint = "|".join(blocks)
            fingerprints.append(f"r{round_id}:c{client_id}:{fingerprint}")
            flat.append(
                {
                    "method": row.get("method", ""),
                    "version": row.get("version", ""),
                    "seed": row.get("seed", ""),
                    "subset": row.get("subset", ""),
                    "round_id": round_id,
                    "client_id": client_id,
                    "payload_ratio": row.get("payload_ratio", ""),
                    "selected_block_ids": blocks,
                    "selected_layer_ids": layers,
                    "num_selected_blocks": len(blocks),
                    "score_mode": row.get("score_mode", config.get("score_mode", "")),
                    "budget_mode": row.get("budget_mode", config.get("budget_mode", "")),
                    "layerwise_budget": row.get("layerwise_budget", config.get("layerwise_budget", "")),
                    "use_utility_memory": config.get("use_utility_memory", ""),
                    "use_hard_query_weighting": config.get("use_hard_query_weighting", ""),
                    "block_score_mean": None,
                    "block_score_std": None,
                    "top_selected_score_mean": None,
                    "top_selected_score_std": None,
                    "layer_distribution": dict(Counter(layers)),
                    "run_dir": str(run_dir),
                }
            )
    summary = {
        "method": row.get("method", ""),
        "version": row.get("version", ""),
        "seed": row.get("seed", ""),
        "subset": row.get("subset", ""),
        "num_rounds": len({item["round_id"] for item in flat}),
        "num_client_rounds": len(flat),
        "unique_selection_patterns": len(set(fingerprints)),
        "unique_block_sets": len({tuple(item["selected_block_ids"]) for item in flat}),
        "avg_payload_ratio": mean(round_payloads) if round_payloads else row.get("payload_ratio", ""),
        "score_mode": row.get("score_mode", config.get("score_mode", "")),
        "budget_mode": row.get("budget_mode", config.get("budget_mode", "")),
        "layerwise_budget": row.get("layerwise_budget", config.get("layerwise_budget", "")),
        "use_utility_memory": config.get("use_utility_memory", ""),
        "use_hard_query_weighting": config.get("use_hard_query_weighting", ""),
        "top_blocks": ";".join(f"{k}:{v}" for k, v in block_counter.most_common(8)),
        "layer_distribution": json.dumps(dict(layer_counter), ensure_ascii=False, sort_keys=True),
        "selection_signature": json.dumps(sorted(fingerprints), ensure_ascii=False),
        "scores_recorded": False,
        "run_dir": str(run_dir),
    }
    return flat, summary


def main() -> None:
    args = parse_args()
    rows = _read_csv_rows(args.raw_csv)
    flat_rows: list[dict] = []
    summaries: list[dict] = []
    for row in rows:
        if not row.get("run_dir"):
            continue
        flat, summary = _flatten_run(row)
        flat_rows.extend(flat)
        summaries.append(summary)

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for item in flat_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    out_csv = Path(args.summary_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "version",
        "seed",
        "subset",
        "num_rounds",
        "num_client_rounds",
        "unique_selection_patterns",
        "unique_block_sets",
        "avg_payload_ratio",
        "score_mode",
        "budget_mode",
        "layerwise_budget",
        "use_utility_memory",
        "use_hard_query_weighting",
        "top_blocks",
        "layer_distribution",
        "scores_recorded",
        "run_dir",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: summary.get(field, "") for field in fields} for summary in summaries)

    grouped = defaultdict(list)
    for summary in summaries:
        grouped[(summary["subset"], summary["seed"])].append(summary)
    lines = [
        "# Method Identity Audit Report",
        "",
        "## Scope",
        "",
        f"- Raw CSV inputs: {', '.join(args.raw_csv)}",
        f"- Audited runs: {len(summaries)}",
        "- Score distribution fields are not available in current round logs, so score means/stds are marked as not recorded.",
        "",
        "## Selection Summary",
        "",
        "| subset | seed | method | unique block sets | top blocks | layer distribution | utility memory | hard weighting |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['subset']} | {summary['seed']} | {summary['method']} | "
            f"{summary['unique_block_sets']} | {summary['top_blocks']} | "
            f"`{summary['layer_distribution']}` | {summary['use_utility_memory']} | {summary['use_hard_query_weighting']} |"
        )
    lines += ["", "## Pairwise Identity Checks", ""]
    for (subset, seed), items in grouped.items():
        lines.append(f"### {subset} / seed={seed}")
        for i, left in enumerate(items):
            for right in items[i + 1 :]:
                identical = left["selection_signature"] == right["selection_signature"]
                lines.append(f"- {left['method']} vs {right['method']}: identical_selection={identical}")
        lines.append("")
    any_identical_baselines = False
    for (_subset, _seed), items in grouped.items():
        baseline_items = [item for item in items if item["method"].startswith(("V3", "V4", "V5"))]
        sigs = {item["selection_signature"] for item in baseline_items}
        if len(baseline_items) >= 2 and len(sigs) == 1:
            any_identical_baselines = True
    lines += ["## Current Audit Conclusion", ""]
    if any_identical_baselines:
        lines.append(
            "V3/V4/V5 same-payload topk2 fixed configurations may collapse to equivalent block-selection behavior under the current implementation."
        )
    else:
        lines.append("V3/V4/V5 do not appear to have fully identical selection signatures in the audited runs.")
    lines.append(
        "Because block score distributions are not recorded, the current audit can verify selected block/layer identity but cannot compare score distribution identity."
    )
    Path(args.report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_jsonl)
    print(out_csv)
    print(args.report_md)


if __name__ == "__main__":
    main()
