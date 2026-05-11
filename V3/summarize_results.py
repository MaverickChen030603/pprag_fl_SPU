from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

from experiment_config import OUTPUT_ROOT
from metrics import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize V3 communication logs.")
    parser.add_argument("--root", default=str(OUTPUT_ROOT / "pprag_fl_v3"))
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_run(run_dir: Path) -> Dict:
    metadata_path = run_dir / "run_metadata.json"
    logs_path = run_dir / "round_logs.json"
    config_path = run_dir / "upstream_config.json"
    if not metadata_path.exists() or not logs_path.exists():
        return {}
    metadata = load_json(metadata_path)
    logs = load_json(logs_path)
    config = load_json(config_path) if config_path.exists() else {}
    if not logs:
        return {}
    avg_payload = sum(float(row.get("avg_payload_ratio", 0.0)) for row in logs) / len(logs)
    total_uploaded = sum(int(row.get("total_uploaded_params", 0)) for row in logs)
    total_full = sum(int(row.get("total_full_params", 0)) for row in logs)
    avg_hn_loss = sum(float(row.get("avg_hn_loss", 0.0)) for row in logs) / len(logs)
    avg_budget_topk = sum(float(row.get("avg_budget_topk", 0.0)) for row in logs) / len(logs)
    avg_predicted_budget_ratio = sum(float(row.get("avg_predicted_budget_ratio", 0.0)) for row in logs) / len(logs)
    utility_payload = sum(float(row.get("utility_per_payload", 0.0)) for row in logs) / len(logs)
    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "task_name": metadata.get("task_name", config.get("task_name", run_dir.parent.name)),
        "suite_tag": metadata.get("suite_tag", config.get("suite_tag", run_dir.parent.parent.name if run_dir.parent.parent else "")),
        "seed": metadata.get("seed", config.get("seed")),
        "estimate_encryption": metadata.get("estimate_encryption", config.get("estimate_encryption", False)),
        "strategy": metadata.get("selection_strategy", config.get("selection_strategy")),
        "topk_blocks": metadata.get("topk_blocks", config.get("topk_blocks")),
        "warmup_rounds": metadata.get("warmup_rounds", config.get("warmup_rounds")),
        "score_mode": metadata.get("score_mode", config.get("score_mode", "importance")),
        "budget_mode": metadata.get("budget_mode", config.get("budget_mode", "fixed")),
        "use_client_embedding": metadata.get("use_client_embedding", config.get("use_client_embedding", True)),
        "use_history_features": metadata.get("use_history_features", config.get("use_history_features", True)),
        "layerwise_budget": metadata.get("layerwise_budget", config.get("layerwise_budget", False)),
        "rounds": len(logs),
        "avg_payload_ratio": avg_payload,
        "total_uploaded_params": total_uploaded,
        "total_full_params": total_full,
        "overall_payload_ratio": total_uploaded / max(total_full, 1),
        "avg_hn_loss": avg_hn_loss,
        "avg_budget_topk": avg_budget_topk,
        "avg_predicted_budget_ratio": avg_predicted_budget_ratio,
        "utility_per_payload": utility_payload,
        "total_encrypted_bytes_est": sum(float(row.get("total_encrypted_bytes_est", 0.0)) for row in logs),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def build_grouped_summary(summaries: List[Dict]) -> List[Dict]:
    groups: dict[tuple, list[Dict]] = {}
    for item in summaries:
        key = (
            item.get("suite_tag", ""),
            item.get("task_name", ""),
            item.get("strategy", ""),
            item.get("topk_blocks", 0),
            item.get("warmup_rounds", 0),
            bool(item.get("estimate_encryption", False)),
            item.get("score_mode", ""),
            item.get("budget_mode", ""),
            bool(item.get("use_client_embedding", False)),
            bool(item.get("use_history_features", False)),
            bool(item.get("layerwise_budget", False)),
        )
        groups.setdefault(key, []).append(item)
    records: list[Dict] = []
    for key, items in groups.items():
        (
            suite_tag,
            task_name,
            strategy,
            topk_blocks,
            warmup_rounds,
            estimate_encryption,
            score_mode,
            budget_mode,
            use_client_embedding,
            use_history_features,
            layerwise_budget,
        ) = key
        payloads = [float(item.get("overall_payload_ratio", 0.0)) for item in items]
        reductions = [1.0 - payload for payload in payloads]
        hn_losses = [float(item.get("avg_hn_loss", 0.0)) for item in items]
        utility_values = [float(item.get("utility_per_payload", 0.0)) for item in items]
        budget_topks = [float(item.get("avg_budget_topk", 0.0)) for item in items]
        predicted_budget_ratios = [float(item.get("avg_predicted_budget_ratio", 0.0)) for item in items]
        records.append(
            {
                "suite_tag": suite_tag,
                "task_name": task_name,
                "strategy": strategy,
                "topk_blocks": topk_blocks,
                "warmup_rounds": warmup_rounds,
                "estimate_encryption": estimate_encryption,
                "score_mode": score_mode,
                "budget_mode": budget_mode,
                "use_client_embedding": use_client_embedding,
                "use_history_features": use_history_features,
                "layerwise_budget": layerwise_budget,
                "seed_count": len(items),
                "seed_list": ",".join(str(item.get("seed", "")) for item in items),
                "overall_payload_ratio_mean": _mean(payloads),
                "overall_payload_ratio_std": _std(payloads),
                "communication_reduction_mean": _mean(reductions),
                "communication_reduction_std": _std(reductions),
                "avg_hn_loss_mean": _mean(hn_losses),
                "avg_hn_loss_std": _std(hn_losses),
                "utility_per_payload_mean": _mean(utility_values),
                "utility_per_payload_std": _std(utility_values),
                "avg_budget_topk_mean": _mean(budget_topks),
                "avg_budget_topk_std": _std(budget_topks),
                "avg_predicted_budget_ratio_mean": _mean(predicted_budget_ratios),
                "avg_predicted_budget_ratio_std": _std(predicted_budget_ratios),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    summaries: List[Dict] = []
    for metadata_path in sorted(root.rglob("run_metadata.json")):
        run_dir = metadata_path.parent
        summary = summarize_run(run_dir)
        if summary:
            summaries.append(summary)
    output = Path(args.output) if args.output else root / "summary"
    grouped = build_grouped_summary(summaries)
    write_json(output.with_suffix(".json"), summaries)
    write_csv(output.with_suffix(".csv"), summaries)
    write_json(output.parent / f"{output.stem}_grouped.json", grouped)
    write_csv(output.parent / f"{output.stem}_grouped.csv", grouped)
    print(
        f"Wrote {len(summaries)} summaries to {output.with_suffix('.json')} and {output.with_suffix('.csv')}; "
        f"grouped stats to {output.parent / f'{output.stem}_grouped.json'}"
    )


if __name__ == "__main__":
    main()
