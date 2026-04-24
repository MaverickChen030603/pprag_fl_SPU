from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from experiment_config import OUTPUT_ROOT
from metrics import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SUP_v2 communication logs.")
    parser.add_argument("--root", default=str(OUTPUT_ROOT / "pprag_fl_sup_v2"))
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
    return {
        "run_dir": str(run_dir),
        "strategy": metadata.get("selection_strategy", config.get("selection_strategy")),
        "topk_blocks": metadata.get("topk_blocks", config.get("topk_blocks")),
        "warmup_rounds": metadata.get("warmup_rounds", config.get("warmup_rounds")),
        "rounds": len(logs),
        "avg_payload_ratio": avg_payload,
        "total_uploaded_params": total_uploaded,
        "total_full_params": total_full,
        "overall_payload_ratio": total_uploaded / max(total_full, 1),
        "avg_hn_loss": avg_hn_loss,
        "total_encrypted_bytes_est": sum(float(row.get("total_encrypted_bytes_est", 0.0)) for row in logs),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    summaries: List[Dict] = []
    for run_dir in sorted(path for path in root.glob("*") if path.is_dir()):
        summary = summarize_run(run_dir)
        if summary:
            summaries.append(summary)
    output = Path(args.output) if args.output else root / "summary"
    write_json(output.with_suffix(".json"), summaries)
    write_csv(output.with_suffix(".csv"), summaries)
    print(f"Wrote {len(summaries)} summaries to {output.with_suffix('.json')} and {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

