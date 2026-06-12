from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
V6_HP1_DIR = ROOT_DIR / "V6-HP1"
if str(V6_HP1_DIR) not in sys.path:
    sys.path.insert(0, str(V6_HP1_DIR))

from report_generator import summarize_downstream_run  # noqa: E402
from summarize_results import summarize_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append one same-payload benchmark result row to CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rag-dir", required=True)
    parser.add_argument("--eval-subset-type", default="all")
    parser.add_argument("--eval-num-examples", type=int, default=1000)
    return parser.parse_args()


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = [
        "label",
        "version",
        "strategy",
        "topk_blocks",
        "warmup_rounds",
        "score_mode",
        "budget_mode",
        "layerwise_budget",
        "overall_payload_ratio",
        "mrr",
        "ndcg",
        "f1",
        "em",
        "recall_3",
        "hit1",
        "hit10",
        "eval_subset_type",
        "eval_num_examples",
        "run_dir",
        "rag_dir",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    rag_dir = Path(args.rag_dir).expanduser().resolve()
    upstream = summarize_run(run_dir)
    downstream = summarize_downstream_run(rag_dir, run_dir, run_dir.name)
    metrics = downstream.get("metrics", {})
    row = {
        "label": args.label,
        "version": args.version,
        "strategy": upstream.get("strategy", ""),
        "topk_blocks": upstream.get("topk_blocks", ""),
        "warmup_rounds": upstream.get("warmup_rounds", ""),
        "score_mode": upstream.get("score_mode", ""),
        "budget_mode": upstream.get("budget_mode", ""),
        "layerwise_budget": upstream.get("layerwise_budget", ""),
        "overall_payload_ratio": upstream.get("overall_payload_ratio", ""),
        "mrr": metrics.get("mrr", ""),
        "ndcg": metrics.get("NDCG", metrics.get("ndcg", "")),
        "f1": metrics.get("F1", metrics.get("f1", "")),
        "em": metrics.get("em", ""),
        "recall_3": metrics.get("recall_3", ""),
        "hit1": metrics.get("hit1", ""),
        "hit10": metrics.get("hit10", ""),
        "eval_subset_type": args.eval_subset_type,
        "eval_num_examples": args.eval_num_examples,
        "run_dir": str(run_dir),
        "rag_dir": str(rag_dir),
    }
    _append_csv(Path(args.csv), row)
    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
