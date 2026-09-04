from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
V6_HP1_DIR = ROOT_DIR / "V6-HP1"
if str(V6_HP1_DIR) not in sys.path:
    sys.path.insert(0, str(V6_HP1_DIR))

from report_generator import summarize_downstream_run  # noqa: E402
from summarize_results import summarize_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append one v6_hp_hyper_next benchmark result row.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--subset-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rag-dir", required=True)
    parser.add_argument("--target-payload", type=float, default=0.070134)
    parser.add_argument("--payload-tolerance", type=float, default=0.002)
    parser.add_argument("--runtime-sec", default="")
    return parser.parse_args()


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = [
        "method",
        "version",
        "seed",
        "subset",
        "subset_path",
        "strategy",
        "topk_blocks",
        "warmup_rounds",
        "score_mode",
        "budget_mode",
        "layerwise_budget",
        "payload_ratio",
        "payload_target",
        "payload_tolerance",
        "payload_mismatch",
        "MRR",
        "NDCG",
        "F1",
        "EM",
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "Recall@10",
        "Hit@1",
        "Hit@10",
        "Gold rank",
        "runtime_sec",
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
    payload = upstream.get("overall_payload_ratio", "")
    try:
        payload_float = float(payload)
        mismatch = abs(payload_float - args.target_payload) > args.payload_tolerance
    except (TypeError, ValueError):
        mismatch = True
    row = {
        "method": args.method,
        "version": args.version,
        "seed": args.seed,
        "subset": args.subset_name,
        "subset_path": str(Path(args.subset_path).expanduser().resolve()),
        "strategy": upstream.get("strategy", ""),
        "topk_blocks": upstream.get("topk_blocks", ""),
        "warmup_rounds": upstream.get("warmup_rounds", ""),
        "score_mode": upstream.get("score_mode", ""),
        "budget_mode": upstream.get("budget_mode", ""),
        "layerwise_budget": upstream.get("layerwise_budget", ""),
        "payload_ratio": payload,
        "payload_target": args.target_payload,
        "payload_tolerance": args.payload_tolerance,
        "payload_mismatch": mismatch,
        "MRR": metrics.get("mrr", ""),
        "NDCG": metrics.get("NDCG", metrics.get("ndcg", "")),
        "F1": metrics.get("F1", metrics.get("f1", "")),
        "EM": metrics.get("em", ""),
        "Recall@1": metrics.get("recall_1", ""),
        "Recall@3": metrics.get("recall_3", ""),
        "Recall@5": metrics.get("recall_5", ""),
        "Recall@10": metrics.get("recall_10", ""),
        "Hit@1": metrics.get("hit1", ""),
        "Hit@10": metrics.get("hit10", ""),
        "Gold rank": metrics.get("gold_rank", ""),
        "runtime_sec": args.runtime_sec,
        "run_dir": str(run_dir),
        "rag_dir": str(rag_dir),
    }
    _append_csv(Path(args.csv), row)
    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
