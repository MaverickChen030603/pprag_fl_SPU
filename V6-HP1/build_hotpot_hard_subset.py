from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
DEFAULT_OUTPUT = CURRENT_DIR / "data" / "hotpot_hard_query_subset.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Hard-Query-Only HotpotQA subset from baseline per-query retrieval logs.")
    parser.add_argument("--per-query-input", default="", help="Existing JSONL from run_rag_eval.py --save-per-query.")
    parser.add_argument("--baseline-model", default="", help="Optional retriever model path used to first generate per-query logs.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--eval-num-examples", type=int, default=1500)
    parser.add_argument("--hotpot-split", default="validation")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--work-dir", default=str(CURRENT_DIR / "outputs" / "hard_query_builder"))
    parser.add_argument("--target-size", type=int, default=800)
    parser.add_argument("--min-size", type=int, default=500)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--recall-threshold", type=float, default=0.0)
    parser.add_argument("--require-answer-miss", action="store_true")
    parser.add_argument("--script", default="main_100_test.py")
    return parser.parse_args()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_as_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_as_text(item) for item in value.values())
    return str(value)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _run_baseline_eval(args: argparse.Namespace) -> Path:
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = work_dir / "baseline_per_query_results.jsonl"
    command = [
        args.python,
        str(CURRENT_DIR / "run_rag_eval.py"),
        "--model",
        str(Path(args.baseline_model).expanduser().resolve()),
        "--script",
        args.script,
        "--output-dir",
        str(work_dir / "rag_eval"),
        "--python",
        args.python,
        "--dataset",
        "hotpot_qa",
        "--hotpot-split",
        args.hotpot_split,
        "--eval-num-examples",
        str(args.eval_num_examples),
        "--save-per-query",
        "--per-query-output",
        str(per_query_path),
    ]
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)
    return per_query_path


def _score_row(row: dict[str, Any], topk: int) -> dict[str, Any]:
    metrics = row.get("metrics", {}) or {}
    retrieval_context = _as_text(row.get("retrieval_context", [])[:topk]).lower()
    answer = _as_text(row.get("expected_answer", "")).strip().lower()
    recall_key = f"recall_{topk}"
    hit_key = f"hit_{topk}"
    topk_recall = float(metrics.get(recall_key, metrics.get("recall_3", 0.0)) or 0.0)
    hit_topk = float(metrics.get(hit_key, metrics.get("cos_3", 0.0)) or 0.0)
    answer_in_topk = bool(answer and answer in retrieval_context)
    return {
        "topk_recall": topk_recall,
        "hit_topk": hit_topk,
        "answer_in_topk": answer_in_topk,
        "gold_rank": int(metrics.get("gold_rank", 0) or 0),
    }


def build_subset(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = []
    fallback = []
    for row in rows:
        score = _score_row(row, args.topk)
        is_hard = score["topk_recall"] <= args.recall_threshold or score["hit_topk"] <= 0.0
        if args.require_answer_miss:
            is_hard = is_hard and not score["answer_in_topk"]
        item = {
            "query_id": str(row.get("query_id")),
            "question": row.get("question", ""),
            "expected_answer": row.get("expected_answer", ""),
            "golden_ids": row.get("golden_ids", []),
            "hardness": score,
        }
        if is_hard:
            candidates.append(item)
        else:
            fallback.append(item)

    candidates.sort(key=lambda item: (item["hardness"]["topk_recall"], item["hardness"]["hit_topk"], item["hardness"]["gold_rank"]))
    if len(candidates) < args.min_size:
        fallback.sort(key=lambda item: (item["hardness"]["topk_recall"], item["hardness"]["hit_topk"], item["hardness"]["gold_rank"]))
        candidates.extend(fallback[: args.min_size - len(candidates)])
    return candidates[: args.target_size]


def main() -> None:
    args = parse_args()
    if args.per_query_input:
        per_query_path = Path(args.per_query_input).expanduser().resolve()
    elif args.baseline_model:
        per_query_path = _run_baseline_eval(args)
    else:
        raise SystemExit("Provide --per-query-input or --baseline-model.")

    rows = _load_jsonl(per_query_path)
    subset = build_subset(rows, args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "subset_type": "hotpot_hard_query_only",
        "source_per_query": str(per_query_path),
        "selection_rule": {
            "topk": args.topk,
            "recall_threshold": args.recall_threshold,
            "require_answer_miss": args.require_answer_miss,
            "target_size": args.target_size,
            "min_size": args.min_size,
        },
        "count": len(subset),
        "queries": subset,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(subset)} hard queries to {output}")


if __name__ == "__main__":
    main()
