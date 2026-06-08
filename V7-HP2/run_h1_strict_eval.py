#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path

from metrics import ensure_dir, write_json

HARD_BLOCKS = {"pooler", "encoder.layer.11", "encoder.layer.8", "encoder.layer.7"}
TAIL_BLOCKS = {"embeddings", "encoder.layer.0", "encoder.layer.1", "encoder.layer.2", "encoder.layer.3"}
ALL_TARGETS = HARD_BLOCKS | TAIL_BLOCKS


def parse_args():
    p = argparse.ArgumentParser(description="Compute non-saturated V7-HP2 hard/tail diagnostic metrics from selection traces.")
    p.add_argument("--upstream-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def method_from_run_name(name: str) -> str:
    return name.split("_k", 1)[0].replace("-", "_")


def alpha_from_task(task: str) -> float:
    m = re.search(r"a(\d+)_", task)
    if not m:
        return 0.3
    return {"005": 0.05, "01": 0.1, "03": 0.3, "05": 0.5}.get(m.group(1), 0.3)


def read_rounds(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def safe_details(raw: str) -> list[dict]:
    try:
        value = ast.literal_eval(raw)
    except Exception:
        return []
    return value if isinstance(value, list) else []


def score_run(run_dir: Path, upstream_root: Path, output_root: Path) -> dict:
    cfg_path = run_dir / "upstream_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    rounds = read_rounds(run_dir / "round_logs.csv")
    task = cfg.get("task_name", run_dir.parent.name)
    alpha = alpha_from_task(task)
    method = cfg.get("method_name") or method_from_run_name(run_dir.name)
    profile = cfg.get("agent_profile", "")
    total_events = 0
    hard_hits = 0.0
    tail_hits = 0.0
    target_hits = 0.0
    budgets = []
    tail_client_budgets = []
    all_blocks = []
    client_budget = {}
    warmup = int(cfg.get("warmup_rounds", 1) or 0)
    for row in rounds:
        round_id = int(float(row.get("round", 0) or 0))
        if round_id <= warmup:
            continue
        for detail in safe_details(row.get("selection_details", "[]")):
            blocks = set(detail.get("upload_blocks") or [])
            if "__ALL__" in blocks:
                blocks = ALL_TARGETS
            budget = int(detail.get("budget_topk", 0) or 0)
            cid = int(detail.get("client_id", -1))
            total_events += 1
            budgets.append(budget)
            client_budget.setdefault(cid, []).append(budget)
            if cid in {0, 1}:
                tail_client_budgets.append(budget)
            hard_hits += len(blocks & HARD_BLOCKS) / len(HARD_BLOCKS)
            tail_hits += len(blocks & TAIL_BLOCKS) / len(TAIL_BLOCKS)
            target_hits += len(blocks & ALL_TARGETS) / len(ALL_TARGETS)
            all_blocks.extend(sorted(blocks))
    total_events = max(total_events, 1)
    avg_budget = sum(budgets) / max(len(budgets), 1)
    budget_std = math.sqrt(sum((b - avg_budget) ** 2 for b in budgets) / max(len(budgets), 1)) if budgets else 0.0
    tail_budget = sum(tail_client_budgets) / max(len(tail_client_budgets), 1)
    unique_blocks = len(set(all_blocks))
    diversity = unique_blocks / max(len(ALL_TARGETS), 1)
    hard_recall = hard_hits / total_events
    tail_recall = tail_hits / total_events
    target_recall = target_hits / total_events
    tail_weight = 1.35 if alpha <= 0.1 else 1.0
    h1_score = min(1.0, 0.30 * hard_recall + 0.40 * min(1.0, tail_recall * tail_weight) + 0.20 * diversity + 0.10 * min(1.0, tail_budget / max(avg_budget, 1e-8)))
    relative = run_dir.relative_to(upstream_root)
    out_dir = ensure_dir(output_root / relative)
    record = {
        "suite": cfg.get("suite_tag", relative.parts[0] if relative.parts else ""),
        "task_name": task,
        "alpha": alpha,
        "run_name": run_dir.name,
        "method": method,
        "agent_profile": profile,
        "seed": cfg.get("seed", ""),
        "avg_budget_topk_h1": avg_budget,
        "budget_std_h1": budget_std,
        "tail_client_budget_h1": tail_budget,
        "hard_block_recall_h1": hard_recall,
        "tail_block_recall_h1": tail_recall,
        "target_block_recall_h1": target_recall,
        "selection_diversity_h1": diversity,
        "h1_non_saturated_score": h1_score,
        "event_count": total_events,
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "h1_strict_metrics.json", record)
    return record


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    upstream_root = Path(args.upstream_root)
    output_root = ensure_dir(args.output_root)
    rows = []
    for meta in sorted(upstream_root.rglob("run_metadata.json")):
        run_dir = meta.parent
        if not (run_dir / "round_logs.csv").exists():
            continue
        out_file = output_root / run_dir.relative_to(upstream_root) / "h1_strict_metrics.json"
        if out_file.exists() and not args.force:
            rows.append(json.loads(out_file.read_text(encoding="utf-8")))
        else:
            rows.append(score_run(run_dir, upstream_root, output_root))
    write_csv(output_root / "h1_strict_summary.csv", rows)
    write_json(output_root / "h1_strict_summary.json", rows)
    print(f"Processed {len(rows)} H1 strict diagnostic runs into {output_root}")


if __name__ == "__main__":
    main()
