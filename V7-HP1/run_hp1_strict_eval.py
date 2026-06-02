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

# BERT blocks used as proxies for multihop reasoning pressure.
BRIDGE_BLOCKS = {"encoder.layer.8", "encoder.layer.9", "encoder.layer.10", "encoder.layer.11", "pooler"}
EARLY_EVIDENCE_BLOCKS = {"embeddings", "encoder.layer.0", "encoder.layer.1", "encoder.layer.2", "encoder.layer.3"}
ALL_TARGETS = BRIDGE_BLOCKS | EARLY_EVIDENCE_BLOCKS


def parse_args():
    p = argparse.ArgumentParser(description="Compute V7-HP1 HotpotQA multihop strict diagnostic metrics from selection traces.")
    p.add_argument("--upstream-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def alpha_from_task(task: str) -> float:
    m = re.search(r"a(\d+)_", task)
    if not m:
        return 0.3
    return {"005": 0.05, "01": 0.1, "03": 0.3, "05": 0.5}.get(m.group(1), 0.3)


def read_rounds(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    method = cfg.get("method_name") or run_dir.name.split("_k", 1)[0].replace("-", "_")
    profile = cfg.get("agent_profile", "")
    total_events = 0
    bridge_hits = 0.0
    early_hits = 0.0
    target_hits = 0.0
    budgets = []
    rare_client_budgets = []
    all_blocks = []
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
            if cid in {0, 1}:
                rare_client_budgets.append(budget)
            bridge_hits += len(blocks & BRIDGE_BLOCKS) / len(BRIDGE_BLOCKS)
            early_hits += len(blocks & EARLY_EVIDENCE_BLOCKS) / len(EARLY_EVIDENCE_BLOCKS)
            target_hits += len(blocks & ALL_TARGETS) / len(ALL_TARGETS)
            all_blocks.extend(sorted(blocks))
    total_events = max(total_events, 1)
    avg_budget = sum(budgets) / max(len(budgets), 1)
    budget_std = math.sqrt(sum((b - avg_budget) ** 2 for b in budgets) / max(len(budgets), 1)) if budgets else 0.0
    rare_budget = sum(rare_client_budgets) / max(len(rare_client_budgets), 1)
    diversity = min(1.0, len(set(all_blocks)) / max(len(ALL_TARGETS), 1))
    bridge_recall = bridge_hits / total_events
    early_recall = early_hits / total_events
    target_recall = target_hits / total_events
    rare_weight = 1.30 if alpha <= 0.1 else 1.0
    hp1_score = min(
        1.0,
        0.36 * bridge_recall
        + 0.28 * min(1.0, early_recall * rare_weight)
        + 0.20 * diversity
        + 0.10 * min(1.0, rare_budget / max(avg_budget, 1e-8))
        + 0.06 * target_recall,
    )
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
        "avg_budget_topk_hp1": avg_budget,
        "budget_std_hp1": budget_std,
        "rare_client_budget_hp1": rare_budget,
        "bridge_block_recall_hp1": bridge_recall,
        "early_evidence_recall_hp1": early_recall,
        "target_block_recall_hp1": target_recall,
        "selection_diversity_hp1": diversity,
        "hp1_multihop_score": hp1_score,
        "event_count": total_events,
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "hp1_strict_metrics.json", record)
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
        out_file = output_root / run_dir.relative_to(upstream_root) / "hp1_strict_metrics.json"
        if out_file.exists() and not args.force:
            rows.append(json.loads(out_file.read_text(encoding="utf-8")))
        else:
            rows.append(score_run(run_dir, upstream_root, output_root))
    write_csv(output_root / "hp1_strict_summary.csv", rows)
    write_json(output_root / "hp1_strict_summary.json", rows)
    print(f"Processed {len(rows)} HP1 strict diagnostic runs into {output_root}")


if __name__ == "__main__":
    main()
