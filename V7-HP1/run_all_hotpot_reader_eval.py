#!/usr/bin/env python3
"""Batch reader/generator Hotpot answer evaluation for V7-HP1 runs."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def split_csv(values: Optional[List[str]]) -> Optional[set[str]]:
    if not values:
        return None
    out: set[str] = set()
    for value in values:
        for part in value.split(','):
            part = part.strip()
            if part:
                out.add(part)
    return out or None


def discover_runs(upstream_root: Path, suites: Optional[set[str]], methods: Optional[set[str]]) -> List[Path]:
    runs = []
    for meta_path in sorted(upstream_root.glob("**/run_metadata.json")):
        run_dir = meta_path.parent
        meta = load_json(meta_path)
        suite = str(meta.get("suite_tag") or "")
        method = str(meta.get("method") or meta.get("selection_strategy") or "")
        if suites and suite not in suites and not any(part in run_dir.parts for part in suites):
            continue
        if methods and method not in methods and not any(m in run_dir.name for m in methods):
            continue
        runs.append(run_dir)
    return runs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-root", type=Path, default=Path("V7-HP1/outputs/pprag_fl_v7_hp1"))
    p.add_argument("--output-root", type=Path, default=Path("V7-HP1/outputs/hotpot_reader_eval"))
    p.add_argument("--rawdata-path", type=Path, default=Path("FedE/select_data_hotpot_train_5000.json"))
    p.add_argument("--suite", action="append")
    p.add_argument("--include-method", action="append")
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--prefer-official", action="store_true")
    p.add_argument("--reader-model", default="google-t5/t5-small")
    p.add_argument("--local-reader-only", action="store_true")
    p.add_argument("--retrieval-topk", type=int, default=5)
    p.add_argument("--support-topk", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--reader-batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit-runs", type=int, default=0)
    args = p.parse_args()

    suites = split_csv(args.suite) or {"hp1_budget_aligned"}
    methods = split_csv(args.include_method)
    runs = discover_runs(args.upstream_root, suites, methods)
    if args.limit_runs > 0:
        runs = runs[:args.limit_runs]
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"discovered_runs": len(runs), "suites": sorted(suites), "methods": sorted(methods) if methods else None}, ensure_ascii=False))

    rows = []
    script = Path("V7-HP1/run_hotpot_reader_eval.py")
    for idx, run_dir in enumerate(runs, 1):
        rel = run_dir.relative_to(args.upstream_root)
        out_dir = args.output_root / rel
        metrics_path = out_dir / "reader_metrics.json"
        if metrics_path.exists() and not args.force:
            metrics = load_json(metrics_path)
            status = metrics.get("status", "completed")
            print(f"[{idx}/{len(runs)}] skip existing {run_dir}")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(script),
                "--run-dir", str(run_dir),
                "--rawdata-path", str(args.rawdata_path),
                "--output-dir", str(out_dir),
                "--max-examples", str(args.max_examples),
                "--reader-model", args.reader_model,
                "--retrieval-topk", str(args.retrieval_topk),
                "--support-topk", str(args.support_topk),
                "--batch-size", str(args.batch_size),
                "--reader-batch-size", str(args.reader_batch_size),
                "--device", args.device,
            ]
            if args.prefer_official:
                cmd.append("--prefer-official")
            if args.local_reader_only:
                cmd.append("--local-reader-only")
            print(f"[{idx}/{len(runs)}] reader {run_dir}")
            proc = subprocess.run(cmd, text=True)
            if proc.returncode != 0:
                status = f"failed:{proc.returncode}"
                metrics = {"status": status, "run_dir": str(run_dir), "output_dir": str(out_dir)}
                (out_dir / "reader_metrics_failed.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            else:
                metrics = load_json(metrics_path)
                status = metrics.get("status", "completed")
        row = {
            "status": status,
            "run_dir": str(run_dir),
            "output_dir": str(out_dir),
            "suite_tag": metrics.get("suite_tag"),
            "task_name": metrics.get("task_name"),
            "method": metrics.get("method"),
            "profile": metrics.get("profile"),
            "seed": metrics.get("seed"),
            "n": metrics.get("n"),
            "dataset_mode": metrics.get("dataset_mode"),
            "reader_model": metrics.get("reader_model"),
        }
        row.update(metrics.get("metrics", {}))
        rows.append(row)
        (args.output_root / "reader_eval_all_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        fields = sorted({k for r in rows for k in r.keys()})
        with (args.output_root / "reader_eval_all_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
    print(json.dumps({"completed_or_seen": len(rows), "summary": str(args.output_root / "reader_eval_all_summary.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
