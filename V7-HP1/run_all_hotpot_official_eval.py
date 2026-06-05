#!/usr/bin/env python3
"""Batch Hotpot-style QA/supporting-fact evaluation for V7-HP1 runs."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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


def discover_runs(upstream_root: Path, suite_filter: Optional[set[str]], method_filter: Optional[set[str]]) -> List[Path]:
    runs: List[Path] = []
    for meta_path in sorted(upstream_root.glob("**/run_metadata.json")):
        run_dir = meta_path.parent
        meta = load_json(meta_path)
        suite = str(meta.get("suite_tag") or "")
        method = str(meta.get("method") or "")
        if suite_filter and suite not in suite_filter and not any(part in run_dir.parts for part in suite_filter):
            continue
        if method_filter and method not in method_filter and not any(m in run_dir.name for m in method_filter):
            continue
        runs.append(run_dir)
    return runs


def rel_output_dir(output_root: Path, upstream_root: Path, run_dir: Path) -> Path:
    try:
        rel = run_dir.relative_to(upstream_root)
    except ValueError:
        rel = Path(run_dir.name)
    return output_root / rel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=Path("V7-HP1/outputs/pprag_fl_v7_hp1"))
    parser.add_argument("--output-root", type=Path, default=Path("V7-HP1/outputs/hotpot_official_eval"))
    parser.add_argument("--rawdata-path", type=Path, default=Path("FedE/select_data_hotpot_train_5000.json"))
    parser.add_argument("--suite", action="append", help="Suite tag(s), comma-separated or repeated. Default: hp1_budget_aligned")
    parser.add_argument("--include-method", action="append", help="Method(s), comma-separated or repeated")
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--support-topk", type=int, default=2)
    parser.add_argument("--answer-topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--prefer-official", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-runs", type=int, default=0)
    args = parser.parse_args()

    suites = split_csv(args.suite) or {"hp1_budget_aligned"}
    methods = split_csv(args.include_method)
    runs = discover_runs(args.upstream_root, suites, methods)
    if args.limit_runs > 0:
        runs = runs[: args.limit_runs]

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"discovered_runs": len(runs), "suites": sorted(suites), "methods": sorted(methods) if methods else None}, ensure_ascii=False))

    rows: List[Dict[str, Any]] = []
    eval_script = Path("V7-HP1/run_hotpot_official_eval.py")
    for idx, run_dir in enumerate(runs, 1):
        out_dir = rel_output_dir(args.output_root, args.upstream_root, run_dir)
        metrics_path = out_dir / "official_metrics.json"
        if metrics_path.exists() and not args.force:
            metrics = load_json(metrics_path)
            status = metrics.get("status", "completed")
            print(f"[{idx}/{len(runs)}] skip existing {run_dir}")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(eval_script),
                "--run-dir", str(run_dir),
                "--rawdata-path", str(args.rawdata_path),
                "--output-dir", str(out_dir),
                "--max-examples", str(args.max_examples),
                "--support-topk", str(args.support_topk),
                "--answer-topk", str(args.answer_topk),
                "--batch-size", str(args.batch_size),
                "--device", args.device,
            ]
            if args.prefer_official:
                cmd.append("--prefer-official")
            print(f"[{idx}/{len(runs)}] eval {run_dir}")
            proc = subprocess.run(cmd, text=True)
            if proc.returncode != 0:
                status = f"failed:{proc.returncode}"
                metrics = {"status": status, "run_dir": str(run_dir), "output_dir": str(out_dir)}
                (out_dir / "official_metrics_failed.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            else:
                metrics = load_json(metrics_path)
                status = metrics.get("status", "completed")
        flat = {
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
        }
        flat.update(metrics.get("metrics", {}))
        rows.append(flat)
        summary_path = args.output_root / "official_eval_all_summary.json"
        summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        if rows:
            csv_path = args.output_root / "official_eval_all_summary.csv"
            fields = sorted({k for row in rows for k in row.keys()})
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    print(json.dumps({"completed_or_seen": len(rows), "summary": str(args.output_root / "official_eval_all_summary.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
