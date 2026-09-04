#!/usr/bin/env python3
"""Collect V7 run metadata and downstream logs into compact CSV/JSON summaries."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/iiserver31/projects/FedE4RAG-main")
OUTPUT_ROOT = ROOT / "V7" / "outputs"
REPORT_ROOT = ROOT / "实验分析报告" / "V7"


def safe_load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_json_error": str(exc), "_path": str(path)}


def infer_fields(path: Path) -> dict:
    parts = path.relative_to(OUTPUT_ROOT).parts
    inferred = {"suite": "", "run_dir": str(path.parent)}
    if parts:
        inferred["suite"] = parts[0]
    text = "/".join(parts)
    for key in ("method", "seed", "topk"):
        match = re.search(rf"{key}[_=-]([^/_.-]+)", text)
        if match:
            inferred[key] = match.group(1)
    return inferred


def flatten_metadata(path: Path) -> dict:
    data = safe_load_json(path)
    row = infer_fields(path)
    row["metadata_path"] = str(path)
    row["method"] = data.get("method", row.get("method", ""))
    row["suite"] = data.get("suite", row.get("suite", ""))
    row["seed"] = data.get("seed", row.get("seed", ""))
    row["topk"] = data.get("topk", row.get("topk", ""))
    row["avg_payload"] = data.get("avg_payload", data.get("payload_avg", ""))
    row["total_payload"] = data.get("total_payload", data.get("payload_total", ""))
    row["final_round"] = data.get("final_round", data.get("round", ""))
    row["status"] = data.get("status", "completed" if "_json_error" not in data else "json_error")
    row["json_error"] = data.get("_json_error", "")
    return row


def summarize_downstream(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    row = infer_fields(log_path)
    row["rag_eval_log"] = str(log_path)
    row["has_traceback"] = "Traceback" in text or "ERROR" in text
    patterns = {
        "recall_at_5": r"Recall@5\s*[:=]\s*([0-9.]+)",
        "recall_at_10": r"Recall@10\s*[:=]\s*([0-9.]+)",
        "mrr": r"\bMRR\b\s*[:=]\s*([0-9.]+)",
        "ndcg": r"\bnDCG\b\s*[:=]\s*([0-9.]+)",
        "f1": r"\bF1\b\s*[:=]\s*([0-9.]+)",
        "em": r"\bEM\b\s*[:=]\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        row[key] = match.group(1) if match else ""
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = REPORT_ROOT / f"v7_collected_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = [flatten_metadata(path) for path in sorted(OUTPUT_ROOT.rglob("run_metadata.json"))]
    downstream_rows = [summarize_downstream(path) for path in sorted(OUTPUT_ROOT.rglob("rag_eval_stdout.log"))]

    write_csv(out_dir / "v7_upstream_summary.csv", metadata_rows)
    write_csv(out_dir / "v7_downstream_summary.csv", downstream_rows)

    summary = {
        "generated_at": stamp,
        "output_root": str(OUTPUT_ROOT),
        "upstream_runs": len(metadata_rows),
        "downstream_runs": len(downstream_rows),
        "upstream_summary": str(out_dir / "v7_upstream_summary.csv"),
        "downstream_summary": str(out_dir / "v7_downstream_summary.csv"),
    }
    (out_dir / "v7_collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
