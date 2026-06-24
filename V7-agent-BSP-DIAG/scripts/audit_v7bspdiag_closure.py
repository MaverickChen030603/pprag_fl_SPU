#!/usr/bin/env python3
"""Audit V7-agent-BSP-DIAG closure status without mutating running jobs."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP-DIAG")
ANALYSIS = ROOT / "analysis"
REPORTS = ROOT / "reports"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def count(pattern: str, base: Path = ROOT) -> int:
    return sum(1 for _ in base.glob(pattern))


def run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def ps(pid: str) -> str:
    return run(["ps", "-p", pid, "-o", "pid,ppid,stat,etime,%cpu,%mem,rss,cmd"])


def hard_errors() -> list[str]:
    patterns = re.compile(
        r"Traceback|CUDA out of memory|killed|No such file|"
        r"T5Tokenizer requires SentencePiece|empty predictions",
        re.IGNORECASE,
    )
    rows: list[str] = []
    for path in [
        ROOT / "runs/v7bspdiag_all.nohup.log",
        ROOT / "runs/logs/official_fid_diag_20260619_044915.log",
    ]:
        if not path.exists():
            continue
        for idx, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if patterns.search(line):
                rows.append(f"{path}:{idx}: {line[:300]}")
    return rows[-50:]


def official_metrics() -> list[Path]:
    return sorted((ROOT / "eval_outputs/official_fid_t5").glob("**/official_metrics.json"))


def check_official_metadata(paths: list[Path]) -> tuple[list[str], list[dict[str, Any]]]:
    required = [
        "method",
        "seed",
        "n",
        "metrics",
        "avg_budget_topk",
        "budget_std",
        "reader_model",
        "beam_size",
        "max_input_length",
        "passage_ordering",
    ]
    problems: list[str] = []
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = read_json(path)
        row = {
            "path": str(path),
            "method": data.get("method") or Path(data.get("run_dir", "")).name.split("_k3_")[0],
            "seed": data.get("seed"),
            "n_examples": data.get("n"),
            "avg_topk": data.get("avg_budget_topk", data.get("avg_topk")),
            "budget_std": data.get("budget_std"),
            "reader_model": data.get("reader_model") or data.get("fid_model"),
            "beam_size": data.get("beam_size"),
            "max_input_length": data.get("max_input_length"),
            "passage_ordering": data.get("passage_ordering"),
            "status": data.get("status", "complete" if data.get("metrics") else "unknown"),
        }
        rows.append(row)
        for key in required:
            if key not in data or data.get(key) is None:
                problems.append(f"{path}: missing {key}")
        for key in ["avg_topk", "budget_std"]:
            val = row.get(key)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                problems.append(f"{path}: {key} is NaN/missing")
        if row["n_examples"] in (None, 0):
            problems.append(f"{path}: n_examples is zero/missing")
        if row["reader_model"] != "t5-base":
            problems.append(f"{path}: reader_model is {row['reader_model']!r}, expected t5-base")
    return problems, rows


def file_status() -> list[dict[str, Any]]:
    required = [
        ("official_summary", ANALYSIS / "official_fid_t5_diag_summary.csv"),
        ("strict_hf_final", ANALYSIS / "strict_diag_hf_final.csv"),
        ("reader_ordering_verification", ANALYSIS / "reader_input_ordering_verification.csv"),
        ("cache_reuse_audit", ANALYSIS / "cache_reuse_audit.md"),
        ("gold_oracle_effect", ANALYSIS / "gold_oracle_debug_effect.csv"),
        ("per_query_alignment", ANALYSIS / "per_query_alignment_final.csv"),
        ("selection_to_qa_correlation", ANALYSIS / "selection_to_qa_correlation.csv"),
        ("true_subgroup_analysis", ANALYSIS / "true_subgroup_analysis_final.csv"),
        ("representative_cases", ANALYSIS / "representative_cases_final.md"),
        ("statistical_tests", ANALYSIS / "statistical_tests_final.csv"),
        ("final_landing_report", REPORTS / "v7_agent_bsp_diag_final_landing_report_20260619.md"),
    ]
    rows = []
    for name, path in required:
        rows.append(
            {
                "artifact": name,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "status": "present" if path.exists() and path.stat().st_size > 0 else "missing",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    official = official_metrics()
    metadata_problems, metadata_rows = check_official_metadata(official)
    artifacts = file_status()
    errors = hard_errors()

    write_csv(ANALYSIS / "closure_acceptance_checklist_current.csv", artifacts)
    write_csv(ANALYSIS / "official_metadata_audit_current.csv", metadata_rows)

    current_output_dir = ROOT / (
        "eval_outputs/official_fid_t5/v7bspdiag_hf/"
        "agent-bsp-hf-bandit-retrieval_k3_w1_s0_enc0_score-downstream_value_budget-fixed_"
        "hist11_client1_block1_hard1_util1_ep1_cov1_mem0_dyn1_ecw000_slotdyn_pmf1r0i0bg1dh0"
    )

    lines = [
        "# V7-agent-BSP-DIAG Run / Closure Audit",
        "",
        f"Generated: {now}",
        f"Project: `{ROOT}`",
        "",
        "## Process State",
        "",
        "```text",
        ps("97504"),
        ps("155141"),
        "```",
        "",
        "## Current Official FiD/T5 State",
        "",
        f"- official metrics completed: {len(official)}/40",
        f"- current output dir exists: {current_output_dir.exists()}",
        f"- current output dir size bytes: {sum(p.stat().st_size for p in current_output_dir.glob('**/*') if p.is_file()) if current_output_dir.exists() else 0}",
        f"- prediction files: {count('eval_outputs/**/*prediction*')}",
        f"- reader input files: {count('debug_reader_inputs/**/*')}",
        "",
        "The first official FiD/T5 run has exceeded the 3-hour slow-run threshold, but the Python process is still CPU-active. This audit does not terminate it. The current implementation writes predictions and metrics only after the run finishes, so file-growth checks are not sufficient by themselves.",
        "",
        "## Hard Error Scan",
        "",
    ]
    if errors:
        lines.extend([f"- {e}" for e in errors])
    else:
        lines.append("- No hard errors found in the scanned logs.")

    lines.extend(["", "## Metadata Problems", ""])
    if metadata_problems:
        lines.extend([f"- {p}" for p in metadata_problems])
    elif official:
        lines.append("- No official metadata problems found in completed official metrics.")
    else:
        lines.append("- No completed official metrics yet; metadata validation is pending.")

    lines.extend(["", "## Acceptance Checklist", ""])
    for row in artifacts:
        lines.append(f"- {row['artifact']}: {row['status']} ({row['path']})")

    lines.extend(
        [
            "",
            "## Current Decision",
            "",
            "Decision B is not yet proven, but it remains the leading risk: reader ordering may not be connected to actual FiD/T5 inputs. The final decision is blocked until official runs finish or the current slow run is explicitly terminated and repaired.",
            "",
            "## Immediate Next Actions",
            "",
            "1. Keep the current official run alive while CPU activity remains high and no hard error appears.",
            "2. If the first official run remains unfinished after the next monitoring window, inspect stack/profiling or migrate official eval to a faster device/config while preserving logs.",
            "3. After official metrics exist, regenerate reader input export with full method/seed/order coverage and run hash verification before interpreting QA metrics.",
            "4. Replace placeholder representative cases and statistical summaries only after per-query official predictions are available.",
        ]
    )
    (ANALYSIS / "run_failure_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORTS / "v7_agent_bsp_diag_current_closure_audit_20260619.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(ANALYSIS / "run_failure_audit.md")
    print(REPORTS / "v7_agent_bsp_diag_current_closure_audit_20260619.md")


if __name__ == "__main__":
    main()
