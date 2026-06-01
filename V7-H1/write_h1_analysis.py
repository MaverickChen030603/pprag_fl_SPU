#!/usr/bin/env python3
"""Write a compact Chinese analysis report for V7-H1."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = REPO_ROOT / "V7-H1" / "outputs" / "pprag_fl_v7_h1"
DEFAULT_STRICT = REPO_ROOT / "V7-H1" / "outputs" / "h1_strict_eval"
DEFAULT_REPORT = REPO_ROOT / "实验分析报告" / "V7-H1"

METHOD_ORDER = [
    "hypernet_v6",
    "adaptive_v6",
    "agent_tail_v7h1",
    "agent_memory_v7h1",
    "agent_oracle_v7h1",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} ± {pstdev(values):.4f}"


def infer_suite(path: Path, strict_root: Path) -> str:
    try:
        return path.relative_to(strict_root).parts[0]
    except Exception:
        return path.parent.name


def load_strict(strict_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(strict_root.glob("*/h1_strict_summary.csv")):
        suite = infer_suite(path, strict_root)
        if suite in {"all_h1", "smoke"}:
            continue
        for row in read_csv(path):
            row = dict(row)
            row["suite"] = suite
            rows.append(row)
    return rows


def load_upstream(upstream_root: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for meta in upstream_root.glob("*/*/*/run_metadata.json"):
        counts[meta.parts[-4]] += 1
    for meta in upstream_root.glob("*/*/run_metadata.json"):
        # tolerate older nesting without double-counting the common case
        if meta.parent.parent.parent == upstream_root:
            continue
        counts[meta.parent.parent.name] += 1
    return dict(counts)


def group(rows: list[dict[str, Any]], suite: str | None = None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if suite and row.get("suite") != suite:
            continue
        method = row.get("method") or row.get("strategy") or "unknown"
        grouped[method].append(row)
    return grouped


def metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        val = as_float(row.get(key))
        if val is not None:
            out.append(val)
    return out


def best_baseline(grouped: dict[str, list[dict[str, Any]]], metric: str) -> float | None:
    vals: list[float] = []
    for method in ("hypernet_v6", "adaptive_v6"):
        values = metric_values(grouped.get(method, []), metric)
        if values:
            vals.append(mean(values))
    return max(vals) if vals else None


def render_table(grouped: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines = [
        "| method | n | avg_budget_topk_h1 | hard_block_recall_h1 | tail_block_recall_h1 | selection_diversity_h1 | h1_non_saturated_score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    seen = set(grouped)
    ordered = [m for m in METHOD_ORDER if m in seen] + sorted(seen - set(METHOD_ORDER))
    for method in ordered:
        rows = grouped[method]
        lines.append(
            "| {method} | {n} | {budget} | {hard} | {tail} | {div} | {score} |".format(
                method=method,
                n=len(rows),
                budget=stats(metric_values(rows, "avg_budget_topk_h1")),
                hard=stats(metric_values(rows, "hard_block_recall_h1")),
                tail=stats(metric_values(rows, "tail_block_recall_h1")),
                div=stats(metric_values(rows, "selection_diversity_h1")),
                score=stats(metric_values(rows, "h1_non_saturated_score")),
            )
        )
    return lines


def render_gaps(grouped: dict[str, list[dict[str, Any]]]) -> list[str]:
    metric = "h1_non_saturated_score"
    base = best_baseline(grouped, metric)
    lines: list[str] = []
    if base is None:
        return ["- baseline 数据不足，暂不能计算 agent gap。"]
    for method in ("agent_tail_v7h1", "agent_memory_v7h1", "agent_oracle_v7h1"):
        values = metric_values(grouped.get(method, []), metric)
        if not values:
            continue
        gap = mean(values) - base
        sign = "正" if gap > 1e-6 else "负/无"
        lines.append(f"- `{method}` 相对最佳 baseline 的 `{metric}` gap = {gap:+.4f}，方向：{sign}信号。")
    return lines or ["- agent 数据不足，暂不能判断 gap。"]


def render_ablation(rows: list[dict[str, Any]]) -> list[str]:
    subset = [r for r in rows if r.get("suite") == "h1_ablation"]
    if not subset:
        return ["- h1_ablation 尚无结果。"]
    grouped = group(subset)
    lines = render_table(grouped)
    lines.append("")
    lines.extend(render_gaps(grouped))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--strict-root", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    rows = load_strict(args.strict_root)
    upstream_counts = load_upstream(args.upstream_root)
    suites = sorted({r.get("suite", "unknown") for r in rows})

    out = args.report_dir / f"v7_h1_auto_analysis_{args.stamp}.md"
    lines: list[str] = []
    lines.append("# V7-H1 自动实验分析报告")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## 结论摘要")
    lines.append("")
    if not rows:
        lines.append("当前尚未发现 H1 strict eval 结果。实验可能仍在运行，或 strict evaluator 尚未执行。")
    else:
        all_grouped = group(rows)
        lines.extend(render_gaps(all_grouped))
        lines.append("")
        budget_vals = metric_values(rows, "avg_budget_topk_h1")
        lines.append(f"- H1 已收集 strict-eval 记录 {len(rows)} 条，平均预算 top-k：{stats(budget_vals)}。")
        lines.append("- 注意：H1 strict 指标是非饱和选择行为诊断，不等同于标准 RAG F1/EM。")
    lines.append("")
    lines.append("## Upstream 完成数")
    lines.append("")
    if upstream_counts:
        for suite, count in sorted(upstream_counts.items()):
            lines.append(f"- `{suite}`: {count} run")
    else:
        lines.append("- 尚未发现 upstream run_metadata。")
    lines.append("")

    if rows:
        lines.append("## 全局 strict 指标")
        lines.append("")
        lines.extend(render_table(group(rows)))
        lines.append("")
        for suite in suites:
            lines.append(f"## Suite: {suite}")
            lines.append("")
            suite_grouped = group(rows, suite=suite)
            lines.extend(render_table(suite_grouped))
            lines.append("")
            lines.extend(render_gaps(suite_grouped))
            lines.append("")
        lines.append("## Ablation 判断")
        lines.append("")
        lines.extend(render_ablation(rows))
        lines.append("")
        lines.append("## 下一步建议")
        lines.append("")
        lines.append("1. 若 `agent_oracle_v7h1` 有明显正 gap 而 fair agent 没有，优先学习 oracle 的 block/action pattern，而不是继续调评估。")
        lines.append("2. 若 `agent_tail_v7h1` 只在 rare-domain suite 有正 gap，应把 rarity/hard-client 信号接入更细粒度的 block-value predictor。")
        lines.append("3. 若 ablation 后 full agent 与 no-memory/no-rarity 接近，说明当前 memory/rarity 只改变日志标签，未实质改变选择行为，需要改 selection policy。")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    latest = args.report_dir / "v7_h1_auto_analysis_latest.md"
    latest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
